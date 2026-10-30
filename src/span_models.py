"""Span predictors for project Section 3.

Three systems, all of which output a BIO label sequence over the context tokens
produced by `spans.tokenise_with_offsets`, so that `evaluation.span_report`
scores them identically:

  * `EmptySpanBaseline` - always predicts the empty span. Section 3 says an
    empty-output baseline is sufficient as the simple comparison, and it is the
    span-prediction analogue of the always-O baseline from the lecture.
  * `LexicalOverlapSpanBaseline` - the overlap baseline Section 3 permits,
    under the constraints it imposes: context tokens are selected using ONLY
    overlap with the question, the best contiguous run becomes the span, and
    nothing in it ever touches the gold answer text or offsets.
  * `QuestionConditionedLabeller` - the required question-conditioned sequence
    labeller. A log-linear token classifier over {O, B-ANS, I-ANS} with
    constrained beam decoding, i.e. the architecture the Week 37 lecture builds
    (slides cells 17-28, 71) rather than an RNN or a Transformer, which are the
    subjects of later lectures.

ACKNOWLEDGEMENT
---------------
The skeleton of `QuestionConditionedLabeller` - a DictVectorizer over per-token
feature dicts, an SGDClassifier with log loss, and a beam decoder that only
extends a path with a legal transition - is adapted from Course Lab 2, down to
its hyperparameters (alpha=1e-5, max_iter=30, `balanced` class weights, beam
size 4). The lab uses balanced weights too; they matter here because answer
tokens are ~3% of the data. What is mine:
the question-conditioned features and their ablation, the single-span decoding
constraint, the no-answer score and abstention, and the two span baselines.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier

from . import config
from .spans import BEGIN, INSIDE, OUTSIDE, find_spans
from .tokenisation import normalise_digits, unicode_word_tokenise

# Anchors are the parts of a non-English question that can survive into an
# English context untranslated. Same definition as Week 36's CrossLingualAnchorRule,
# reused here as FEATURES rather than as a standalone decision rule.
_LATIN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_DIGIT_RE = re.compile(r"\d+")
_MIN_LATIN = 3
_MIN_DIGITS = 2


def question_features(question: str, lang: str) -> Dict[str, object]:
    """Precompute everything about the question that the token features need.

    Done once per example rather than once per token, because a context averages
    ~100 tokens and none of this changes along the sequence.
    """
    normalised = normalise_digits(question or "")
    tokens = [t.casefold() for t in unicode_word_tokenise(normalised)]
    anchors = set()
    for match in _LATIN_RE.findall(normalised):
        if len(match) >= _MIN_LATIN:
            anchors.add(match.casefold())
    for match in _DIGIT_RE.findall(normalised):
        if len(match) >= _MIN_DIGITS:
            anchors.add(match)
    return {
        "lang": lang,
        "tokens": set(tokens),
        "anchors": anchors,
        # The leading token of a question is usually the interrogative word
        # ("what"/"where"/"who"), which is the strongest available cue to what
        # KIND of thing the answer should be.
        "first": tokens[0] if tokens else "<empty>",
        "len_bucket": min(len(tokens), 10),
    }


def token_features(tokens: Sequence[str], i: int, qf: Dict[str, object],
                   use_question: bool = True) -> Dict[str, float]:
    """Features for context token `i`, conditioned on the question.

    Section 3 requires that the question influence the label of EVERY context
    token, so the question-derived features below are split into two kinds, and
    the distinction is worth being honest about in the report:

      * Example-level (`q_lang`, `q_first`, `q_len`): identical at every token
        of an example. Because they are constant along the sequence they shift
        the O-versus-answer balance uniformly, so they act as a per-question
        prior on whether and how long a span should be - they modulate
        abstention, but they do not by themselves choose WHICH token.
      * Token-specific (`q_overlap`, `q_anchor`, `q_anchor_near`) and the
        conjunctions at the end: these differ from token to token, so they are
        what lets the question change which token is selected.

    Nothing here reads the gold answer text or its offsets, which Section 3
    forbids for the overlap baseline and which would be cheating here too.
    """
    word = tokens[i]
    lower = word.casefold()
    normalised = normalise_digits(lower)
    feats: Dict[str, float] = {
        "bias": 1.0,
        f"w={lower}": 1.0,
        f"pre2={lower[:2]}": 1.0,
        f"suf2={lower[-2:]}": 1.0,
        f"suf3={lower[-3:]}": 1.0,
        f"istitle={word.istitle()}": 1.0,
        f"isupper={word.isupper()}": 1.0,
        f"isdigit={word.isdigit()}": 1.0,
        f"isyear={word.isdigit() and len(word) == 4}": 1.0,
        # The context tokeniser emits every punctuation character as its own
        # token, so the model needs to be able to generalise over them rather
        # than learning each bracket and comma separately.
        f"ispunct={len(word) == 1 and not word.isalnum()}": 1.0,
        # Relative position, bucketed into tenths. Answers are not uniformly
        # distributed through a context paragraph.
        f"pos={min(int(10 * i / max(len(tokens), 1)), 9)}": 1.0,
    }

    if i == 0:
        feats["BOS"] = 1.0
    else:
        previous = tokens[i - 1]
        feats[f"prev={previous.casefold()}"] = 1.0
        feats[f"prev_istitle={previous.istitle()}"] = 1.0
    if i == len(tokens) - 1:
        feats["EOS"] = 1.0
    else:
        following = tokens[i + 1]
        feats[f"next={following.casefold()}"] = 1.0
        feats[f"next_istitle={following.istitle()}"] = 1.0

    # Everything below is question-derived. `use_question=False` drops all of
    # it, which gives the ablation baseline: same model, same features, no
    # knowledge of what was asked.
    if not use_question:
        return feats

    # Example-level question features.
    feats[f"q_lang={qf['lang']}"] = 1.0
    feats[f"q_first={qf['first']}"] = 1.0
    feats[f"q_len={qf['len_bucket']}"] = 1.0

    # Token-specific question features.
    overlap = normalised in qf["tokens"]
    anchor = normalised in qf["anchors"]
    feats[f"q_overlap={overlap}"] = 1.0
    feats[f"q_anchor={anchor}"] = 1.0
    near = 0
    for j in range(max(0, i - 3), min(len(tokens), i + 4)):
        if normalise_digits(tokens[j].casefold()) in qf["anchors"]:
            near += 1
    feats[f"q_anchor_near={min(near, 3)}"] = 1.0

    # Conjunctions: these are the features that let the question change WHICH
    # token wins, e.g. a "where" question should prefer capitalised tokens.
    feats[f"q_first={qf['first']}&istitle={word.istitle()}"] = 1.0
    feats[f"q_first={qf['first']}&isdigit={word.isdigit()}"] = 1.0
    feats[f"q_lang={qf['lang']}&istitle={word.istitle()}"] = 1.0
    return feats


class EmptySpanBaseline:
    """Always abstains. The baseline Section 3 says is sufficient.

    Scores 0 on token F1 by construction, and scores the unanswerable rate on
    exact match - which is exactly why both metrics have to be reported.
    """

    name = "empty-span baseline"

    def predict(self, encoded) -> List[List[str]]:
        """One all-O sequence per example."""
        return [[OUTSIDE] * len(e["tokens"]) for e in encoded]


class LexicalOverlapSpanBaseline:
    """Longest run of context tokens that also occur in the question.

    Section 3's constraints, all honoured: selection uses only question overlap,
    the fixed preprocessing is casefolding plus the project's digit
    normalisation, the best contiguous run becomes the span, and the gold answer
    is never consulted. Ties are broken by the earliest run.

    Expect this to abstain nearly always on this dataset: the questions are
    Arabic, Korean and Telugu while the contexts are English, so there is very
    little shared vocabulary to overlap on. That degeneracy is the point of
    reporting it.
    """

    name = "lexical-overlap span baseline"

    def predict(self, encoded) -> List[List[str]]:
        """One BIO sequence per example, all-O when nothing overlaps."""
        out = []
        for e in encoded:
            qf = question_features(e["question"], e["lang"])
            tokens = e["tokens"]
            hits = [normalise_digits(t.casefold()) in qf["tokens"] for t in tokens]
            best = self._longest_run(hits)
            labels = [OUTSIDE] * len(tokens)
            if best is not None:
                start, end = best
                labels[start] = BEGIN
                for k in range(start + 1, end):
                    labels[k] = INSIDE
            out.append(labels)
        return out

    @staticmethod
    def _longest_run(hits: Sequence[bool]):
        """Longest contiguous True run as (start, end), or None if there is none."""
        best = None
        best_len = 0
        start = None
        for i, hit in enumerate(hits):
            if hit and start is None:
                start = i
            elif not hit and start is not None:
                if i - start > best_len:
                    best, best_len = (start, i), i - start
                start = None
        if start is not None and len(hits) - start > best_len:
            best = (start, len(hits))
        return best


# Decoder states for the single-span constraint. Extractive QA allows at most
# one answer, so the legal label sequences are exactly the regular language
#     O*  (B-ANS  I-ANS*)?  O*
# i.e. zero or one span: the all-O sequence (abstaining) is legal. That is
# strictly narrower than plain IOB2 validity, which would also permit B O B,
# i.e. two spans. Enforcing it during search rather than repairing afterwards
# makes the objective "highest-scoring legal sequence"; the beam searches for
# that sequence but does not guarantee to find it
# (tests/test_report_claims.py pins both statements).
_BEFORE, _WITHIN, _AFTER = 0, 1, 2


def _next_state(state: int, label: str) -> Optional[int]:
    """Successor state, or None if `label` is illegal in `state`."""
    if state == _BEFORE:
        if label == OUTSIDE:
            return _BEFORE
        if label == BEGIN:
            return _WITHIN
        return None                 # I-ANS cannot open a span (IOB2)
    if state == _WITHIN:
        if label == INSIDE:
            return _WITHIN
        if label == OUTSIDE:
            return _AFTER
        return None                 # a second B-ANS would be a second span
    return _AFTER if label == OUTSIDE else None


class QuestionConditionedLabeller:
    """Log-linear token classifier over {O, B-ANS, I-ANS}, decoded under constraints.

    This is the lecture's "transparent baseline" (slides cell 17): labels are
    scored independently at each position, and all of the label interaction
    lives in decoding, which the lecture keeps separate from encoding (cell 71).
    Three decoding choices are compared in the results table:

      * `argmax`  - independent per-token argmax, ignoring the constraint. It
        can emit several spans, so a post-hoc step keeps the highest-scoring
        one; this is the only decoding that needs such a step, and it is what
        the constrained variants are being compared against.
      * `greedy`  - the constraint applied left to right, beam size 1.
      * `beam`    - the same constraint searched with a beam of four.

    See the module docstring for what is adapted from Course Lab 2.
    """

    name = "question-conditioned labeller"

    def __init__(self, beam_size: int = 4, seed: int = config.SEED,
                 use_question: bool = True):
        self.beam_size = beam_size
        self.seed = seed
        # Set False to fit the ablation: identical model without any
        # question-derived feature.
        self.use_question = use_question
        self.vectoriser = None
        self.classifier = None
        self.classes_ = None

    def _featurise(self, encoded):
        """Flatten a split into one feature dict per context token."""
        rows = []
        lengths = []
        for e in encoded:
            qf = question_features(e["question"], e["lang"])
            tokens = e["tokens"]
            lengths.append(len(tokens))
            for i in range(len(tokens)):
                rows.append(token_features(tokens, i, qf, self.use_question))
        return rows, lengths

    def fit(self, encoded):
        """Fit on the gold BIO labels of one split.

        `class_weight="balanced"` matters a lot here: only about 3.4% of context
        tokens are inside an answer, so an unweighted fit would predict O
        everywhere and reproduce the empty baseline.
        """
        features, _ = self._featurise(encoded)
        labels = [l for e in encoded for l in e["labels"]]
        self.vectoriser = DictVectorizer(sparse=True)
        X = self.vectoriser.fit_transform(features)
        self.classifier = SGDClassifier(
            loss="log_loss",
            alpha=1e-5,
            class_weight="balanced",
            max_iter=30,
            random_state=self.seed,
        )
        self.classifier.fit(X, labels)
        self.classes_ = list(self.classifier.classes_)
        return self

    def token_log_probs(self, encoded):
        """Per-token log probabilities, split back into one array per example."""
        features, lengths = self._featurise(encoded)
        X = self.vectoriser.transform(features)
        flat = self.classifier.predict_log_proba(X)
        out = []
        offset = 0
        for n in lengths:
            out.append(flat[offset:offset + n])
            offset += n
        return out

    def predict(self, encoded, decoding: str = "beam", log_probs=None):
        """Predict BIO sequences. `decoding` is one of argmax/greedy/beam.

        Pass `log_probs` from `token_log_probs` to avoid re-featurising the
        split when several decoding strategies are compared.
        """
        sequences = []
        for scores in (log_probs if log_probs is not None else self.token_log_probs(encoded)):
            if decoding == "argmax":
                labels = [self.classes_[int(j)] for j in scores.argmax(axis=1)]
                # Only the unconstrained decoding can produce several spans, so
                # only it needs reducing to one.
                labels = self._keep_best_span(labels, scores)
            elif decoding == "greedy":
                labels = self._constrained_decode(scores, beam_size=1)
            elif decoding == "beam":
                labels = self._constrained_decode(scores, beam_size=self.beam_size)
            else:
                raise ValueError(f"unknown decoding {decoding!r}")
            sequences.append(labels)
        return sequences

    def _constrained_decode(self, scores: np.ndarray, beam_size: int) -> List[str]:
        """Highest-scoring label sequence in the language O* B-ANS I-ANS* O*.

        Scores are log probabilities, so partial paths are compared by SUMMING
        them (slides cell 105). A path carries its decoder state, and a label is
        only considered when `_next_state` accepts it, so every sequence the
        beam can return is legal by construction and the all-O sequence (pure
        abstention) is always among them. beam_size=1 is constrained greedy.
        """
        beam = [([], _BEFORE, 0.0)]
        for row in scores:
            candidates = []
            for labels, state, total in beam:
                for j, label in enumerate(self.classes_):
                    nxt = _next_state(state, label)
                    if nxt is None:
                        continue
                    candidates.append((labels + [label], nxt, total + float(row[j])))
            candidates.sort(key=lambda item: item[2], reverse=True)
            beam = candidates[:beam_size]
        return beam[0][0]

    def _keep_best_span(self, labels: List[str], scores: np.ndarray) -> List[str]:
        """Reduce an unconstrained sequence to at most one span.

        Used only after independent argmax. The kept span is the one with the
        highest mean answer-label log probability; the rest are blanked. The
        constrained decoders never call this, because they cannot emit a second
        span in the first place.
        """
        found = find_spans(labels)
        if len(found) <= 1:
            return labels
        best = None
        best_score = -np.inf
        for start, end in found:
            total = 0.0
            for i in range(start, end):
                j = self.classes_.index(labels[i])
                total += float(scores[i][j])
            mean = total / (end - start)
            if mean > best_score:
                best, best_score = (start, end), mean
        out = [OUTSIDE] * len(labels)
        out[best[0]] = BEGIN
        for i in range(best[0] + 1, best[1]):
            out[i] = INSIDE
        return out

    def no_answer_scores(self, encoded, predicted, log_probs=None) -> np.ndarray:
        """Section 1's no-answer score, adapted to this model.

        Defined as (score of the all-O sequence) - (score of the predicted
        sequence), both as summed token log probabilities. A HIGH value means
        abstaining is relatively more attractive, so a high score means NO
        ANSWER. Section 1 requires that direction to be stated explicitly.
        """
        out = []
        o_index = self.classes_.index(OUTSIDE)
        source = log_probs if log_probs is not None else self.token_log_probs(encoded)
        for scores, labels in zip(source, predicted):
            all_o = float(scores[:, o_index].sum())
            chosen = sum(float(scores[i][self.classes_.index(l)]) for i, l in enumerate(labels))
            out.append(all_o - chosen)
        return np.asarray(out)


def apply_abstention(predicted, scores: np.ndarray, threshold: float):
    """Blank out any prediction whose no-answer score is at or above `threshold`.

    Kept as a free function rather than a method so the same threshold logic can
    be applied to any model's output later in the project.
    """
    out = []
    for labels, score in zip(predicted, scores):
        if score >= threshold:
            out.append([OUTSIDE] * len(labels))
        else:
            out.append(list(labels))
    return out
