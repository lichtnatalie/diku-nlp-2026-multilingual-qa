"""Character-level answer offsets <-> token-level BIO labels (project Section 3).

The dataset stores an answer as a character offset into the English context.
A sequence labeller needs one label per context token, so the conversion has to
happen somewhere, and it has to be reversible: the lecture's rule (slides cell
70) is "test alignment in both directions; map predictions back to the annotated
units; evaluate spans only after that mapping".

ONE TOKENISER PER SIDE OF THE EXAMPLE
-------------------------------------
The question and the context are in different languages and are used for
different things, so they do not have to share a tokeniser, and here they
deliberately do not:

  * QUESTION side (Week 36 statistics, and the question features of
    span_models.py) uses `tokenisation.unicode_word_tokenise`, which treats
    letters, digits and combining marks as word characters and discards
    everything else.  That is the right call for Arabic, Korean and Telugu:
    Telugu writes its vowels as combining marks, so a tokeniser that drops them
    shatters a single orthographic word.
  * CONTEXT side (this module) uses `tokenise_with_offsets` below, which keeps
    the same word runs but ALSO emits every punctuation or symbol character as
    a token of its own.  The context is always English, so nothing is lost by
    doing so, and a great deal is gained: answer spans in this dataset
    routinely end on a bracket or a full stop, and a tokeniser with no token
    for ")" cannot reproduce "British Broadcasting Corporation (BBC)" at all.

Two conventions are then fixed, and the lecture separates them:

  * INTERNAL (training target).  A context token is inside the answer if it
    OVERLAPS the gold character span at all.  This always produces a
    well-defined BIO sequence, even when the gold span does not land on token
    boundaries.
  * EVALUATION.  Exact-span match is scored against the ORIGINAL gold character
    span - never against the token-aligned version.

With punctuation kept, the round trip character span -> BIO -> character span
is lossless for every gold span whose endpoints fall on a token boundary, which
is all of them except the handful where the stored offset cuts a word in half.
Those are annotation errors in the source ("eremy Bentham" for "Jeremy
Bentham"), not tokenisation failures, and `alignment_audit` counts them so that
the residual ceiling is reported rather than defined away.

Tagging scheme is IOB2: every span starts with B-ANS.  There is at most one
answer span per example, so I-ANS may only follow B-ANS or I-ANS.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .tokenisation import _is_word_char, normalise_digits

# The three labels. IOB2, one entity type ("ANS"), so 2n+1 = 3 tags.
OUTSIDE = "O"
BEGIN = "B-ANS"
INSIDE = "I-ANS"
LABELS = (OUTSIDE, BEGIN, INSIDE)

# Alignment statuses reported by `char_span_to_bio`. Used as the row labels of
# the audit table in the report, so the names are meant to be read by a human.
STATUS_UNANSWERABLE = "unanswerable"
STATUS_EXACT = "exact"
STATUS_TRIMMED = "trimmed"
STATUS_WIDENED = "widened"
STATUS_NO_TOKENS = "no_word_tokens"
STATUSES = (
    STATUS_UNANSWERABLE,
    STATUS_EXACT,
    STATUS_TRIMMED,
    STATUS_WIDENED,
    STATUS_NO_TOKENS,
)


def tokenise_with_offsets(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Tokenise the English context, returning each token's character offsets.

    A token is either a maximal run of word characters (the same predicate
    Week 36 uses, `tokenisation._is_word_char`) or a single punctuation or
    symbol character. Whitespace separates and is never a token.

    Punctuation gets one token per character rather than per run so that a gold
    span can end anywhere a non-space character ends. That is what makes the
    conversion lossless: "Corporation (BBC)" tokenises as
    ["Corporation", "(", "BBC", ")"] and reconstructs exactly.

    Dropping the punctuation tokens recovers `unicode_word_tokenise(text)`
    exactly, so the two tokenisers still describe the same words; the context
    one just also has somewhere to put the brackets. A test asserts this.

    The offsets are what make the conversion reversible; nothing in this module
    ever searches the context for the answer string.
    """
    tokens: List[str] = []
    offsets: List[Tuple[int, int]] = []
    if not text:
        return tokens, offsets
    current: List[str] = []
    start = 0
    for i, ch in enumerate(text):
        if _is_word_char(ch):
            if not current:
                start = i
            current.append(ch)
            continue
        # A word run just ended (if one was open); flush it before the
        # separator character is considered.
        if current:
            tokens.append("".join(current))
            offsets.append((start, i))
            current = []
        if not ch.isspace():
            tokens.append(ch)
            offsets.append((i, i + 1))
    if current:
        tokens.append("".join(current))
        offsets.append((start, len(text)))
    return tokens, offsets


def char_span_to_bio(
    context: str,
    answerable: bool,
    answer_start: Optional[int] = None,
    answer_text: str = "",
) -> Dict[str, object]:
    """Convert one row's stored character offsets into BIO labels.

    Note the signature: answerability is passed in EXPLICITLY and is what
    decides the target. Week 36 found that all 527 unanswerable rows carry a
    non-empty `answer` string (389 of them literally "no"), so inferring
    answerability from `answer == ""` would emit a labelled span where the
    correct output is the empty span. Never gate on the answer text.

    Raises ValueError only when the stored data is internally inconsistent
    (offset out of range, or the stored text not matching the stored offset).
    A span that merely fails to land on token boundaries is NOT an error - it
    gets a status and is handled by the documented convention.
    """
    tokens, offsets = tokenise_with_offsets(context)
    labels = [OUTSIDE] * len(tokens)
    info: Dict[str, object] = {
        "tokens": tokens,
        "offsets": offsets,
        "labels": labels,
        "status": STATUS_UNANSWERABLE,
        "gold_char_span": None,
        "token_char_span": None,
        "token_index_span": None,
    }

    # Unanswerable: the target is all-O and the answer fields are not consulted.
    if not answerable:
        return info

    if answer_start is None:
        raise ValueError("answerable row with answer_start=None")
    answer_end = answer_start + len(answer_text)
    if answer_start < 0 or answer_end > len(context):
        raise ValueError(f"answer span [{answer_start},{answer_end}) outside context")
    if context[answer_start:answer_end] != answer_text:
        raise ValueError("stored answer text does not match the stored offset")

    info["gold_char_span"] = (answer_start, answer_text)

    # A token belongs to the answer if it overlaps the gold span at all.
    covered = [i for i, (s, e) in enumerate(offsets) if s < answer_end and e > answer_start]
    if not covered:
        # No token overlaps the span at all. Since punctuation is tokenised too,
        # this now requires a gold span made entirely of whitespace.
        info["status"] = STATUS_NO_TOKENS
        return info

    first, last = covered[0], covered[-1]
    tok_start, tok_end = offsets[first][0], offsets[last][1]
    info["token_char_span"] = (tok_start, context[tok_start:tok_end])
    info["token_index_span"] = (first, last + 1)

    labels[first] = BEGIN
    for i in covered[1:]:
        labels[i] = INSIDE

    # Classify how the token span relates to the gold span, for the audit table.
    if tok_start == answer_start and tok_end == answer_end:
        info["status"] = STATUS_EXACT
    elif tok_start >= answer_start and tok_end <= answer_end:
        # The gold span had characters at one or both edges that no token
        # carries. With punctuation tokenised this means whitespace only.
        info["status"] = STATUS_TRIMMED
    else:
        # The gold span cuts a token in half, so the token span is WIDER than
        # the gold one. These are spans that do not align to token boundaries
        # under this tokeniser; the ones inspected are annotation errors in the
        # source ("eremy Bentham" for "Jeremy Bentham"), but the status is
        # named for what is measured, not for the cause.
        info["status"] = STATUS_WIDENED
    return info


def bio_to_char_span(
    context: str,
    offsets: Sequence[Tuple[int, int]],
    labels: Sequence[str],
) -> Tuple[Optional[int], str]:
    """Map a BIO label sequence back to a character span. The inverse direction.

    Returns (None, "") for an all-O sequence, which is the correct output for an
    unanswerable question. If the sequence somehow contains more than one span,
    the first one is returned - decoding is supposed to prevent that, and
    `find_spans` is the function to call when you want to see all of them.
    """
    if len(offsets) != len(labels):
        raise ValueError(f"{len(offsets)} offsets but {len(labels)} labels")
    spans = find_spans(labels)
    if not spans:
        return None, ""
    first, last = spans[0]
    start = offsets[first][0]
    end = offsets[last - 1][1]
    return start, context[start:end]


def find_spans(labels: Sequence[str]) -> List[Tuple[int, int]]:
    """All maximal spans in a label sequence, as half-open token index pairs.

    This is the project's ONE span extractor, and the choice matters: the
    lecture's `get_spans` discards an orphan I-ANS while lab_2's `bio_spans`
    promotes it to a span start, and the two give different P/R/F on identical
    predictions. We follow lab_2: an I-ANS with no preceding B-ANS/I-ANS starts
    a span. That is the more forgiving convention, and it means a BIO-repair
    step would be a no-op at the span level, so we do not implement one.
    """
    spans: List[Tuple[int, int]] = []
    start = None
    for i, label in enumerate(labels):
        if label == BEGIN:
            if start is not None:
                spans.append((start, i))
            start = i
        elif label == INSIDE:
            if start is None:
                start = i
        else:
            if start is not None:
                spans.append((start, i))
                start = None
    if start is not None:
        spans.append((start, len(labels)))
    return spans


def valid_bio_transition(previous: Optional[str], current: str) -> bool:
    """Is `current` a legal label to follow `previous` under IOB2?

    Only I-ANS is constrained: it must continue an existing span. `previous` is
    None at the start of the sequence, where I-ANS is illegal.
    """
    if current != INSIDE:
        return True
    return previous in (BEGIN, INSIDE)


def assert_scheme(label_sequences) -> int:
    """Fail loudly if any gold sequence violates the IOB2 assumption.

    Week 37's lab showed what happens when this assumption is wrong and nobody
    checks: run the same constrained decoder on IOB1 data and span F1 collapses
    from 0.78 to 0.011 while precision stays at 0.82 - the model is silenced,
    not broken, so the failure is invisible in the precision column. Returns the
    number of violations (always 0, or it raises).
    """
    violations = 0
    for seq in label_sequences:
        previous = None
        for label in seq:
            if not valid_bio_transition(previous, label):
                violations += 1
            previous = label
    if violations:
        raise AssertionError(f"gold data violates the assumed IOB2 scheme ({violations} times)")
    return violations


def encode_split(df: pd.DataFrame) -> List[Dict[str, object]]:
    """Run char_span_to_bio over a whole split, keeping row identity.

    Returns one dict per row with the tokens, offsets, gold labels, status and
    both span representations, so downstream code never has to re-tokenise.
    """
    encoded = []
    for row in df.itertuples():
        info = char_span_to_bio(
            context=row.context,
            answerable=bool(row.answerable),
            answer_start=int(row.answer_start) if bool(row.answerable) else None,
            answer_text=row.answer if bool(row.answerable) else "",
        )
        info["example_id"] = row.example_id
        info["lang"] = row.lang
        info["answerable"] = bool(row.answerable)
        info["question"] = row.question
        info["context"] = row.context
        # Carried so that `unanswerable_check` can count the unanswerable rows
        # whose stored answer string is non-empty. Nothing else may read it:
        # the labels come from `answerable`, never from this field.
        info["answer"] = "" if row.answer is None else str(row.answer)
        encoded.append(info)
    return encoded


def alignment_audit(encoded: Sequence[Dict[str, object]], split: str) -> pd.DataFrame:
    """Count alignment statuses per language - the table Section 3 asks for.

    The `exact` rate is the ceiling on exact-span match AMONG ANSWERABLE
    EXAMPLES, and nothing else: a `trimmed` or `widened` row can never be
    reproduced character-for-character, because the token boundaries do not fall
    where the stored offsets do. Unanswerable examples have no span to align and
    are always reachable (the model has only to abstain), so the ceiling on
    OVERALL exact match is higher than this number - `overall_ceiling` below is
    that figure. Reporting both next to the model scores is the honest
    presentation.
    """
    rows = []
    langs = sorted({e["lang"] for e in encoded})
    for subset in ["overall"] + langs:
        items = [e for e in encoded if subset == "overall" or e["lang"] == subset]
        answerable = [e for e in items if e["answerable"]]
        counts = {s: sum(1 for e in answerable if e["status"] == s) for s in STATUSES}
        n_ans = len(answerable)
        rows.append({
            "split": split,
            "subset": subset,
            "n": len(items),
            "n_answerable": n_ans,
            "n_unanswerable": len(items) - n_ans,
            "exact": counts[STATUS_EXACT],
            "trimmed": counts[STATUS_TRIMMED],
            "widened": counts[STATUS_WIDENED],
            "no_word_tokens": counts[STATUS_NO_TOKENS],
            # Ceiling on exact match among answerable rows only.
            "exact_rate": counts[STATUS_EXACT] / n_ans if n_ans else 0.0,
            "usable_rate": (counts[STATUS_EXACT] + counts[STATUS_TRIMMED]) / n_ans if n_ans else 0.0,
            # Ceiling on exact match over ALL rows: every unanswerable row is
            # reachable by abstaining, so it adds to the numerator in full.
            "overall_ceiling": (counts[STATUS_EXACT] + (len(items) - n_ans)) / len(items)
            if items else 0.0,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Automatic checks required by Section 3.
# --------------------------------------------------------------------------
# Section 3 asks for checks covering at least: an answer at character 0, a
# multi-token answer, punctuation adjacent to an answer, and an unanswerable
# example. Three more are added because Week 36 and the Week 37 lab both showed
# where a plausible implementation silently goes wrong.

# Each case is (name, required_by_brief, context, answerable, answer_text,
# expected_non_O_labels). `required_by_brief` marks the four cases Section 3
# names explicitly, so the report can state that all four are covered and the
# table shows which rows they are.
_CHECK_CASES = [
    ("answer at character 0", True,
     "Ada Lovelace wrote the first algorithm.", True, "Ada Lovelace",
     [BEGIN, INSIDE]),
    ("multi-token answer", True,
     "The event was on 1 July 2000.", True, "1 July 2000",
     [BEGIN, INSIDE, INSIDE]),
    ("punctuation adjacent to the answer", True,
     "He worked at Bell Labs (New Jersey) for years.", True, "New Jersey",
     [BEGIN, INSIDE]),
    ("unanswerable example", True,
     "Ada Lovelace wrote the first algorithm.", False, "no",
     []),
    # Four more, added because Week 36 and the Week 37 lab showed where a
    # plausible implementation goes wrong quietly.
    ("punctuation inside the answer", False,
     "The British Broadcasting Corporation (BBC) is a broadcaster.", True,
     "Broadcasting Corporation (BBC)",
     [BEGIN, INSIDE, INSIDE, INSIDE, INSIDE]),
    ("answer followed by a comma", False,
     "She moved to Aarhus, Denmark in 1994.", True, "Aarhus",
     [BEGIN]),
    ("answer ends the context", False,
     "The capital of Denmark is Copenhagen.", True, "Copenhagen",
     [BEGIN]),
    ("single-token answer", False,
     "The colour is blue today.", True, "blue", [BEGIN]),
]


def conversion_checks() -> pd.DataFrame:
    """Run the Section 3 checks and return one row per case, pass/fail.

    Every case goes character span -> BIO -> character span and asserts three
    things: the labels are the expected IOB2 sequence, the round trip recovers
    the token-aligned span, and - now that the context tokeniser keeps
    punctuation - the round trip also recovers the ORIGINAL gold string
    character for character. A case passes only if all three hold.

    Returning a table rather than just asserting means the result can go
    straight into the report. `run_week37.py` still raises if any row fails, so
    a failure stops the pipeline instead of being written to a CSV nobody reads.
    """
    rows = []
    for name, required, context, answerable, answer, expected in _CHECK_CASES:
        start = context.index(answer) if (answerable and answer) else None
        # `context.index` is fine HERE because these are hand-written fixtures
        # with a unique answer. Never do this on the real data: 1,082 rows have
        # their answer string more than once in the context.
        info = char_span_to_bio(context, answerable, start, answer if answerable else "")
        labels = info["labels"]
        non_o = [l for l in labels if l != OUTSIDE]
        back_start, back_text = bio_to_char_span(context, info["offsets"], labels)

        if answerable:
            want_start, want_text = info["token_char_span"]
            round_trip_ok = (back_start == want_start and back_text == want_text)
            recovers_gold = (back_start == start and back_text == answer)
        else:
            # The correct output for an unanswerable question is the empty span,
            # and "recovering the gold" means recovering exactly that.
            round_trip_ok = (back_start is None and back_text == "")
            recovers_gold = round_trip_ok

        rows.append({
            "check": name,
            "required_by_brief": bool(required),
            "status": info["status"],
            "labels_ok": non_o == expected,
            "round_trip_ok": bool(round_trip_ok),
            "recovers_gold": bool(recovers_gold),
            "recovered": back_text,
            "passed": bool(non_o == expected and round_trip_ok and recovers_gold),
        })
    return pd.DataFrame(rows)


def unanswerable_check(encoded) -> Dict[str, int]:
    """Check over the REAL data that no unanswerable row produces a span.

    This is the seventh check and the one Week 36 says is most likely to fail in
    a plausible implementation, because every unanswerable row in this dataset
    carries a non-empty `answer` string. Returns counts so the report can quote
    them.
    """
    unanswerable = [e for e in encoded if not e["answerable"]]
    with_span = sum(1 for e in unanswerable if any(l != OUTSIDE for l in e["labels"]))
    return {
        "n_unanswerable": len(unanswerable),
        "n_with_non_O_label": with_span,
        "n_with_non_empty_answer_string": sum(
            1 for e in unanswerable if str(e.get("answer", "")) != ""
        ),
    }
