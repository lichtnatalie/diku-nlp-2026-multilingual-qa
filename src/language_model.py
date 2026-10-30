"""Week 38 (project Section 4): a word-level n-gram language model of questions.

Design, in the order the project asks for it
--------------------------------------------
Tokenisation.  The Week 36 tokeniser (`tokenisation.unicode_word_tokenise`),
after a light, standard orthographic normalisation for Arabic
(`normalise_arabic`): ASCII digits, no short-vowel marks or tatweel, one
alef for the hamza-carrying forms, and ya for alef maqsura.  Questions are
typed by many people, and the same word appears as "أين" and "اين" or as
"ماهى" and "ماهي"; left alone, each spelling is its own rare type.  The
normalisation is applied before counting and is part of the model, so every
perplexity below is over normalised tokens.

Sentence boundaries.  Each question is one sequence, padded on the left with
n-1 `<s>` and closed with `</s>`.  `</s>` is predicted and counted in the
token total T; `<s>` is never predicted and is not counted (J&M 2026, Sec. 3.3,
footnote 2).

Unknown tokens.  The vocabulary is every word seen at least `min_count`
(default 2) times in the TRAINING questions.  Every other word, in training
and in evaluation, is replaced by `<unk>`.  Mapping the training hapaxes to
`<unk>` is what gives `<unk>` a probability learned from data (the recipe in
the course's language-modelling lecture).  The price is that perplexity is
then measured over a vocabulary in which every rare word is one cheap token,
so the unknown rate is always reported next to it.

Model.  Jelinek-Mercer interpolation, in its recursive form:

    P_1(w)       = c(w) / N                                   (MLE unigram)
    P_k(w | h_k) = lam_k * c(h_k, w) / c(h_k) + (1 - lam_k) * P_{k-1}(w | h_{k-1})

where h_k is the last k-1 words.  If the history h_k was never seen in
training, P_k = P_{k-1}: the unseen component carries no information, and
dropping it keeps the distribution normalised.  Every word in the vocabulary
(including `<unk>` and `</s>`) has a positive unigram count, so no probability
is ever zero and perplexity is always finite.  The lam_k are tuned on a
held-out slice of the training questions, never on validation.

An add-k bigram is included as the course's other smoothing method.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

from .tokenisation import normalise_digits, strip_arabic_diacritics, unicode_word_tokenise

BOS, EOS, UNK = "<s>", "</s>", "<unk>"

_ALEF_FORMS = re.compile("[آأإٱ]")  # آ أ إ ٱ


def normalise_arabic(text: str) -> str:
    """Orthographic normalisation applied before tokenising a question."""
    text = strip_arabic_diacritics(normalise_digits(text or ""))
    text = _ALEF_FORMS.sub("ا", text)        # -> ا
    return text.replace("ى", "ي")        # ى -> ي


def tokenise_question(text: str) -> List[str]:
    return unicode_word_tokenise(normalise_arabic(text))


def build_vocabulary(sentences: Iterable[Sequence[str]], min_count: int = 2) -> set:
    counts = Counter(w for s in sentences for w in s)
    return {w for w, n in counts.items() if n >= min_count}


def map_unknown(sentences: Iterable[Sequence[str]], vocab: set) -> List[List[str]]:
    return [[w if w in vocab else UNK for w in s] for s in sentences]


def events(sentence: Sequence[str], order: int) -> List[Tuple[Tuple[str, ...], str]]:
    """(history, target) pairs for one sentence: n-1 BOS on the left, EOS last."""
    padded = [BOS] * (order - 1) + list(sentence) + [EOS]
    return [(tuple(padded[i - order + 1:i]), padded[i])
            for i in range(order - 1, len(padded))]


class InterpolatedNGramLM:
    """Recursive Jelinek-Mercer interpolation of MLE n-gram estimates.

    `lambdas[k]` is the weight of the order-k MLE estimate against the
    interpolated order-(k-1) model, for k = 2..order.
    """

    def __init__(self, order: int = 3, lambdas: Dict[int, float] | None = None):
        if order < 1:
            raise ValueError("order must be at least 1")
        self.order = order
        self.lambdas = dict(lambdas or {k: 0.5 for k in range(2, order + 1)})
        for k in range(2, order + 1):
            if not 0.0 <= self.lambdas.get(k, -1) < 1.0:
                raise ValueError(f"lambda for order {k} must be in [0, 1)")
        self.ngram = [Counter() for _ in range(order + 1)]    # [k][(h, w)]
        self.history = [Counter() for _ in range(order + 1)]  # [k][h]
        self.n_events = 0
        self.targets: set = set()

    def fit(self, sentences: Iterable[Sequence[str]]):
        """Count events. Sentences must already be mapped to the vocabulary."""
        for s in sentences:
            for h, w in events(s, self.order):
                self.n_events += 1
                self.targets.add(w)
                for k in range(1, self.order + 1):
                    hk = h[len(h) - (k - 1):] if k > 1 else ()
                    self.ngram[k][(hk, w)] += 1
                    self.history[k][hk] += 1
        return self

    def prob(self, word: str, history: Sequence[str]) -> float:
        h = tuple(history)[-(self.order - 1):] if self.order > 1 else ()
        p = self.ngram[1][((), word)] / self.n_events
        for k in range(2, self.order + 1):
            hk = h[len(h) - (k - 1):]
            if len(hk) < k - 1:
                break
            denom = self.history[k][hk]
            if denom == 0:
                continue   # unseen history: keep the lower-order estimate
            lam = self.lambdas[k]
            p = lam * self.ngram[k][(hk, word)] / denom + (1.0 - lam) * p
        return p

    def sentence_logprobs(self, sentence: Sequence[str]) -> List[float]:
        return [math.log(self.prob(w, h)) for h, w in events(sentence, self.order)]


class AddKBigramLM:
    """Add-k (Laplace for k = 1) bigram, the course's first smoothing method."""

    order = 2

    def __init__(self, k: float = 1.0):
        self.k = k
        self.ngram: Counter = Counter()
        self.history: Counter = Counter()
        self.targets: set = set()

    def fit(self, sentences: Iterable[Sequence[str]], vocab: set):
        # Every word that can be predicted: the vocabulary, <unk> and </s>.
        self.targets = set(vocab) | {UNK, EOS}
        for s in sentences:
            for h, w in events(s, 2):
                self.ngram[(h, w)] += 1
                self.history[h] += 1
        return self

    def prob(self, word: str, history: Sequence[str]) -> float:
        h = tuple(history)[-1:]
        return ((self.ngram[(h, word)] + self.k)
                / (self.history[h] + self.k * len(self.targets)))

    def sentence_logprobs(self, sentence: Sequence[str]) -> List[float]:
        return [math.log(self.prob(w, h)) for h, w in events(sentence, 2)]


class UniformLM:
    order = 1

    def __init__(self, targets: set):
        self.targets = set(targets)

    def prob(self, word: str, history: Sequence[str]) -> float:
        return 1.0 / len(self.targets) if word in self.targets else 0.0

    def sentence_logprobs(self, sentence: Sequence[str]) -> List[float]:
        return [math.log(self.prob(w, h)) for h, w in events(sentence, 1)]


def perplexity(model, sentences: Sequence[Sequence[str]]) -> Tuple[float, int]:
    """Per-token perplexity over all sentences pooled, and the token count T.

    T counts every word and one </s> per sentence; <s> is never counted.
    """
    total, n = 0.0, 0
    for s in sentences:
        lp = model.sentence_logprobs(s)
        total += sum(lp)
        n += len(lp)
    return math.exp(-total / n), n


def event_seen(model, history, word) -> bool:
    """Was this model's highest-order (history, word) event seen in training?

    The split that shows what a smoothing choice buys: two models with the same
    counts differ only in how they treat events they never saw.
    """
    order = getattr(model, "order", 1)
    if order == 1 or not hasattr(model, "ngram"):
        return True          # the uniform and unigram models have no history
    h = tuple(history)[-(order - 1):]
    counts = model.ngram[order] if isinstance(model.ngram, list) else model.ngram
    return counts[(h, word)] > 0


def perplexity_by_event(model, sentences):
    """Per-token perplexity split by whether the event was seen in training."""
    out = {}
    for seen in (True, False):
        lps = [lp for s in sentences
               for (h, w), lp in zip(events(s, getattr(model, "order", 1)),
                                     model.sentence_logprobs(s))
               if event_seen(model, h, w) is seen]
        out[seen] = (math.exp(-sum(lps) / len(lps)) if lps else float("nan"), len(lps))
    return out


def shuffle_words(sentences, rng):
    """Shuffle the words inside each sentence, keeping the sentence itself.

    A model that uses no context scores a shuffled corpus exactly as it scores
    the original one; a model that uses context does not.
    """
    out = []
    for s in sentences:
        words = list(s)
        rng.shuffle(words)
        out.append(words)
    return out
