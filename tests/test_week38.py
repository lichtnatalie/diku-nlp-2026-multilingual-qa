"""Tests for Week 38: the question language model.

Most of these are the checks the course's language-model exercise do
(does the model define a probability distribution?) plus the bookkeeping that
decides what a perplexity number means: which tokens are counted, and what
happens to words and histories never seen in training.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import language_model as lm  # noqa: E402

# Jurafsky & Martin (2026), Sec. 3.1.2: the "I am Sam" corpus.
SAM = [s.split() for s in ("I am Sam", "Sam I am", "I do not like green eggs and ham")]


def _targets(model):
    return sorted(model.targets)


def test_normalisation_merges_spelling_variants():
    assert lm.normalise_arabic("أين") == lm.normalise_arabic("اين")
    assert lm.normalise_arabic("إنتهت") == lm.normalise_arabic("انتهت")
    assert lm.normalise_arabic("ماهى") == lm.normalise_arabic("ماهي")
    assert lm.normalise_arabic("الْكِتَاب") == "الكتاب"
    assert lm.normalise_arabic("١٩٤٥") == "1945"
    assert lm.tokenise_question("متى إنتهت حرب نورماندي ؟") == ["متي", "انتهت", "حرب", "نورماندي"]


def test_events_pad_with_bos_and_end_with_eos():
    ev = lm.events(["a", "b"], order=3)
    assert ev == [(("<s>", "<s>"), "a"), (("<s>", "a"), "b"), (("a", "b"), "</s>")]
    # <s> is never a target, </s> always is: T = words + 1.
    assert [w for _, w in ev].count(lm.BOS) == 0


def test_unknown_mapping_uses_training_counts_only():
    vocab = lm.build_vocabulary([["a", "a", "b"], ["b", "c"]], min_count=2)
    assert vocab == {"a", "b"}
    assert lm.map_unknown([["a", "c", "z"]], vocab) == [["a", lm.UNK, lm.UNK]]


def test_mle_counts_match_the_textbook():
    # P(I|<s>) = 2/3, P(am|I) = 2/3, P(Sam|am) = 1/2, P(</s>|Sam) = 1/2 (J&M 3.1.2).
    model = lm.InterpolatedNGramLM(order=2).fit(SAM)
    mle = lambda h, w: model.ngram[2][((h,), w)] / model.history[2][(h,)]
    assert mle("<s>", "I") == pytest.approx(2 / 3)
    assert mle("I", "am") == pytest.approx(2 / 3)
    assert mle("am", "Sam") == pytest.approx(1 / 2)
    assert mle("Sam", "</s>") == pytest.approx(1 / 2)
    # Unigram over targets: 14 words + 3 </s>.
    assert model.n_events == 17
    assert model.prob("I", []) != model.prob("I", ["<s>"])


def test_interpolation_formula_by_hand():
    lam = 0.7
    model = lm.InterpolatedNGramLM(order=2, lambdas={2: lam}).fit(SAM)
    expected = lam * (2 / 3) + (1 - lam) * (3 / 17)       # c(I) = 3 of 17 events
    assert model.prob("I", ["<s>"]) == pytest.approx(expected)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_interpolated_model_is_a_distribution(order):
    model = lm.InterpolatedNGramLM(order=order,
                                   lambdas={k: 0.6 for k in range(2, order + 1)}).fit(SAM)
    for history in (["<s>", "<s>"], ["<s>", "I"], ["I", "am"], ["green", "eggs"],
                    ["never", "seen"], ["Sam", "Sam"]):
        total = sum(model.prob(w, history) for w in _targets(model))
        assert total == pytest.approx(1.0)


def test_unseen_history_falls_back_to_the_lower_order():
    model = lm.InterpolatedNGramLM(order=3, lambdas={2: 0.5, 3: 0.5}).fit(SAM)
    # ("Sam", "Sam") never occurs, so the trigram term is dropped entirely.
    assert model.prob("I", ["Sam", "Sam"]) == pytest.approx(model.prob("I", ["Sam"]))


def test_no_target_word_gets_zero_probability():
    model = lm.InterpolatedNGramLM(order=3, lambdas={2: 0.9, 3: 0.9}).fit(SAM)
    assert all(model.prob(w, ["ham", "Sam"]) > 0 for w in _targets(model))


def test_add_k_bigram_is_a_distribution():
    vocab = {w for s in SAM for w in s}
    model = lm.AddKBigramLM(k=0.1).fit(SAM, vocab)
    for history in (["<s>"], ["I"], ["never"]):
        assert sum(model.prob(w, history) for w in model.targets) == pytest.approx(1.0)


def test_uniform_perplexity_is_the_number_of_targets():
    targets = {"a", "b", "c", lm.EOS}
    pp, n = lm.perplexity(lm.UniformLM(targets), [["a", "b"], ["c"]])
    assert pp == pytest.approx(len(targets))
    assert n == 5          # 3 words + 2 </s>; <s> is not counted


def test_perplexity_matches_the_definition():
    model = lm.InterpolatedNGramLM(order=2, lambdas={2: 0.5}).fit(SAM)
    sent = ["I", "am", "Sam"]
    probs = [model.prob(w, h) for h, w in lm.events(sent, 2)]
    pp, n = lm.perplexity(model, [sent])
    assert n == 4
    assert pp == pytest.approx(math.prod(probs) ** (-1 / 4))


def test_lambda_of_one_is_rejected():
    # lambda = 1 would give unseen continuations probability zero.
    with pytest.raises(ValueError):
        lm.InterpolatedNGramLM(order=2, lambdas={2: 1.0})
