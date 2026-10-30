"""Tests for Week 39: the two learned answerability classifiers.

The properties checked here are the ones that would silently corrupt a result
rather than crash a run: that the dense features mean what the report says they
mean, that the vocabulary is built from training text only and in a
reproducible order, that padding never reaches the recurrent model's summary,
that the attention weights are a distribution over real context tokens, and
that the ablation switches actually remove the side they name.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import answerability_models as am  # noqa: E402
from tests import make_fixture  # noqa: E402

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def train() -> pd.DataFrame:
    return make_fixture.make_train()


@pytest.fixture(scope="module")
def validation() -> pd.DataFrame:
    return make_fixture.make_validation()


# ---------------------------------------------------------------- features --
def test_dense_feature_names_match_the_matrix_width(train):
    X = am.dense_features(train)
    assert X.shape == (len(train), len(am.DENSE_FEATURE_NAMES))
    assert np.isfinite(X).all()


def test_anchor_features_count_hits_not_occurrences():
    df = pd.DataFrame(
        [
            {
                # Two distinct anchors, one of which occurs twice in the context.
                "question": "متى تأسست WikiLeaks في ٢٠٠٦؟",
                "context": "WikiLeaks was founded in 2006. WikiLeaks publishes leaks.",
                "lang": "ar",
            }
        ]
    )
    f = dict(zip(am.DENSE_FEATURE_NAMES, am.dense_features(df)[0]))
    assert f["n_anchors"] == 2  # "wikileaks" and the normalised "2006"
    assert f["n_anchor_hits"] == 2  # distinct anchors hit, not occurrences
    assert f["anchor_hit_rate"] == pytest.approx(1.0)
    assert f["all_anchors_hit"] == 1.0
    assert f["no_anchor_hit"] == 0.0
    assert f["n_digit_anchors"] == 1 and f["n_digit_hits"] == 1
    assert f["n_latin_anchors"] == 1 and f["n_latin_hits"] == 1


def test_questions_without_anchors_are_marked_as_such():
    df = pd.DataFrame(
        [{"question": "ఎవरు ఈ చిత్రానికి దర్శకుడు?", "context": "Seoul is the capital.", "lang": "te"}]
    )
    f = dict(zip(am.DENSE_FEATURE_NAMES, am.dense_features(df)[0]))
    assert f["n_anchors"] == 0
    assert f["has_anchor"] == 0.0
    # With no anchors there is nothing to hit, and the two "all"/"no" flags must
    # not fire, or the model would read "no anchors" as "anchors, none hit".
    assert f["all_anchors_hit"] == 0.0 and f["no_anchor_hit"] == 0.0
    assert f["anchor_hit_rate"] == 0.0


def test_feature_classifier_fits_and_reports_its_weights(train, validation):
    clf = am.FeatureAnswerabilityClassifier(min_df=1).fit(train)
    p = clf.predict_proba_answerable(validation)
    assert p.shape == (len(validation),)
    assert ((0.0 <= p) & (p <= 1.0)).all()
    # The threshold must be the only thing separating probabilities from labels.
    assert (clf.predict(validation, threshold=0.5) == (p >= 0.5)).all()
    assert (clf.predict(validation, threshold=0.0) == True).all()  # noqa: E712
    coef = clf.coefficients(top=3)
    assert set(coef["pushes_towards"]) <= {"answerable", "unanswerable"}
    assert len(coef) <= 6


# -------------------------------------------------------------- vocabulary --
def test_vocabulary_is_reserved_padded_and_deterministic(train):
    vocab = am.build_vocabulary(train, min_count=1)
    assert vocab[am.PAD] == 0 and vocab[am.UNK] == 1
    # Lab 4's bug: indices that depend on set iteration order. Rebuilding from
    # a shuffled frame must give exactly the same mapping.
    shuffled = train.sample(frac=1.0, random_state=7).reset_index(drop=True)
    assert am.build_vocabulary(shuffled, min_count=1) == vocab


def test_vocabulary_respects_min_count_and_is_shared_by_both_sides(train):
    vocab = am.build_vocabulary(train, min_count=1)
    # A context word and a question word both index into the same table, which
    # is what lets a Latin-script name bridge the script boundary.
    assert "wikileaks" in vocab
    rare = am.build_vocabulary(train, min_count=10_000)
    assert set(rare) == {am.PAD, am.UNK}


def test_encoding_truncates_and_never_returns_an_empty_sequence(train):
    vocab = am.build_vocabulary(train, min_count=1)
    ids = am._encode("WikiLeaks was founded in Iceland in 2006", vocab, max_len=3)
    assert len(ids) == 3
    # A question of only unseen tokens still has to produce one input symbol.
    assert am._encode("!!!", vocab, max_len=8) == [vocab[am.UNK]]


# --------------------------------------------------------------- recurrent --
def _small(vocab, **kw):
    kw.setdefault("embedding_dim", 8)
    kw.setdefault("hidden_dim", 8)
    kw.setdefault("epochs", 1)
    kw.setdefault("batch_size", 4)
    kw.setdefault("max_context_len", 16)
    return am.RecurrentAnswerabilityClassifier(vocab, **kw)


def test_recurrent_classifier_fits_and_is_reproducible(train, validation):
    vocab = am.build_vocabulary(train, min_count=1)
    a = _small(vocab, seed=11).fit(train)
    b = _small(vocab, seed=11).fit(train)
    pa = a.predict_proba_answerable(validation)
    pb = b.predict_proba_answerable(validation)
    assert ((0.0 <= pa) & (pa <= 1.0)).all()
    np.testing.assert_allclose(pa, pb, rtol=0, atol=1e-6)
    assert a.n_train_examples_ == len(train)


def test_padding_does_not_change_a_prediction(train):
    """The summary and the attention must be masked.

    Scoring one example alone and scoring it in a batch beside a much longer
    one must give the same probability; if padding leaked into the max-pool or
    into the softmax, the two would differ.
    """
    vocab = am.build_vocabulary(train, min_count=1)
    clf = _small(vocab, seed=3).fit(train)
    one = train.iloc[[0]].reset_index(drop=True)
    long_row = one.copy()
    long_row.loc[0, "context"] = (one.loc[0, "context"] + " ") * 6
    long_row.loc[0, "question"] = one.loc[0, "question"] + " " + one.loc[0, "question"]
    alone = clf.predict_proba_answerable(one)[0]
    batched = clf.predict_proba_answerable(pd.concat([one, long_row], ignore_index=True))[0]
    assert alone == pytest.approx(batched, abs=1e-6)


def test_attention_is_a_distribution_over_real_context_tokens(train):
    vocab = am.build_vocabulary(train, min_count=1)
    clf = _small(vocab, seed=5).fit(train)
    row = train.iloc[0]
    prob, attended = clf.attention_for(row.question, row.context, top=4)
    assert 0.0 <= prob <= 1.0
    assert len(attended) <= 4
    weights = [w for _, w in attended]
    assert all(w >= 0 for w in weights)
    # A top-k slice of a softmax over the real tokens: non-negative and no more
    # than the whole distribution.
    assert sum(weights) <= 1.0 + 1e-6
    # Weights are sorted, so the first one is the token the model attends to most.
    assert weights == sorted(weights, reverse=True)
    # Every attended token is a context token, not padding.
    context_tokens = set(am._tokens(row.context))
    assert {t for t, _ in attended} <= context_tokens


def test_ablation_switches_remove_the_side_they_name(train, validation):
    """`use_context=False` must make the context irrelevant, and vice versa."""
    vocab = am.build_vocabulary(train, min_count=1)
    clf = _small(vocab, seed=9, use_context=False).fit(train)
    changed = validation.copy()
    changed["context"] = "Entirely unrelated English text about something else."
    np.testing.assert_allclose(
        clf.predict_proba_answerable(validation),
        clf.predict_proba_answerable(changed),
        rtol=0,
        atol=1e-6,
    )

    clf_q = _small(vocab, seed=9, use_question=False).fit(train)
    changed_q = validation.copy()
    changed_q["question"] = "متى؟"
    np.testing.assert_allclose(
        clf_q.predict_proba_answerable(validation),
        clf_q.predict_proba_answerable(changed_q),
        rtol=0,
        atol=1e-6,
    )


def test_class_weighting_follows_the_training_imbalance(train):
    """`pos_weight` is (1 - p) / p, so the rarer class is up-weighted."""
    vocab = am.build_vocabulary(train, min_count=1)
    clf = _small(vocab, seed=1).fit(train)
    p = float(train["answerable"].mean())
    assert clf.pos_weight_ == pytest.approx((1 - p) / p, rel=1e-6)


# ------------------------------------------------------- reporting pins ----
def test_error_direction_matches_the_reported_definition():
    """Section 5 says the positive class is *unanswerable*.

    The error dump in `scripts/run_week39.py` labels a row "false positive"
    when the gold label is answerable, i.e. the model predicted the positive
    (unanswerable) class wrongly. This pins that wording to the taxonomy file,
    so the report's definition and the data cannot drift apart.
    """
    path = Path(__file__).resolve().parents[1] / "results" / "50_week39_error_taxonomy.csv"
    if not path.exists():
        pytest.skip("the Week 39 run has not been made in this checkout")
    tax = pd.read_csv(path)
    fp = tax[tax["error_type"] == "false positive"]
    fn = tax[tax["error_type"] == "false negative"]
    assert (fp["gold"] == "answerable").all()
    assert (fn["gold"] == "unanswerable").all()


def test_every_inspected_error_has_a_category_and_a_justification():
    path = Path(__file__).resolve().parents[1] / "results" / "50_week39_error_taxonomy.csv"
    if not path.exists():
        pytest.skip("the Week 39 run has not been made in this checkout")
    tax = pd.read_csv(path)
    assert len(tax) >= 10                      # the project asks for at least ten
    assert tax["category"].notna().all()
    assert tax["note"].str.len().gt(20).all()
    # Both directions and all three languages, as Section 5 claims.
    assert set(tax["error_type"]) == {"false positive", "false negative"}
    assert set(tax["lang"]) == {"ar", "ko", "te"}
