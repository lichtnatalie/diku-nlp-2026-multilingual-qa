"""Tests for Week 40: the integrated multilingual QA model (Section 6).

The properties checked are the ones that would silently corrupt a result
rather than crash a run: that every context character reaches the model, that
the training target is the stored gold span (located by offset, gated on
`answerable`), that the decoder only returns legal spans and scores "no answer"
the way the report says, that a HIGH no-answer score means abstention, that the
SQuAD-v2 metrics treat unanswerable questions as the official script does, and
that best-epoch selection keeps a real copy of the weights.

Everything runs offline: the model is `qa_model.tiny_test_model`, a tiny random
XLM-R with a tokenizer trained on the fixture text.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("torch")
pytest.importorskip("transformers")

from src import config, data, evaluation, spans  # noqa: E402
from src import qa_model as qm  # noqa: E402
from tests import make_fixture  # noqa: E402

LONG_CONTEXT = (
    "Copenhagen is the capital of Denmark. " * 6
    + "The Little Mermaid statue was unveiled in 1913 at Langelinie. "
    + "Aarhus is the second largest city in Denmark."
)


@pytest.fixture(scope="module")
def splits():
    return data.load_project_splits(make_fixture.make_raw())


@pytest.fixture(scope="module")
def tiny(splits):
    tr, va = splits["train"], splits["validation"]
    texts = list(tr.question) + list(tr.context) + list(va.question) + list(va.context)
    texts += [LONG_CONTEXT, "When was the statue unveiled?"]
    return qm.tiny_test_model(texts)


@pytest.fixture(scope="module")
def small_cfg():
    # Small windows so that the fixture contexts are split several times.
    return qm.QAConfig(model_name="tiny", max_length=40, stride=8, epochs=3,
                       batch_size=4, eval_batch_size=8, learning_rate=1e-3)


def _row(question, context, answerable, answer="", answer_start=-1, lang="ar", eid="x"):
    return {"example_id": eid, "question": question, "context": context, "lang": lang,
            "answerable": answerable, "answer": answer, "answer_start": answer_start}


# ------------------------------------------------------------------- dev ---
def test_dev_slice_is_the_week39_slice(splits):
    week39 = pytest.importorskip("scripts.run_week39")
    a_fit, a_dev = qm.split_dev(splits["train"])
    b_fit, b_dev = week39.split_dev(splits["train"])
    assert list(a_fit["example_id"]) == list(b_fit["example_id"])
    assert list(a_dev["example_id"]) == list(b_dev["example_id"])


# --------------------------------------------------------------- windows ---
@pytest.mark.parametrize("max_length,stride", [(40, 8), (32, 4), (64, 16)])
def test_every_context_character_reaches_a_window(splits, tiny, max_length, stride):
    tok, _ = tiny
    contexts = list(splits["train"]["context"].unique()) + [LONG_CONTEXT]
    for context in contexts:
        windows = qm.encode_windows("When?", context, tok, max_length, stride)
        covered = set()
        for w in windows:
            assert len(w["input_ids"]) == max_length
            covered |= set(range(*w["char_range"]))
        missing = [i for i, ch in enumerate(context) if not ch.isspace() and i not in covered]
        assert not missing, f"characters {missing[:5]} of a context are in no window"


def test_window_offsets_point_into_the_context(tiny):
    tok, _ = tiny
    for w in qm.encode_windows("When was the statue unveiled?", LONG_CONTEXT, tok, 40, 8):
        ctx = [o for o in w["offsets"] if o is not None]
        assert ctx, "a window with no context tokens"
        assert all(0 <= s < e <= len(LONG_CONTEXT) for s, e in ctx)
        assert ctx == sorted(ctx)
        assert w["offsets"][w["cls_index"]] is None


def test_too_small_max_length_is_refused(tiny):
    tok, _ = tiny
    with pytest.raises(ValueError):
        qm.encode_windows("When was the statue unveiled?", LONG_CONTEXT, tok, 16, 8)


# ---------------------------------------------------------------- labels ---
def _labelled_text(df, feats, row_index=0):
    """The gold text recovered from the first window that labels the answer."""
    context = df.iloc[row_index]["context"]
    for f, r in enumerate(feats["example_index"]):
        if r == row_index and feats["start_positions"][f] != feats["cls_index"][f]:
            s = feats["offsets"][f][feats["start_positions"][f]][0]
            e = feats["offsets"][f][feats["end_positions"][f]][1]
            return s, context[s:e].strip()
    return None, None


def test_answer_only_in_a_later_window_is_labelled_there(tiny, small_cfg):
    tok, _ = tiny
    answer = "1913"
    df = pd.DataFrame([_row("When was the statue unveiled?", LONG_CONTEXT, True,
                            answer, LONG_CONTEXT.index(answer))])
    feats = qm.build_features(df, tok, small_cfg, with_labels=True)
    labelled = [f for f in range(len(feats["input_ids"]))
                if feats["start_positions"][f] != feats["cls_index"][f]]
    assert len(feats["input_ids"]) > 2
    assert labelled and labelled[0] > 0, "the answer should not be in the first window"
    for f in range(labelled[0]):
        assert feats["start_positions"][f] == feats["end_positions"][f] == feats["cls_index"][f]
    assert _labelled_text(df, feats)[1] == answer


@pytest.mark.parametrize("answer,where", [
    ("Copenhagen", "start"),     # answer at character 0
    ("Denmark.", "end"),         # answer ends the context, punctuation included
    ("second largest city", "middle"),
])
def test_labelled_span_recovers_the_gold_text(tiny, answer, where):
    tok, _ = tiny
    # The stride (24 tokens) is longer than any of these answers, so each answer
    # lies wholly inside at least one window.
    cfg = qm.QAConfig(model_name="tiny", max_length=64, stride=24)
    start = {"start": 0, "end": LONG_CONTEXT.rindex(answer),
             "middle": LONG_CONTEXT.index(answer)}[where]
    df = pd.DataFrame([_row("Which?", LONG_CONTEXT, True, answer, start)])
    feats = qm.build_features(df, tok, cfg, with_labels=True)
    s, text = _labelled_text(df, feats)
    assert text == answer
    assert abs(s - start) <= 1  # at most the leading space of a word-initial piece


def test_label_audit_reports_an_answer_split_by_every_window_boundary(tiny, small_cfg):
    # With a stride SHORTER than the answer, an answer can straddle every window
    # boundary and be labelled nowhere. The audit must say so (run_week40 then
    # stops), rather than letting the example train as "no answer".
    tok, _ = tiny
    answer = "second largest city"
    df = pd.DataFrame([_row("Which?", LONG_CONTEXT, True, answer, LONG_CONTEXT.index(answer))])
    audit = qm.label_audit(df, qm.build_features(df, tok, small_cfg, with_labels=True))
    assert audit["answerable"] == 1
    assert audit["labelled_in_a_window"] == 0


def test_repeated_answer_is_located_by_offset_not_by_search(splits, tiny, small_cfg):
    tok, _ = tiny
    df = splits["train"]
    # Fixture row 2: "WikiLeaks" occurs twice; the stored offset is the second one.
    i = int(np.flatnonzero((df["answer"] == "WikiLeaks") & (df["answer_start"] > 0))[0])
    one = df.iloc[[i]].reset_index(drop=True)
    feats = qm.build_features(one, tok, small_cfg, with_labels=True)
    s, text = _labelled_text(one, feats)
    assert text == "WikiLeaks"
    assert s >= one.iloc[0]["answer_start"] - 1
    assert s > one.iloc[0]["context"].index("WikiLeaks")


def test_unanswerable_target_ignores_a_non_empty_answer_string(tiny, small_cfg):
    tok, _ = tiny
    # Week 36: every unanswerable row in the real data has a non-empty answer.
    df = pd.DataFrame([_row("Who wrote it?", LONG_CONTEXT, False, "Copenhagen", 0)])
    feats = qm.build_features(df, tok, small_cfg, with_labels=True)
    assert all(s == e == c for s, e, c in zip(feats["start_positions"],
                                              feats["end_positions"], feats["cls_index"]))


def test_a_stored_offset_mismatch_is_caught_only_when_training(splits, tiny, small_cfg):
    tok, _ = tiny
    broken = splits["validation"]  # fixture row 1 has an offset shifted by one
    with pytest.raises(ValueError):
        qm.build_features(broken, tok, small_cfg, with_labels=True)
    qm.build_features(broken, tok, small_cfg, with_labels=False)  # prediction never reads gold


def test_label_audit_on_the_fixture(splits, tiny, small_cfg):
    tok, _ = tiny
    df = splits["train"]
    audit = qm.label_audit(df, qm.build_features(df, tok, small_cfg, with_labels=True))
    assert audit["labelled_in_a_window"] == audit["answerable"] == int(df["answerable"].sum())
    assert audit["unanswerable_all_cls"] == audit["unanswerable"]
    assert audit["exact_char_match"] == audit["answerable"]


# ---------------------------------------------------------- postprocess ---
def _fake_feats(n_windows, length, ctx_from, context):
    """Windows whose context tokens are single characters of `context`."""
    feats = {"example_index": [0] * n_windows, "cls_index": [0] * n_windows, "offsets": []}
    for w in range(n_windows):
        offs = [None] * length
        for t in range(ctx_from, length - 1):
            c = w * (length - 1 - ctx_from) + (t - ctx_from)
            offs[t] = (c, c + 1) if c < len(context) else None
        feats["offsets"].append(offs)
    return feats


def test_decoder_only_returns_legal_spans():
    context = "abcdefghij"
    df = pd.DataFrame([_row("q", context, True, "cde", 2)])
    feats = _fake_feats(1, 14, 3, context)  # tokens 3..12 are context chars a..j
    s = np.full((1, 14), -100.0)
    e = np.full((1, 14), -100.0)
    s[0, 1], e[0, 2] = 50.0, 50.0    # a "span" inside the QUESTION: must be ignored
    s[0, 12], e[0, 3] = 40.0, 40.0   # start j, end a: end before start, illegal
    s[0, 5], e[0, 7] = 5.0, 5.0      # the legal best: c..e
    s[0, 0], e[0, 0] = 1.0, 1.0      # null score 2
    out = qm.postprocess(df, feats, s, e, max_answer_tokens=30, n_best=20).iloc[0]
    assert out["span_text"] == "cde"
    assert out["span_score"] == pytest.approx(10.0)
    assert out["null_score"] == pytest.approx(2.0)
    assert out["no_answer_score"] == pytest.approx(2.0 - 10.0)


def test_decoder_respects_the_maximum_span_length():
    context = "abcdefghij"
    df = pd.DataFrame([_row("q", context, True, "a", 0)])
    feats = _fake_feats(1, 14, 3, context)
    s = np.full((1, 14), -10.0)
    e = np.full((1, 14), -10.0)
    s[0, 3], e[0, 12] = 9.0, 9.0     # a..j, ten tokens
    e[0, 4] = 8.0                    # a..b, two tokens
    out = qm.postprocess(df, feats, s, e, max_answer_tokens=3, n_best=20).iloc[0]
    assert out["span_text"] == "ab"


def test_null_score_is_the_minimum_over_windows_and_spans_compete_across_windows():
    context = "abcdefghijklmnopqrst"
    df = pd.DataFrame([_row("q", context, True, "m", 12)])
    feats = _fake_feats(2, 14, 3, context)  # window 1: a..j, window 2: k..t
    s = np.full((2, 14), -10.0)
    e = np.full((2, 14), -10.0)
    s[0, 0], e[0, 0] = 4.0, 4.0      # window 1 null 8
    s[1, 0], e[1, 0] = 1.0, 2.0      # window 2 null 3  -> example null = 3
    s[0, 4], e[0, 4] = 2.0, 2.0      # window 1 best span "b", score 4
    s[1, 5], e[1, 5] = 3.0, 3.0      # window 2 best span "m", score 6
    out = qm.postprocess(df, feats, s, e).iloc[0]
    assert out["span_text"] == "m"
    assert out["null_score"] == pytest.approx(3.0)
    assert out["no_answer_score"] == pytest.approx(3.0 - 6.0)
    assert out["n_windows"] == 2


def test_high_no_answer_score_means_abstain():
    pred = pd.DataFrame({"no_answer_score": [-2.0, 0.0, 3.0], "span_start": [5, 6, 7],
                         "span_text": ["x", "y", "z"]})
    out = qm.apply_threshold(pred, 0.0)
    assert list(out["answered"]) == [True, False, False]
    assert list(out["pred_text"]) == ["x", "", ""]
    assert list(out["pred_start"]) == [5, -1, -1]


def test_threshold_tuning_states_its_direction_and_survives_inf():
    pred = pd.DataFrame({"no_answer_score": [-5.0, -4.0, 2.0, math.inf],
                         "answerable": [True, True, False, False]})
    cut = qm.tune_no_answer_threshold(pred)
    assert cut["higher_means_answerable"] is False
    assert cut["macro_f1"] == pytest.approx(1.0)
    assert -4.0 < cut["threshold"] <= 2.0


# --------------------------------------------------------------- metrics ---
def test_squad_normalisation_and_f1():
    assert evaluation.normalize_answer("The  Nile, River.") == "nile river"
    assert evaluation.squad_exact("the Nile", "Nile!") == 1
    assert evaluation.squad_f1("Nile river", "the Nile") == pytest.approx(2 / 3)
    assert evaluation.squad_f1("Nile", "Amazon") == 0.0
    assert evaluation.squad_f1("", "") == 1.0
    assert evaluation.squad_f1("", "Nile") == 0.0


def test_qa_metrics_follow_the_squad2_rules_for_abstention():
    pred = pd.DataFrame({
        "lang": ["ar", "ar", "ko", "te"],
        "answerable": [True, True, False, False],
        "gold_answer": ["Nile", "1913", "", ""],
        "answered": [True, False, False, True],   # right, abstained, abstained, answered
        "pred_text": ["the Nile", "", "", "Aarhus"],
    })
    m = evaluation.qa_metrics(pred)
    # per example EM: 1, 0 (abstained on answerable), 1 (abstained on unanswerable), 0
    assert m["exact_match"] == pytest.approx(0.5)
    assert m["has_ans_exact_match"] == pytest.approx(0.5)
    assert m["no_ans_exact_match"] == pytest.approx(0.5)
    ref = evaluation.answerability_metrics(pred["answerable"], pred["answered"])
    assert m["macro_f1"] == pytest.approx(ref["macro_f1"])
    report = evaluation.qa_report(pred, "m")
    assert list(report["subset"]) == ["overall", "ar", "ko", "te"]
    assert list(report["n"]) == [4, 2, 1, 1]


def test_abstention_is_read_from_the_decision_not_the_text():
    # An answered prediction whose text normalises to "" is still an answer.
    pred = pd.DataFrame({"lang": ["ar"], "answerable": [False], "gold_answer": [""],
                         "answered": [True], "pred_text": ["the"]})
    m = evaluation.qa_metrics(pred)
    assert m["unanswerable_recall"] == 0.0


# ------------------------------------------------- Week 37 BIO bridge ------
def test_gold_spans_map_back_to_the_week37_gold_labels(splits):
    df = splits["train"]
    encoded = spans.encode_split(df)
    gold = pd.DataFrame({
        "example_id": df["example_id"],
        "pred_start": np.where(df["answerable"], df["answer_start"], -1),
        "pred_text": np.where(df["answerable"], df["answer"], ""),
    })
    bio = qm.char_spans_to_bio(encoded, gold)
    for e, labels in zip(encoded, bio):
        assert labels == e["labels"]
    m = evaluation.span_metrics(encoded, bio)
    assert m["token_f1"] == pytest.approx(1.0)
    assert m["exact_match"] == pytest.approx(1.0)


# --------------------------------------------------------------- training ---
def test_a_snapshot_is_not_changed_by_later_training(tiny):
    torch = pytest.importorskip("torch")
    _, model = tiny
    snap = qm.snapshot_state(model)
    name = next(iter(snap))
    before = snap[name].clone()
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    assert torch.equal(snap[name], before)
    with torch.no_grad():
        next(model.parameters()).sub_(1.0)


def test_fit_keeps_the_best_dev_epoch_and_reproduces(splits, small_cfg):
    tr, va = splits["train"], splits["validation"]
    texts = list(tr.question) + list(tr.context) + list(va.question) + list(va.context)
    tok, _ = qm.tiny_test_model(texts)
    runs = []
    for _ in range(2):
        _, model = qm.tiny_test_model(tokenizer=tok, seed=1)
        m = qm.IntegratedQAModel(small_cfg, tokenizer=tok, model=model, device="cpu")
        m.fit(tr, va, log=lambda *_: None)
        runs.append(m)
    m = runs[0]
    assert [h["epoch"] for h in m.history_] == [1, 2, 3]
    f1 = [h["dev_f1"] for h in m.history_]
    assert m.best_epoch_ == 1 + int(np.argmax(f1))  # first epoch reaching the maximum
    assert m.history_[-1]["train_loss"] < m.history_[0]["train_loss"]
    # Same seed, same data: identical training history.
    assert [h["train_loss"] for h in runs[0].history_] == pytest.approx(
        [h["train_loss"] for h in runs[1].history_])
    pred = m.predict_scores(va)
    assert len(pred) == len(va)
    assert set(pred["example_id"]) == set(va["example_id"])
