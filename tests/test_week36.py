"""Self-tests for the Week 36 pipeline, run against the synthetic fixture.

These check the properties:
  * the ar/ko/te filter actually drops out-of-scope languages;
  * example_id survives filtering;
  * Item 3 finds exactly the row we broke on purpose, and no others;
  * duplicate detection finds the duplicate pair and notices the label clash;
  * macro-F1 is the unweighted mean of the two class F1s (hand-checked case);
  * the metrics agree with scikit-learn where scikit-learn is applicable;
  * both baselines run and produce per-language rows with N attached.

Run:  python -m pytest tests/ -q      (or)      python tests/test_week36.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import audit, baselines, config, data, evaluation  # noqa: E402
from src.tokenisation import normalise_digits, unicode_word_tokenise  # noqa: E402
from tests.make_fixture import make_raw  # noqa: E402


def _splits():
    return data.load_project_splits(make_raw())


# --------------------------------------------------------------------- data
def test_language_filter_drops_out_of_scope():
    splits = _splits()
    for split, df in splits.items():
        assert set(df["lang"]) <= set(config.LANGS), f"{split} kept a foreign language"
    assert len(splits["train"]) == 12, len(splits["train"])   # 13 rows minus one bn
    assert len(splits["validation"]) == 6, len(splits["validation"])  # 7 minus one bn


def test_example_ids_are_unique_and_stable():
    splits = _splits()
    for df in splits.values():
        assert df["example_id"].is_unique
    # ids encode the position in the UNFILTERED split, so the bn row leaves a gap
    ids = splits["train"]["example_id"].tolist()
    assert "train-000011" not in ids, "the dropped bn row's id should be absent"


# --------------------------------------------------------------- tokenisation
def test_unicode_tokeniser_keeps_combining_marks():
    assert unicode_word_tokenise("పరిపాలన") == ["పరిపాలన"]
    assert unicode_word_tokenise("الْكِتَاب") == ["الْكِتَاب"]
    assert unicode_word_tokenise("한국어를 배웁니다.") == ["한국어를", "배웁니다"]


def test_digit_normalisation():
    assert normalise_digits("٢٠٠٦") == "2006"
    assert normalise_digits("౧౯౪౫") == "1945"
    assert normalise_digits("2006") == "2006"


# ------------------------------------------------------------------- Item 1
def test_describe_split_shapes_and_proportions():
    splits = _splits()
    desc = audit.describe_split(splits["train"], "train")
    allrow = desc[desc["lang"] == "all"].iloc[0]
    assert allrow["n"] == 12
    assert allrow["n_answerable"] + allrow["n_unanswerable"] == allrow["n"]
    assert abs(allrow["prop_answerable"] + allrow["prop_unanswerable"] - 1.0) < 1e-12
    # per-language rows sum back to the total
    per_lang = desc[desc["lang"] != "all"]
    assert per_lang["n"].sum() == allrow["n"]
    assert (desc["q_len_iqr"] >= 0).all()


def test_duplicate_detection():
    splits = _splits()
    d = audit.duplicate_pairs(splits["train"])
    assert d["n_groups"] == 1, d
    assert d["n_rows_in_groups"] == 2, d
    assert d["n_redundant_rows"] == 1, d
    assert audit.label_inconsistent_duplicates(splits["train"]) == 1


def test_missing_values_counts_empty_answers():
    splits = _splits()
    miss = audit.missing_values(splits["train"], "train")
    allrow = miss[miss["lang"] == "all"].iloc[0]
    # unanswerable rows carry an empty (not null) answer string
    n_unans = int((~splits["train"]["answerable"]).sum())
    assert allrow["answer_empty"] == n_unans
    assert allrow["answer_null"] == 0


# ------------------------------------------------------------------- Item 3
def test_offset_verification_finds_exactly_the_broken_row():
    splits = _splits()

    train_res = audit.verify_answer_offsets(splits["train"], "train")
    assert train_res["n_failures"] == 0, train_res["failures"]

    val_res = audit.verify_answer_offsets(splits["validation"], "validation")
    assert val_res["n_failures"] == 1, val_res["failures"]
    assert val_res["failures"].iloc[0]["lang"] == "ar"
    # the answer text is still present in the context, just not at that offset
    assert bool(val_res["failures"].iloc[0]["answer_occurs_elsewhere"])

    # the convention violation on the unanswerable side is counted separately
    assert val_res["n_unanswerable_violating_convention"] == 1
    assert train_res["n_unanswerable_violating_convention"] == 0

    # one train row has an answer string occurring twice in its context
    assert train_res["n_answers_occurring_more_than_once"] >= 1


def test_offset_verification_counts_match_totals():
    splits = _splits()
    for split, df in splits.items():
        res = audit.verify_answer_offsets(df, split)
        assert res["n_answerable_checked"] == int(df["answerable"].sum())
        assert res["n_answerable_checked"] + res["n_unanswerable"] == len(df)
        assert sum(v["checked"] for v in res["per_language"].values()) == res["n_answerable_checked"]


# --------------------------------------------------------------- evaluation
def test_macro_f1_is_unweighted_mean_of_class_f1s():
    y_true = np.array([True, True, True, False, False])
    y_pred = np.array([True, False, True, False, True])
    m = evaluation.answerability_metrics(y_true, y_pred)
    # answerable: tp=2 fp=1 fn=1 -> P=2/3, R=2/3, F1=2/3
    assert abs(m["answerable_f1"] - 2 / 3) < 1e-12
    # unanswerable: tp=1 fp=1 fn=1 -> P=1/2, R=1/2, F1=1/2
    assert abs(m["unanswerable_f1"] - 0.5) < 1e-12
    assert abs(m["macro_f1"] - 0.5 * (2 / 3 + 0.5)) < 1e-12
    assert abs(m["accuracy"] - 3 / 5) < 1e-12
    assert m["n"] == 5


def test_matches_sklearn():
    from sklearn.metrics import f1_score, precision_recall_fscore_support

    rng = np.random.default_rng(config.SEED)
    y_true = rng.random(500) < 0.7
    y_pred = rng.random(500) < 0.5
    m = evaluation.answerability_metrics(y_true, y_pred)
    assert abs(m["macro_f1"] - f1_score(y_true, y_pred, average="macro")) < 1e-12
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[True], zero_division=0)
    assert abs(m["answerable_precision"] - p[0]) < 1e-12
    assert abs(m["answerable_recall"] - r[0]) < 1e-12
    assert abs(m["answerable_f1"] - f[0]) < 1e-12


def test_degenerate_predictions_do_not_crash():
    y_true = np.array([True, True, True, False])
    for const in (True, False):
        m = evaluation.answerability_metrics(y_true, np.full(4, const))
        assert 0.0 <= m["macro_f1"] <= 1.0
        assert m["n"] == 4


def test_report_has_n_on_every_row():
    splits = _splits()
    df = splits["validation"]
    rep = evaluation.answerability_report(df, np.ones(len(df), dtype=bool), "always-answerable")
    assert set(rep["subset"]) <= {"overall", *config.LANGS}
    assert rep["subset"].iloc[0] == "overall"
    assert (rep["n"] > 0).all()
    assert rep[rep["subset"] != "overall"]["n"].sum() == len(df)


def test_threshold_tuning_direction():
    # scores where HIGH means unanswerable, as in the Section 1 definition
    y_true = np.array([True, True, False, False])          # answerable flags
    no_answer_score = np.array([-2.0, -1.0, 1.0, 2.0])
    best = evaluation.tune_threshold(
        no_answer_score, y_true, higher_means_answerable=False
    )
    assert abs(best["macro_f1"] - 1.0) < 1e-12
    assert -1.0 < best["threshold"] <= 1.0
    assert best["higher_means_answerable"] is False


# ---------------------------------------------------------------- baselines
def test_majority_baseline():
    splits = _splits()
    maj = baselines.MajorityClassBaseline().fit(splits["train"])
    preds = maj.predict(splits["validation"])
    assert preds.dtype == bool
    assert len(set(preds.tolist())) == 1
    assert maj.majority_ == (splits["train"]["answerable"].mean() >= 0.5)


def test_anchor_rule_behaviour():
    rule = baselines.CrossLingualAnchorRule().fit(_splits()["train"])
    # Latin token and a normalised year are both anchors
    assert "wikileaks" in rule.anchors("WikiLeaks ٢٠٠٦ سنة تأسست؟")
    assert "2006" in rule.anchors("WikiLeaks ٢٠٠٦ سنة تأسست؟")
    # short Latin runs are not anchors
    assert rule.anchors("II 는 무엇입니까?") == []
    # a question in pure Telugu script yields nothing -> the fallback decides
    assert rule.anchors("పరిపాలన ఎక్కడ జరుగుతుంది?") == []
    ctx = "WikiLeaks was founded in 2006."
    assert rule.hits("WikiLeaks ٢٠٠٦ سنة تأسست؟", ctx) == 2
    assert rule.hits("Nile ఎంత పొడవు?", ctx) == 0


def test_anchor_rule_predicts_and_diagnoses():
    splits = _splits()
    rule = baselines.CrossLingualAnchorRule().fit(splits["train"])
    preds = rule.predict(splits["validation"])
    assert preds.dtype == bool and len(preds) == len(splits["validation"])
    diag = rule.anchor_diagnostics(splits["train"])
    assert set(diag["lang"]) <= {"all", *config.LANGS}
    assert diag[diag["lang"] == "all"]["n"].iloc[0] == len(splits["train"])
    assert (diag["prop_questions_with_any_anchor"].between(0, 1)).all()
    sweep = rule.sweep_min_hits(splits["train"], values=(1, 2))
    assert list(sweep["min_hits"]) == [1, 2]
    assert rule.min_hits == 1, "sweep must not mutate the rule"


def test_anchor_rule_uses_both_classes():
    """The rule must not silently collapse to a constant prediction.

    It differs from the majority baseline only on questions that HAVE anchors
    none of which occur in the context; the fixture contains such rows, so
    both labels must appear.
    """
    splits = _splits()
    rule = baselines.CrossLingualAnchorRule().fit(splits["train"])
    preds = rule.predict(splits["train"])
    assert set(preds.tolist()) == {True, False}, "rule degenerated to a constant"
    maj = baselines.MajorityClassBaseline().fit(splits["train"]).predict(splits["train"])
    assert not np.array_equal(preds, maj), "rule is indistinguishable from majority"


def test_label_free_fallback_variant():
    splits = _splits()
    r = baselines.CrossLingualAnchorRule(fallback=True).fit(splits["train"])
    assert r.fallback_value_ is True
    # a question with no anchors must be predicted answerable under this variant
    no_anchor = splits["train"][
        splits["train"]["question"].apply(lambda q: len(r.anchors(q)) == 0)
    ]
    if len(no_anchor):
        assert r.predict(no_anchor).all()


# ------------------------------------------------------------------- driver
def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
