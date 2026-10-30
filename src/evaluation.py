"""Shared evaluation protocol (project description, Section 1).

Every model in the project — the Week 36 baselines, the Week 39 classifiers,
the abstention decisions of the Week 41 integrated QA model and every
extension — is scored by the functions in this file, so that comparisons are
never confounded by differences in the scoring code.

Answerability metrics implemented here:
  * precision / recall / F1 for BOTH classes ("answerable", "unanswerable");
  * macro-averaged F1 = the unweighted mean of those two class F1 scores
    (this is what the project calls macro-F1 — it is NOT weighted by support);
  * accuracy, reported for completeness but never quoted alone, because the
    classes are imbalanced;
  * the number of evaluated examples, and the support of each class, on every
    row — the project requires N to accompany every result.

Zero-division convention: if a class has no true instances and no predicted
instances in a subset, its precision/recall/F1 are reported as 0.0 and the
``*_support`` column shows 0, following the usual scikit-learn convention.
Read any macro-F1 whose class supports are very small with care; the
per-language Telugu subsets are the ones most likely to be affected.

Section 1 also defines the no-answer score for the integrated QA model as
(null/CLS-span score - best non-empty-span score), with a threshold tuned on
validation.  ``tune_threshold`` implements that tuning generically so the
Week 39 classifiers and the Week 41 model use the same procedure.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from . import config

# The two classes, in the order used in every table.
ANSWERABLE = "answerable"
UNANSWERABLE = "unanswerable"


def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def answerability_metrics(
    y_true: Sequence[bool], y_pred: Sequence[bool]
) -> Dict[str, float]:
    """Metrics for one subset.  ``True`` means *answerable*.

    Returns a flat dict so it can go straight into a DataFrame row.
    """
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")

    n = int(y_true.size)
    out: Dict[str, float] = {"n": n}

    for name, positive in ((ANSWERABLE, True), (UNANSWERABLE, False)):
        t = y_true == positive
        p = y_pred == positive
        tp = int(np.sum(t & p))
        fp = int(np.sum(~t & p))
        fn = int(np.sum(t & ~p))
        scores = _prf(tp, fp, fn)
        out[f"{name}_support"] = int(np.sum(t))
        out[f"{name}_predicted"] = int(np.sum(p))
        out[f"{name}_precision"] = scores["precision"]
        out[f"{name}_recall"] = scores["recall"]
        out[f"{name}_f1"] = scores["f1"]

    out["macro_f1"] = 0.5 * (out[f"{ANSWERABLE}_f1"] + out[f"{UNANSWERABLE}_f1"])
    out["accuracy"] = float(np.mean(y_true == y_pred)) if n else 0.0
    return out


def answerability_report(
    df: pd.DataFrame,
    y_pred: Sequence[bool],
    model_name: str,
    y_true_col: str = "answerable",
    lang_col: str = "lang",
    langs: Iterable[str] = config.LANGS,
) -> pd.DataFrame:
    """Overall + per-language metrics, in the layout used in the report.

    The overall row comes first, then one row per language in a fixed order,
    so that tables for different models are directly comparable.
    """
    y_pred = np.asarray(y_pred, dtype=bool)
    if len(y_pred) != len(df):
        raise ValueError(f"{len(y_pred)} predictions for {len(df)} rows")
    y_true = df[y_true_col].to_numpy(dtype=bool)

    rows = [
        {"model": model_name, "subset": "overall", **answerability_metrics(y_true, y_pred)}
    ]
    for lang in langs:
        mask = (df[lang_col] == lang).to_numpy()
        if not mask.any():
            continue
        rows.append(
            {
                "model": model_name,
                "subset": lang,
                **answerability_metrics(y_true[mask], y_pred[mask]),
            }
        )
    return pd.DataFrame(rows)


def tune_threshold(
    scores: Sequence[float],
    y_true: Sequence[bool],
    higher_means_answerable: bool = True,
    objective: str = "macro_f1",
    grid: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Pick the decision threshold that maximises ``objective`` on this subset.

    Used for the Section 1 no-answer threshold.  Call it on the VALIDATION
    split only — the project forbids tuning it on the challenge set, and
    tuning it on train would not reflect the deployed decision rule.

    ``higher_means_answerable`` records which side of the cutoff means "no
    answer"; the project requires that this be stated explicitly.  With the
    Section 1 definition (no-answer score = null score - best non-null score),
    a HIGH score means *unanswerable*, so pass ``False`` for that quantity.
    """
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=bool)
    if grid is None:
        # Candidate cutoffs midway between consecutive observed scores, plus
        # the two extremes so that "always answerable"/"never answerable" are
        # reachable.
        uniq = np.unique(scores)
        mids = (uniq[:-1] + uniq[1:]) / 2.0 if uniq.size > 1 else np.array([])
        grid = np.concatenate([[uniq.min() - 1.0], mids, [uniq.max() + 1.0]])

    best = {"threshold": float("nan"), objective: -1.0}
    for t in grid:
        pred = scores >= t if higher_means_answerable else scores < t
        m = answerability_metrics(y_true, pred)
        if m[objective] > best[objective]:
            best = {"threshold": float(t), **m}
    best["higher_means_answerable"] = higher_means_answerable
    return best


# ==========================================================================
# Span-prediction metrics (project Section 3).
# ==========================================================================
# Section 1 asks for two things here, and they disagree with each other often
# enough that reporting only one would hide the interesting errors:
#
#   * token-level answer-token F1, with B-ANS and I-ANS as the positive class
#     and O as the negative one (so the two answer labels are pooled);
#   * exact span match.
#
# How unanswerable examples enter each metric - Section 1 requires this to be
# stated, so it is stated here and repeated in the report:
#
#   * TOKEN F1: an unanswerable row contributes only O tokens to the gold. It
#     can therefore never produce a true positive or a false negative, only
#     false positives. Unanswerable rows can only ever LOWER token precision.
#   * EXACT MATCH: an unanswerable row counts as correct exactly when the model
#     outputs the empty span, so it CAN contribute a hit. A model that abstains
#     everywhere scores 0 on token F1 but scores the unanswerable rate on exact
#     match.
#
# Exact match is scored against the ORIGINAL gold character span, never against
# the token-aligned version. Rows whose gold span does not land on token
# boundaries are therefore unreachable; `spans.alignment_audit` reports that
# ceiling separately.

from . import spans as _spans


def _token_counts(gold_labels, pred_labels):
    """Pool tp/fp/fn over a list of label sequences, B-ANS and I-ANS positive."""
    tp = fp = fn = 0
    for gold, pred in zip(gold_labels, pred_labels):
        if len(gold) != len(pred):
            raise ValueError(f"{len(gold)} gold labels but {len(pred)} predicted")
        for g, p in zip(gold, pred):
            g_pos = g != _spans.OUTSIDE
            p_pos = p != _spans.OUTSIDE
            if g_pos and p_pos:
                tp += 1
            elif p_pos:
                fp += 1
            elif g_pos:
                fn += 1
    return tp, fp, fn


def span_metrics(encoded, predicted_labels):
    """Metrics for one subset. `encoded` comes from spans.encode_split.

    Returns a flat dict so it can go straight into a DataFrame row, matching the
    layout `answerability_metrics` already uses.
    """
    if len(encoded) != len(predicted_labels):
        raise ValueError(f"{len(predicted_labels)} predictions for {len(encoded)} rows")

    gold_labels = [e["labels"] for e in encoded]
    tp, fp, fn = _token_counts(gold_labels, predicted_labels)
    token = _prf(tp, fp, fn)

    n_exact = 0
    n_exact_ans = 0
    n_exact_unans = 0
    n_text = 0
    n_answerable = 0
    n_pred_span = 0
    n_right_text_wrong_offset = 0
    for e, pred in zip(encoded, predicted_labels):
        pred_start, pred_text = _spans.bio_to_char_span(e["context"], e["offsets"], pred)
        predicted_empty = pred_start is None
        if not predicted_empty:
            n_pred_span += 1
        if e["answerable"]:
            n_answerable += 1
            # Compare against the raw gold span, not the token-aligned one.
            gold_start, gold_text = e["gold_char_span"]
            correct = (not predicted_empty) and (pred_start, pred_text) == (gold_start, gold_text)
            text_correct = (not predicted_empty) and pred_text == gold_text
            n_exact_ans += int(correct)
            n_right_text_wrong_offset += int(text_correct and not correct)
        else:
            correct = predicted_empty
            text_correct = predicted_empty
            n_exact_unans += int(correct)
        n_exact += int(correct)
        n_text += int(text_correct)

    n = len(encoded)
    n_unanswerable = n - n_answerable
    return {
        "n": n,
        "n_answerable": n_answerable,
        "n_unanswerable": n_unanswerable,
        "n_predicted_span": n_pred_span,
        "token_tp": tp,
        "token_fp": fp,
        "token_fn": fn,
        "token_precision": token["precision"],
        "token_recall": token["recall"],
        "token_f1": token["f1"],
        "n_right_text_wrong_offset": n_right_text_wrong_offset,
        "exact_match": n_exact / n if n else 0.0,
        "exact_match_text": n_text / n if n else 0.0,
        "exact_match_answerable": n_exact_ans / n_answerable if n_answerable else 0.0,
        "exact_match_unanswerable": n_exact_unans / n_unanswerable if n_unanswerable else 0.0,
    }


def span_report(encoded, predicted_labels, model_name, langs=config.LANGS):
    """Overall + per-language span metrics, in the layout used in the report.

    Same shape as `answerability_report`: overall first, then one row per
    language in a fixed order, so tables for different models line up.
    """
    rows = [{
        "model": model_name,
        "subset": "overall",
        **span_metrics(encoded, predicted_labels),
    }]
    for lang in langs:
        idx = [i for i, e in enumerate(encoded) if e["lang"] == lang]
        if not idx:
            continue
        rows.append({
            "model": model_name,
            "subset": lang,
            **span_metrics([encoded[i] for i in idx], [predicted_labels[i] for i in idx]),
        })
    return pd.DataFrame(rows)


# ==========================================================================
# Integrated QA metrics (project Sections 1 and 6).
# ==========================================================================
# Section 1: "Integrated QA: SQuAD-v2 EM + token F1 + answerability macro-F1 at
# the chosen threshold." EM and F1 follow the official SQuAD 2.0 evaluation
# script (Rajpurkar et al. 2018):
#
#   * normalisation: lower-case, strip punctuation, drop the articles a/an/the,
#     collapse whitespace;
#   * an UNANSWERABLE gold counts as the empty string, so it scores EM = F1 = 1
#     exactly when the model abstains, and 0 otherwise;
#   * an answerable gold scores 0 on both when the model abstains.
#
# The dataset has one gold answer per question, so there is no max over
# references. Unlike `span_metrics` (Week 37), these are TEXT metrics: a span
# with the right words at a different position still counts as correct, which
# is the SQuAD convention. `span_metrics` stays the position-exact comparison.
#
# Abstention is read from the `answered` column, i.e. from the thresholded
# decision, never inferred from an empty predicted string.

import re as _re
import string as _string
from collections import Counter as _Counter

_PUNCT = set(_string.punctuation)
_ARTICLES = _re.compile(r"\b(a|an|the)\b", _re.UNICODE)


def normalize_answer(text: str) -> str:
    """SQuAD normalisation: lower, no punctuation, no articles, single spaces."""
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def squad_exact(gold: str, pred: str) -> int:
    return int(normalize_answer(gold) == normalize_answer(pred))


def squad_f1(gold: str, pred: str) -> float:
    gold_toks = normalize_answer(gold).split()
    pred_toks = normalize_answer(pred).split()
    if not gold_toks or not pred_toks:
        # Either side empty: 1 if both are, else 0 (official script).
        return float(gold_toks == pred_toks)
    common = _Counter(gold_toks) & _Counter(pred_toks)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(pred_toks)
    recall = same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def qa_example_scores(pred: pd.DataFrame) -> pd.DataFrame:
    """Per-example EM and F1. Needs `answerable`, `gold_answer`, `answered`, `pred_text`."""
    out = pred.copy()
    gold = np.where(out["answerable"].to_numpy(bool), out["gold_answer"].astype(str), "")
    text = np.where(out["answered"].to_numpy(bool), out["pred_text"].astype(str), "")
    out["em"] = [squad_exact(g, p) for g, p in zip(gold, text)]
    out["f1"] = [squad_f1(g, p) for g, p in zip(gold, text)]
    return out


def qa_metrics(pred: pd.DataFrame) -> Dict[str, float]:
    """SQuAD-v2 EM/F1 (overall, answerable-only, unanswerable-only) + answerability."""
    scored = qa_example_scores(pred)
    ans = scored["answerable"].to_numpy(bool)
    out: Dict[str, float] = {
        "exact_match": float(scored["em"].mean()) if len(scored) else 0.0,
        "f1": float(scored["f1"].mean()) if len(scored) else 0.0,
        "has_ans_exact_match": float(scored.loc[ans, "em"].mean()) if ans.any() else 0.0,
        "has_ans_f1": float(scored.loc[ans, "f1"].mean()) if ans.any() else 0.0,
        "no_ans_exact_match": float(scored.loc[~ans, "em"].mean()) if (~ans).any() else 0.0,
    }
    out.update(answerability_metrics(ans, scored["answered"].to_numpy(bool)))
    return out


def qa_report(pred: pd.DataFrame, model_name: str, langs: Iterable[str] = config.LANGS) -> pd.DataFrame:
    """Overall + per-language QA metrics, same layout as the other reports."""
    rows = [{"model": model_name, "subset": "overall", **qa_metrics(pred)}]
    for lang in langs:
        sub = pred[pred["lang"] == lang]
        if len(sub):
            rows.append({"model": model_name, "subset": lang, **qa_metrics(sub)})
    return pd.DataFrame(rows)
