#!/usr/bin/env python3
"""
Week 41 (Section 6): the stratified sample of integrated-QA validation errors
for the manual error taxonomy.

Run: "python -m scripts.week41_qa_error_sample" from the project root.

Error types (headline seed, SEEDS[0], at its tuned threshold):
  wrongly abstains  gold answerable,   model returns no answer
  wrongly answers   gold unanswerable, model returns a span
  wrong span        gold answerable,   model answers, exact match 0
For wrong spans the position of the predicted span relative to the gold one is
recorded (pred contains gold / gold contains pred / partial overlap /
disjoint), because boundary errors and wrong-answer errors are different
failures and SQuAD F1 treats them differently.

Sample: 24 errors, 8 per error type, spread over the three languages as evenly
as the counts allow (Arabic has only 2 wrongly answered examples), drawn with
the project seed from errors the model makes under ALL THREE seeds, so the
taxonomy describes systematic behaviour, not seed noise. Wrong spans are split
evenly between boundary errors and disjoint answers within each language.

Outputs: results/72_week41_qa_error_counts.*, results/73_week41_qa_error_sample.*
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, reporting  # noqa: E402

PER_TYPE = 8


def error_type(r):
    if r.answerable and not r.answered:
        return "wrongly abstains"
    if not r.answerable and r.answered:
        return "wrongly answers"
    if r.answerable and r.answered and r.em == 0:
        return "wrong span"
    return "correct"


def span_relation(r):
    if r.error_type != "wrong span":
        return ""
    gs, ps = int(r.gold_start), int(r.pred_start)
    ge, pe = gs + len(r.gold_answer), ps + len(r.pred_text)
    if ps <= gs and pe >= ge:
        return "pred contains gold"
    if ps >= gs and pe <= ge:
        return "gold contains pred"
    if pe <= gs or ps >= ge:
        return "disjoint"
    return "partial overlap"


def main():
    qa = pd.read_csv(config.RESULTS_DIR / "66_week40_validation_predictions.csv", keep_default_na=False)
    for c in ("answered", "answerable"):
        qa[c] = qa[c].astype(str).eq("True")
    qa["em"] = qa["em"].astype(int)
    qa["error_type"] = qa.apply(error_type, axis=1)
    qa["span_relation"] = qa.apply(span_relation, axis=1)

    counts = (qa.groupby(["seed", "error_type", "span_relation", "lang"]).size()
                .unstack("lang", fill_value=0).reset_index())
    reporting.save_table(counts, "72_week41_qa_error_counts")

    head = qa[qa["seed"] == config.SEED].set_index("example_id")
    head["n_seeds_same_error"] = qa.groupby("example_id")["error_type"].agg(lambda x: int((x == x.iloc[0]).sum()))
    pool = head[(head["error_type"] != "correct") & (head["n_seeds_same_error"] == 3)].reset_index()

    rng = np.random.default_rng(config.SEED)
    picks = []
    for etype in ("wrongly abstains", "wrongly answers", "wrong span"):
        sub = pool[pool["error_type"] == etype]
        avail = {l: sub[sub["lang"] == l] for l in config.LANGS}
        # as even a split over languages as availability allows
        quota = {l: 0 for l in config.LANGS}
        for _ in range(PER_TYPE):
            open_ = [l for l in config.LANGS if quota[l] < len(avail[l])]
            l = min(open_, key=lambda x: quota[x])
            quota[l] += 1
        for l, k in quota.items():
            cand = avail[l]
            if etype == "wrong span":
                boundary = cand[cand["span_relation"] != "disjoint"]
                disjoint = cand[cand["span_relation"] == "disjoint"]
                kb = k // 2
                idx = list(rng.choice(boundary.index, kb, replace=False)) + \
                      list(rng.choice(disjoint.index, k - kb, replace=False))
            else:
                idx = list(rng.choice(cand.index, k, replace=False))
            picks.append(cand.loc[idx])
    sample = pd.concat(picks)
    cols = ["example_id", "lang", "error_type", "span_relation", "n_seeds_same_error", "question",
            "gold_answer", "pred_text", "span_text", "no_answer_score", "threshold", "f1", "context"]
    sample = sample[cols].reset_index(drop=True)
    reporting.save_table(sample, "73_week41_qa_error_sample")
    print(pd.crosstab(sample["error_type"], sample["lang"]))
    print(f"pool of errors made under all 3 seeds: {len(pool)}")


if __name__ == "__main__":
    main()
