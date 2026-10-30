#!/usr/bin/env python3
"""
Week 40: score the integrated QA model's spans with the Week 37 span metrics.

Run: "python -m scripts.week40_spans_bio" from the project root. CPU, seconds,
no model needed: it reads the per-example predictions that run_week40.py saved
(results/66_week40_validation_predictions.csv, every seed at its tuned
threshold) and the validation split from data/.

Why: Section 6 asks for the integrated model's span predictions to be compared
with the Week 37 sequence labeller. The labeller is scored by
`evaluation.span_metrics` (answer-token F1 over the Week 37 context tokens and
exact match against the stored character span). Converting each predicted
character span to BIO labels over the SAME tokens (`qa_model.char_spans_to_bio`)
lets the integrated model be scored by that same function, so the comparison
uses one metric implementation. run_week40.py did this for the first seed only
(results/65_*); this does it for every seed, so the report can quote seed
means as it does for the labeller.

Output: results/67_week40_span_metrics_bio_seeds.csv (one row per seed and
subset). It stops if the first seed does not reproduce results/65_* exactly.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, data, evaluation, reporting, spans  # noqa: E402
from src.qa_model import char_spans_to_bio  # noqa: E402

MODEL = "integrated QA (XLM-R, tuned)"


def main():
    pred = pd.read_csv(config.RESULTS_DIR / "66_week40_validation_predictions.csv",
                       keep_default_na=False)
    valid = data.load_project_splits()["validation"]
    encoded = spans.encode_split(valid)
    ids = [e["example_id"] for e in encoded]

    frames = []
    for seed, p in pred.groupby("seed", sort=True):
        if sorted(p["example_id"]) != sorted(ids):
            raise AssertionError(f"seed {seed}: predictions do not cover the validation split")
        p = p.assign(pred_start=p["pred_start"].astype(int), pred_text=p["pred_text"].astype(str))
        bio = char_spans_to_bio(encoded, p)
        rep = evaluation.span_report(encoded, bio, MODEL)
        rep.insert(0, "seed", int(seed))
        frames.append(rep)
    out = pd.concat(frames, ignore_index=True)

    # The first seed was already scored inside run_week40.py: the two must agree.
    first = out[out["seed"] == out["seed"].min()].set_index("subset")
    ref_path = config.RESULTS_DIR / "65_week40_span_metrics_bio.csv"
    if ref_path.exists():
        ref = pd.read_csv(ref_path).set_index("subset")
        for subset in ref.index:
            for col in ("token_f1", "exact_match", "exact_match_unanswerable"):
                if abs(ref.loc[subset, col] - first.loc[subset, col]) > 1e-9:
                    raise AssertionError(f"first seed does not reproduce results/65_*: "
                                         f"{subset} {col}")
        print("first seed reproduces results/65_week40_span_metrics_bio exactly")

    reporting.save_table(out, "67_week40_span_metrics_bio_seeds")
    cols = ["token_f1", "exact_match", "exact_match_answerable", "exact_match_unanswerable"]
    print(out.groupby("subset", sort=False)[cols].agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
