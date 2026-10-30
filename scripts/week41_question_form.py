#!/usr/bin/env python3
"""
Week 41 diagnostic: how much of answerability is predictable from the FORM of
the question (polar yes/no vs wh-question), per language, and what the
integrated QA model does with polar questions.

Run: "python -m scripts.week41_question_form" from the project root.

Polar-question detection (deterministic, no labels used):
  ar  the question starts with the interrogative particle هل (hal). Exact.
  ko  the question contains no wh-word stem (누구/누가/누군, 무엇/무슨/뭐,
      어디/어딘, 언제/언젠, 몇, 어느/어떤/어떻, 왜, 얼마). A HEURISTIC: the list of
      flagged validation questions is saved so it can be checked by hand.
  te  not attempted. Telugu wh-words are fused with case endings (ఏమని,
      పేరేమిటి, ఎన్నవది, దేనితో, ...) and a stem list misfires; the Week 39
      inspection found no polar questions among the Telugu errors.

Outputs: results/76_week41_question_form.*, results/77_week41_korean_polar_flags.*
"""

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, data, reporting  # noqa: E402

KO_WH = re.compile(r"(누구|누가|누군|무엇|무슨|뭐|어디|어딘|언제|언젠|몇|어느|어떤|어떻|어땠|왜|얼마)")


def is_polar(lang: str, q: str):
    q = q.strip()
    if lang == "ar":
        return q.startswith("هل")
    if lang == "ko":
        return KO_WH.search(q) is None
    return None


def main():
    splits = data.load_project_splits()
    qa = pd.read_csv(config.RESULTS_DIR / "66_week40_validation_predictions.csv", keep_default_na=False)
    qa["answered"] = qa["answered"].astype(str).eq("True")
    answered_rate = qa.groupby("example_id")["answered"].mean()  # mean over the 3 seeds

    rows = []
    for split in config.SPLITS:
        df = splits[split]
        df = df[df["lang"].isin(("ar", "ko"))].copy()
        df["polar"] = [is_polar(l, q) for l, q in zip(df["lang"], df["question"])]
        if split == "validation":
            df["qa_answered"] = df["example_id"].map(answered_rate)
        for (lang, polar), g in df.groupby(["lang", "polar"]):
            rows.append({
                "split": split, "lang": lang, "question_form": "polar" if polar else "wh",
                "n": len(g), "unanswerable": int((~g["answerable"]).sum()),
                "share_unanswerable": float((~g["answerable"]).mean()),
                "share_of_language_unanswerables": float((~g["answerable"]).sum() / (~df[df["lang"] == lang]["answerable"]).sum()),
                "qa_answered_rate_mean_3_seeds": float(g["qa_answered"].mean()) if split == "validation" else float("nan"),
            })
        if split == "validation":
            ko = df[(df["lang"] == "ko") & df["polar"]][["example_id", "question", "answerable", "qa_answered"]]
            reporting.save_table(ko, "77_week41_korean_polar_flags")
    out = pd.DataFrame(rows)
    reporting.save_table(out, "76_week41_question_form")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
