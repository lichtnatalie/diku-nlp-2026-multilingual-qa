#!/usr/bin/env python3
"""
Week 41 (project Section 6, part 2): paired comparisons on the same validation
examples, and the combined pipeline "classifier abstains, QA model extracts".

Run: "python -m scripts.week41_paired" from the project root.
     "--skip-rnn" leaves out the recurrent classifier (no PyTorch needed).

What it does
  1. Rebuilds the per-example answerability decisions of the Week 39 models,
     which run_week39.py did not save: the logistic regression (deterministic)
     and the BiGRU + attention classifier under its five seeds, each at its own
     validation-tuned threshold, with exactly the Week 39 settings. A check
     compares every rebuilt macro-F1 with results/40_ and 42_; the script stops
     if they disagree, so nothing below rests on models that differ from the
     ones in the report.
  2. Reads the integrated QA model's per-example predictions for its three
     seeds (results/66_week40_validation_predictions.csv, tuned thresholds).
  3. Paired answerability comparison: for each pair of models, the 2 x 2 table
     of correct / incorrect decisions on the same examples, per language, with
     McNemar's exact test on the discordant pairs. QA seed i is paired with
     BiGRU seed i (i = 1..3); LR is deterministic and is paired with every QA
     seed.
  4. Combined pipeline: the classifier decides answerable or not, the QA model
     supplies its best non-empty span whenever the classifier says answerable.
     Scored with the same SQuAD-v2 code as the QA model (evaluation.qa_report).
     An "oracle abstention" row (gold answerability + QA span) shows how much
     of the QA model's EM is lost to abstention and how much to extraction.
  5. The per-example table that the Week 41 error analysis samples from.

Outputs: results/68_week41_* .. 71_week41_*

Nothing here touches a threshold on anything but validation, and the challenge
set is not used.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import answerability_models as am  # noqa: E402
from src import config, data, evaluation, reporting  # noqa: E402

DEV_FRACTION = 0.10  # as run_week39.py
RESULTS = config.RESULTS_DIR


def split_dev(train, seed=config.SEED):
    """Identical to run_week39.split_dev."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train))
    n_dev = int(round(DEV_FRACTION * len(train)))
    return train.iloc[idx[n_dev:]].reset_index(drop=True), train.iloc[idx[:n_dev]].reset_index(drop=True)


def week39_settings():
    env = json.loads((RESULTS / "environment_week39.json").read_text())
    lr_c = float(env["logistic_regression"]["C"])
    rec = env["recurrent"]
    rnn_cfg = {k: rec[k] for k in ("hidden_dim", "embedding_dim", "dropout", "lr")}
    return lr_c, rnn_cfg, int(rec["epochs"]), int(rec["max_context_len"]), list(rec["seeds"])


def mcnemar(a_correct, b_correct):
    """2 x 2 counts and McNemar's exact (binomial) p-value on the discordant pairs."""
    a, b = np.asarray(a_correct, bool), np.asarray(b_correct, bool)
    both, only_a, only_b, neither = int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum()), int((~a & ~b).sum())
    disc = only_a + only_b
    p = binomtest(only_a, disc, 0.5).pvalue if disc else 1.0
    return {"n": int(a.size), "both_correct": both, "only_a_correct": only_a,
            "only_b_correct": only_b, "both_wrong": neither, "mcnemar_exact_p": float(p)}


def paired_rows(valid, name_a, correct_a, name_b, correct_b, seed_pair):
    rows = []
    for subset in ("overall",) + config.LANGS:
        m = np.ones(len(valid), bool) if subset == "overall" else (valid["lang"] == subset).to_numpy()
        for gold_name, g in (("all", np.ones(len(valid), bool)),
                             ("answerable", valid["answerable"].to_numpy(bool)),
                             ("unanswerable", ~valid["answerable"].to_numpy(bool))):
            mask = m & g
            rows.append({"model_a": name_a, "model_b": name_b, "seed_pair": seed_pair,
                         "subset": subset, "gold": gold_name,
                         **mcnemar(correct_a[mask], correct_b[mask])})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-rnn", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    splits = data.load_project_splits()
    train, valid = splits["train"], splits["validation"]
    y = valid["answerable"].to_numpy(bool)
    lr_c, rnn_cfg, rnn_epochs, max_ctx, rnn_seeds = week39_settings()
    w39 = pd.read_csv(RESULTS / "40_week39_answerability_results.csv")
    w39_seeds = pd.read_csv(RESULTS / "42_week39_rnn_seed_sweep.csv")

    # ---------------------------------------------------------------- 1. LR
    lr = am.FeatureAnswerabilityClassifier(C=lr_c).fit(train)
    lr_p = lr.predict_proba_answerable(valid)
    lr_t = evaluation.tune_threshold(lr_p, y, higher_means_answerable=True)
    want = w39[(w39["model"] == "logistic regression (tuned threshold)") & (w39["subset"] == "overall")]["macro_f1"].iloc[0]
    print(f"LR rebuilt: threshold {lr_t['threshold']:.4f}, macro-F1 {lr_t['macro_f1']:.4f} (report {want:.4f})")
    checks = [{"model": "logistic regression", "seed": "", "rebuilt_macro_f1": lr_t["macro_f1"], "reported_macro_f1": want}]
    decisions = {"lr": lr_p >= lr_t["threshold"]}
    table = valid[["example_id", "lang", "answerable", "question", "answer", "context"]].copy()
    table["lr_p_answerable"] = lr_p
    table["lr_answered"] = decisions["lr"]

    # ------------------------------------------------------------ 1b. BiGRU
    if not args.skip_rnn:
        fit_part, dev = split_dev(train)
        vocab = am.build_vocabulary(fit_part)
        for seed in rnn_seeds:
            t = time.time()
            m = am.RecurrentAnswerabilityClassifier(
                vocab, epochs=rnn_epochs, max_context_len=max_ctx, seed=seed, **rnn_cfg
            ).fit(fit_part, dev)
            p = m.predict_proba_answerable(valid)
            cut = evaluation.tune_threshold(p, y, higher_means_answerable=True)
            ref = w39_seeds[(w39_seeds["seed"] == seed) & (w39_seeds["threshold_kind"] == "tuned")
                            & (w39_seeds["subset"] == "overall")]["macro_f1"].iloc[0]
            print(f"BiGRU seed {seed}: macro-F1 {cut['macro_f1']:.4f} (report {ref:.4f}), {time.time()-t:.0f}s", flush=True)
            checks.append({"model": "BiGRU", "seed": seed, "rebuilt_macro_f1": cut["macro_f1"], "reported_macro_f1": ref})
            decisions[f"bigru_{seed}"] = p >= cut["threshold"]
            table[f"bigru_{seed}_p_answerable"] = p
            table[f"bigru_{seed}_answered"] = decisions[f"bigru_{seed}"]

    checks = pd.DataFrame(checks)
    checks["abs_diff"] = (checks["rebuilt_macro_f1"] - checks["reported_macro_f1"]).abs()
    reporting.save_table(checks, "68_week41_rebuild_check")
    if (checks["abs_diff"] > 0.005).any():
        print(checks.to_string(index=False))
        print("WARNING: a rebuilt model differs from the report by more than .005 macro-F1. "
              "Paired results below use the rebuilt models; check the environment before using them.")

    # ------------------------------------------------------------- 2. QA
    qa = pd.read_csv(RESULTS / "66_week40_validation_predictions.csv", keep_default_na=False)
    qa_seeds = sorted(qa["seed"].unique())
    qa_by_seed = {}
    for s in qa_seeds:
        q = qa[qa["seed"] == s].set_index("example_id").loc[valid["example_id"]].reset_index()
        q["answered"] = q["answered"].astype(str).eq("True")
        q["answerable"] = q["answerable"].astype(str).eq("True")
        assert (q["answerable"].to_numpy() == y).all()
        qa_by_seed[s] = q
        table[f"qa_{s}_answered"] = q["answered"].to_numpy()
        table[f"qa_{s}_span_text"] = q["span_text"].to_numpy()
        table[f"qa_{s}_no_answer_score"] = q["no_answer_score"].to_numpy()
        table[f"qa_{s}_em"] = q["em"].astype(int).to_numpy()
        table[f"qa_{s}_f1"] = q["f1"].astype(float).to_numpy()

    # --------------------------------------------------- 3. paired decisions
    rows = []
    rnn_seed_list = [s for s in rnn_seeds if f"bigru_{s}" in decisions]
    for i, s in enumerate(qa_seeds):
        qa_ok = qa_by_seed[s]["answered"].to_numpy() == y
        rows += paired_rows(valid, "integrated QA", qa_ok, "logistic regression", decisions["lr"] == y, f"QA {s} / LR")
        if i < len(rnn_seed_list):
            r = rnn_seed_list[i]
            rows += paired_rows(valid, "integrated QA", qa_ok, "BiGRU", decisions[f"bigru_{r}"] == y, f"QA {s} / BiGRU {r}")
    paired = pd.DataFrame(rows)
    reporting.save_table(paired, "69_week41_answerability_paired")
    print(paired[paired["gold"] == "all"][["seed_pair", "subset", "both_correct", "only_a_correct",
                                           "only_b_correct", "both_wrong", "mcnemar_exact_p"]].to_string(index=False))

    # ------------------------------------------------ 4. combined pipeline
    frames = []
    for i, s in enumerate(qa_seeds):
        base = qa_by_seed[s][["example_id", "lang", "answerable", "gold_answer", "span_text", "answered"]].copy()
        base["pred_text"] = np.where(base["answered"], base["span_text"], "")
        frames.append(evaluation.qa_report(base, "integrated QA").assign(seed=str(s), abstention="QA null score"))
        variants = [("logistic regression", decisions["lr"]), ("gold (oracle)", y)]
        if i < len(rnn_seed_list):
            variants.insert(1, ("BiGRU", decisions[f"bigru_{rnn_seed_list[i]}"]))
        for name, dec in variants:
            comb = base.copy()
            comb["answered"] = dec
            comb["pred_text"] = np.where(dec, comb["span_text"], "")
            label = f"{name} abstention + QA span"
            frames.append(evaluation.qa_report(comb, label).assign(seed=str(s), abstention=name))
    combined = pd.concat(frames, ignore_index=True)
    reporting.save_table(combined, "70_week41_combined_pipeline")
    keep = ["model", "seed", "subset", "exact_match", "f1", "has_ans_exact_match", "no_ans_exact_match", "macro_f1"]
    mean = combined.groupby(["model", "subset"], sort=False)[keep[3:]].agg(["mean", "std"])
    print(mean.round(3).to_string())

    reporting.save_table(table, "71_week41_validation_decisions")
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
