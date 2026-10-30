#!/usr/bin/env python3
"""
Week 40 (project Section 6, part 1): the integrated multilingual QA model.

Run on a GPU (Colab): "python -m scripts.run_week40" from the project root.
`notebooks/week40_integrated_qa.ipynb` does the setup and runs this.
Offline pipeline check on CPU, no download needed: "python -m scripts.run_week40 --smoke".

Protocol (Section 1, and the same choices as Week 39)
  * ONE joint model over ar/ko/te: FacebookAI/xlm-roberta-base with a span
    head, trained on 90% of TRAIN. The epoch is chosen on the held-out 10%
    `dev` slice (the same slice Week 39 tuned on), never on validation.
  * No-answer score = null-span score - best non-empty-span score. HIGH means
    no answer. The threshold is tuned on VALIDATION only, for answerability
    macro-F1, with the shared `evaluation.tune_threshold`. The untuned rule
    (threshold 0: answer iff the best span beats the null span) is reported
    next to it, as Week 39 reported its 0.5 cut.
  * Three seeds (fine-tuning is stochastic, and the Week 39 BiGRU varied by
    SD .024). Reported numbers are the MEAN over seeds at each seed's own
    tuned threshold; the per-seed rows are saved alongside.
  * Metrics: SQuAD-v2 EM and token F1, answerability P/R/F1 per class and
    macro-F1, overall and per language, with N. The spans are ALSO scored with
    the Week 37 BIO metrics (`evaluation.span_metrics`), so the Week 41
    comparison with the sequence labeller uses one metric implementation.

Outputs (results/):
  60_week40_sequence_lengths      pieces per pair, windows, share over 512
  61_week40_label_audit           training targets checked against gold offsets
  62_week40_epoch_selection       dev-slice history per seed
  63_week40_qa_seeds              per seed x {tuned, untuned} x subset
  64_week40_qa_results            mean over seeds (the reported table)
  65_week40_span_metrics_bio      first seed, scored like the Week 37 labeller
  66_week40_validation_predictions per example, every seed (for Week 41)
  environment_week40.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, data, evaluation, qa_model as qm, reporting, spans  # noqa: E402

N_SEEDS = 3
UNTUNED_THRESHOLD = 0.0
KEY_METRICS = ["exact_match", "f1", "has_ans_exact_match", "has_ans_f1",
               "no_ans_exact_match", "answerable_f1", "unanswerable_f1",
               "unanswerable_precision", "unanswerable_recall", "macro_f1", "accuracy"]


def banner(text):
    print("\n" + "=" * 78 + "\n" + text + "\n" + "=" * 78, flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    d = qm.QAConfig()
    p.add_argument("--model", default=d.model_name)
    p.add_argument("--seeds", type=int, default=N_SEEDS, help="how many of config.SEEDS")
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--lr", type=float, default=d.learning_rate)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--max-length", type=int, default=d.max_length)
    p.add_argument("--stride", type=int, default=d.stride)
    p.add_argument("--save-model", default=None,
                   help="directory to save the first seed's model, tokenizer and threshold "
                        "(needed later to freeze the model for the challenge set)")
    p.add_argument("--smoke", action="store_true",
                   help="tiny offline model on a data subset; writes to results/smoke_week40/")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    if args.smoke:
        config.RESULTS_DIR = config.RESULTS_DIR / "smoke_week40"
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    banner("0. Data and the dev slice")
    splits = data.load_project_splits()
    train, valid = splits["train"], splits["validation"]
    if args.smoke:
        train = train.groupby("lang", group_keys=False).head(60).reset_index(drop=True)
        valid = valid.groupby("lang", group_keys=False).head(30).reset_index(drop=True)
    fit_part, dev = qm.split_dev(train)
    print(f"train {len(train)} (fit {len(fit_part)} / dev {len(dev)}), validation {len(valid)}")

    base_cfg = qm.QAConfig(model_name=args.model, epochs=args.epochs, learning_rate=args.lr,
                           batch_size=args.batch_size, max_length=args.max_length,
                           stride=args.stride)
    if args.smoke:
        texts = pd.concat([train.question, train.context, valid.question, valid.context])
        tokenizer, _ = qm.tiny_test_model(texts, vocab_size=800)
        base_cfg.model_name, base_cfg.epochs = "tiny-test-model", min(args.epochs, 2)
        base_cfg.learning_rate, base_cfg.batch_size = 1e-3, 8
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_cfg.model_name)
    print(f"model {base_cfg.model_name}, max_length {base_cfg.max_length}, stride {base_cfg.stride}")

    banner("1. Sequence lengths (justifies max_length and stride)")
    lengths = pd.concat([qm.sequence_length_audit(df, tokenizer, base_cfg).assign(split=name)
                         for name, df in (("train", train), ("validation", valid))],
                        ignore_index=True)
    lengths = lengths[["split"] + [c for c in lengths.columns if c != "split"]]
    reporting.save_table(lengths, "60_week40_sequence_lengths", floatfmt="%.1f")
    print(lengths.to_string(index=False))

    banner("2. Training targets vs gold character offsets")
    audit_rows = []
    for name, df in (("train", train), ("validation", valid)):
        feats = qm.build_features(df, tokenizer, base_cfg, with_labels=True)
        audit_rows.append({"split": name, "windows": len(feats["input_ids"]),
                           **qm.label_audit(df, feats)})
    audit = pd.DataFrame(audit_rows)
    reporting.save_table(audit, "61_week40_label_audit", floatfmt="%d")
    print(audit.to_string(index=False))
    for r in audit_rows:
        if r["labelled_in_a_window"] != r["answerable"]:
            raise AssertionError(f"{r['split']}: an answerable example is labelled in no window")
        if r["unanswerable_all_cls"] != r["unanswerable"]:
            raise AssertionError(f"{r['split']}: an unanswerable example has a span target")
        if r["exact_char_match"] + r["wider_than_gold"] != r["answerable"]:
            raise AssertionError(f"{r['split']}: a labelled span does not cover its gold characters")

    banner(f"3. Fine-tune under {args.seeds} seeds")
    seeds = list(config.SEEDS[: args.seeds])
    seed_rows, history_rows, pred_frames, per_seed = [], [], [], {}
    first_model = None
    for seed in seeds:
        print(f"\n--- seed {seed} ---", flush=True)
        cfg = qm.QAConfig(**{**base_cfg.to_dict(), "seed": seed})
        if args.smoke:
            _, tiny = qm.tiny_test_model(tokenizer=tokenizer, seed=seed)
            model = qm.IntegratedQAModel(cfg, tokenizer=tokenizer, model=tiny)
        else:
            model = qm.IntegratedQAModel(cfg, tokenizer=tokenizer)
        model.fit(fit_part, dev)
        for h in model.history_:
            history_rows.append({"seed": seed, **h, "selected": h["epoch"] == model.best_epoch_})

        raw = model.predict_scores(valid)
        cut = qm.tune_no_answer_threshold(raw, cfg.threshold_objective)
        for kind, thr in (("tuned", cut["threshold"]), ("untuned (0)", UNTUNED_THRESHOLD)):
            decided = qm.apply_threshold(raw, thr)
            rep = evaluation.qa_report(decided, "integrated QA (XLM-R)")
            rep.insert(0, "threshold_kind", kind)
            rep.insert(0, "seed", seed)
            rep["threshold"] = thr
            seed_rows.append(rep)
        tuned = evaluation.qa_example_scores(qm.apply_threshold(raw, cut["threshold"]))
        tuned.insert(0, "seed", seed)
        pred_frames.append(tuned)
        per_seed[seed] = {"best_epoch": model.best_epoch_,
                          "train_seconds": model.train_seconds_,
                          "n_train_windows": model.n_train_features_,
                          "threshold_tuned_on_validation": cut["threshold"]}
        o = seed_rows[-2].iloc[0]
        print(f"  seed {seed}: epoch {model.best_epoch_}, threshold {cut['threshold']:.3f}, "
              f"EM {o['exact_match']:.3f}, F1 {o['f1']:.3f}, macro-F1 {o['macro_f1']:.3f}",
              flush=True)
        if seed == seeds[0]:
            first_model = model
            if args.save_model:
                out = Path(args.save_model)
                out.mkdir(parents=True, exist_ok=True)
                model.model.save_pretrained(out)
                tokenizer.save_pretrained(out)
                (out / "decision_rule.json").write_text(json.dumps({
                    "no_answer_score": "null score - best span score; HIGH = no answer",
                    "answer_iff": "no_answer_score < threshold",
                    "threshold": cut["threshold"], "seed": seed,
                    "config": cfg.to_dict()}, indent=2))
                print(f"  saved model and decision rule to {out}")
        else:
            del model

    banner("4. Results (mean over seeds, each at its own threshold)")
    history = pd.DataFrame(history_rows)
    reporting.save_table(history, "62_week40_epoch_selection")
    seeds_df = pd.concat(seed_rows, ignore_index=True)
    reporting.save_table(seeds_df, "63_week40_qa_seeds")
    grouped = seeds_df.groupby(["threshold_kind", "subset"], sort=False)
    mean = grouped[["n"] + KEY_METRICS + ["threshold"]].mean().reset_index()
    sd = grouped[["exact_match", "f1", "macro_f1"]].std(ddof=1).add_suffix("_sd").reset_index()
    results = mean.merge(sd, on=["threshold_kind", "subset"])
    results.insert(2, "n_seeds", len(seeds))
    results["n"] = results["n"].astype(int)
    reporting.save_table(results, "64_week40_qa_results")
    print(results[["threshold_kind", "subset", "n", "exact_match", "f1", "has_ans_f1",
                   "no_ans_exact_match", "macro_f1", "f1_sd", "macro_f1_sd"]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))

    banner("5. First seed, scored with the Week 37 span metrics")
    encoded = spans.encode_split(valid)
    first_pred = pred_frames[0]
    bio = qm.char_spans_to_bio(encoded, first_pred)
    span_table = evaluation.span_report(encoded, bio, f"integrated QA (seed {seeds[0]}, tuned)")
    reporting.save_table(span_table, "65_week40_span_metrics_bio")
    print(span_table[["subset", "n", "token_precision", "token_recall", "token_f1",
                      "exact_match", "exact_match_answerable", "exact_match_unanswerable"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    predictions = pd.concat(pred_frames, ignore_index=True)
    predictions = predictions.merge(valid[["example_id", "question", "context"]], on="example_id")
    reporting.save_table(predictions, "66_week40_validation_predictions")

    banner("6. Environment")
    env = reporting.environment_record({
        "week": 40,
        "hardware": reporting.hardware_record(),
        "gpu": _gpu_name(),
        "smoke_test": bool(args.smoke),
        "integrated_qa": {
            **base_cfg.to_dict(),
            "seeds": seeds,
            "per_seed": per_seed,
            "mean_train_seconds": float(np.mean([v["train_seconds"] for v in per_seed.values()])),
            "n_train_examples": len(fit_part),
            "n_dev_examples": len(dev),
            "n_validation_examples": len(valid),
            "threshold_untuned": UNTUNED_THRESHOLD,
            "parameters": int(sum(p.numel() for p in first_model.model.parameters())),
        },
        "data": data.data_provenance(),
        "total_run_seconds": round(time.time() - t0, 1),
    })
    path = config.RESULTS_DIR / "environment_week40.json"
    path.write_text(json.dumps(env, indent=2, ensure_ascii=False))
    print(json.dumps(env["integrated_qa"], indent=2)[:1500])

    banner("Done")
    print(f"total {time.time() - t0:.0f}s. Tables written to {config.RESULTS_DIR}/6*_week40_*")


def _gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        return None


if __name__ == "__main__":
    main()
