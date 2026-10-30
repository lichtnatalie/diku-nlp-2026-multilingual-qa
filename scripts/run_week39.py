#!/usr/bin/env python3
"""
Week 39 (project Section 5): two learned answerability classifiers.

Run: "python -m scripts.run_week39" from the project root. Takes no arguments.

Protocol
  * Both models are fitted on the training split and evaluated on validation,
    with the Section 1 metrics (per-class P/R/F1, macro-F1, N, per language).
  * Hyperparameters are chosen on a held-out 10% slice of the TRAINING split
    (`dev`), never on validation: the regularisation strength of the logistic
    regression, and the recurrent model's size, dropout, learning rate and
    number of epochs.
  * Only the DECISION THRESHOLD is tuned on validation, which is what Section 1
    prescribes for the no-answer decision. Every model is scored at BOTH the
    default 0.5 cut and its tuned cut, so the size of that tuning is visible.
  * The recurrent model is stochastic, so it is fitted under five seeds. Its
    reported numbers are the MEAN over those seeds, per subset and per metric,
    and the per-seed spread is saved alongside. Each seed gets its own
    threshold, tuned on validation, so the tuned rows are means over five
    independently tuned models.

Reporting convention (one, used everywhere)
  Every reported number is the mean over the seeds a model is fitted under
  (one for the deterministic models, five for the recurrent one) at that
  model's tuned threshold; the default-0.5 rows are kept for comparison and
  the spread goes to the appendix.

Outputs: results/40_week39_* .. 47_week39_*
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import (  # noqa: E402
    answerability_models as am,
    baselines,
    config,
    data,
    evaluation,
    reporting,
)

DEV_FRACTION = 0.10
C_GRID = (0.03, 0.1, 0.3, 1.0, 3.0)
RNN_GRID = (
    {"hidden_dim": 64, "embedding_dim": 64, "dropout": 0.3, "lr": 1e-3},
    {"hidden_dim": 32, "embedding_dim": 32, "dropout": 0.3, "lr": 1e-3},
    {"hidden_dim": 64, "embedding_dim": 64, "dropout": 0.5, "lr": 5e-4},
    {"hidden_dim": 32, "embedding_dim": 64, "dropout": 0.5, "lr": 1e-3},
)
RNN_EPOCHS = 5
MAX_CONTEXT = 200


def banner(text):
    print("\n" + "=" * 78 + "\n" + text + "\n" + "=" * 78, flush=True)


def split_dev(train, seed=config.SEED):
    """A held-out slice of the training split, for hyperparameter choices."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train))
    n_dev = int(round(DEV_FRACTION * len(train)))
    return train.iloc[idx[n_dev:]].reset_index(drop=True), train.iloc[idx[:n_dev]].reset_index(drop=True)


def macro_f1(df, proba, threshold=0.5):
    m = evaluation.answerability_metrics(
        df["answerable"].to_numpy(dtype=bool), proba >= threshold
    )
    return m["macro_f1"]


def main():
    t0 = time.time()
    splits = data.load_project_splits()
    train, valid = splits["train"], splits["validation"]
    fit_part, dev = split_dev(train)
    y_valid = valid["answerable"].to_numpy(dtype=bool)

    banner("0. Data")
    print(f"train {len(train)} (fit {len(fit_part)} / dev {len(dev)}), validation {len(valid)}")
    print(train.groupby("lang")["answerable"].agg(["size", "mean"]).to_string())
    print(valid.groupby("lang")["answerable"].agg(["size", "mean"]).to_string())

    # ------------------------------------------------------------------
    banner("1. Interpretable baseline: logistic regression")
    tuning = []
    for C in C_GRID:
        model = am.FeatureAnswerabilityClassifier(C=C).fit(fit_part)
        tuning.append(
            {
                "model": "logistic regression",
                "hyperparameter": f"C={C}",
                "dev_macro_f1": macro_f1(dev, model.predict_proba_answerable(dev)),
            }
        )
        print(tuning[-1], flush=True)
    best_C = float(
        max(tuning, key=lambda r: r["dev_macro_f1"])["hyperparameter"].split("=")[1]
    )
    print(f"chosen on the dev slice: C = {best_C}")

    t_lr = time.time()
    lr_model = am.FeatureAnswerabilityClassifier(C=best_C).fit(train)
    lr_seconds = time.time() - t_lr
    lr_proba = lr_model.predict_proba_answerable(valid)
    lr_thresh = evaluation.tune_threshold(lr_proba, y_valid, higher_means_answerable=True)
    print(lr_model.describe())
    print(f"fit in {lr_seconds:.1f}s; tuned threshold {lr_thresh['threshold']:.3f} "
          f"(macro-F1 {lr_thresh['macro_f1']:.3f}, 0.5 gives {macro_f1(valid, lr_proba):.3f})")

    coef = lr_model.coefficients(top=12)
    reporting.save_table(coef, "43_week39_lr_coefficients")
    print(coef.to_string(index=False))

    # Feature-block ablation: which representation carries the signal.
    ablation = []
    for name, kw in (
        ("all features", {}),
        ("question BoW only", {"use_overlap": False, "use_length": False}),
        ("overlap + length only", {"use_question_bow": False}),
        ("overlap only", {"use_question_bow": False, "use_length": False}),
        ("no overlap", {"use_overlap": False}),
    ):
        m = am.FeatureAnswerabilityClassifier(C=best_C, **kw).fit(train)
        p = m.predict_proba_answerable(valid)
        row = {"features": name, "n_features": len(m.feature_names_)}
        row.update(evaluation.answerability_metrics(y_valid, p >= 0.5))
        ablation.append(row)
        print(f"{name:24s} macro-F1 {row['macro_f1']:.3f}", flush=True)
    ablation = pd.DataFrame(ablation)
    reporting.save_table(ablation, "44_week39_feature_ablation")

    # ------------------------------------------------------------------
    banner("2. Recurrent classifier: grid on the dev slice")
    vocab = am.build_vocabulary(fit_part)
    print(f"vocabulary {len(vocab)} types (questions + contexts of the fit part)")
    for cfg in RNN_GRID:
        t = time.time()
        m = am.RecurrentAnswerabilityClassifier(
            vocab, epochs=RNN_EPOCHS, max_context_len=MAX_CONTEXT, seed=config.SEED, **cfg
        ).fit(fit_part, dev)
        best = max(m.history_, key=lambda r: r["dev_macro_f1"])
        tuning.append(
            {
                "model": "recurrent",
                "hyperparameter": json.dumps(cfg),
                "dev_macro_f1": best["dev_macro_f1"],
                "best_epoch": best["epoch"],
                "seconds": time.time() - t,
            }
        )
        print(tuning[-1], flush=True)
    best_rnn = max(
        (r for r in tuning if r["model"] == "recurrent"), key=lambda r: r["dev_macro_f1"]
    )
    rnn_cfg = json.loads(best_rnn["hyperparameter"])
    rnn_epochs = int(best_rnn["best_epoch"])
    print(f"chosen on the dev slice: {rnn_cfg}, {rnn_epochs} epochs")
    reporting.save_table(pd.DataFrame(tuning), "41_week39_hyperparameter_tuning")

    banner("3. Recurrent classifier: five seeds")
    # Every seed is scored on every subset, at the default cut and at its own
    # threshold tuned on validation, so the reported row can be the MEAN over
    # seeds of each metric rather than one seed's numbers.
    seed_frames, seed_models, seed_probas, seed_thresholds = [], {}, {}, {}
    for seed in config.SEEDS:
        t = time.time()
        m = am.RecurrentAnswerabilityClassifier(
            vocab,
            epochs=rnn_epochs,
            max_context_len=MAX_CONTEXT,
            seed=seed,
            **rnn_cfg,
        ).fit(fit_part, dev)
        p = m.predict_proba_answerable(valid)
        seed_models[seed], seed_probas[seed] = m, p
        cut_ = evaluation.tune_threshold(p, y_valid, higher_means_answerable=True)
        seed_thresholds[seed] = float(cut_["threshold"])
        for kind, thr in (("default", 0.5), ("tuned", seed_thresholds[seed])):
            f = evaluation.answerability_report(valid, p >= thr, "recurrent")
            f.insert(0, "seed", seed)
            f.insert(1, "threshold_kind", kind)
            f.insert(2, "threshold", thr)
            f["train_seconds"] = m.train_seconds_
            f["n_train_examples"] = m.n_train_examples_
            seed_frames.append(f)
        overall = seed_frames[-1].iloc[0]
        print(f"seed {seed}: macro-F1 {overall['macro_f1']:.3f} at its tuned cut "
              f"{seed_thresholds[seed]:.3f} in {time.time()-t:.0f}s", flush=True)
    seeds_df = pd.concat(seed_frames, ignore_index=True)
    reporting.save_table(seeds_df, "42_week39_rnn_seed_sweep")
    print(
        seeds_df[seeds_df["subset"] == "overall"][
            ["seed", "threshold_kind", "threshold", "macro_f1", "answerable_f1", "unanswerable_f1"]
        ].to_string(index=False)
    )

    def seed_mean(kind, name):
        """Mean over seeds of every metric, subset by subset."""
        sub = seeds_df[seeds_df["threshold_kind"] == kind]
        num = [c for c in sub.columns
               if c not in ("seed", "threshold_kind", "model", "subset")]
        out = (sub.groupby("subset", sort=False)[num].mean().reset_index())
        # Counts are identical across seeds; keep them as integers.
        for c in ("n", "answerable_support", "unanswerable_support"):
            if c in out:
                out[c] = out[c].round().astype(int)
        out.insert(0, "model", name)
        order = ["overall"] + [l for l in config.LANGS]
        out["_o"] = out["subset"].map({s: i for i, s in enumerate(order)})
        return out.sort_values("_o").drop(columns="_o").reset_index(drop=True)

    # The first seed is kept only for the things that need ONE fitted model:
    # the attention inspection and the interaction ablation.
    headline_seed = config.SEEDS[0]
    rnn_model, rnn_proba = seed_models[headline_seed], seed_probas[headline_seed]
    # Recorded for the appendix only: the MEAN of the per-seed tuned cuts. No
    # result is computed at this value -- each seed is scored at its own cut,
    # and 42_week39_rnn_seed_sweep.csv carries all five.
    rnn_thresh = {"threshold": float(np.mean(list(seed_thresholds.values())))}

    # Interaction ablation: does attention (and the context at all) matter?
    banner("4. Interaction ablation")
    inter_rows = []
    for name, kw in (
        ("question + context + attention", {}),
        ("question + context, no attention", {"use_attention": False}),
        ("question only", {"use_context": False}),
        ("context only", {"use_question": False}),
    ):
        m = am.RecurrentAnswerabilityClassifier(
            vocab, epochs=rnn_epochs, max_context_len=MAX_CONTEXT,
            seed=headline_seed, **rnn_cfg, **kw
        ).fit(fit_part, dev)
        p = m.predict_proba_answerable(valid)
        row = {"variant": name}
        row.update(evaluation.answerability_metrics(y_valid, p >= 0.5))
        inter_rows.append(row)
        print(f"{name:34s} macro-F1 {row['macro_f1']:.3f}", flush=True)
    reporting.save_table(pd.DataFrame(inter_rows), "45_week39_interaction_ablation")

    # ------------------------------------------------------------------
    banner("5. Results table, against the Week 36 baselines")
    majority = baselines.MajorityClassBaseline().fit(train)
    anchor = baselines.CrossLingualAnchorRule().fit(train)
    frames = [
        evaluation.answerability_report(valid, majority.predict(valid), "majority class (W36)"),
        evaluation.answerability_report(valid, anchor.predict(valid), "anchor rule (W36)"),
        evaluation.answerability_report(valid, lr_proba >= 0.5, "logistic regression"),
        evaluation.answerability_report(
            valid, lr_proba >= lr_thresh["threshold"], "logistic regression (tuned threshold)"
        ),
        # The recurrent rows are means over the five seeds, not one seed.
        seed_mean("default", "recurrent (BiGRU + attention)"),
        seed_mean("tuned", "recurrent (tuned threshold)"),
    ]
    results = pd.concat(frames, ignore_index=True)
    results.insert(1, "split", "validation")
    reporting.save_table(results, "40_week39_answerability_results")
    print(results[["model", "subset", "n", "answerable_f1", "unanswerable_f1", "macro_f1"]]
          .to_string(index=False))

    # ------------------------------------------------------------------
    banner("6. Errors, for the manual analysis")
    rule = baselines.CrossLingualAnchorRule()
    train_questions = set(train["question"])
    conflicting = {
        q for q, g in valid.groupby("question")["answerable"] if g.nunique() > 1
    }
    rows = []
    for i, row in enumerate(valid.itertuples(index=False)):
        anchors = rule.anchors(row.question)
        hits = rule.hits(row.question, row.context)
        for model_name, proba in (("logistic regression", lr_proba), ("recurrent", rnn_proba)):
            pred = bool(proba[i] >= 0.5)
            if pred == bool(row.answerable):
                continue
            rows.append(
                {
                    "model": model_name,
                    "example_id": row.example_id,
                    "lang": row.lang,
                    "gold": "answerable" if row.answerable else "unanswerable",
                    # Positive class = unanswerable (config.POSITIVE_CLASS), so a
                    # false positive is "predicted unanswerable, gold answerable".
                    "error_type": "false positive" if row.answerable else "false negative",
                    "p_answerable": float(proba[i]),
                    "question": row.question,
                    "answer": row.answer,
                    "n_anchors": len(anchors),
                    "n_hits": hits,
                    "question_seen_in_train": row.question in train_questions,
                    "question_has_conflicting_labels": row.question in conflicting,
                    "context_tokens": len(row.context.split()),
                    "context": row.context,
                }
            )
    errors = pd.DataFrame(rows)
    reporting.save_table(errors, "46_week39_errors_all")
    print(errors.groupby(["model", "lang", "error_type"]).size().to_string())

    # A stratified sample for manual inspection: both error types, all three
    # languages, for both models. Section 5 asks for at least ten.
    rng = np.random.default_rng(config.SEED)
    sample = []
    for model_name in errors["model"].unique():
        for lang in config.LANGS:
            for etype in ("false positive", "false negative"):
                sub = errors[
                    (errors["model"] == model_name)
                    & (errors["lang"] == lang)
                    & (errors["error_type"] == etype)
                ]
                if len(sub) == 0:
                    continue
                take = sub.iloc[rng.permutation(len(sub))[:2]]
                sample.append(take)
    sample = pd.concat(sample, ignore_index=True)
    reporting.save_table(sample, "47_week39_error_sample")
    print(f"\nsampled {len(sample)} errors for manual inspection")

    # Attention for the sampled recurrent errors, as inspection evidence.
    att_rows = []
    for row in sample[sample["model"] == "recurrent"].itertuples(index=False):
        p, tokens = rnn_model.attention_for(row.question, row.context, top=6)
        att_rows.append(
            {
                "example_id": row.example_id,
                "lang": row.lang,
                "error_type": row.error_type,
                "p_answerable": p,
                "top_attended": ", ".join(f"{t} ({w:.2f})" for t, w in tokens),
            }
        )
    if att_rows:
        reporting.save_table(pd.DataFrame(att_rows), "48_week39_attention")

    # ------------------------------------------------------------------
    banner("7. Week 38 language-model score as a diagnostic (appendix)")
    # The Section 5 brief allows the Week 38 LM score as a feature or a
    # diagnostic. Week 38 found that a low question perplexity marks a familiar
    # interrogative FRAME rather than an answerable question, so the honest
    # claim to test is weak: does the Arabic trigram's per-question perplexity
    # separate the two classes on validation at all?
    # Unit of evaluation: the Week 38 models score each DISTINCT question once,
    # but answerability is a property of the question-context PAIR, and the same
    # question occurs with more than one passage (sometimes with different
    # labels). The score is therefore attached to every Arabic validation pair
    # by question text, so all of them are scored, not only the deduplicated
    # representatives.
    lm_rows = []
    pp_path = config.RESULTS_DIR / "33_week38_lm_question_perplexity.csv"
    if pp_path.exists():
        pp = pd.read_csv(pp_path)
        pp = pp[pp["model"] == "interpolated trigram"][["question", "perplexity"]]
        arabic = valid[valid["lang"] == "ar"]
        merged = arabic.merge(pp, on="question", how="inner")
        if len(merged) != len(arabic):
            print(f"warning: {len(arabic) - len(merged)} Arabic validation rows "
                  "have no Week 38 perplexity", flush=True)
        if len(merged):
            a = merged.loc[merged["answerable"], "perplexity"]
            u = merged.loc[~merged["answerable"], "perplexity"]
            # AUC via the rank-sum identity, no extra dependency.
            ranks = merged["perplexity"].rank()
            auc = (ranks[~merged["answerable"]].sum() - len(u) * (len(u) + 1) / 2) / (
                len(u) * len(a)
            )
            lm_rows.append(
                {
                    "subset": "Arabic validation question-context pairs",
                    "n": len(merged),
                    "n_distinct_questions": int(merged["question"].nunique()),
                    "n_answerable": int(len(a)),
                    "n_unanswerable": int(len(u)),
                    "median_pp_answerable": float(a.median()),
                    "median_pp_unanswerable": float(u.median()),
                    "auc_unanswerable_from_perplexity": float(auc),
                }
            )
            print(pd.DataFrame(lm_rows).to_string(index=False))
            reporting.save_table(pd.DataFrame(lm_rows), "49_week39_lm_diagnostic")

    # ------------------------------------------------------------------
    banner("8. Environment")
    env = reporting.environment_record(
        {
            "week": 39,
            # Week 39 did NOT run on the laptop that produced Weeks 36-38, so
            # the appendix must report this machine separately.
            "hardware": reporting.hardware_record(),
            "logistic_regression": {
                "C": best_C,
                "n_features": len(lr_model.feature_names_),
                "fit_seconds": lr_seconds,
                "n_train_examples": len(train),
                "threshold_default": 0.5,
                "threshold_tuned_on_validation": lr_thresh["threshold"],
            },
            "recurrent": {
                **rnn_cfg,
                "epochs": rnn_epochs,
                "max_context_len": MAX_CONTEXT,
                "vocabulary": len(vocab),
                "seeds": list(config.SEEDS),
                "mean_train_seconds": float(seeds_df["train_seconds"].mean()),
                "n_train_examples": int(seeds_df["n_train_examples"].iloc[0]),
                "threshold_tuned_on_validation": rnn_thresh["threshold"],
            },
            "total_run_seconds": time.time() - t0,
        }
    )
    path = config.RESULTS_DIR / "environment_week39.json"
    path.write_text(json.dumps(env, indent=2, ensure_ascii=False))
    print(json.dumps(env, indent=2, ensure_ascii=False)[:1200])

    banner("Done")
    print(f"total {time.time() - t0:.0f}s. Tables written to results/40_week39_* .. 49_week39_*")


if __name__ == "__main__":
    main()
