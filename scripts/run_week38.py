#!/usr/bin/env python3
"""
Week 38 (project Section 4): one language-model experiment, on Arabic questions.

Run: "python -m scripts.run_week38" from the project root. Takes no arguments.

Protocol
  * Train only on the Arabic questions of the training split.
  * Hyperparameters (add-k's k, the interpolation weights) are chosen on a
    held-out 10% of those TRAINING questions, with the vocabulary rebuilt
    from the other 90% so that the held-out slice has unknown words just as
    validation does. Validation is used once, for the reported numbers.
  * Every model shares one vocabulary and one tokenisation, so their
    perplexities are comparable with each other - and with nothing else.
"""

import itertools
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, data, language_model as lm, reporting  # noqa: E402

LANG = "ar"
MIN_COUNT = 2
DEV_FRACTION = 0.10
K_GRID = (1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001)
LAMBDA_GRID = tuple(round(x, 2) for x in np.arange(0.05, 1.0, 0.05))


def banner(text):
    print("\n" + "=" * 78 + "\n" + text + "\n" + "=" * 78)


def fit_models(train_sents, vocab, params):
    """Every model of the comparison, fitted on the same mapped sentences."""
    mapped = lm.map_unknown(train_sents, vocab)
    targets = set(vocab) | {lm.UNK, lm.EOS}
    return {
        "uniform": lm.UniformLM(targets),
        "unigram": lm.InterpolatedNGramLM(order=1).fit(mapped),
        "add-k bigram": lm.AddKBigramLM(k=params["k"]).fit(mapped, vocab),
        "interpolated bigram": lm.InterpolatedNGramLM(
            order=2, lambdas={2: params["bigram_l2"]}).fit(mapped),
        "interpolated trigram": lm.InterpolatedNGramLM(
            order=3, lambdas={2: params["trigram_l2"], 3: params["trigram_l3"]}).fit(mapped),
    }


def tune(train_sents, rng):
    """Choose k and the lambdas on a held-out slice of the training questions."""
    idx = rng.permutation(len(train_sents))
    n_dev = int(round(DEV_FRACTION * len(train_sents)))
    dev = [train_sents[i] for i in idx[:n_dev]]
    fit = [train_sents[i] for i in idx[n_dev:]]
    vocab = lm.build_vocabulary(fit, MIN_COUNT)
    fit_m, dev_m = lm.map_unknown(fit, vocab), lm.map_unknown(dev, vocab)
    rows = []
    for k in K_GRID:
        pp, _ = lm.perplexity(lm.AddKBigramLM(k=k).fit(fit_m, vocab), dev_m)
        rows.append({"model": "add-k bigram", "k": k, "lambda2": np.nan,
                     "lambda3": np.nan, "dev_perplexity": pp})
    for l2 in LAMBDA_GRID:
        pp, _ = lm.perplexity(lm.InterpolatedNGramLM(2, {2: l2}).fit(fit_m), dev_m)
        rows.append({"model": "interpolated bigram", "k": np.nan, "lambda2": l2,
                     "lambda3": np.nan, "dev_perplexity": pp})
    for l2, l3 in itertools.product(LAMBDA_GRID, LAMBDA_GRID):
        pp, _ = lm.perplexity(lm.InterpolatedNGramLM(3, {2: l2, 3: l3}).fit(fit_m), dev_m)
        rows.append({"model": "interpolated trigram", "k": np.nan, "lambda2": l2,
                     "lambda3": l3, "dev_perplexity": pp})
    grid = pd.DataFrame(rows)
    best = grid.loc[grid.groupby("model")["dev_perplexity"].idxmin()].set_index("model")
    params = {
        "k": float(best.loc["add-k bigram", "k"]),
        "bigram_l2": float(best.loc["interpolated bigram", "lambda2"]),
        "trigram_l2": float(best.loc["interpolated trigram", "lambda2"]),
        "trigram_l3": float(best.loc["interpolated trigram", "lambda3"]),
    }
    info = {"n_fit": len(fit), "n_dev": len(dev), "dev_vocab": len(vocab),
            "dev_unk_rate": sum(w == lm.UNK for s in dev_m for w in s)
            / sum(len(s) for s in dev_m)}
    return grid, best.reset_index(), params, info


def position_label(i, n_words):
    if i == n_words:
        return "</s>"
    return {0: "word 1", 1: "word 2"}.get(i, "word 3+")


def main():
    if len(sys.argv) > 1:
        raise SystemExit("run_week38.py takes no arguments; it always reads data/")

    banner("0. Arabic questions, tokenised and normalised")
    splits = data.load_project_splits()
    # The unit is the QUESTION. Validation pairs some questions with more than
    # one context (74 repeated rows among 415 Arabic ones; training has none),
    # and scoring a question once per pairing would weight it by how many
    # passages it happened to be paired with.
    rows_per_split = {name: int((df["lang"] == LANG).sum()) for name, df in splits.items()}
    # THE UNIT IS THE QUESTION. The language model never sees the passage, so a
    # question that the split pairs with two passages is one linguistic event,
    # not two; scoring it once per pairing would weight it by how the QA data
    # was assembled. Every headline number below is therefore over the distinct
    # validation questions, and the figure over all supplied rows is kept as a
    # diagnostic column. Training carries no repeated question at all, so the
    # choice only ever concerns validation.
    allrow_frames = {name: df[df["lang"] == LANG].reset_index(drop=True)
                     for name, df in splits.items()}
    assert allrow_frames["train"]["question"].nunique() == len(allrow_frames["train"]), \
        "training questions are no longer unique; the note in the report is stale"
    frames = {name: df.drop_duplicates("question").reset_index(drop=True)
              for name, df in allrow_frames.items()}
    sents = {name: [lm.tokenise_question(q) for q in df["question"]]
             for name, df in frames.items()}
    allrow_sents = {name: [lm.tokenise_question(q) for q in df["question"]]
                    for name, df in allrow_frames.items()}
    raw_types = len({w for q in frames["train"]["question"]
                     for w in lm.unicode_word_tokenise(lm.normalise_digits(q))})
    vocab = lm.build_vocabulary(sents["train"], MIN_COUNT)
    train_counts = Counter(w for s in sents["train"] for w in s)
    stats = []
    for name, ss in sents.items():
        n_words = sum(len(s) for s in ss)
        stats.append({
            "split": name, "n_rows": rows_per_split[name],
            "n_questions": len(ss),
            "n_distinct_questions": len(sents[name]), "n_words": n_words,
            "n_tokens_scored": n_words + len(ss),
            "n_types": len({w for s in ss for w in s}),
            "unk_rate": sum(w not in vocab for s in ss for w in s) / n_words,
            "unk_rate_scored": sum(w not in vocab for s in ss for w in s) / (n_words + len(ss)),
            "types_unseen_in_train": len({w for s in ss for w in s} - set(train_counts)),
            "tokens_unseen_in_train": sum(w not in train_counts for s in ss for w in s) / n_words,
        })
    stats = pd.DataFrame(stats)
    stats["vocabulary_size"] = len(vocab)
    stats["min_count"] = MIN_COUNT
    stats["train_types_before_normalisation"] = raw_types
    stats["train_hapax_types"] = sum(1 for n in train_counts.values() if n == 1)
    reporting.save_table(stats, "30_week38_lm_data")
    print(stats.to_string(index=False))

    banner("1. Tune on a held-out 10% of the TRAINING questions")
    rng = np.random.default_rng(config.SEED)
    grid, best, params, info = tune(sents["train"], rng)
    reporting.save_table(grid, "31_week38_lm_tuning_dev")
    print(best.to_string(index=False))
    print(f"  chosen: {params}")
    print(f"  tuning split: {info}")

    banner("2. Refit on all training questions, evaluate once on validation")
    t0 = time.perf_counter()
    models = fit_models(sents["train"], vocab, params)
    fit_seconds = time.perf_counter() - t0
    mapped = {name: lm.map_unknown(ss, vocab) for name, ss in sents.items()}
    mapped_allrows = {name: lm.map_unknown(ss, vocab) for name, ss in allrow_sents.items()}
    results = []
    for name, model in models.items():
        row = {"model": name, "order": model.order}
        for split in ("train", "validation"):
            pp, n = lm.perplexity(model, mapped[split])
            row[f"{split}_perplexity"] = pp
            row[f"{split}_tokens"] = n
        # Diagnostic, not a replacement: the same model scored only on the
        # positions whose target is a real vocabulary word. Unknown targets are
        # one cheap token, so this shows how much the headline number owes them.
        lps = []
        for s in mapped["validation"]:
            for (h, w), lp in zip(lm.events(s, model.order), model.sentence_logprobs(s)):
                if w != lm.UNK:
                    lps.append(lp)
        row["validation_perplexity_known_targets"] = math.exp(-sum(lps) / len(lps))
        row["validation_known_targets"] = len(lps)
        # Diagnostic: the same model over every supplied row, i.e. the split
        # as given, which weights a question by how many passages it was
        # paired with. Reported beside the headline so the choice is visible.
        pp_rows, n_rows = lm.perplexity(model, mapped_allrows["validation"])
        row["validation_perplexity_all_rows"] = pp_rows
        row["validation_all_row_tokens"] = n_rows
        row.update({"k": params["k"] if name == "add-k bigram" else np.nan,
                    "lambda2": {"interpolated bigram": params["bigram_l2"],
                                "interpolated trigram": params["trigram_l2"]}.get(name, np.nan),
                    "lambda3": params["trigram_l3"] if name == "interpolated trigram" else np.nan,
                    "dev_perplexity": (best.set_index("model")["dev_perplexity"].get(name, np.nan))})
        results.append(row)
    results = pd.DataFrame(results)
    results["n_train_questions"] = len(sents["train"])
    results["fit_seconds_all_models"] = fit_seconds
    reporting.save_table(results, "32_week38_lm_results")
    print(results[["model", "dev_perplexity", "train_perplexity", "validation_perplexity",
                   "validation_perplexity_known_targets", "validation_tokens"]]
          .to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    # The model analysed below is the one the DEV slice prefers, so the choice
    # is not made on validation either.
    candidates = results[results["model"] != "uniform"]
    chosen = candidates.loc[candidates["dev_perplexity"].idxmin(), "model"]
    selected = models[chosen]
    print(f"\n  model chosen on the dev slice for the analysis: {chosen}")

    banner("3. What does each model capture?")
    # Section 4 asks for a hypothesis per model, so each model needs evidence of
    # its own. Two contrasts do most of the work:
    #   * word-order shuffle: a model that conditions on nothing scores a
    #     shuffled question exactly as it scores the original one;
    #   * seen vs unseen events: two models fitted on the same counts can only
    #     differ in how they treat n-grams they never saw.
    shuffled = lm.shuffle_words(mapped["validation"], np.random.default_rng(config.SEED))
    diag_rows = []
    for name, each in models.items():
        pp, _ = lm.perplexity(each, mapped["validation"])
        pp_shuffled, _ = lm.perplexity(each, shuffled)
        by_event = lm.perplexity_by_event(each, mapped["validation"])
        diag_rows.append({
            "model": name, "order": each.order,
            "validation_perplexity": pp,
            "perplexity_shuffled_words": pp_shuffled,
            "shuffle_ratio": pp_shuffled / pp,
            "perplexity_seen_event": by_event[True][0],
            "n_seen_event": by_event[True][1],
            "perplexity_unseen_event": by_event[False][0],
            "n_unseen_event": by_event[False][1],
        })
    diagnostics = pd.DataFrame(diag_rows)
    diagnostics["share_unseen_event"] = (
        diagnostics["n_unseen_event"]
        / (diagnostics["n_seen_event"] + diagnostics["n_unseen_event"]))
    reporting.save_table(diagnostics, "37_week38_lm_model_diagnostics")
    print(diagnostics.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # The five easiest and five hardest questions for EVERY model, not just the
    # one analysed in depth below.
    extremes = []
    for name, each in models.items():
        if name == "uniform":
            continue   # every question has the same per-token perplexity
        rows = []
        for i, (orig, s_mapped) in enumerate(zip(sents["validation"], mapped["validation"])):
            lp = each.sentence_logprobs(s_mapped)
            rows.append({"model": name,
                         "example_id": frames["validation"].iloc[i]["example_id"],
                         "question": frames["validation"].iloc[i]["question"],
                         "tokens": " ".join(s_mapped),
                         "n_words": len(s_mapped),
                         "n_unk": sum(w == lm.UNK for w in s_mapped),
                         "perplexity": math.exp(-sum(lp) / len(lp))})
        frame = pd.DataFrame(rows).sort_values("perplexity")
        extremes.append(frame.head(5).assign(end="lowest"))
        extremes.append(frame.tail(5).assign(end="highest"))
    extremes = pd.concat(extremes, ignore_index=True)
    reporting.save_table(extremes, "38_week38_lm_extremes")
    print(extremes[["model", "end", "perplexity", "n_unk", "tokens"]]
          .to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    banner("4. Per-question perplexity on validation")
    rows, pos_rows = [], []
    for i, (orig, s) in enumerate(zip(sents["validation"], mapped["validation"])):
        lp = selected.sentence_logprobs(s)
        rec = frames["validation"].iloc[i]
        rows.append({
            "example_id": rec["example_id"], "question": rec["question"],
            "first_word": orig[0] if orig else "",
            "n_words": len(s), "n_scored": len(lp),
            "n_unk": sum(w == lm.UNK for w in s),
            "unk_fraction": sum(w == lm.UNK for w in s) / max(len(s), 1),
            "log_prob": sum(lp), "perplexity": math.exp(-sum(lp) / len(lp)),
            "tokens": " ".join(s),
        })
        for j, ((h, w), l) in enumerate(zip(lm.events(s, selected.order), lp)):
            pos_rows.append({"position": position_label(j, len(s)),
                             "target_is_unk": w == lm.UNK, "surprisal_bits": -l / math.log(2)})
    per_q = pd.DataFrame(rows).sort_values("perplexity").reset_index(drop=True)
    # The per-question table must come from the model it is labelled with: its
    # pooled perplexity has to equal that model's headline validation figure.
    pooled = math.exp(-per_q["log_prob"].sum() / per_q["n_scored"].sum())
    headline = results.set_index("model").loc[chosen, "validation_perplexity"]
    assert abs(pooled - headline) < 1e-6 * headline, (pooled, headline, chosen)
    per_q.insert(0, "model", chosen)
    reporting.save_table(per_q, "33_week38_lm_question_perplexity")
    cols = ["perplexity", "n_words", "n_unk", "first_word", "tokens"]
    print("  lowest:\n" + per_q.head(10)[cols].to_string(index=False))
    print("  highest:\n" + per_q.tail(10)[cols].to_string(index=False))

    pos = pd.DataFrame(pos_rows)
    by_pos = (pos.groupby(["position", "target_is_unk"])["surprisal_bits"]
              .agg(["count", "mean"]).reset_index())
    total_bits = pos["surprisal_bits"].sum()
    share = pos.groupby("position")["surprisal_bits"].agg(["count", "mean", "sum"]).reset_index()
    share["share_of_total_bits"] = share["sum"] / total_bits
    share["share_of_positions"] = share["count"] / len(pos)
    share["target_is_unk"] = "any"
    by_pos = pd.concat([by_pos, share.drop(columns="sum")], ignore_index=True)
    by_pos.insert(0, "model", chosen)
    reporting.save_table(by_pos, "34_week38_lm_position_surprisal")
    print(by_pos.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    corr = {
        "spearman_perplexity_unk_fraction": per_q["perplexity"].corr(per_q["unk_fraction"], method="spearman"),
        "spearman_perplexity_n_words": per_q["perplexity"].corr(per_q["n_words"], method="spearman"),
    }
    frequent = [w for w, _ in Counter(s[0] for s in sents["train"] if s).most_common(5)]
    per_q["frame"] = np.where(per_q["first_word"].isin(frequent), "top-5 opening word", "other opening")
    groups = []
    for key in ("frame",):
        for value, sub in per_q.groupby(key):
            pooled = math.exp(-sub["log_prob"].sum() / sub["n_scored"].sum())
            groups.append({"grouping": key, "group": str(value), "n_questions": len(sub),
                           "pooled_perplexity": pooled,
                           "median_question_perplexity": sub["perplexity"].median(),
                           "unk_rate": sub["n_unk"].sum() / sub["n_words"].sum()})
    groups = pd.DataFrame(groups)
    groups.insert(0, "model", chosen)
    for k, v in corr.items():
        groups[k] = v
    groups["top5_opening_words"] = " ".join(frequent)
    reporting.save_table(groups, "35_week38_lm_groups")
    print(groups.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    banner("5. The unknown-token threshold changes what perplexity measures")
    # NOT a model comparison: each row has a different vocabulary, which is
    # exactly why the rows are not comparable. Reported to show the direction.
    rows = []
    for mc in (2, 3, 5, 10):
        v = lm.build_vocabulary(sents["train"], mc)
        tr, va = lm.map_unknown(sents["train"], v), lm.map_unknown(sents["validation"], v)
        m = lm.InterpolatedNGramLM(3, {2: params["trigram_l2"], 3: params["trigram_l3"]}).fit(tr)
        pp, n = lm.perplexity(m, va)
        rows.append({"min_count": mc, "vocabulary_size": len(v) + 2,
                     "validation_unk_rate": sum(w == lm.UNK for s in va for w in s)
                     / sum(len(s) for s in va),
                     "validation_perplexity": pp})
    sens = pd.DataFrame(rows)
    reporting.save_table(sens, "36_week38_lm_threshold_sensitivity")
    print(sens.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    banner("Done")
    print("Tables written to results/30_week38_* .. 38_week38_*")


if __name__ == "__main__":
    main()
