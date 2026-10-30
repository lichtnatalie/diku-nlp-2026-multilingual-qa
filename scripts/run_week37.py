#!/usr/bin/env python3
"""
Week 37 end-to-end run: BIO conversion, checks, and the Section 3 models.

Run: "python -m scripts.run_week37" from the project root

Takes no arguments and always reads the files in data/. Every table is written to results/ as .csv.

Follows Section 1: fit on train, tune the abstention threshold on validation, report both splits side by side.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, data, evaluation, reporting, spans, span_models  

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)


def banner(text):
    """Print a section header so the console log is readable."""
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def main():
    if len(sys.argv) > 1:
        raise SystemExit("run_week37.py takes no arguments; it always reads data/")

    banner("0. Load and convert character offsets to BIO")
    splits = data.load_project_splits()
    encoded = {name: spans.encode_split(df) for name, df in splits.items()}
    for name, enc in encoded.items():
        print(f"  {name}: {len(enc)} rows, {sum(len(e['tokens']) for e in enc)} context tokens")
        # The assumption the whole decoding constraint rests on. Week 37's lab
        # showed that getting this wrong silences the model rather than breaking
        # it, so the failure would be invisible in the precision column.
        spans.assert_scheme([e["labels"] for e in enc])
        print(f"  {name}: gold BIO scheme verified (IOB2, 0 violations)")

    banner("1. Section 3 automatic checks")
    checks = spans.conversion_checks()
    reporting.save_table(checks, "20_week37_conversion_checks")
    print(checks.to_string(index=False))
    # Section 3 names four cases explicitly. Assert that all four are actually
    # present, not just that whatever cases exist happen to pass - otherwise
    # deleting a case would silently satisfy the check.
    required = checks[checks["required_by_brief"]]
    if len(required) != 4:
        raise AssertionError(f"expected the 4 checks Section 3 names, found {len(required)}")
    if not checks["passed"].all():
        raise AssertionError("a Section 3 conversion check failed")
    print(f"\n  {len(required)}/4 checks named by Section 3 present, all passed;"
          f" {len(checks)} checks in total.")

    # The seventh check runs over the real data rather than fixtures.
    unans_rows = []
    for name, enc in encoded.items():
        result = spans.unanswerable_check(enc)
        result["split"] = name
        unans_rows.append(result)
        if result["n_with_non_O_label"]:
            raise AssertionError(f"{name}: an unanswerable row produced a span")
    unans = pd.DataFrame(unans_rows)[
        ["split", "n_unanswerable", "n_with_non_O_label", "n_with_non_empty_answer_string"]
    ]
    reporting.save_table(unans, "21_week37_unanswerable_check", floatfmt="%d")
    print("\n" + unans.to_string(index=False))

    banner("2. Alignment audit (the exact-match ceiling)")
    audits = [spans.alignment_audit(enc, name) for name, enc in encoded.items()]
    audit_table = pd.concat(audits, ignore_index=True)
    reporting.save_table(audit_table, "22_week37_alignment_audit")
    print(audit_table.to_string(index=False))

    banner("3. Fit the question-conditioned labeller on train")
    labeller = span_models.QuestionConditionedLabeller(beam_size=4, seed=config.SEED)
    labeller.fit(encoded["train"])
    print(f"  classes: {labeller.classes_}")
    print(f"  features: {len(labeller.vectoriser.get_feature_names_out())}")

    # Ablation: the identical model with every question-derived feature removed.
    # Section 3 requires the question to influence every context token, and this
    # is the evidence that it actually does rather than merely being wired up.
    ablation = span_models.QuestionConditionedLabeller(
        beam_size=4, seed=config.SEED, use_question=False)
    ablation.fit(encoded["train"])
    print(f"  ablation features: {len(ablation.vectoriser.get_feature_names_out())}")

    # Featurise each split once and reuse the scores for every decoding variant.
    log_probs = {name: labeller.token_log_probs(enc) for name, enc in encoded.items()}
    abl_log_probs = {name: ablation.token_log_probs(enc) for name, enc in encoded.items()}

    banner("4. Decoding comparison (Section 3 model vs the two baselines)")
    empty = span_models.EmptySpanBaseline()
    overlap = span_models.LexicalOverlapSpanBaseline()

    predictions = {}
    for name, enc in encoded.items():
        predictions[name] = {
            empty.name: empty.predict(enc),
            overlap.name: overlap.predict(enc),
            "labeller (independent argmax)": labeller.predict(
                enc, decoding="argmax", log_probs=log_probs[name]),
            "labeller (constrained greedy)": labeller.predict(
                enc, decoding="greedy", log_probs=log_probs[name]),
            "labeller (constrained beam 4)": labeller.predict(
                enc, decoding="beam", log_probs=log_probs[name]),
            "ablation: no question features (beam 4)": ablation.predict(
                enc, decoding="beam", log_probs=abl_log_probs[name]),
        }

    banner("5. Tune the abstention threshold on validation (Section 1)")
    # No-answer score = score(all-O) - score(predicted). HIGH means no answer.
    base = predictions["validation"]["labeller (constrained beam 4)"]
    best_t, best_exact_match, sweep = tune_abstention(
        labeller, encoded["validation"], base, log_probs["validation"])
    reporting.save_table(sweep, "23_week37_abstention_sweep_validation")
    print(f"  best threshold on validation = {best_t:.4f} (exact match {best_exact_match:.4f})")
    print("  direction: a HIGH no-answer score means NO ANSWER, so predictions at")
    print("  or above the threshold are blanked to the empty span.")

    for name, enc in encoded.items():
        scores = labeller.no_answer_scores(
            enc, predictions[name]["labeller (constrained beam 4)"], log_probs=log_probs[name])
        predictions[name]["labeller (beam 4 + abstention)"] = span_models.apply_abstention(
            predictions[name]["labeller (constrained beam 4)"], scores, best_t)

    banner("6. Results (Section 1 layout: overall then per language, with N)")
    for name, enc in encoded.items():
        tables = [
            evaluation.span_report(enc, preds, model)
            for model, preds in predictions[name].items()
        ]
        table = pd.concat(tables, ignore_index=True)
        reporting.save_table(table, f"24_week37_span_results_{name}")
        print(f"\n--- {name} ---")
        cols = ["model", "subset", "n", "n_predicted_span", "token_precision",
                "token_recall", "token_f1", "exact_match",
                "exact_match_answerable", "exact_match_unanswerable"]
        print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    banner("7. Error examples for the report")
    errors = collect_errors(encoded["validation"],
                            predictions["validation"]["labeller (constrained beam 4)"])
    reporting.save_table(errors, "25_week37_error_examples")
    print(errors.head(12).to_string(index=False))
    # The taxonomy is meant to PARTITION the errors, so check that it does. If
    # the answerable categories do not account for every answerable example the
    # model got wrong, the report's error paragraph is quoting a subset without
    # saying so.
    check_taxonomy_is_exhaustive(
        encoded["validation"], errors,
        evaluation.span_metrics(
            encoded["validation"],
            predictions["validation"]["labeller (constrained beam 4)"]))

    banner("8. Predicted span lengths vs gold")
    lengths = span_length_diagnostic(encoded, predictions)
    reporting.save_table(lengths, "26_week37_span_lengths")
    print(lengths.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    banner("9. Seed sweep (Section 1: more than one seed where it could matter)")
    # The two claims that depend on the fit are "each decoding step helps" and
    # "the question features help". Both are differences of a few points, and
    # SGDClassifier shuffles, so a single seed cannot establish either.
    sweep = seed_sweep(encoded, config.SEEDS)
    reporting.save_table(sweep, "27_week37_seed_sweep")
    overall = sweep[sweep["subset"] == "overall"]
    summary = overall.groupby("model", sort=False)[["token_f1", "exact_match"]].agg(
        ["mean", "std", "min", "max"])
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))
    # Table 4 reports the seed means, and the error analysis uses the single
    # run above. Both are only coherent if SEEDS[0] reproduces that run
    # exactly, so check it rather than assume it.
    first = overall[overall["seed"] == config.SEEDS[0]].set_index("model")
    for model in first.index:
        single = evaluation.span_metrics(encoded["validation"], predictions["validation"][model])
        for column in ("token_f1", "exact_match", "exact_match_unanswerable"):
            if abs(single[column] - first.loc[model, column]) > 1e-12:
                raise AssertionError(
                    f"seed sweep's first seed does not reproduce the single run: "
                    f"{model} {column} {first.loc[model, column]} vs {single[column]}")
    print(f"\n  first seed of the sweep reproduces the single run for all "
          f"{len(first)} learned systems.")

    banner("Done")
    print("Tables written to results/20_week37_* .. 27_week37_*")


def tune_abstention(labeller, val_encoded, base, val_log_probs):
    """Choose the abstention cutoff on validation (Section 1).

    No-answer score = score(all-O) - score(predicted); HIGH means no answer.
    Candidate cutoffs are 41 quantiles of the validation scores. Returns the
    chosen cutoff, the exact match it reaches, and the whole sweep.
    """
    val_scores = labeller.no_answer_scores(val_encoded, base, log_probs=val_log_probs)
    grid = np.quantile(val_scores, np.linspace(0, 1, 41))
    best_t = None
    best_exact_match = -1.0
    sweep_rows = []
    for t in grid:
        gated = span_models.apply_abstention(base, val_scores, float(t))
        m = evaluation.span_metrics(val_encoded, gated)
        sweep_rows.append({"threshold": float(t), **m})
        # Tuned on EXACT MATCH, not token F1. Token F1 can never reward
        # abstaining - an all-O prediction scores 0 on it whatever the gold -
        # so it would always pick the loosest threshold. Exact match is the
        # metric where abstention genuinely trades off.
        if m["exact_match"] > best_exact_match:
            best_exact_match, best_t = m["exact_match"], float(t)
    return best_t, best_exact_match, pd.DataFrame(sweep_rows)


def seed_sweep(encoded, seeds):
    """Refit the labeller and its ablation under each seed, score on validation.

    Every learned system in Table 4 is swept, including the abstention variant,
    whose cutoff is re-tuned on validation under each seed exactly as in the
    single run. Rows are written overall AND per language, because Table 4
    reports the seed means for all of its columns. The two baselines are
    deterministic, so a seed cannot move them. Training decoding is skipped
    here because no claim rests on it.
    """
    rows = []
    val = encoded["validation"]
    for seed in seeds:
        full = span_models.QuestionConditionedLabeller(beam_size=4, seed=seed)
        full.fit(encoded["train"])
        ablation = span_models.QuestionConditionedLabeller(
            beam_size=4, seed=seed, use_question=False)
        ablation.fit(encoded["train"])
        lp = full.token_log_probs(val)
        lp_ablation = ablation.token_log_probs(val)
        systems = {
            "labeller (independent argmax)": full.predict(val, "argmax", log_probs=lp),
            "labeller (constrained greedy)": full.predict(val, "greedy", log_probs=lp),
            "labeller (constrained beam 4)": full.predict(val, "beam", log_probs=lp),
            "ablation: no question features (beam 4)": ablation.predict(
                val, "beam", log_probs=lp_ablation),
        }
        beam = systems["labeller (constrained beam 4)"]
        threshold, _, _ = tune_abstention(full, val, beam, lp)
        scores = full.no_answer_scores(val, beam, log_probs=lp)
        systems["labeller (beam 4 + abstention)"] = span_models.apply_abstention(
            beam, scores, threshold)
        for model, preds in systems.items():
            report = evaluation.span_report(val, preds, model)
            report.insert(0, "seed", seed)
            report["abstention_threshold"] = (
                threshold if model == "labeller (beam 4 + abstention)" else np.nan)
            rows.append(report)
        print(f"  seed {seed} done (abstention cutoff {threshold:.4f})")
    return pd.concat(rows, ignore_index=True)


def span_length_diagnostic(encoded, predictions):
    """Mean predicted span length against the gold, per split and model.

    A model that is right about WHERE the answer is but wrong about how far
    it extends looks the same in exact match as one that is lost, so this
    separates the two failure modes.
    """
    rows = []
    for split, enc in encoded.items():
        # Gold span lengths, deliberately the same shape as the prediction
        # loop below: both ask "how many tokens long is the first span?".
        gold_lengths = []
        for e in enc:
            found = spans.find_spans(e["labels"])
            if found:
                start, end = found[0]
                gold_lengths.append(end - start)
        for model, preds in predictions[split].items():
            pred_lengths = []
            for labels in preds:
                found = spans.find_spans(labels)
                if found:
                    start, end = found[0]
                    pred_lengths.append(end - start)
            rows.append({
                "split": split,
                "model": model,
                "n_gold_spans": len(gold_lengths),
                "mean_gold_length": float(np.mean(gold_lengths)) if gold_lengths else 0.0,
                "n_predicted_spans": len(pred_lengths),
                "mean_predicted_length": float(np.mean(pred_lengths)) if pred_lengths else 0.0,
                "median_predicted_length": float(np.median(pred_lengths)) if pred_lengths else 0.0,
            })
    return pd.DataFrame(rows)


ANSWERABLE_ERROR_KINDS = (
    "wrong location",
    "abstained on answerable",
    "boundary too narrow",
    "boundary too wide",
    "partial overlap",
    "right text, wrong occurrence",
)


def check_taxonomy_is_exhaustive(encoded, errors, metrics):
    """Assert the answerable error categories account for every answerable miss.

    n_answerable - (exact matches on answerable) must equal the sum of the
    answerable categories. The unanswerable category is counted separately and
    deliberately not folded in, because those rows are not span errors.
    """
    n_answerable = sum(1 for e in encoded if e["answerable"])
    n_correct = round(metrics["exact_match_answerable"] * n_answerable)
    expected = n_answerable - n_correct
    counts = errors.groupby("kind")["n_total"].first()
    got = int(sum(int(counts[k]) for k in ANSWERABLE_ERROR_KINDS if k in counts.index))
    if got != expected:
        raise AssertionError(
            f"error taxonomy does not partition the answerable errors: "
            f"categories sum to {got}, but {expected} answerable examples are wrong")
    print(f"\n  taxonomy check: {got} answerable errors, fully categorised; "
          f"{int(counts.get('false span on unanswerable', 0))} false spans on "
          f"unanswerable rows are counted separately.")


def error_kind(pred_start, pred_text, gold_start, gold_text):
    """Classify a wrong, non-empty prediction on an answerable example.

    Defined on CHARACTER POSITIONS, not on the strings: a predicted "St" at
    the start of the passage is not a too-narrow version of a gold "St.
    Peter's Basilica" further down, even though one string contains the other.
    With predicted span [ps, pe) and gold span [gs, ge):

      right text, wrong occurrence  same string, different position
      boundary too narrow           gs <= ps and pe <= ge  (inside the gold)
      boundary too wide             ps <= gs and ge <= pe  (contains the gold)
      partial overlap               the spans overlap but each has characters
                                    outside the other (one boundary each side)
      wrong location                the spans share no character
    """
    ps, pe = pred_start, pred_start + len(pred_text)
    gs, ge = gold_start, gold_start + len(gold_text)
    if pred_text == gold_text:
        # Week 36 counted 1,082 answerable rows whose answer appears more than
        # once in the context, which is what makes this category non-empty.
        return "right text, wrong occurrence"
    if gs <= ps and pe <= ge:
        return "boundary too narrow"
    if ps <= gs and ge <= pe:
        return "boundary too wide"
    if ps < ge and gs < pe:
        return "partial overlap"
    return "wrong location"


def collect_errors(encoded, predicted, n_per_kind=5):
    """Sample a few errors of each kind, for the Section 3 error analysis.

    Kinds are chosen so that the categories map onto the failure modes Week 36
    predicted: predicting a span on an unanswerable question, abstaining on an
    answerable one, and getting the boundaries wrong.
    """
    rows = []
    for e, pred in zip(encoded, predicted):
        pred_start, pred_text = spans.bio_to_char_span(e["context"], e["offsets"], pred)
        gold = e["gold_char_span"]
        if not e["answerable"]:
            kind = "false span on unanswerable" if pred_start is not None else None
        elif pred_start is None:
            kind = "abstained on answerable"
        elif (pred_start, pred_text) == gold:
            kind = None
        else:
            kind = error_kind(pred_start, pred_text, gold[0], gold[1])
        if kind is None:
            continue
        rows.append({
            "kind": kind,
            "lang": e["lang"],
            "example_id": e["example_id"],
            "question": e["question"][:60],
            "gold": gold[1] if gold else "",
            "predicted": pred_text,
            "alignment_status": e["status"],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    counts = df["kind"].value_counts().rename_axis("kind").reset_index(name="n_total")
    # Sample per (kind, language) so the table is not dominated by whichever
    # language happens to sit at the start of the split.
    sample = df.groupby(["kind", "lang"], group_keys=False).head(2).groupby(
        "kind", group_keys=False).head(n_per_kind)
    return sample.merge(counts, on="kind").sort_values(["kind", "lang"])


if __name__ == "__main__":
    main()
