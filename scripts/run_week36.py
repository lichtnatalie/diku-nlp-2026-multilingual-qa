#!/usr/bin/env python3
"""Week 36 end-to-end run: dataset audit (Items 1-3) + the two baselines.

Usage
-----
    python -m scripts.run_week36            # from the project root

Reads the parquet files in data/ (see README for how they get there).  The
synthetic fixture in tests/ is used by the test suite only -- it exists to
exercise failure paths that the real data does not contain, not as a stand-in
for the data.

Every table is written to results/ as both .csv and .md.  Nothing is printed
that is not also saved, so the report can be written from results/ alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import audit, baselines, config, data, evaluation, reporting  # noqa: E402
from src.tokenisation import DEFAULT_TOKENISER  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)


def banner(text: str) -> None:
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def main() -> None:
    banner("0. Load")
    raw = data.load_raw()
    inventory = data.language_inventory(raw)
    reporting.save_table(inventory, "00_language_inventory")
    print(inventory.to_string(index=False))
    kept = inventory[inventory["in_project_scope"] & inventory["split"].isin(config.SPLITS)]["n"].sum()
    total = inventory[inventory["split"].isin(config.SPLITS)]["n"].sum()
    print(f"\nkeeping {kept} / {total} train+validation rows after the ar/ko/te filter")

    splits = data.load_project_splits(raw)
    for name, df in splits.items():
        print(f"  {name}: {len(df)} rows, columns = {list(df.columns)}")

    # ------------------------------------------------------------------ Item 1
    banner("1. Item 1 - counts, proportions, length distributions")
    desc = pd.concat(
        [audit.describe_split(df, split, DEFAULT_TOKENISER) for split, df in splits.items()],
        ignore_index=True,
    )
    reporting.save_table(desc, "01_item1_describe")
    print(
        desc[
            [
                "split", "lang", "n", "n_answerable", "n_unanswerable",
                "prop_answerable", "q_len_median", "q_len_iqr",
                "ctx_len_median", "ctx_len_iqr",
                "dup_pair_groups", "dup_pair_redundant_rows",
            ]
        ].to_string(index=False)
    )

    banner("1b. Item 1 - missing values")
    miss = pd.concat(
        [audit.missing_values(df, split) for split, df in splits.items()], ignore_index=True
    )
    reporting.save_table(miss, "02_item1_missing_values")
    print(miss.to_string(index=False))

    banner("1c. Item 1 - duplicate question-context pairs")
    for split, df in splits.items():
        d = audit.duplicate_pairs(df)
        inconsistent = audit.label_inconsistent_duplicates(df)
        print(
            f"{split}: {d['n_groups']} duplicated pairs covering {d['n_rows_in_groups']} rows "
            f"({d['n_redundant_rows']} redundant); {inconsistent} of those groups have "
            f"conflicting answerability labels"
        )
    ex = audit.duplicate_pair_examples(splits["train"])
    if len(ex):
        reporting.save_table(ex, "03_item1_duplicate_examples")
        print("\nexample duplicate groups (train):")
        print(ex.to_string(index=False))

    overlap = audit.cross_split_overlap(splits)
    # Saved, not just printed: "no question-context pair occurs in both splits"
    # is a validity claim the report makes, so it needs a result file behind it.
    reporting.save_table(pd.DataFrame([overlap]), "03b_item1_cross_split_overlap")
    print(f"\ntrain/validation overlap: {overlap}")

    banner("1d. Tokeniser comparison (justifies the choice in the report)")
    tok = audit.tokeniser_comparison(splits["train"])
    reporting.save_table(tok, "04_tokeniser_comparison")
    print(tok.to_string(index=False))

    # ------------------------------------------------------------------ Item 2
    banner("2. Item 2 - five most common question tokens per language (train only)")
    top = audit.top_question_tokens(splits["train"], k=5)
    reporting.save_table(top, "05_item2_top_question_tokens")
    print(top.to_string(index=False))
    todo = top[~top["translation_verified"]]
    if len(todo):
        print(
            f"\n!! {len(todo)} token(s) are not in the glossary and need a hand translation "
            f"before this goes in the report:"
        )
        print(todo[["lang", "token", "count"]].to_string(index=False))

    # ------------------------------------------------------------------ Item 3
    banner("3. Item 3 - answer_start / answer consistency")
    verif_rows = []
    for split, df in splits.items():
        res = audit.verify_answer_offsets(df, split)
        print(
            f"{split}: checked {res['n_answerable_checked']} answerable examples, "
            f"{res['n_failures']} failures ({res['failure_rate']:.6f})"
        )
        for lang, v in res["per_language"].items():
            print(f"    {lang}: checked {v['checked']}, failures {v['failures']}")
        print(
            f"    unanswerable rows: {res['n_unanswerable']}, "
            f"violating the (-1, '') convention: {res['n_unanswerable_violating_convention']}"
        )
        print(
            f"    answerable rows whose answer string occurs more than once in the context: "
            f"{res['n_answers_occurring_more_than_once']}  (matters for Week 37 BIO alignment)"
        )
        verif_rows.append(
            {
                "split": split,
                "n_answerable_checked": res["n_answerable_checked"],
                "n_failures": res["n_failures"],
                "failure_rate": res["failure_rate"],
                "n_unanswerable": res["n_unanswerable"],
                "n_unanswerable_violating_convention": res["n_unanswerable_violating_convention"],
                "n_answers_occurring_more_than_once": res["n_answers_occurring_more_than_once"],
                **{f"{l}_checked": v["checked"] for l, v in res["per_language"].items()},
                **{f"{l}_failures": v["failures"] for l, v in res["per_language"].items()},
            }
        )
        if len(res["failures"]):
            reporting.save_table(res["failures"], f"06_item3_failures_{split}")
    reporting.save_table(pd.DataFrame(verif_rows), "06_item3_offset_verification")

    # --------------------------------------------------------------- baselines
    banner("4. Baseline (i): majority class")
    maj = baselines.MajorityClassBaseline().fit(splits["train"])
    print(maj.describe())

    banner("5. Baseline (ii): cross-lingual anchor rule")
    rule = baselines.CrossLingualAnchorRule().fit(splits["train"])
    print(rule.describe())

    diag = rule.anchor_diagnostics(splits["train"])
    reporting.save_table(diag, "07_anchor_diagnostics_train")
    print("\nanchor diagnostics (train):")
    print(diag.to_string(index=False))

    sweep = rule.sweep_min_hits(splits["train"], values=(1, 2, 3))
    reporting.save_table(sweep, "08_anchor_min_hits_sweep_train")
    print("\nmin_hits sweep, TRAIN only (validation labels must not be used here):")
    print(sweep[["min_hits", "n", "macro_f1", "answerable_f1", "unanswerable_f1", "accuracy"]].to_string(index=False))

    ex = baselines.anchor_examples(rule, splits["train"])
    reporting.save_table(ex, "09_anchor_worked_examples")
    print("\nworked examples:")
    print(ex.to_string(index=False))

    banner("6. Shared evaluation protocol (Section 1)")
    rule_nolabel = baselines.CrossLingualAnchorRule(fallback=True).fit(splits["train"])
    models = {
        "majority-class": maj,
        "anchor-rule": rule,
        "anchor-rule (label-free fallback)": rule_nolabel,
    }
    reports = []
    for split, df in splits.items():
        for name, model in models.items():
            rep = evaluation.answerability_report(df, model.predict(df), name)
            rep.insert(1, "split", split)
            reports.append(rep)
    results = pd.concat(reports, ignore_index=True)
    reporting.save_table(results, "10_week36_answerability_results")

    cols = [
        "model", "split", "subset", "n",
        "answerable_support", "unanswerable_support",
        "answerable_precision", "answerable_recall", "answerable_f1",
        "unanswerable_precision", "unanswerable_recall", "unanswerable_f1",
        "macro_f1", "accuracy",
    ]
    print(results[cols].to_string(index=False))

    print("\nheadline (validation, overall):")
    print(
        results[(results["split"] == "validation") & (results["subset"] == "overall")][
            ["model", "n", "macro_f1", "answerable_f1", "unanswerable_f1", "accuracy"]
        ].to_string(index=False)
    )

    path = reporting.save_environment(
        extra={
            "tokeniser": DEFAULT_TOKENISER,
            # Which files actually produced these numbers (size + SHA-256
            # prefix per split).  Goes into the reproducibility appendix and
            # catches a silently swapped local parquet between runs.
            "data_provenance": data.data_provenance(),
        }
    )
    banner(f"Done. Tables in {config.RESULTS_DIR}, environment record at {path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(f"{Path(__file__).name} takes no arguments (got {sys.argv[1:]})")
    main()
