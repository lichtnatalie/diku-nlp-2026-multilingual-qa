#!/usr/bin/env python3
"""
Week 39, Section 5: the manual error taxonomy.

`run_week39.py` writes a stratified sample of validation errors (both models,
both error types, all three languages) to
``results/47_week39_error_sample.csv``. I read those examples, their questions,
their gold answers and their passages, and assigned each one a category by
hand. This file records that judgement, so the counts in the report are
reproducible from the repository rather than from my notes, and re-running it
re-derives the counts table.

The categories are task- and language-motivated rather than length or
confidence bins, which the project description asks for explicitly:

  polar question
      The question is a yes/no (polar) interrogative: Arabic "hal" questions,
      Korean questions ending in -나요/-는가. An extractive system has no span
      to return for a yes/no question, and the dataset annotates such items
      inconsistently, so both models treat the interrogative form itself as
      evidence and are wrong in both directions.

  unrelated passage
      The question is a well-formed wh-question and the passage is on a
      different subject altogether, so the example is unanswerable for a reason
      that has nothing to do with the question's wording. Both models read the
      question as answerable-looking and answer.

  topic present, fact missing
      The passage is about the entity the question names but does not state the
      fact asked for (the period, but not when it ended; that some states have
      no capital, but not how many). This is the hard case XOR-AttriQA is built
      from.

  no cross-lingual entity link
      The answer is in the passage, but nothing links the entity the question
      ASKS ABOUT to the passage lexically. Two sub-cases, both of which leave
      the overlap features and the shared embedding table with nothing to work
      with: the question yields no anchor at all (7 of the 8 errors here), or
      its only matching anchor is a bare year that identifies no entity (1 of
      them: validation-000005, where the Telugu question about the Right to
      Information Act 2005 matches the passage only on "2005"). The model
      abstains on an answerable example.

      The narrower reading — "no anchor at all" — would misdescribe that one
      error, which does have a matching anchor, so the category is defined by
      what the anchor identifies rather than by the raw hit count.

  questionable gold label
      On inspection the annotated label is hard to defend: the passage does
      state the fact although the row is marked unanswerable, or the passage is
      about a different entity than the question although the row is marked
      answerable.

Each error gets exactly one primary category. Precedence (added 12 Oct, shared
with week41_qa_error_taxonomy.py): "questionable gold label" comes first
whenever the gold answer, or the gold "unanswerable", does not address the
question as asked; the question form is then a secondary note. Three rows were
reclassified from "polar question" on that basis (001630, 001131, 002131). `question_has_conflicting_labels`
(recorded by `run_week39.py`) is kept as a separate flag rather than a
category, because it cuts across all five.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, reporting  # noqa: E402

# example_id -> (category, one-line justification written while inspecting)
# Two rows can share an example_id: the same validation example can be an error
# for both models, and each is judged separately, so the key is (model, id).
JUDGEMENTS = {
    ("logistic regression", "validation-002075"): ("polar question", "Arabic hal-question about UC Davis veterinary degrees, annotated answerable with the span 'medicine'"),
    ("logistic regression", "validation-001630"): ("questionable gold label", "Arabic hal-question whether Paul counts among Christ's disciples (the Twelve); the passage calls him 'an apostle (though not one of the Twelve Apostles)', so the gold 'apostle' does not settle it (secondary: polar question; reclassified 12 Oct under the precedence rule)"),
    ("logistic regression", "validation-002096"): ("unrelated passage", "cause of the Angolan civil war against a passage on the Mongol advance into Syria"),
    ("logistic regression", "validation-001782"): ("questionable gold label", "why St John's wort is called perforatum, and the passage states the reason"),
    ("logistic regression", "validation-001380"): ("no cross-lingual entity link", "average lifetime of a star, answer '10-billion' present, no Latin-script or numeric anchor in the Korean question"),
    ("logistic regression", "validation-000852"): ("no cross-lingual entity link", "NHK founder, answer present, the anchor 'NHK-TV' does not match 'NHK' in the passage"),
    ("logistic regression", "validation-001085"): ("topic present, fact missing", "passage covers the Yamato period but never says when it ended"),
    ("logistic regression", "validation-001131"): ("questionable gold label", "Korean polar question whether a provisional government is one formed after the previous government collapses; the passage's definition supports 'yes', yet the row is annotated unanswerable (secondary: polar question; reclassified 12 Oct under the precedence rule)"),
    ("logistic regression", "validation-000138"): ("no cross-lingual entity link", "which country has the most chickens, 'United States' present, Telugu question has no anchor"),
    ("logistic regression", "validation-000005"): ("no cross-lingual entity link", "which government passed the 2005 information act, 'India' present; the only anchor is the year 2005, which identifies no entity, and the model abstains"),
    ("logistic regression", "validation-002996"): ("unrelated passage", "music director of Munna against a passage on Mullum Malarum"),
    ("logistic regression", "validation-002962"): ("unrelated passage", "music director of Munna against a passage on Malaya Marutha"),
    ("recurrent", "validation-001935"): ("no cross-lingual entity link", "weight of Big Ben, '13 1/2 tons' present, Arabic question carries no anchor"),
    ("recurrent", "validation-002131"): ("questionable gold label", "Arabic hal-question about Messi's Spanish nationality, annotated answerable with 'Spanish descent': ancestry establishes neither citizenship nor its timing (secondary: polar question; reclassified 12 Oct under the precedence rule, as in the Week 41 QA taxonomy)"),
    ("recurrent", "validation-001782"): ("questionable gold label", "same perforatum row as above, the passage states the reason"),
    ("recurrent", "validation-002096"): ("unrelated passage", "same Angolan civil war row as above"),
    ("recurrent", "validation-000883"): ("no cross-lingual entity link", "cut of beef used for bulgogi, 'Sirloin, rib eye or brisket' present, no anchor"),
    ("recurrent", "validation-000891"): ("no cross-lingual entity link", "length of Etruscan rule over Rome, the passage gives durations, no anchor"),
    ("recurrent", "validation-000678"): ("topic present, fact missing", "the passage says several states have no capital but gives no count"),
    ("recurrent", "validation-001085"): ("topic present, fact missing", "same Yamato row as above"),
    ("recurrent", "validation-000075"): ("questionable gold label", "minimum presidential age in India, annotated answerable against a constitution that is not India's"),
    ("recurrent", "validation-000195"): ("no cross-lingual entity link", "China's rank in farm output, 'first' present, the 2010 anchor is absent from the passage"),
    ("recurrent", "validation-002995"): ("unrelated passage", "producer of Bharat Ane Nenu against a passage on Nenunnanu"),
    ("recurrent", "validation-002976"): ("unrelated passage", "producer of Bharat Ane Nenu against a passage on Nani's filmography"),
}

CATEGORIES = (
    "no cross-lingual entity link",
    "unrelated passage",
    "polar question",
    "topic present, fact missing",
    "questionable gold label",
)


def main() -> None:
    sample = pd.read_csv(config.RESULTS_DIR / "47_week39_error_sample.csv")
    rows = []
    for row in sample.itertuples(index=False):
        key = (row.model, row.example_id)
        if key not in JUDGEMENTS:
            raise SystemExit(
                f"no manual judgement recorded for {key}. The sample changed, so "
                "the errors have to be inspected again before the counts can be "
                "reported."
            )
        category, note = JUDGEMENTS[key]
        rows.append(
            {
                "model": row.model,
                "example_id": row.example_id,
                "lang": row.lang,
                "error_type": row.error_type,
                "gold": row.gold,
                "category": category,
                "note": note,
                "question_has_conflicting_labels": row.question_has_conflicting_labels,
                "n_anchors": row.n_anchors,
                "n_hits": row.n_hits,
                "question": row.question,
                "answer": row.answer,
            }
        )
    tax = pd.DataFrame(rows)
    reporting.save_table(tax, "50_week39_error_taxonomy")

    counts = (
        tax.pivot_table(
            index="category", columns="error_type", values="example_id", aggfunc="count"
        )
        .reindex(CATEGORIES)
        .fillna(0)
        .astype(int)
    )
    counts["total"] = counts.sum(axis=1)
    for lang in config.LANGS:
        counts[lang] = (
            tax[tax["lang"] == lang]
            .groupby("category")["example_id"]
            .count()
            .reindex(CATEGORIES)
            .fillna(0)
            .astype(int)
        )
    counts = counts.reset_index()
    reporting.save_table(counts, "51_week39_error_categories")
    print(counts.to_string(index=False))
    print(
        f"\n{len(tax)} errors inspected, "
        f"{tax['question_has_conflicting_labels'].sum()} of them on questions that "
        "appear in validation with both labels"
    )


if __name__ == "__main__":
    main()
