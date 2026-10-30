#!/usr/bin/env python3
"""
Week 41, Section 6: the manual error taxonomy for the integrated QA model.

The categories, notes and English glosses were drafted with Claude (see
genai-logs/) and checked by me against each question, gold answer and full
passage on 12 Oct. A second check the same day revised items 1, 2, 4 and 7,
the glosses of 3, 6, 12 and 16, and added the secondary notes and the
precedence rule below.

`week41_qa_error_sample.py` writes 24 validation errors (8 wrongly abstains,
8 wrongly answers, 8 wrong spans; all three languages; only errors the model
makes under all three seeds) to ``results/73_week41_qa_error_sample.csv``.
This file records the judgement for each, so the counts in the report are
reproducible from the repository.

How to read a category
  Each error gets ONE primary category and, where useful, a secondary note.
  A category describes a property of the example (its question, its passage,
  its annotation or the predicted span). It does not claim to be the cause of
  the model's decision: for a wrong abstention, the model's unreturned best
  span is recorded, but its properties do not show why the model abstained.

Precedence
  1. questionable gold label comes first whenever the gold answer (or the
     gold "unanswerable") does not address the question as asked. The error
     is then a disagreement with the benchmark, not clearly a semantic failure.
     Question form is kept as a secondary note.
  2. Otherwise the question form (yes/no or alternative question) or the
     passage (unrelated, topic present but fact missing) for answerability
     errors, and the span relation for wrong spans.

Categories. The first five are the Week 39 ones (`week39_error_taxonomy.py`),
reused so the two analyses can be compared. The key "polar question" is kept
so the two analyses share a row, and here covers yes/no AND alternative
(A-or-B) questions; the report labels it "yes/no or alternative question".

  polar question
      Yes/no question (Arabic "hal", Korean -나요/-는가/-까) or alternative
      question (Korean "A 아니면 B"). The dataset annotates such items both as
      answerable (with an evidence clause as the "answer") and as unanswerable.
  unrelated passage, topic present / fact missing, questionable gold label,
  no cross-lingual entity link
      As in Week 39 (the last does not occur in this sample).
  predicate paraphrase
      The passage states an answer, but not with the question's key
      predicate ("symptoms" against "there is muscle weakness"). A feature of
      the example, not a demonstrated cause of the abstention.
  answer type mismatch
      The span has another type than the question asks for (a year for a
      duration). For a wrong abstention it describes the unreturned
      candidate span.
  right type, wrong referent
      The span has the right type but names another entity of that type.
  span boundary
      The span contains the gold answer or lies inside it. Several are
      semantically acceptable answers that fail exact match only.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, reporting  # noqa: E402

# example_id -> (primary category, English gloss, secondary note, note)
JUDGEMENTS = {
    # ---- wrongly abstains
    "validation-001670": ("questionable gold label", "Did the prophet Solomon build Jerusalem?",
                          "yes/no question",
                          "gold is the clause crediting him with building the First Temple in Jerusalem, which does not establish that he built Jerusalem; the abstention is a benchmark error, not clearly a semantic failure"),
    "validation-002131": ("questionable gold label", "Did Messi hold Spanish nationality before joining the Argentine national team?",
                          "yes/no question",
                          "gold 'Spanish descent': ancestry establishes neither citizenship nor its timing, and the full passage does not resolve the question. the Week 39 BiGRU row is now also a questionable gold label (same precedence rule)"),
    "validation-002075": ("polar question", "Does the University of California offer a veterinary degree?",
                          "underspecified gold span",
                          "the passage supports the broad question (veterinary medicine is listed); the gold span 'medicine' is underspecified; same row in the Week 39 taxonomy"),
    "validation-000778": ("polar question", "Does 'the West' mean America or Europe?",
                          "alternative question",
                          "an alternative (A-or-B) question, not a yes/no question; gold is the list phrase 'part of Europe, Australasia, and the Americas'"),
    "validation-000986": ("polar question", "Is same-sex marriage legal in Indonesia?", "",
                          "yes/no question; gold is the clause 'Indonesia does not recognize same-sex marriage'"),
    "validation-000681": ("predicate paraphrase", "What are the symptoms of polio?",
                          "several defensible answers",
                          "the passage says 'in about 0.5 percent of cases there is muscle weakness' and later 'minor symptoms such as fever and a sore throat'; the category describes the example, not a demonstrated cause of the abstention"),
    "validation-000195": ("answer type mismatch", "As of 2010, what rank does China hold in agriculture?",
                          "wrong abstention; candidate span has wrong answer type",
                          "answer 'first' ('China ranks first worldwide in farm output'); the model abstains, and its unreturned best span is 'China', a country for a rank question. This does not show why it abstained. Week 39 put this row under 'no cross-lingual entity link' for the BiGRU"),
    "validation-000094": ("questionable gold label", "(tentative, the Telugu is malformed) As of 2017, which one (place) had the most people die from mosquitoes in a year?",
                          "gloss tentative",
                          "the question word is 'edi' (which one), so a place is possible, but there are several mismatches: year versus country, and deaths versus 'the largest number of recorded cases' in 2008; the passage mentions neither deaths nor mosquitoes. The model abstains"),
    # ---- wrongly answers
    "validation-000678": ("topic present, fact missing", "How many countries have no capital?", "",
                          "passage: 'several states that have no capital', no count; same row in Week 39"),
    "validation-000369": ("polar question", "Is there a formula for infinity?", "",
                          "yes/no question, annotated unanswerable; the model answers 'number'. The category describes the question's form, not the cause of the failure"),
    "validation-001085": ("topic present, fact missing", "When did Japan's Yamato regime end?", "",
                          "passage gives the Kofun period (300 to 538 AD) and says Kofun + Asuka are 'sometimes' called Yamato, never when Yamato ended; same row in Week 39"),
    "validation-000497": ("polar question", "Do laws vary from state to state in the US?",
                          "topic present, fact missing also defensible",
                          "yes/no question, annotated unanswerable; the passage establishes separate state laws but not that they differ; the model answers 'each separate U.S. state'"),
    "validation-003000": ("questionable gold label", "Who was the US president in 1960?",
                          "prediction still wrong",
                          "annotated unanswerable, but the passage names 'the incumbent president, Republican Dwight D. Eisenhower'. The model's 'Richard Nixon' (a candidate) is still incorrect: a doubtful annotation does not make the answer right"),
    "validation-002980": ("unrelated passage", "Who is the music director of the film Munna?", "",
                          "passage is about the Tamil film Ninaive Oru Sangeetham; the model returns its director; the Munna questions also appeared in Week 39"),
    "validation-002988": ("unrelated passage", "Who is the producer of the film Bharat Ane Nenu?", "",
                          "passage is about Ninna Nedu Repu; the model returns its lead actors"),
    "validation-002973": ("topic present, fact missing", "Who was Canada's first chief minister? (likely intended: prime minister)", "",
                          "literal gloss of Telugu 'mukhyamantri' (chief minister), most likely meaning prime minister; the passage on Mackenzie King names Macdonald but never says who was first; the model answers 'King'"),
    # ---- wrong span
    "validation-001527": ("span boundary", "Where did the Najahid state arise?",
                          "semantically acceptable",
                          "pred 'Zabid in the Tihama (lowlands) region of Yemen' contains the gold and is arguably more precise: an exact-match failure, not an incorrect answer"),
    "validation-001855": ("answer type mismatch", "How long did King Saif ad-Din Qutuz reign?", "",
                          "duration question, gold 'less than a year'; the model returns the year '1259'"),
    "validation-001492": ("right type, wrong referent", "Which minorities live in Iraqi Kurdistan?", "",
                          "gold is the list of minorities; the model returns 'Kurdish majority', next to the question's words"),
    "validation-000861": ("span boundary", "What was Adolf Hitler's original occupation?",
                          "semantically acceptable",
                          "pred adds 'of Vienna's sights' to the gold 'painting and selling watercolours': an exact-match failure, not an incorrect answer"),
    "validation-000913": ("right type, wrong referent", "Who was the leader at the time of the Third Republic?",
                          "ambiguous question",
                          "gold Trochu (president of the provisional government); the model returns Gambetta, the other leader named; the question does not say which leader or which point in the 70-year period"),
    "validation-001370": ("answer type mismatch", "What happened when Our Lady of Fatima first appeared?", "",
                          "asks for an event, gold 'children reported a prophecy'; the model returns the entity 'Virgin Mary'"),
    "validation-000193": ("span boundary", "How many oxygen atoms are in carbon monoxide?", "",
                          "pred 'one carbon atom and one' runs across the coordination; gold 'one'"),
    "validation-000165": ("right type, wrong referent", "Which vitamin is responsible for an organism's growth?",
                          "excessive span length",
                          "gold 'A' ('vitamin A function as regulators of cell and tissue growth'); the model returns a long span about vitamins D, C and E"),
}

CATEGORIES = (
    "polar question",
    "unrelated passage",
    "topic present, fact missing",
    "questionable gold label",
    "predicate paraphrase",
    "answer type mismatch",
    "right type, wrong referent",
    "span boundary",
)
ERROR_TYPES = ("wrongly abstains", "wrongly answers", "wrong span")


def main() -> None:
    sample = pd.read_csv(config.RESULTS_DIR / "73_week41_qa_error_sample.csv", keep_default_na=False)
    rows = []
    for row in sample.itertuples(index=False):
        if row.example_id not in JUDGEMENTS:
            raise SystemExit(f"no manual judgement recorded for {row.example_id}; the sample changed, "
                             "so the errors have to be inspected again")
        category, gloss, secondary, note = JUDGEMENTS[row.example_id]
        assert category in CATEGORIES, category
        rows.append({"example_id": row.example_id, "lang": row.lang, "error_type": row.error_type,
                     "span_relation": row.span_relation, "category": category, "secondary": secondary,
                     "question_gloss": gloss, "note": note, "question": row.question,
                     "gold_answer": row.gold_answer, "pred_text": row.pred_text})
    tax = pd.DataFrame(rows)
    reporting.save_table(tax, "74_week41_qa_error_taxonomy")

    counts = (tax.pivot_table(index="category", columns="error_type", values="example_id", aggfunc="count")
                 .reindex(index=CATEGORIES, columns=ERROR_TYPES).fillna(0).astype(int))
    counts["total"] = counts.sum(axis=1)
    for lang in config.LANGS:
        counts[lang] = tax[tax["lang"] == lang].groupby("category")["example_id"].count() \
            .reindex(CATEGORIES).fillna(0).astype(int)
    counts = counts.reset_index()
    reporting.save_table(counts, "75_week41_qa_error_categories")
    print(counts.to_string(index=False))
    print(f"\n{(tax['secondary'] != '').sum()} errors carry a secondary note; "
          f"{(tax['secondary'] == 'semantically acceptable').sum()} wrong spans are semantically acceptable")


if __name__ == "__main__":
    main()
