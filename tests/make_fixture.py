"""Synthetic fixture matching the real dataset schema.

The real dataset cannot be downloaded in every environment, and more
importantly the audit and metric code must be checked against cases whose
correct answers are known by construction.  This fixture deliberately contains
every edge case the project description asks me to handle:

  * an answer at character 0;
  * a multi-token answer;
  * an answer with punctuation immediately adjacent;
  * an answer string that occurs more than once in the context (so offsets,
    not string search, must drive alignment);
  * unanswerable rows with answer_start == -1 and an empty answer;
  * a deliberately BROKEN answerable row whose offset is off by one, so that
    Item 3 has something to find;
  * an unanswerable row violating the (-1, "") convention;
  * an exact duplicate question-context pair, with conflicting labels;
  * a language outside the project scope (bn), which the filter must drop;
  * a Telugu question whose word contains combining vowel signs;
  * an Arabic question with Arabic-Indic digits;
  * questions with and without Latin-script anchors.

Offsets are computed from the text rather than hard-coded, so the fixture
cannot silently drift out of sync with the contexts.
"""

from __future__ import annotations

import pandas as pd

CTX_A = (
    "WikiLeaks is an international non-profit organisation founded in 2006 in "
    "Iceland. WikiLeaks publishes secret information."
)
CTX_B = (
    "The Nile is a major north-flowing river in north-eastern Africa. "
    "It is the longest river in Africa."
)
CTX_C = "Seoul is the capital of South Korea, with a population of about 9.7 million."
CTX_D = "Hyderabad is the capital of the Indian state of Telangana."


def _at(context: str, answer: str, occurrence: int = 1, shift: int = 0) -> int:
    """Character offset of the ``occurrence``-th appearance of ``answer``."""
    idx = -1
    for _ in range(occurrence):
        idx = context.index(answer, idx + 1)
    return idx + shift


def _row(question, context, lang, answerable, answer_start, answer, answer_inlang=None):
    return {
        "question": question,
        "context": context,
        "lang": lang,
        "answerable": answerable,
        "answer_start": answer_start,
        "answer": answer,
        "answer_inlang": answer_inlang,
    }


def _ans(question, context, lang, answer, occurrence=1, shift=0, answer_inlang=None):
    return _row(
        question, context, lang, True,
        _at(context, answer, occurrence, shift), answer, answer_inlang,
    )


def _unans(question, context, lang, answer="", answer_start=-1):
    return _row(question, context, lang, False, answer_start, answer, None)


def make_train() -> pd.DataFrame:
    rows = [
        # answer at character 0; Latin anchor "WikiLeaks" present in context
        _ans("WikiLeaks ٢٠٠٦ سنة تأسست؟", CTX_A, "ar", "WikiLeaks"),
        # multi-token answer; Arabic-Indic digits in the question
        _ans("متى تأسست ويكيليكس في ٢٠٠٦؟", CTX_A, "ar", "in 2006 in Iceland"),
        # answer string occurs twice in the context; offset points at the 2nd
        _ans("WikiLeaks 무엇을 발행합니까?", CTX_A, "ko", "WikiLeaks", occurrence=2),
        # unanswerable, no Latin anchor at all in the question
        _unans("이 문서의 저자는 누구입니까?", CTX_A, "ko"),
        # punctuation ("." ) immediately after the answer
        _ans("Nile ఎక్కడ ప్రవహిస్తుంది?", CTX_B, "te", "Africa"),
        # unanswerable although a Latin anchor DOES occur -> rule false positive
        _unans("Nile నది పొడవు ఎంత?", CTX_B, "te"),
        # Telugu question with combining vowel signs, no anchor
        _unans("పరిపాలన ఎక్కడ జరుగుతుంది?", CTX_D, "te"),
        # Korean, answerable, digit anchor
        _ans("Seoul 인구는 9.7 백만입니까?", CTX_C, "ko", "Seoul"),
        # exact duplicate of row 0 with a CONFLICTING label
        _unans("WikiLeaks ٢٠٠٦ سنة تأسست؟", CTX_A, "ar"),
        # anchor present in the question but ABSENT from the context, gold
        # unanswerable -> the rule gets this right (true negative)
        _unans("Bitcoin 가격은 얼마입니까?", CTX_A, "ko"),
        # anchor absent from the context but gold answerable -> the rule gets
        # this wrong (false negative); exercises the other error branch
        _ans("Sunshine أين تأسست؟", CTX_A, "ar", "Iceland"),
        # out-of-scope language: the filter must drop this
        _ans("উইকিলিকস কবে প্রতিষ্ঠিত হয়?", CTX_A, "bn", "2006"),
        # answer_inlang present (most rows have it null)
        _ans("Hyderabad ఏ రాష్ట్ర రాజధాని?", CTX_D, "te", "Telangana", answer_inlang="తెలంగాణ"),
    ]
    return pd.DataFrame(rows)


def make_validation() -> pd.DataFrame:
    rows = [
        _ans("WikiLeaks أين تأسست؟", CTX_A, "ar", "Iceland"),
        # BROKEN: offset shifted by one; Item 3 must catch exactly this row
        _ans("متى تأسست ويكيليكس؟", CTX_A, "ar", "2006", shift=1),
        _ans("Seoul 어디에 있습니까?", CTX_C, "ko", "South Korea"),
        _unans("이 사람의 생일은 언제입니까?", CTX_C, "ko"),
        # violates the unanswerable convention: non-empty answer string
        _unans("Nile ఎంత పొడవు?", CTX_B, "te", answer="unknown"),
        _ans("Telangana రాజధాని ఏది?", CTX_D, "te", "Hyderabad"),
        _ans("এই নদীর নাম কি?", CTX_B, "bn", "Nile"),
    ]
    return pd.DataFrame(rows)


def make_raw() -> dict:
    return {"train": make_train(), "validation": make_validation()}


if __name__ == "__main__":
    for split, df in make_raw().items():
        print(f"--- {split} ({len(df)} rows) ---")
        print(df.to_string())
