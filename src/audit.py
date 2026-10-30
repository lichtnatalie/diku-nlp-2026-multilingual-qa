"""Week 36, Section 2: dataset audit (Items 1-3).

Item 1  per language and split: counts, answerable/unanswerable proportions,
        median and IQR of tokenised question and context lengths, missing
        values, exact duplicate question-context pairs; plus overall counts
        and answerability proportions.
Item 2  the five most common question tokens per language, from the TRAINING
        split only, with counts and an English translation.
Item 3  programmatic verification that every answerable item's ``answer``
        equals the context substring beginning at ``answer_start``, over both
        splits, reporting the number checked and any failures.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from . import config
from .tokenisation import DEFAULT_TOKENISER, TOKENISERS, token_counts

# ---------------------------------------------------------------------------
# Item 1
# ---------------------------------------------------------------------------

_TEXT_COLUMNS = ("question", "context", "answer", "answer_inlang")


def _length_stats(texts: Iterable[str], tokeniser: str) -> Dict[str, float]:
    lengths = np.asarray(token_counts(texts, tokeniser), dtype=float)
    if lengths.size == 0:
        return {"median": np.nan, "q1": np.nan, "q3": np.nan, "iqr": np.nan}
    q1, med, q3 = np.percentile(lengths, [25, 50, 75])
    return {"median": float(med), "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1)}


def describe_split(
    df: pd.DataFrame, split: str, tokeniser: str = DEFAULT_TOKENISER
) -> pd.DataFrame:
    """Item 1 for one split: one row per language plus an ``all`` row."""
    rows = []
    groups: List[tuple] = [("all", df)] + [
        (lang, df[df["lang"] == lang]) for lang in config.LANGS
    ]
    for lang, sub in groups:
        n = len(sub)
        if n == 0:
            continue
        n_ans = int(sub["answerable"].sum())
        q = _length_stats(sub["question"], tokeniser)
        c = _length_stats(sub["context"], tokeniser)
        dup = duplicate_pairs(sub)
        rows.append(
            {
                "split": split,
                "lang": lang,
                "n": n,
                "n_answerable": n_ans,
                "n_unanswerable": n - n_ans,
                "prop_answerable": n_ans / n,
                "prop_unanswerable": (n - n_ans) / n,
                "q_len_median": q["median"],
                "q_len_q1": q["q1"],
                "q_len_q3": q["q3"],
                "q_len_iqr": q["iqr"],
                "ctx_len_median": c["median"],
                "ctx_len_q1": c["q1"],
                "ctx_len_q3": c["q3"],
                "ctx_len_iqr": c["iqr"],
                "dup_pair_groups": dup["n_groups"],
                "dup_pair_rows": dup["n_rows_in_groups"],
                "dup_pair_redundant_rows": dup["n_redundant_rows"],
                # Reported alongside the duplicate count because a duplicate
                # pair carrying two different labels is an upper bound on any
                # model's accuracy, whereas a label-consistent duplicate is
                # merely redundant.
                "dup_pair_conflicting_groups": label_inconsistent_duplicates(sub),
            }
        )
    return pd.DataFrame(rows)


def missing_values(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Nulls and empty strings per column, per language and overall.

    Both are reported because the two are not the same thing here:
    ``answer`` is an empty string (not null) for unanswerable items by design,
    whereas ``answer_inlang`` is genuinely null for most rows.
    """
    rows = []
    groups: List[tuple] = [("all", df)] + [
        (lang, df[df["lang"] == lang]) for lang in config.LANGS
    ]
    for lang, sub in groups:
        if len(sub) == 0:
            continue
        row = {"split": split, "lang": lang, "n": len(sub)}
        for col in df.columns:
            if col == "example_id":
                continue
            n_null = int(sub[col].isna().sum())
            row[f"{col}_null"] = n_null
            if col in _TEXT_COLUMNS:
                filled = sub[col].fillna("")
                row[f"{col}_empty"] = int((filled.astype(str).str.strip() == "").sum())
        rows.append(row)
    return pd.DataFrame(rows)


def duplicate_pairs(df: pd.DataFrame) -> Dict[str, int]:
    """Exact duplicate (question, context) pairs within a frame."""
    if len(df) == 0:
        return {"n_groups": 0, "n_rows_in_groups": 0, "n_redundant_rows": 0}
    key = df["question"].astype(str) + "\x00" + df["context"].astype(str)
    counts = key.value_counts()
    dup = counts[counts > 1]
    return {
        "n_groups": int(len(dup)),
        "n_rows_in_groups": int(dup.sum()),
        "n_redundant_rows": int(dup.sum() - len(dup)),
    }


def duplicate_pair_examples(df: pd.DataFrame, max_groups: int = 5) -> pd.DataFrame:
    """A few concrete duplicate groups, and whether their labels agree.

    Label-inconsistent duplicates matter: an identical question-context pair
    labelled both answerable and unanswerable is an upper bound on any
    model's achievable accuracy, and belongs in the limitations section.
    """
    key = df["question"].astype(str) + "\x00" + df["context"].astype(str)
    df = df.assign(_key=key)
    counts = key.value_counts()
    dup_keys = counts[counts > 1].index[:max_groups]
    rows = []
    for k in dup_keys:
        sub = df[df["_key"] == k]
        rows.append(
            {
                "lang": sub["lang"].iloc[0],
                "n_copies": len(sub),
                "labels_agree": bool(sub["answerable"].nunique() == 1),
                "answer_starts": sorted(set(sub["answer_start"].tolist())),
                "question": sub["question"].iloc[0][:80],
                "context_prefix": sub["context"].iloc[0][:80],
            }
        )
    return pd.DataFrame(rows)


def label_inconsistent_duplicates(df: pd.DataFrame) -> int:
    """How many duplicate (question, context) groups carry conflicting labels."""
    if len(df) == 0:
        return 0
    key = df["question"].astype(str) + "\x00" + df["context"].astype(str)
    g = df.assign(_key=key).groupby("_key")["answerable"].nunique()
    return int((g > 1).sum())


def cross_split_overlap(splits: Dict[str, pd.DataFrame]) -> Dict[str, int]:
    """Question-context pairs appearing in both train and validation.

    Not required by Item 1, but any overlap inflates every validation number
    in the project, so it is worth one line in the report.
    """
    keys = {
        s: set(df["question"].astype(str) + "\x00" + df["context"].astype(str))
        for s, df in splits.items()
    }
    if "train" not in keys or "validation" not in keys:
        return {}
    shared = keys["train"] & keys["validation"]
    return {
        "n_shared_pairs": len(shared),
        "n_train_pairs": len(keys["train"]),
        "n_validation_pairs": len(keys["validation"]),
    }


def tokeniser_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Median question length under each tokeniser, per language.

    Supports the tokenisation justification: the gap between the whitespace
    and unicode-word counts is much larger for some scripts than others.
    """
    rows = []
    for lang in config.LANGS:
        sub = df[df["lang"] == lang]
        if len(sub) == 0:
            continue
        row = {"lang": lang, "n": len(sub)}
        for name in TOKENISERS:
            row[f"q_len_median_{name}"] = _length_stats(sub["question"], name)["median"]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Item 2
# ---------------------------------------------------------------------------

# Glossary of tokens we expect to dominate question-initial position in these
# three languages.  Anything the data throws up that is NOT in here is flagged
# so it gets translated by hand rather than guessed.
QUESTION_TOKEN_GLOSSARY: Dict[str, Dict[str, str]] = {
    "ar": {
        "من": "who / from",
        "ما": "what",
        "ماذا": "what",
        "متى": "when",
        "أين": "where",
        "اين": "where (without hamza)",
        "كم": "how many / how much",
        "كيف": "how",
        "هو": "he / it (masc.)",
        "هي": "she / it (fem.)",
        "في": "in",
        "على": "on",
        "الذي": "which / that (masc.)",
        "التي": "which / that (fem.)",
        "عام": "year",
        "سنة": "year",
        "تم": "was done / passive auxiliary",
        "أول": "first",
        "اول": "first (without hamza)",
        "و": "and",
        "ال": "the (definite article)",
    },
    "ko": {
        "몇": "how many",
        "가장": "most / -est (superlative marker)",
        "무엇인가": "what is (it)?  [interrogative 'what' + copula + question ending]",
        "어디인가": "where is (it)?  [interrogative 'where' + copula + question ending]",
        "누구": "who",
        "언제": "when",
        "어디": "where",
        "무엇": "what",
        "무슨": "what kind of",
        "년": "year (counter)",
        "년도": "year",
        "이름": "name",
        "사람": "person",
        "것": "thing (bound noun)",
        "은": "topic particle",
        "는": "topic particle",
        "이": "subject particle / this",
        "가": "subject particle",
        "의": "genitive particle",
        "에": "locative particle",
        "에서": "from / at",
    },
    "te": {
        "ఎప్పుడు": "when",
        "ఏది": "which one / which",
        "ఎక్కడ": "where",
        "ఎవరు": "who",
        "ఏమిటి": "what",
        "ఏ": "which",
        "ఎంత": "how much",
        "ఎన్ని": "how many",
        "పేరు": "name",
        "సంవత్సరం": "year",
        "లో": "in (locative suffix)",
        "మొదటి": "first",
        "ఎవరి": "whose",
    },
}


def top_question_tokens(
    train: pd.DataFrame,
    k: int = 5,
    tokeniser: str = DEFAULT_TOKENISER,
    lowercase: bool = True,
) -> pd.DataFrame:
    """Item 2: the k most common question tokens per language (train only).

    Ties at the k-th position are broken deterministically by token string, so
    the table is reproducible across runs.
    """
    fn = TOKENISERS[tokeniser]
    rows = []
    for lang in config.LANGS:
        sub = train[train["lang"] == lang]
        if len(sub) == 0:
            continue
        counter: Counter = Counter()
        for q in sub["question"].fillna(""):
            toks = fn(q)
            if lowercase:
                toks = [t.lower() for t in toks]
            counter.update(toks)
        total = sum(counter.values())
        ordered = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        for rank, (tok, count) in enumerate(ordered, start=1):
            gloss = QUESTION_TOKEN_GLOSSARY.get(lang, {}).get(tok)
            rows.append(
                {
                    "lang": lang,
                    "rank": rank,
                    "token": tok,
                    "count": count,
                    "share_of_tokens": count / total if total else np.nan,
                    "n_questions_containing": int(
                        sub["question"]
                        .fillna("")
                        .apply(lambda q, t=tok: t in [x.lower() for x in fn(q)])
                        .sum()
                    ),
                    "english_translation": gloss if gloss else "*** TRANSLATE BY HAND ***",
                    "translation_verified": gloss is not None,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Item 3
# ---------------------------------------------------------------------------


def verify_answer_offsets(df: pd.DataFrame, split: str) -> Dict[str, object]:
    """Item 3: check ``answer == context[answer_start : answer_start+len(answer)]``.

    Applied to every answerable example.  Unanswerable examples are checked
    separately for the convention the project specifies for them
    (``answer_start == -1`` and an empty answer string); failures there are
    reported but counted apart, since Item 3 only asks about answerable items.
    """
    answerable = df[df["answerable"]]
    failures = []
    for row in answerable.itertuples(index=False):
        ctx = row.context or ""
        ans = row.answer or ""
        start = int(row.answer_start)
        ok = (
            start >= 0
            and start + len(ans) <= len(ctx)
            and ctx[start : start + len(ans)] == ans
        )
        if not ok:
            failures.append(
                {
                    "example_id": row.example_id,
                    "lang": row.lang,
                    "answer_start": start,
                    "len_context": len(ctx),
                    "answer": ans[:60],
                    "context_at_offset": ctx[start : start + len(ans)][:60]
                    if 0 <= start <= len(ctx)
                    else "<out of range>",
                    "answer_occurs_elsewhere": ans in ctx if ans else False,
                }
            )

    unanswerable = df[~df["answerable"]]
    bad_unanswerable = unanswerable[
        (unanswerable["answer_start"] != -1)
        | (unanswerable["answer"].fillna("").str.len() > 0)
    ]

    per_lang = {}
    for lang in config.LANGS:
        sub = answerable[answerable["lang"] == lang]
        n_fail = sum(1 for f in failures if f["lang"] == lang)
        per_lang[lang] = {"checked": int(len(sub)), "failures": n_fail}

    return {
        "split": split,
        "n_answerable_checked": int(len(answerable)),
        "n_failures": len(failures),
        "failure_rate": len(failures) / len(answerable) if len(answerable) else 0.0,
        "per_language": per_lang,
        "failures": pd.DataFrame(failures),
        "n_unanswerable": int(len(unanswerable)),
        "n_unanswerable_violating_convention": int(len(bad_unanswerable)),
        "unanswerable_violations": bad_unanswerable,
        # Extra diagnostic used in Week 37: an answer that is not unique in the
        # context means character offsets, not string matching, must drive the
        # BIO alignment.
        "n_answers_occurring_more_than_once": int(
            sum(
                1
                for row in answerable.itertuples(index=False)
                if row.answer and (row.context or "").count(row.answer) > 1
            )
        ),
    }
