"""Week 36, Section 2: the two mandatory answerability baselines.

(i)  MajorityClassBaseline  - the majority answerability label estimated on
     the training split, predicted constantly.

(ii) CrossLingualAnchorRule - a deterministic rule-based classifier that sees
     only the question and the context.  It uses no labelled validation
     examples and no trained answerability or QA model.

Why an "anchor" rule
--------------------
This is a cross-lingual reading-comprehension setting: the question is in
Arabic, Korean or Telugu and the context is in English.  Ordinary lexical
overlap is therefore vacuous - the two strings share almost no characters by
construction.  What *does* survive the language boundary without translation
is the material that is not translated in the first place:

  * Latin-script substrings embedded in the question - proper names,
    scientific terms, film and band titles, abbreviations - which are often
    quoted verbatim even in Arabic, Korean or Telugu text;
  * numbers, once the Arabic-Indic and Telugu digit systems are mapped to
    ASCII: years above all, but also quantities and ordinals.

The rule predicts "answerable" when at least ``min_hits`` such anchors from
the question also occur in the context.  This is a genuine cross-lingual
signal that costs no model and no translation system, and it makes a clear,
falsifiable claim: *if the entity or year the question asks about is not even
mentioned in the passage, the passage probably does not support an answer.*

What the rule can and cannot exploit (for the report)
-----------------------------------------------------
Can:   named entities that are script-preserved or not transliterated; years
       and other numerals; the presence/absence of the question's topic in the
       passage.
Cannot: anything that requires translation - a question whose entities are
       fully transliterated into Arabic or Telugu script yields no anchors at
       all; semantic entailment (the entity can be mentioned while the
       specific fact asked for is absent, which is exactly the hard case that
       XOR-AttriQA's unanswerable items are built from); and any notion of
       question type, so it cannot tell "when was X founded" from "who founded
       X" when both mention X.
The rule should therefore have decent recall on the answerable class and poor
precision on the unanswerable class, and it should degrade differently across
the three languages depending on how much Latin script each keeps.  That is
the hypothesis to check against the per-language table.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import config
from .tokenisation import normalise_digits

# A Latin-script run of >= min_latin_len letters, or a run of >= min_digit_len
# digits.  Both are extracted from the DIGIT-NORMALISED question.
_LATIN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_DIGIT_RE = re.compile(r"\d+")


class MajorityClassBaseline:
    """Baseline (i): predict the training-split majority label everywhere."""

    def __init__(self) -> None:
        self.majority_: Optional[bool] = None
        self.train_proportion_: Optional[float] = None

    def fit(self, train: pd.DataFrame) -> "MajorityClassBaseline":
        prop_answerable = float(train["answerable"].mean())
        self.train_proportion_ = prop_answerable
        self.majority_ = bool(prop_answerable >= 0.5)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.majority_ is None:
            raise RuntimeError("call fit() on the training split first")
        return np.full(len(df), self.majority_, dtype=bool)

    def describe(self) -> str:
        return (
            f"majority class = {'answerable' if self.majority_ else 'unanswerable'} "
            f"(training proportion answerable = {self.train_proportion_:.4f})"
        )


class CrossLingualAnchorRule:
    """Baseline (ii): deterministic anchor-overlap rule.

    Parameters
    ----------
    min_hits
        How many distinct question anchors must occur in the context for the
        rule to predict "answerable".
    min_latin_len
        Minimum length of a Latin-script run to count as an anchor.  Short runs
        ("a", "de", "II") match too easily inside English text.
    min_digit_len
        Minimum length of a digit run.  Single digits match almost any passage.
    fallback
        What to predict when the question yields no anchors at all.  Either a
        literal ``True``/``False``, or ``"train_majority"``, which reuses the
        training-split majority label.  Note that ``"train_majority"`` reads
        training labels; it reads no validation labels and fits no model, so it
        stays inside the project's constraint, but ``fallback=True`` is the
        fully label-free variant and is reported alongside it.
    """

    def __init__(
        self,
        min_hits: int = 1,
        min_latin_len: int = 3,
        min_digit_len: int = 2,
        fallback="train_majority",
    ) -> None:
        self.min_hits = min_hits
        self.min_latin_len = min_latin_len
        self.min_digit_len = min_digit_len
        self.fallback = fallback
        self.fallback_value_: Optional[bool] = None

    # -- rule internals ----------------------------------------------------
    def anchors(self, question: str) -> List[str]:
        q = normalise_digits(question or "")
        found = [
            m.group(0).lower()
            for m in _LATIN_RE.finditer(q)
            if len(m.group(0)) >= self.min_latin_len
        ]
        found += [
            m.group(0)
            for m in _DIGIT_RE.finditer(q)
            if len(m.group(0)) >= self.min_digit_len
        ]
        # De-duplicate, preserving order, so "hits" counts distinct anchors.
        seen, out = set(), []
        for a in found:
            if a not in seen:
                seen.add(a)
                out.append(a)
        return out

    def hits(self, question: str, context: str) -> int:
        ctx = normalise_digits(context or "").lower()
        return sum(1 for a in self.anchors(question) if a in ctx)

    # -- sklearn-ish API ---------------------------------------------------
    def fit(self, train: pd.DataFrame) -> "CrossLingualAnchorRule":
        """No parameters are learned; this only resolves the fallback label.

        Kept as ``fit`` so the runner can treat both baselines uniformly.
        """
        if self.fallback == "train_majority":
            self.fallback_value_ = bool(train["answerable"].mean() >= 0.5)
        else:
            self.fallback_value_ = bool(self.fallback)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.fallback_value_ is None:
            raise RuntimeError("call fit() first (it resolves the fallback label)")
        preds = np.empty(len(df), dtype=bool)
        for i, row in enumerate(df.itertuples(index=False)):
            anchors = self.anchors(row.question)
            if not anchors:
                preds[i] = self.fallback_value_
            else:
                preds[i] = self.hits(row.question, row.context) >= self.min_hits
        return preds

    def describe(self) -> str:
        return (
            f"anchor rule (min_hits={self.min_hits}, min_latin_len={self.min_latin_len}, "
            f"min_digit_len={self.min_digit_len}, fallback="
            f"{'answerable' if self.fallback_value_ else 'unanswerable'})"
        )

    # -- diagnostics used in the report -----------------------------------
    def anchor_diagnostics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-language anchor coverage, and hit rate split by gold label.

        This is the evidence for the "can / cannot exploit" discussion: it
        shows how often the rule has any signal at all in each language, and
        whether the signal separates the two classes.
        """
        rows = []
        groups = [("all", df)] + [(l, df[df["lang"] == l]) for l in config.LANGS]
        for lang, sub in groups:
            if len(sub) == 0:
                continue
            n_anchors = np.array([len(self.anchors(q)) for q in sub["question"]])
            n_hits = np.array(
                [self.hits(q, c) for q, c in zip(sub["question"], sub["context"])]
            )
            gold = sub["answerable"].to_numpy(dtype=bool)
            rows.append(
                {
                    "lang": lang,
                    "n": len(sub),
                    "prop_questions_with_any_anchor": float(np.mean(n_anchors > 0)),
                    "mean_anchors_per_question": float(np.mean(n_anchors)),
                    "mean_hits_answerable": float(np.mean(n_hits[gold])) if gold.any() else np.nan,
                    "mean_hits_unanswerable": float(np.mean(n_hits[~gold])) if (~gold).any() else np.nan,
                    "prop_hit_ge1_answerable": float(np.mean(n_hits[gold] >= 1)) if gold.any() else np.nan,
                    "prop_hit_ge1_unanswerable": float(np.mean(n_hits[~gold] >= 1)) if (~gold).any() else np.nan,
                }
            )
        return pd.DataFrame(rows)

    def sweep_min_hits(
        self, train: pd.DataFrame, values: Sequence[int] = (1, 2, 3)
    ) -> pd.DataFrame:
        """Effect of ``min_hits`` measured on the TRAINING split only.

        The project forbids using labelled validation examples to build the
        rule, so if a value other than the default is chosen it must be chosen
        here, and the report must say so.
        """
        from .evaluation import answerability_metrics

        rows = []
        original = self.min_hits
        try:
            for v in values:
                self.min_hits = v
                preds = self.predict(train)
                m = answerability_metrics(train["answerable"].to_numpy(dtype=bool), preds)
                rows.append({"min_hits": v, "split": "train", **m})
        finally:
            self.min_hits = original
        return pd.DataFrame(rows)


def anchor_examples(rule: CrossLingualAnchorRule, df: pd.DataFrame, n: int = 6) -> pd.DataFrame:
    """A handful of worked examples, for the qualitative paragraph."""
    rows = []
    for row in df.head(n * 20).itertuples(index=False):
        a = rule.anchors(row.question)
        rows.append(
            {
                "lang": row.lang,
                "answerable": row.answerable,
                "question": (row.question or "")[:70],
                "anchors": a,
                "hits": rule.hits(row.question, row.context),
            }
        )
        if len(rows) >= n * 3:
            break
    out = pd.DataFrame(rows)
    # Prefer a mix: some with anchors, some without.
    with_a = out[out["anchors"].str.len() > 0].head(n // 2 + n % 2)
    without = out[out["anchors"].str.len() == 0].head(n // 2)
    return pd.concat([with_a, without], ignore_index=True)
