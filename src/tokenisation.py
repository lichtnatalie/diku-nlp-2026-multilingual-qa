"""Tokenisation used for all descriptive statistics in Week 36.

Why this file exists
--------------------
The project ask to *explain* the tokenisation.  The naive choices both
have defects on this data, and the defects are language-specific, so the
choice is itself a finding worth reporting:

* ``str.split()`` (whitespace) is wrong for Arabic and Telugu punctuation,
  which is written attached to the preceding word ("word،" / "word."), and
  produces one "token" per *eojeol* in Korean (a whitespace unit that usually
  bundles a content word with one or more grammatical particles).

* ``re.findall(r"\\w+", s)`` looks script-agnostic but silently breaks
  Indic and Arabic script.  Python's ``\\w`` matches characters for which
  ``str.isalnum()`` is true, and that is false for Unicode combining marks
  (categories ``Mn``/``Mc``).  Telugu writes its vowels as combining signs
  (e.g. U+0C3E TELUGU VOWEL SIGN AA), and vocalised Arabic writes short
  vowels the same way, so ``\\w+`` splits a single orthographic word into
  several fragments.  ``run_selftest()`` below demonstrates this.

I therefore use ``unicode_word_tokenise`` as the project default: a token is
a maximal run of characters that are letters (L*), numbers (N*), combining
marks (Mn/Mc/Me), or the zero-width (non-)joiners that Indic and Arabic
scripts use inside words.  Everything else (whitespace, punctuation, symbols)
is a separator and is discarded.

This is still a *word-ish* tokenisation, not morphological segmentation.  It
does not split Korean eojeol into morphemes, nor Arabic proclitics such as
``wa-``/``al-``, nor Telugu agglutinated suffixes.  That is a deliberate,
stated limitation: it keeps the unit comparable across the three languages
(one orthographic word) at the cost of undercounting morphemes in the two
agglutinating languages.  Subword segmentation is introduced in Week 37,
where it is forced by the pretrained models.
"""

from __future__ import annotations

import unicodedata
from functools import lru_cache
from typing import Iterable, List

# Zero-width non-joiner / joiner: word-internal in Arabic and Indic scripts.
_WORD_INTERNAL_FORMAT = {"‌", "‍"}

# Digit systems that appear in this dataset's question languages, mapped to
# ASCII so that a year written "١٩٤٥" (Arabic-Indic) or "౧౯౪౫" (Telugu) can be
# compared with "1945" in the English context.  Korean uses ASCII digits.
_DIGIT_MAPS = {
    0x0660: "0",  # ARABIC-INDIC DIGIT ZERO      ٠-٩
    0x06F0: "0",  # EXTENDED ARABIC-INDIC DIGIT ZERO ۰-۹
    0x0C66: "0",  # TELUGU DIGIT ZERO            ౦-౯
}


@lru_cache(maxsize=None)
def _is_word_char(ch: str) -> bool:
    if ch in _WORD_INTERNAL_FORMAT:
        return True
    cat = unicodedata.category(ch)
    return cat[0] in ("L", "N") or cat in ("Mn", "Mc", "Me")


def unicode_word_tokenise(text: str) -> List[str]:
    """Project default tokeniser.  See module docstring for the rationale."""
    if not text:
        return []
    tokens: List[str] = []
    current: List[str] = []
    for ch in text:
        if _is_word_char(ch):
            current.append(ch)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def whitespace_tokenise(text: str) -> List[str]:
    """Baseline tokeniser, reported alongside the default in the appendix."""
    return text.split() if text else []


TOKENISERS = {
    "unicode_word": unicode_word_tokenise,
    "whitespace": whitespace_tokenise,
}
DEFAULT_TOKENISER = "unicode_word"


def normalise_digits(text: str) -> str:
    """Map Arabic-Indic / extended Arabic-Indic / Telugu digits to ASCII."""
    if not text:
        return text
    out = []
    for ch in text:
        cp = ord(ch)
        for base, zero in _DIGIT_MAPS.items():
            if base <= cp <= base + 9:
                out.append(chr(ord(zero) + cp - base))
                break
        else:
            out.append(ch)
    return "".join(out)


def strip_arabic_diacritics(text: str) -> str:
    """Remove Arabic short-vowel marks and tatweel.

    Optional normalisation: the dataset is mostly unvocalised, but a handful of
    questions carry harakat, and leaving them in creates spurious token types.
    """
    out = []
    for ch in text:
        cp = ord(ch)
        if 0x064B <= cp <= 0x0652 or cp == 0x0640:  # harakat, tatweel
            continue
        out.append(ch)
    return "".join(out)


def token_counts(texts: Iterable[str], tokeniser: str = DEFAULT_TOKENISER) -> List[int]:
    fn = TOKENISERS[tokeniser]
    return [len(fn(t or "")) for t in texts]


# --------------------------------------------------------------------------
def run_selftest() -> None:
    """Demonstrates the \\w+ failure mode quoted in the report."""
    import re

    telugu = "పరిపాలన"          # a single Telugu word containing vowel signs
    arabic = "الْكِتَاب"          # vocalised Arabic word
    print("Telugu word:", telugu)
    print(r"  re \w+          ->", re.findall(r"\w+", telugu))
    print("  unicode_word    ->", unicode_word_tokenise(telugu))
    print("Arabic word (vocalised):", arabic)
    print(r"  re \w+          ->", re.findall(r"\w+", arabic))
    print("  unicode_word    ->", unicode_word_tokenise(arabic))
    print("Korean eojeol:", "한국어를 배웁니다.")
    print("  whitespace      ->", whitespace_tokenise("한국어를 배웁니다."))
    print("  unicode_word    ->", unicode_word_tokenise("한국어를 배웁니다."))
    print("Digit normalisation:", normalise_digits("١٩٤٥ / ౧౯౪౫ / 1945"))


if __name__ == "__main__":
    run_selftest()
