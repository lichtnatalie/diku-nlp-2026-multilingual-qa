"""Dataset loading.

Single entry point so that every week uses identical splits and identical
filtering.  Nothing downstream is allowed to call ``load_dataset`` itself.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from . import config


def load_raw(prefer_local: bool = True) -> Dict[str, pd.DataFrame]:
    """Load the published dataset, unfiltered, as pandas frames.

    Prefers a local copy of the parquet files in ``data/`` and falls back to
    the Hub.  Two reasons for the local path: huggingface.co is not reachable
    from every environment this project runs in, and pinning a downloaded copy
    makes the reported numbers reproducible from exactly the files in the
    submitted ZIP rather than from whatever the Hub serves later.

    Download once from
    https://huggingface.co/datasets/coastalcph/tydi_xor_rc/resolve/main/train.parquet
    (likewise validation.parquet, test.parquet) into ``data/``.
    """
    if prefer_local:
        present = {
            split: path
            for split, path in config.PARQUET_FILES.items()
            if path.exists()
        }
        if all(s in present for s in config.SPLITS):
            return {split: pd.read_parquet(path) for split, path in present.items()}

    from datasets import load_dataset

    ds = load_dataset(config.DATASET_NAME)
    return {split: ds[split].to_pandas() for split in ds.keys()}


def data_provenance() -> Dict[str, object]:
    """Record which files the run actually read, with their SHA-256 digests.

    Goes into the reproducibility appendix; also catches the case where the
    local parquet files are silently replaced between runs.
    """
    import hashlib

    out: Dict[str, object] = {"source": "hub", "files": {}}
    for split, path in config.PARQUET_FILES.items():
        if path.exists():
            h = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            out["files"][split] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256_16": h,
            }
    if all(s in out["files"] for s in config.SPLITS):
        out["source"] = "local parquet"
    return out


def load_project_splits(
    raw: Optional[Dict[str, pd.DataFrame]] = None,
    langs=config.LANGS,
) -> Dict[str, pd.DataFrame]:
    """Return {'train': df, 'validation': df} restricted to the project languages.

    A stable ``example_id`` is attached here.  The dataset ships no identifier
    column, and every later week (BIO alignment, error analysis, the paired
    challenge-set comparison) needs one that does not move when rows are
    filtered or reordered, so it is derived from the split name and the row's
    position in the *unfiltered* split.
    """
    if raw is None:
        raw = load_raw()

    out: Dict[str, pd.DataFrame] = {}
    for split in config.SPLITS:
        if split not in raw:
            raise KeyError(f"split {split!r} not present; found {sorted(raw)}")
        df = raw[split].copy()
        df["example_id"] = [f"{split}-{i:06d}" for i in range(len(df))]
        df = df[df["lang"].isin(langs)].reset_index(drop=True)
        # Normalise dtypes: `answerable` must be a real bool, `answer` a str.
        df["answerable"] = df["answerable"].astype(bool)
        df["answer"] = df["answer"].fillna("")
        df["answer_start"] = df["answer_start"].astype(int)
        out[split] = df
    return out


def language_inventory(raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """All languages present upstream, per split.

    Reported in the appendix to justify the ar/ko/te filter and to state how
    many examples the filter discards.
    """
    rows = []
    for split, df in raw.items():
        counts = df["lang"].value_counts()
        for lang, n in counts.items():
            rows.append(
                {
                    "split": split,
                    "lang": lang,
                    "n": int(n),
                    "in_project_scope": lang in config.LANGS,
                }
            )
    return pd.DataFrame(rows).sort_values(["split", "lang"]).reset_index(drop=True)
