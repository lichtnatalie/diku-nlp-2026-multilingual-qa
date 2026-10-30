"""Small helpers for writing result tables and recording the environment."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pandas as pd

from . import config


def save_table(df: pd.DataFrame, name: str, floatfmt: str = "%.4f") -> Path:
    """Write a table as CSV (for reuse) and Markdown (for pasting into notes)."""
    csv_path = config.RESULTS_DIR / f"{name}.csv"
    md_path = config.RESULTS_DIR / f"{name}.md"
    df.to_csv(csv_path, index=False)
    with open(md_path, "w") as fh:
        fh.write(df.to_markdown(index=False, floatfmt=floatfmt.replace("%", "").replace("f", "f")))
        fh.write("\n")
    return csv_path


def hardware_record() -> Dict[str, object]:
    """CPU model, core count and memory of the machine this runs on.

    The same values `scripts/record_environment.py` records for Weeks 36-38,
    so a run on a different machine (Week 39 ran in a Linux container, not on
    the laptop) can state its own hardware rather than inheriting the other
    machine's.
    """
    import pathlib

    out: Dict[str, object] = {
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    try:                                    # Linux
        cpuinfo = pathlib.Path("/proc/cpuinfo").read_text().splitlines()
        for line in cpuinfo:
            if line.startswith("model name"):
                out["cpu"] = line.split(":", 1)[1].strip()
                break
        out["cores"] = len([l for l in cpuinfo if l.startswith("processor")])
        for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                out["memory_gb"] = round(int(line.split()[1]) / 1024 ** 2, 1)
                break
    except FileNotFoundError:               # macOS
        def sysctl(key):
            return subprocess.run(["sysctl", "-n", key],
                                  capture_output=True, text=True).stdout.strip()
        out["cpu"] = sysctl("machdep.cpu.brand_string")
        out["cores"] = int(sysctl("hw.ncpu") or 0)
        mem = sysctl("hw.memsize")
        out["memory_gb"] = round(int(mem) / 1024 ** 3, 1) if mem else None
    out.setdefault("cpu", platform.processor() or "unknown")
    return out


def environment_record(extra: Dict[str, object] | None = None) -> Dict[str, object]:
    """Everything the reproducibility appendix has to state.

    The project requires hardware, software versions, random seeds, training
    time and number of training examples for every learned model.  Week 36 has
    no learned model, but recording the environment from the first run means
    the appendix is not reconstructed from memory in week 44.
    """
    rec: Dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "seed": config.SEED,
        "dataset": config.DATASET_NAME,
        "languages": list(config.LANGS),
        "splits_used": list(config.SPLITS),
    }
    for mod in ("numpy", "pandas", "datasets", "sklearn", "torch", "transformers"):
        try:
            m = __import__(mod)
            rec[f"{mod}_version"] = getattr(m, "__version__", "unknown")
        except Exception:
            rec[f"{mod}_version"] = None
    try:
        rec["git_commit"] = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        rec["git_commit"] = None
    if extra:
        rec.update(extra)
    return rec


def save_environment(name: str = "environment_week36", extra: Dict[str, object] | None = None) -> Path:
    rec = environment_record(extra)
    path = config.RESULTS_DIR / f"{name}.json"
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=2, ensure_ascii=False)
    return path
