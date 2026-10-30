"""Record the environment the reported numbers were produced in.

Section 10 of the project asks for versions, hardware, training time and the
number of training examples for every learned model. Those are facts about a
machine, not about the data, so they are measured here and written to
``results/39_environment.json`` rather than typed into the report: the appendix
reads them through ``scripts/make_report_tables.py`` like every other number.

Run it on the machine that produced the submitted results:

    python3 -m scripts.record_environment

It refits the Week 37 labeller once (about a minute and a half on the machine
this was written on) to time it, and fits the five Week 38 language models,
which take well under a second together. Nothing it does touches results/
tables other than the JSON it writes.
"""
from __future__ import annotations

import json
import platform
import subprocess
import time
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "39_environment.json"
PACKAGES = ("numpy", "pandas", "sklearn", "scipy", "datasets", "pyarrow")


def _version(name: str) -> str:
    try:
        return getattr(import_module(name), "__version__", "unknown")
    except ImportError:
        return "not installed"


def _cpu() -> dict:
    """CPU model, core count and memory, from whichever of the two this is."""
    out = {"machine": platform.machine(), "processor": platform.processor()}
    try:                                    # Linux
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                out["cpu"] = line.split(":", 1)[1].strip()
                break
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                out["memory_gb"] = round(int(line.split()[1]) / 1024 ** 2, 1)
                break
        out["cores"] = len(
            [l for l in Path("/proc/cpuinfo").read_text().splitlines()
             if l.startswith("processor")])
    except FileNotFoundError:               # macOS
        def sysctl(key):
            return subprocess.run(["sysctl", "-n", key],
                                  capture_output=True, text=True).stdout.strip()
        out["cpu"] = sysctl("machdep.cpu.brand_string")
        out["cores"] = int(sysctl("hw.ncpu") or 0)
        mem = sysctl("hw.memsize")
        out["memory_gb"] = round(int(mem) / 1024 ** 3, 1) if mem else None
    return out


def main() -> None:
    from src import config, data, span_models, spans
    import src.language_model as lm
    import scripts.run_week38 as week38

    record = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: _version(name) for name in PACKAGES},
        "hardware": _cpu(),
        "seed": config.SEED,
        "seeds": list(config.SEEDS),
    }

    splits = data.load_project_splits()
    encoded = {name: spans.encode_split(df) for name, df in splits.items()}
    labeller = span_models.QuestionConditionedLabeller(beam_size=4, seed=config.SEED)
    t0 = time.perf_counter()
    labeller.fit(encoded["train"])
    record["labeller"] = {
        "fit_seconds": round(time.perf_counter() - t0, 1),
        "n_train_examples": len(encoded["train"]),
        "n_train_tokens": sum(len(e["tokens"]) for e in encoded["train"]),
        "epochs": int(labeller.classifier.max_iter),
        "note": "one fit; the seed sweep repeats it once per seed",
    }

    questions = [lm.tokenise_question(q) for q in
                 splits["train"].loc[splits["train"]["lang"] == week38.LANG, "question"]]
    vocab = lm.build_vocabulary(questions, week38.MIN_COUNT)
    params = {"k": 0.005, "bigram_l2": 0.75, "trigram_l2": 0.65, "trigram_l3": 0.35}
    t0 = time.perf_counter()
    week38.fit_models(questions, vocab, params)
    record["language_models"] = {
        "fit_seconds_all_five": round(time.perf_counter() - t0, 3),
        "n_train_questions": len(questions),
    }

    OUT.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
