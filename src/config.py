"""Project-wide constants.

Everything that a later week might silently disagree with lives here, so that
Week 36 and Week 41 cannot drift apart.
"""

from pathlib import Path

# ---------------------------------------------------------------- dataset ---
DATASET_NAME = "coastalcph/tydi_xor_rc"

# The published dataset contains more question languages than the project uses
# (the first training row is Bengali).  The project description restricts the
# mandatory experiments to Arabic, Korean and Telugu, so every count reported
# in the report is computed AFTER this filter.
LANGS = ("ar", "ko", "te")
LANG_NAMES = {"ar": "Arabic", "ko": "Korean", "te": "Telugu"}

# The upstream repo also ships a `test` split, but it holds only 4 rows and is
# not usable for evaluation.  The shared protocol (Section 1) fits on `train`
# and selects/tunes on `validation`, so those are the only splits we touch.
SPLITS = ("train", "validation")

# ------------------------------------------------------------ label naming ---
# `answerable` is a bool in the raw data.  Throughout the project we use the
# convention: positive class = "unanswerable", because abstention is the
# phenomenon the research question is about.  Metrics are reported for BOTH
# classes anyway (Section 1), so this only fixes which one we call "positive"
# when a single number is quoted.
POSITIVE_CLASS = "unanswerable"
CLASSES = ("answerable", "unanswerable")

# ------------------------------------------------------------ reproducibility ---
SEED = 20260901

# Section 1 asks for more than one seed wherever stochastic variation could
# change a conclusion. SGDClassifier shuffles its data, so the Week 37 decoding
# comparison and the question-feature ablation are repeated over these five.
# SEED stays first so that every single-seed table in the report is the run
# that `SEEDS[0]` produces.
SEEDS = (SEED, SEED + 1, SEED + 2, SEED + 3, SEED + 4)

# ------------------------------------------------------------------ paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Local copy of the dataset.  huggingface.co is not reachable from every
# environment we run in, and pinning a local copy also makes the run
# reproducible: the numbers in the report come from these exact files.
DATA_DIR = PROJECT_ROOT / "data"
PARQUET_FILES = {s: DATA_DIR / f"{s}.parquet" for s in ("train", "validation", "test")}
