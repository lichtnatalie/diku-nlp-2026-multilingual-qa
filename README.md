# Reliable Multilingual Question Answering

Run the following commands from this folder (`src/`).

## Contents

- `src/`: model implementations, tokenisation, data handling and evaluation.
- `scripts/`: experiments, error analyses, and challenge evaluation.
- `tests/`: checks for project methods and evaluation.
- `notebooks/`: Colab workflows for training and challenge inference.
- `data/`, `models/`, `results/`: saved inputs and outputs needed for reproduction and report generation.
- `challenge.json`, `challenge_notes.json`, `CHALLENGE.md`: paired challenge records, glosses and source evidence.
- `EXTENSION-PROTOCOL.md`: hypothesis and procedure decieded before the extension.

The final PDF is the report. Submitted AI transcripts are in `../genai-logs/`.

## Setup and checks

```bash
python -m pip install -r requirements.txt
python -m pytest tests -q
python student_starter/validate_challenge.py challenge.json
```

`requirements-local.txt` records the completion environment. Integrated QA training requires a GPU and access to the pretrained model; see `notebooks/week40_integrated_qa.ipynb`.

## Reproduce experiments

```bash
python -m scripts.run_week36
python -m scripts.run_week37
python -m scripts.run_week38
python -m scripts.run_week39
python -m scripts.run_week40
python -m scripts.run_week43
```

These commands train models and may replace saved results. Keep a separate copy before rerunning. The Week 41 analysis scripts use saved QA predictions.

## Challenge evaluation

```bash
python -m scripts.prepare_challenge_models
python -m scripts.evaluate_challenge --qa-checkpoint /path/to/week40_xlmr_seed20260901
```

Checkpoint recovery and result-merging scripts are retained to document reproduction of the original frozen challenge evaluation. They do not form a new experiment. See `CHALLENGE.md` and the frozen provenance under `results/challenge/`.
