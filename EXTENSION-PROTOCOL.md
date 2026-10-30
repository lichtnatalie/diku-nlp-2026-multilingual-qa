# Week 43 extension protocol

On 21 October 2026 before computing any extension results or evaluating the challenge set.

Hypothesis: language-specific no-answer thresholds improve overall answerability macro-F1 relative to one global threshold on the same held-out validation examples, averaged over the three existing QA seeds. The expected direction is an increase, particularly in Telugu unanswerable recall. SQuAD-v2 exact match and token F1 are secondary outcomes. A negative result will be reported as such.

Keep the existing trained QA models, raw span predictions and no-answer scores fixed. The only intervention is replacing one global threshold with one threshold per question language. All thresholds maximise the shared answerability macro-F1 criterion on calibration examples only. Answer iff no-answer score is strictly below threshold.

Use one deterministic five-fold partition shared across all seeds and conditions. Keep identical question strings together across folds to avoid calibrating and evaluating copies of the same question. Stratify by language and answerability where the grouped split permits. In each fold, tune thresholds on the other four folds and evaluate on the held-out fold. Compare the global and per-language methods on exactly the same held-out rows. Report overall and per-language N, both-class P/R/F1, macro-F1, SQuAD-v2 EM/F1, per-seed differences and the mean/range of the three differences. Do not select the extension after seeing results.

Then fit both threshold policies on the full validation split and freeze their values, model/config identifiers and input hashes before challenge inference. Full-validation tuned results are descriptive, not held-out estimates. Challenge examples, labels and outcomes are never used for tuning or model selection. Use the first project seed for the small challenge comparison, fixed in advance, with the same QA checkpoint for core QA and the extension.
