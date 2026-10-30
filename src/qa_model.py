"""Integrated multilingual QA model (project Section 6, Weeks 40-41).

One multilingual Transformer encoder, fine-tuned as an extractive QA model with
a no-answer option: a non-English question (ar/ko/te) and an English context
go in as ONE sequence, and a span head predicts a start and an end logit for
every token (Devlin et al. 2019, Fig. 4; HF LLM course, ch. 1 sec. 5).

THE PIPELINE, AND THE CONVENTION AT EACH STEP
---------------------------------------------
1. Windows.  Contexts longer than the model's input are split into
   overlapping windows of context tokens (`max_length`, `stride`). Every
   window is a full model input: <s> question </s></s> window </s> for XLM-R,
   [CLS] question [SEP] window [SEP] for mBERT. The tokenizer's own pair
   template builds it, so the module is model-agnostic.

   The windows are built here, from the context's own token offsets, rather
   than with the tokenizer's `return_overflowing_tokens`. While writing this
   module, the overflow path of the installed `tokenizers` (0.23.2) dropped the
   END of the context for a question/context pair with a stride: the last
   window stopped 13 characters before the end. Whether or not XLM-R's
   tokenizer does the same, a QA model that silently never sees the end of
   some contexts would cap recall without any error, so the windows are built
   explicitly and `tests/test_week40.py` asserts that every non-space
   character of every context is covered by at least one window.

2. Training targets.  An answerable example is labelled in a window only if
   the whole gold span lies inside that window's context; otherwise, and for
   every unanswerable example, the target is the first token (<s>/[CLS]),
   the SQuAD 2.0 convention. The span is located from the stored CHARACTER
   OFFSET, never by searching for the answer string (1,082 rows contain their
   answer more than once). A token belongs to the span if it OVERLAPS the gold
   characters, the same rule as the Week 37 BIO conversion. And answerability
   is read from `answerable`, never from the answer text: Week 36 found that
   all unanswerable rows carry a non-empty `answer` string.

3. Scoring a prediction.  For an example with windows w = 1..W:
       null score  = min over windows of  start_logit[cls] + end_logit[cls]
       span score  = max over windows, over context-token pairs i <= j with
                     j - i + 1 <= max_answer_tokens, of start[i] + end[j]
       no-answer score = null score - best span score          (Section 1)
   A HIGH no-answer score means "no answer". Taking the MINIMUM null score over
   windows follows the Hugging Face SQuAD v2 post-processing: a window that
   does not contain the answer is expected to point at <s>, so only the window
   that is LEAST convinced there is no answer speaks for the example.

4. Decision.  answer  iff  no-answer score < threshold.  The threshold is
   tuned on VALIDATION only, with `evaluation.tune_threshold` (the procedure
   every Section 5 classifier used), objective: answerability macro-F1.
   The untuned rule threshold = 0 ("answer if the best span beats the null
   span") is reported next to it, as Week 39 reported its 0.5 cut.

5. Model selection.  As in Week 39: the model is trained on 90% of TRAIN and
   the epoch is chosen on the held-out 10% `dev` slice, never on validation.
   The criterion is SQuAD-v2 token F1 on the dev slice at a threshold tuned on
   the dev slice, so that span quality and abstention both count.
"""

from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import config, evaluation

# Week 39 held out this share of TRAIN as `dev`, with config.SEED. The same
# slice is used here (a test checks that the two functions agree), so the
# integrated model and the Week 39 classifiers are tuned on identical data.
DEV_FRACTION = 0.10


def split_dev(train: pd.DataFrame, seed: int = config.SEED) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(fit part, dev slice) - identical to `scripts.run_week39.split_dev`."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train))
    n_dev = int(round(DEV_FRACTION * len(train)))
    return (train.iloc[idx[n_dev:]].reset_index(drop=True),
            train.iloc[idx[:n_dev]].reset_index(drop=True))


@dataclass
class QAConfig:
    """Every setting that changes the model. Saved with the results."""

    model_name: str = "FacebookAI/xlm-roberta-base"
    max_length: int = 384          # tokens per window, question included
    stride: int = 128              # context tokens shared by consecutive windows
    max_question_tokens: int = 64  # longest question in the data is 18 words
    max_answer_tokens: int = 30    # longest span the decoder may return
    n_best: int = 20               # start/end candidates per window
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    epochs: int = 3
    batch_size: int = 16
    eval_batch_size: int = 64
    max_grad_norm: float = 1.0
    seed: int = config.SEED
    fp16: bool = True              # only used on CUDA
    threshold_objective: str = "macro_f1"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


# =========================================================================
# 1. Windows
# =========================================================================

def _n_special_tokens(tokenizer) -> int:
    """Number of special tokens the tokenizer's PAIR template adds."""
    a = tokenizer("a", add_special_tokens=False)["input_ids"]
    b = tokenizer("b", add_special_tokens=False)["input_ids"]
    both = tokenizer("a", "b")["input_ids"]
    return len(both) - len(a) - len(b)


def encode_windows(
    question: str,
    context: str,
    tokenizer,
    max_length: int,
    stride: int,
    max_question_tokens: int = 64,
    pad: bool = True,
) -> List[Dict[str, object]]:
    """All model inputs (windows) for one question/context pair.

    Returns one dict per window with `input_ids`, `attention_mask`, `offsets`
    (a (start, end) character span in the CONTEXT for context tokens, None for
    question and special tokens), `cls_index`, and `char_range`, the
    [first, last) context characters the window actually covers.
    """
    q = tokenizer(question, add_special_tokens=False, return_offsets_mapping=True)
    q_off = q["offset_mapping"][:max_question_tokens]
    question_text = question[: q_off[-1][1]] if q_off else question

    c = tokenizer(context, add_special_tokens=False, return_offsets_mapping=True)
    c_off = [o for o in c["offset_mapping"] if o[1] > o[0]]
    if not c_off:
        raise ValueError("context produced no tokens")

    # Two tokens of margin: re-tokenising a substring can add a word-initial
    # marker at the window edge.
    budget = max_length - len(q_off) - _n_special_tokens(tokenizer) - 2
    if budget <= stride:
        raise ValueError(f"max_length {max_length} leaves {budget} context tokens per "
                         f"window, which must exceed the stride {stride}")

    spans: List[Tuple[int, int]] = []
    start = 0
    while True:
        end = min(start + budget, len(c_off))
        spans.append((start, end))
        if end == len(c_off):
            break
        start = end - stride

    windows = []
    for a, b in spans:
        lo, hi = c_off[a][0], c_off[b - 1][1]
        enc = tokenizer(
            question_text, context[lo:hi],
            truncation="only_second", max_length=max_length,
            padding="max_length" if pad else False,
            return_offsets_mapping=True,
        )
        seq_ids = enc.sequence_ids()
        offsets: List[Optional[Tuple[int, int]]] = []
        covered_hi = lo
        for (s, e), sid in zip(enc["offset_mapping"], seq_ids):
            if sid == 1 and e > s:
                offsets.append((s + lo, e + lo))
                covered_hi = max(covered_hi, e + lo)
            else:
                offsets.append(None)
        ids = list(enc["input_ids"])
        cls_index = ids.index(tokenizer.cls_token_id) if tokenizer.cls_token_id in ids else 0
        windows.append({
            "input_ids": ids,
            "attention_mask": list(enc["attention_mask"]),
            "offsets": offsets,
            "cls_index": cls_index,
            "char_range": (lo, covered_hi),
        })
    return windows


def _answer_token_span(offsets, char_start: int, char_end: int) -> Optional[Tuple[int, int]]:
    """Token indices (first, last) overlapping [char_start, char_end), or None.

    None also when the span is not ENTIRELY inside the window's context.
    """
    ctx = [(i, o) for i, o in enumerate(offsets) if o is not None]
    if not ctx:
        return None
    if char_start < ctx[0][1][0] or char_end > ctx[-1][1][1]:
        return None
    covered = [i for i, (s, e) in ctx if s < char_end and e > char_start]
    if not covered:
        return None
    return covered[0], covered[-1]


def build_features(df: pd.DataFrame, tokenizer, cfg: QAConfig, with_labels: bool) -> Dict[str, object]:
    """Windows for every row of `df`, with SQuAD-2.0 targets if `with_labels`.

    Returns a dict of parallel lists plus `example_index` (window -> row of df).
    With labels, raises if a stored answer does not match its stored offset:
    Week 36 checked every answerable row in both splits, and a mismatch here
    would mean the data changed. Without labels (prediction) the gold fields
    are not read at all.
    """
    feats = {k: [] for k in ("input_ids", "attention_mask", "offsets", "cls_index",
                             "char_range", "example_index", "start_positions",
                             "end_positions")}
    for row_i, row in enumerate(df.itertuples(index=False)):
        windows = encode_windows(row.question, row.context, tokenizer, cfg.max_length,
                                 cfg.stride, cfg.max_question_tokens)
        answerable = bool(row.answerable)
        if with_labels and answerable:
            a_start = int(row.answer_start)
            a_end = a_start + len(row.answer)
            if row.context[a_start:a_end] != row.answer:
                raise ValueError(f"row {row_i}: stored answer does not match its offset")
        for w in windows:
            for k in ("input_ids", "attention_mask", "offsets", "cls_index", "char_range"):
                feats[k].append(w[k])
            feats["example_index"].append(row_i)
            if with_labels:
                span = _answer_token_span(w["offsets"], a_start, a_end) if answerable else None
                if span is None:
                    span = (w["cls_index"], w["cls_index"])
                feats["start_positions"].append(span[0])
                feats["end_positions"].append(span[1])
    if not with_labels:
        del feats["start_positions"], feats["end_positions"]
    return feats


def sequence_length_audit(df: pd.DataFrame, tokenizer, cfg: QAConfig) -> pd.DataFrame:
    """How many pieces each pair needs, and how many windows it gets.

    The numbers the report needs to justify `max_length`/`stride`: per language,
    the median and maximum question+context length in model pieces, the share
    over the model's 512-position limit and over `max_length`, and how many
    examples are split into more than one window.
    """
    n_special = _n_special_tokens(tokenizer)
    rows = []
    for row in df.itertuples(index=False):
        nq = len(tokenizer(row.question, add_special_tokens=False)["input_ids"])
        nc = len(tokenizer(row.context, add_special_tokens=False)["input_ids"])
        n_win = len(encode_windows(row.question, row.context, tokenizer, cfg.max_length,
                                   cfg.stride, cfg.max_question_tokens, pad=False))
        rows.append({"lang": row.lang, "pieces": nq + nc + n_special,
                     "question_pieces": nq, "windows": n_win})
    t = pd.DataFrame(rows)
    out = []
    for subset in ["overall"] + list(config.LANGS):
        s = t if subset == "overall" else t[t["lang"] == subset]
        if s.empty:
            continue
        out.append({
            "subset": subset, "n": len(s),
            "pieces_median": float(s["pieces"].median()),
            "pieces_p99": float(s["pieces"].quantile(0.99)),
            "pieces_max": int(s["pieces"].max()),
            "question_pieces_median": float(s["question_pieces"].median()),
            "over_512": int((s["pieces"] > 512).sum()),
            f"over_{cfg.max_length}": int((s["pieces"] > cfg.max_length).sum()),
            "multi_window": int((s["windows"] > 1).sum()),
            "windows_total": int(s["windows"].sum()),
        })
    return pd.DataFrame(out)


def label_audit(df: pd.DataFrame, feats: Dict[str, object]) -> Dict[str, int]:
    """Check the training targets against the gold character spans.

    For every answerable row: is the answer labelled in at least one window,
    and does the labelled token span cover the gold characters exactly (the
    token span may be WIDER when the gold offset cuts a word, which Week 37
    found to be annotation slips)? For every unanswerable row: are all its
    windows labelled <s>? Returns counts; `run_week40` raises on a failure.
    """
    by_row: Dict[int, List[int]] = {}
    for f_i, r_i in enumerate(feats["example_index"]):
        by_row.setdefault(r_i, []).append(f_i)
    out = {"answerable": 0, "labelled_in_a_window": 0, "exact_char_match": 0,
           "wider_than_gold": 0, "unanswerable": 0, "unanswerable_all_cls": 0}
    for r_i, row in enumerate(df.itertuples(index=False)):
        fs = by_row[r_i]
        if not bool(row.answerable):
            out["unanswerable"] += 1
            out["unanswerable_all_cls"] += int(all(
                feats["start_positions"][f] == feats["cls_index"][f] ==
                feats["end_positions"][f] for f in fs))
            continue
        out["answerable"] += 1
        a_s, a_e = int(row.answer_start), int(row.answer_start) + len(row.answer)
        labelled = [f for f in fs if feats["start_positions"][f] != feats["cls_index"][f]]
        if not labelled:
            continue
        out["labelled_in_a_window"] += 1
        f = labelled[0]
        s = feats["offsets"][f][feats["start_positions"][f]][0]
        e = feats["offsets"][f][feats["end_positions"][f]][1]
        # SentencePiece offsets can include the space a word-initial piece
        # stands for; trim it, as `postprocess` does for predictions.
        raw = row.context[s:e]
        s += len(raw) - len(raw.lstrip())
        e -= len(raw) - len(raw.rstrip())
        if (s, e) == (a_s, a_e):
            out["exact_char_match"] += 1
        elif s <= a_s and e >= a_e:
            out["wider_than_gold"] += 1
    return out


# =========================================================================
# 3. Post-processing: logits -> span, null score, no-answer score
# =========================================================================

def postprocess(
    df: pd.DataFrame,
    feats: Dict[str, object],
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    max_answer_tokens: int = 30,
    n_best: int = 20,
) -> pd.DataFrame:
    """One row per example: best span, its score, null score, no-answer score.

    Only context tokens can start or end a span, end >= start, and the span has
    at most `max_answer_tokens` tokens. The predicted text is cut from the
    context by character offsets, with surrounding whitespace trimmed.
    """
    by_row: Dict[int, List[int]] = {}
    for f_i, r_i in enumerate(feats["example_index"]):
        by_row.setdefault(r_i, []).append(f_i)

    rows = []
    for r_i, row in enumerate(df.itertuples(index=False)):
        best_score, best_span = -math.inf, None
        null_score = math.inf
        for f in by_row[r_i]:
            s_log, e_log = start_logits[f], end_logits[f]
            offsets = feats["offsets"][f]
            cls = feats["cls_index"][f]
            null_score = min(null_score, float(s_log[cls] + e_log[cls]))
            ctx = np.array([o is not None for o in offsets])
            if not ctx.any():
                continue
            s_cand = [i for i in np.argsort(s_log)[::-1] if i < len(offsets) and ctx[i]][:n_best]
            e_cand = [j for j in np.argsort(e_log)[::-1] if j < len(offsets) and ctx[j]][:n_best]
            for i in s_cand:
                for j in e_cand:
                    if j < i or j - i + 1 > max_answer_tokens:
                        continue
                    score = float(s_log[i] + e_log[j])
                    if score > best_score:
                        best_score = score
                        best_span = (offsets[i][0], offsets[j][1])
        if best_span is None:
            pred_start, pred_text = -1, ""
            no_answer = math.inf
        else:
            raw = row.context[best_span[0]:best_span[1]]
            lead = len(raw) - len(raw.lstrip())
            pred_text = raw.strip()
            pred_start = best_span[0] + lead
            no_answer = null_score - best_score
        rows.append({
            "example_id": row.example_id,
            "lang": row.lang,
            "answerable": bool(row.answerable),
            "gold_start": int(row.answer_start) if bool(row.answerable) else -1,
            "gold_answer": row.answer if bool(row.answerable) else "",
            "span_start": pred_start,
            "span_text": pred_text,
            "span_score": best_score,
            "null_score": null_score,
            "no_answer_score": no_answer,
            "n_windows": len(by_row[r_i]),
        })
    return pd.DataFrame(rows)


def apply_threshold(pred: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Final decision: answer iff no-answer score < threshold (HIGH = no answer)."""
    out = pred.copy()
    out["answered"] = out["no_answer_score"] < threshold
    out["pred_start"] = np.where(out["answered"], out["span_start"], -1)
    out["pred_text"] = np.where(out["answered"], out["span_text"], "")
    out["threshold"] = threshold
    return out


def tune_no_answer_threshold(pred: pd.DataFrame, objective: str = "macro_f1") -> Dict[str, float]:
    """Section 1 threshold, via the shared `evaluation.tune_threshold`.

    `higher_means_answerable=False`: a HIGH no-answer score means no answer.
    Infinite scores (no valid span at all) are clipped so the grid stays finite;
    such an example abstains at every threshold.
    """
    scores = pred["no_answer_score"].to_numpy(dtype=float)
    finite = scores[np.isfinite(scores)]
    cap = (finite.max() + 1.0) if finite.size else 1.0
    scores = np.where(np.isfinite(scores), scores, cap)
    return evaluation.tune_threshold(scores, pred["answerable"].to_numpy(bool),
                                     higher_means_answerable=False, objective=objective)


# =========================================================================
# 4. Training
# =========================================================================

def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def snapshot_state(model) -> Dict[str, object]:
    """A real COPY of the weights, on CPU.

    `model.state_dict()` returns references to the live tensors, which later
    optimiser steps overwrite in place - Lab 5's "best model" was therefore
    always the last epoch. Copying is what makes best-epoch selection real; a
    test checks that later training does not change a snapshot.
    """
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}


class _Batches:
    """Minimal batching over the feature lists (no DataLoader workers needed)."""

    def __init__(self, feats, keys, batch_size, shuffle, seed):
        self.feats, self.keys, self.bs = feats, keys, batch_size
        self.shuffle, self.rng = shuffle, np.random.default_rng(seed)

    def __len__(self):
        return math.ceil(len(self.feats["input_ids"]) / self.bs)

    def __iter__(self):
        import torch

        n = len(self.feats["input_ids"])
        order = self.rng.permutation(n) if self.shuffle else np.arange(n)
        for b in range(0, n, self.bs):
            idx = order[b:b + self.bs]
            yield {k: torch.tensor([self.feats[k][i] for i in idx]) for k in self.keys}


class IntegratedQAModel:
    """Wraps tokenizer + span-head model + the Section 1 decision rule."""

    def __init__(self, cfg: QAConfig, tokenizer=None, model=None, device: Optional[str] = None):
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self.cfg = cfg
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(cfg.model_name)
        if getattr(self.tokenizer, "padding_side", "right") != "right":
            raise ValueError("question-first inputs assume right padding")
        if model is None:
            set_seed(cfg.seed)  # the span head is randomly initialised
            model = AutoModelForQuestionAnswering.from_pretrained(cfg.model_name)
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.history_: List[Dict[str, float]] = []
        self.best_epoch_: Optional[int] = None
        self.train_seconds_: Optional[float] = None
        self.n_train_features_: Optional[int] = None

    # ---------------------------------------------------------------- train
    def fit(self, fit_df: pd.DataFrame, dev_df: Optional[pd.DataFrame] = None, log=print):
        """Train on `fit_df`; if `dev_df` is given, keep the best dev epoch."""
        import torch

        cfg = self.cfg
        set_seed(cfg.seed)
        feats = build_features(fit_df, self.tokenizer, cfg, with_labels=True)
        self.n_train_features_ = len(feats["input_ids"])
        batches = _Batches(feats, ("input_ids", "attention_mask", "start_positions",
                                   "end_positions"), cfg.batch_size, True, cfg.seed)
        no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
        groups = [
            {"params": [p for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in no_decay)],
             "weight_decay": cfg.weight_decay},
            {"params": [p for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        opt = torch.optim.AdamW(groups, lr=cfg.learning_rate)
        total = cfg.epochs * len(batches)
        warmup = int(cfg.warmup_ratio * total)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / max(1, warmup) if s < warmup
            else max(0.0, (total - s) / max(1, total - warmup)))
        use_amp = cfg.fp16 and self.device == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        dev_feats = (build_features(dev_df, self.tokenizer, cfg, with_labels=False)
                     if dev_df is not None else None)
        best_score, best_state = -math.inf, None
        t0 = time.time()
        for epoch in range(1, cfg.epochs + 1):
            self.model.train()
            losses = []
            for batch in batches:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    loss = self.model(**batch).loss
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                    opt.step()
                sched.step()
                losses.append(loss.item())
            record = {"epoch": epoch, "train_loss": float(np.mean(losses)),
                      "seconds": round(time.time() - t0, 1)}
            if dev_df is not None:
                pred = self._predict_raw(dev_df, dev_feats)
                cut = tune_no_answer_threshold(pred, cfg.threshold_objective)
                m = evaluation.qa_metrics(apply_threshold(pred, cut["threshold"]))
                record.update({"dev_threshold": cut["threshold"], "dev_f1": m["f1"],
                               "dev_exact_match": m["exact_match"],
                               "dev_macro_f1": m["macro_f1"]})
                if m["f1"] > best_score:
                    best_score, best_state = m["f1"], snapshot_state(self.model)
                    self.best_epoch_ = epoch
            self.history_.append(record)
            log(f"  epoch {epoch}: " + ", ".join(
                f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                for k, v in record.items() if k != "epoch"))
        self.train_seconds_ = round(time.time() - t0, 1)
        if best_state is not None:
            self.model.load_state_dict(best_state)
        else:
            self.best_epoch_ = cfg.epochs
        return self

    # -------------------------------------------------------------- predict
    def _logits(self, feats) -> Tuple[np.ndarray, np.ndarray]:
        import torch

        self.model.eval()
        starts, ends = [], []
        batches = _Batches(feats, ("input_ids", "attention_mask"), self.cfg.eval_batch_size,
                           False, 0)
        with torch.no_grad():
            for batch in batches:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                out = self.model(**batch)
                starts.append(out.start_logits.float().cpu().numpy())
                ends.append(out.end_logits.float().cpu().numpy())
        return np.concatenate(starts), np.concatenate(ends)

    def _predict_raw(self, df: pd.DataFrame, feats=None) -> pd.DataFrame:
        feats = feats or build_features(df, self.tokenizer, self.cfg, with_labels=False)
        s, e = self._logits(feats)
        return postprocess(df, feats, s, e, self.cfg.max_answer_tokens, self.cfg.n_best)

    def predict_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Best span and no-answer score per example, before any threshold."""
        return self._predict_raw(df)


# =========================================================================
# 5. Bridge to the Week 37 BIO evaluation
# =========================================================================

def char_spans_to_bio(encoded: Sequence[Dict[str, object]], pred: pd.DataFrame) -> List[List[str]]:
    """Turn predicted character spans into BIO labels over the Week 37 tokens.

    Lets the integrated model be scored by `evaluation.span_metrics`, the exact
    function the Week 37 labeller was scored by (token-level answer-token F1 and
    exact span match), so the span comparison Section 6 asks for uses ONE metric
    implementation. Same overlap rule as `spans.char_span_to_bio`. An abstention
    (pred_start == -1) becomes all-O.
    """
    from . import spans

    by_id = pred.set_index("example_id")
    out = []
    for e in encoded:
        labels = [spans.OUTSIDE] * len(e["tokens"])
        row = by_id.loc[e["example_id"]]
        start, text = int(row["pred_start"]), str(row["pred_text"])
        if start >= 0 and text:
            end = start + len(text)
            covered = [i for i, (s, t) in enumerate(e["offsets"]) if s < end and t > start]
            for k, i in enumerate(covered):
                labels[i] = spans.BEGIN if k == 0 else spans.INSIDE
        out.append(labels)
    return out


# =========================================================================
# Offline test double
# =========================================================================

def tiny_test_model(texts: Sequence[str] = (), vocab_size: int = 300, seed: int = 0,
                    tokenizer=None):
    """A tiny random XLM-R with a tokenizer trained on `texts`. TESTS ONLY.

    huggingface.co is not reachable from every environment this project runs
    in, so the tests and `run_week40.py --smoke` need a model that can be built
    offline. Same architecture class and the same <s> A </s></s> B </s> pair
    template as XLM-R; its predictions mean nothing.
    """
    import torch
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast, XLMRobertaConfig, XLMRobertaForQuestionAnswering

    if tokenizer is not None:
        # Reuse a tokenizer from an earlier call: only the weights change.
        torch.manual_seed(seed)
        cfg = XLMRobertaConfig(
            vocab_size=len(tokenizer), hidden_size=32, num_hidden_layers=2,
            num_attention_heads=2, intermediate_size=64, max_position_embeddings=520,
            pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.cls_token_id,
            eos_token_id=tokenizer.sep_token_id)
        return tokenizer, XLMRobertaForQuestionAnswering(cfg)

    tok = Tokenizer(models.Unigram())
    tok.normalizer = normalizers.NFKC()
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    tok.decoder = decoders.Metaspace()
    tok.train_from_iterator(list(texts), trainers.UnigramTrainer(
        vocab_size=vocab_size, special_tokens=["<s>", "<pad>", "</s>", "<unk>"],
        unk_token="<unk>"))
    s_id, pad_id, e_id = (tok.token_to_id(t) for t in ("<s>", "<pad>", "</s>"))
    tok.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>", pair="<s> $A </s> </s> $B </s>",
        special_tokens=[("<s>", s_id), ("</s>", e_id)])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok, bos_token="<s>", eos_token="</s>", cls_token="<s>",
        sep_token="</s>", pad_token="<pad>", unk_token="<unk>")
    torch.manual_seed(seed)
    cfg = XLMRobertaConfig(
        vocab_size=len(tokenizer), hidden_size=32, num_hidden_layers=2,
        num_attention_heads=2, intermediate_size=64, max_position_embeddings=520,
        pad_token_id=pad_id, bos_token_id=s_id, eos_token_id=e_id)
    return tokenizer, XLMRobertaForQuestionAnswering(cfg)
