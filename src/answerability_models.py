"""Learned answerability classifiers for project Section 5.

Two systems, both of which see the question and the context and output a
probability that the example is ANSWERABLE, so that `evaluation.tune_threshold`
and `evaluation.answerability_report` score them identically to the Week 36
baselines:

  * `FeatureAnswerabilityClassifier` - the required interpretable learned
    baseline. Logistic regression over (i) bag-of-words counts of the question,
    (ii) cross-lingual overlap features built from the Week 36 anchor rule, and
    (iii) length features. Every coefficient is readable, which is the point of
    the "interpretable" in the brief, and the three feature blocks can be
    ablated separately.

  * `RecurrentAnswerabilityClassifier` - the required recurrent neural
    classifier. A BiGRU encoder for the question and a second BiGRU encoder for
    the context, one embedding table shared by both, and dot-product attention
    from the question summary over the context states. Representations are
    learned from scratch: no pretrained vectors are used, and the reason is
    stated in the report (the English static embeddings in the course are
    useless for ar/ko/te questions, and per-language spaces are not aligned).

Why the question and the context must interact inside the model
---------------------------------------------------------------
Answerability is not a property of either string alone. The same question is
answerable against one passage and not against another, and the validation
split contains questions that appear with more than one passage. A model that
encoded the two sides separately and only combined them in the final classifier
could still learn "this passage looks thin" or "this question looks hard", but
it could not learn "the thing this question asks about is missing from this
passage". Both systems here therefore contain an explicit interaction term:

  * the feature model: anchor hits, i.e. how many of the question's
    script-preserved anchors occur in the context, and the derived hit rate;
  * the recurrent model: attention weights over the context tokens computed
    from the question summary, so every context position is scored for
    relevance to the question before the sequence is pooled.

The attention weights are also what makes the neural model inspectable at all,
and the error analysis uses them.

ACKNOWLEDGEMENT
---------------
The anchor extraction is Week 36's `baselines.CrossLingualAnchorRule`, reused
here as a feature source rather than as a decision rule. The BiGRU-with-
attention design follows the Week 39 lecture (`rnn_slides_ucph.ipynb`, the
many-to-one architecture and the bidirectional slides) and J&M Ch. 14: the
encoder is 14.3.2 (sequence classification, with mean pooling replaced by
attention pooling) and the attention is the dot-product form of 14.35-14.37,
with the question summary in the role of the decoder state. The training loop
(AdamW, linear warmup, best-epoch checkpointing on a held-out slice) follows
Lab 4's Transformer loop, with Lab 4's two bugs fixed: the label mapping is
sorted rather than read off a Python set, and padding uses the padding id
rather than index 0.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import config
from .baselines import CrossLingualAnchorRule
from .tokenisation import normalise_digits, unicode_word_tokenise

# --------------------------------------------------------------------------
# Feature model
# --------------------------------------------------------------------------

#: Names of the dense feature block, in the order `dense_features` returns them.
DENSE_FEATURE_NAMES = (
    "n_anchors",
    "n_anchor_hits",
    "anchor_hit_rate",
    "has_anchor",
    "all_anchors_hit",
    "no_anchor_hit",
    "n_digit_anchors",
    "n_digit_hits",
    "n_latin_anchors",
    "n_latin_hits",
    "question_tokens",
    "log_context_tokens",
    "context_tokens_per_question_token",
)


def dense_features(
    df: pd.DataFrame, rule: Optional[CrossLingualAnchorRule] = None
) -> np.ndarray:
    """The overlap and length block: one row per example, `DENSE_FEATURE_NAMES`.

    The overlap features are the cross-lingual ones from Week 36: Latin-script
    runs and digit runs in the question that also occur in the context. They
    are the only lexical overlap available without a translation system,
    because question and context are in different scripts.
    """
    rule = rule or CrossLingualAnchorRule()
    rows = []
    for row in df.itertuples(index=False):
        anchors = rule.anchors(row.question)
        ctx = normalise_digits(row.context or "").lower()
        hits = [a for a in anchors if a in ctx]
        digits = [a for a in anchors if a[0].isdigit()]
        digit_hits = [a for a in hits if a[0].isdigit()]
        latin = [a for a in anchors if not a[0].isdigit()]
        latin_hits = [a for a in hits if not a[0].isdigit()]
        q_tokens = len(unicode_word_tokenise(row.question or ""))
        c_tokens = len(unicode_word_tokenise(row.context or ""))
        rows.append(
            [
                len(anchors),
                len(hits),
                len(hits) / len(anchors) if anchors else 0.0,
                float(bool(anchors)),
                float(bool(anchors) and len(hits) == len(anchors)),
                float(bool(anchors) and not hits),
                len(digits),
                len(digit_hits),
                len(latin),
                len(latin_hits),
                float(q_tokens),
                math.log1p(c_tokens),
                c_tokens / q_tokens if q_tokens else 0.0,
            ]
        )
    return np.asarray(rows, dtype=float)


class FeatureAnswerabilityClassifier:
    """Logistic regression over question bag-of-words + overlap + length.

    Parameters
    ----------
    use_question_bow, use_overlap, use_length
        Switches for the three feature blocks, so the report can ablate them.
    C
        Inverse regularisation strength. Chosen on a held-out slice of the
        TRAINING split by `run_week39.py`, never on validation.
    min_df
        Minimum document frequency for a question token to get its own
        feature. Rare tokens are dropped rather than given a weight fitted from
        one or two examples, which is the Week 36 "false killer feature"
        problem.
    """

    def __init__(
        self,
        use_question_bow: bool = True,
        use_overlap: bool = True,
        use_length: bool = True,
        C: float = 1.0,
        min_df: int = 5,
        seed: int = config.SEED,
    ) -> None:
        self.use_question_bow = use_question_bow
        self.use_overlap = use_overlap
        self.use_length = use_length
        self.C = C
        self.min_df = min_df
        self.seed = seed
        self.rule = CrossLingualAnchorRule()
        self.vectoriser_: Optional[CountVectorizer] = None
        self.scaler_: Optional[StandardScaler] = None
        self.model_: Optional[LogisticRegression] = None
        self.feature_names_: List[str] = []

    # -- feature assembly --------------------------------------------------
    def _dense_block(self, df: pd.DataFrame) -> np.ndarray:
        full = dense_features(df, self.rule)
        keep = []
        for i, name in enumerate(DENSE_FEATURE_NAMES):
            is_length = name in (
                "question_tokens",
                "log_context_tokens",
                "context_tokens_per_question_token",
            )
            if is_length and self.use_length:
                keep.append(i)
            elif not is_length and self.use_overlap:
                keep.append(i)
        return full[:, keep] if keep else np.zeros((len(df), 0))

    def _dense_names(self) -> List[str]:
        names = []
        for name in DENSE_FEATURE_NAMES:
            is_length = name in (
                "question_tokens",
                "log_context_tokens",
                "context_tokens_per_question_token",
            )
            if (is_length and self.use_length) or (not is_length and self.use_overlap):
                names.append(name)
        return names

    def _matrix(self, df: pd.DataFrame, fit: bool) -> sparse.csr_matrix:
        blocks = []
        names: List[str] = []

        if self.use_question_bow:
            if fit:
                self.vectoriser_ = CountVectorizer(
                    tokenizer=unicode_word_tokenise,
                    preprocessor=normalise_digits,
                    lowercase=True,
                    min_df=self.min_df,
                    token_pattern=None,
                )
                bow = self.vectoriser_.fit_transform(df["question"].tolist())
            else:
                bow = self.vectoriser_.transform(df["question"].tolist())
            blocks.append(bow)
            names += [f"q:{t}" for t in self.vectoriser_.get_feature_names_out()]

        dense = self._dense_block(df)
        if dense.shape[1]:
            if fit:
                self.scaler_ = StandardScaler()
                dense = self.scaler_.fit_transform(dense)
            else:
                dense = self.scaler_.transform(dense)
            blocks.append(sparse.csr_matrix(dense))
            names += self._dense_names()

        if not blocks:
            raise ValueError("all feature blocks are switched off")
        if fit:
            self.feature_names_ = names
        return sparse.hstack(blocks).tocsr()

    # -- sklearn-ish API ---------------------------------------------------
    def fit(self, train: pd.DataFrame) -> "FeatureAnswerabilityClassifier":
        X = self._matrix(train, fit=True)
        y = train["answerable"].to_numpy(dtype=bool)
        # `balanced` weights: unanswerable is 5.7% of the training split, and
        # an unweighted fit puts almost no mass on it.
        self.model_ = LogisticRegression(
            C=self.C,
            max_iter=2000,
            class_weight="balanced",
            random_state=self.seed,
        ).fit(X, y)
        return self

    def predict_proba_answerable(self, df: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("call fit() first")
        X = self._matrix(df, fit=False)
        classes = list(self.model_.classes_)
        return self.model_.predict_proba(X)[:, classes.index(True)]

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return self.predict_proba_answerable(df) >= threshold

    # -- interpretability --------------------------------------------------
    def coefficients(self, top: int = 12) -> pd.DataFrame:
        """The largest weights in both directions, for the report's §5 prose."""
        if self.model_ is None:
            raise RuntimeError("call fit() first")
        coef = self.model_.coef_[0]
        order = np.argsort(coef)
        idx = np.concatenate([order[:top], order[-top:]])
        return pd.DataFrame(
            {
                "feature": [self.feature_names_[i] for i in idx],
                "weight": coef[idx],
                "pushes_towards": [
                    "unanswerable" if coef[i] < 0 else "answerable" for i in idx
                ],
            }
        ).sort_values("weight")

    def describe(self) -> str:
        blocks = [
            name
            for flag, name in (
                (self.use_question_bow, "question bag-of-words"),
                (self.use_overlap, "anchor overlap"),
                (self.use_length, "length"),
            )
            if flag
        ]
        n_feat = len(self.feature_names_)
        return f"logistic regression (C={self.C}, {n_feat} features: {', '.join(blocks)})"


# --------------------------------------------------------------------------
# Recurrent model
# --------------------------------------------------------------------------

PAD, UNK = "<pad>", "<unk>"


def build_vocabulary(
    df: pd.DataFrame, min_count: int = 2, max_size: int = 40000
) -> Dict[str, int]:
    """One vocabulary over question AND context tokens of the training split.

    Sharing the table matters: a Latin-script name occurs in an Arabic question
    and in the English context as the same token, so the model can learn that
    seeing it on both sides is evidence, which is the neural counterpart of the
    anchor feature. Tokens are sorted before indexing, so the mapping does not
    depend on set iteration order (the bug in Lab 4, cell 65).
    """
    from collections import Counter

    counts: "Counter[str]" = Counter()
    for col in ("question", "context"):
        for text in df[col]:
            counts.update(_tokens(text))
    kept = sorted(
        (t for t, c in counts.items() if c >= min_count),
        key=lambda t: (-counts[t], t),
    )[:max_size]
    vocab = {PAD: 0, UNK: 1}
    for token in sorted(kept):
        vocab[token] = len(vocab)
    return vocab


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in unicode_word_tokenise(normalise_digits(text or ""))]


def _encode(text: str, vocab: Dict[str, int], max_len: int) -> List[int]:
    ids = [vocab.get(t, vocab[UNK]) for t in _tokens(text)][:max_len]
    return ids or [vocab[UNK]]


class RecurrentAnswerabilityClassifier:
    """BiGRU question encoder + BiGRU context encoder + attention interaction.

    The forward pass, in the notation of the Week 39 notes:

        e_i          = E x_i                              shared embedding
        H^q          = BiGRU_q(e(question))               [n_q, 2d]
        h_q          = max_i H^q_i  (over real tokens)    [2d]
        H^c          = BiGRU_c(e(context))                [n_c, 2d]
        alpha_j      = softmax_j( (W h_q) . H^c_j )       dot-product attention
        c            = sum_j alpha_j H^c_j                attention-pooled context
        p(answerable)= sigmoid( MLP([h_q ; c ; |h_q - c| ; h_q * c]) )

    The concatenation of difference and product is the standard sentence-pair
    interaction; it gives the classifier a direct view of "how far apart" the
    question summary and the attended context are, rather than making it
    rediscover that from the two vectors.
    """

    def __init__(
        self,
        vocab: Dict[str, int],
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        lr: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 8,
        max_question_len: int = 24,
        max_context_len: int = 300,
        seed: int = config.SEED,
        use_attention: bool = True,
        use_context: bool = True,
        use_question: bool = True,
    ) -> None:
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_question_len = max_question_len
        self.max_context_len = max_context_len
        self.seed = seed
        self.use_attention = use_attention
        self.use_context = use_context
        self.use_question = use_question
        self.model_ = None
        self.history_: List[Dict[str, float]] = []
        self.train_seconds_: float = 0.0
        self.n_train_examples_: int = 0

    # -- torch plumbing ----------------------------------------------------
    def _build(self):
        import torch
        from torch import nn

        vocab_size = len(self.vocab)
        d, h, dropout = self.embedding_dim, self.hidden_dim, self.dropout
        use_attention, use_context, use_question = (
            self.use_attention,
            self.use_context,
            self.use_question,
        )

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(vocab_size, d, padding_idx=0)
                self.q_rnn = nn.GRU(d, h, batch_first=True, bidirectional=True)
                self.c_rnn = nn.GRU(d, h, batch_first=True, bidirectional=True)
                self.att = nn.Linear(2 * h, 2 * h, bias=False)
                n_blocks = 4 if (use_question and use_context) else 1
                self.head = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(n_blocks * 2 * h, h),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(h, 1),
                )

            def encode(self, rnn, ids, mask):
                # Pack, so the backward direction starts at the last REAL
                # token rather than at padding. Without this the backward
                # states of short sequences are computed over pad embeddings.
                lengths = mask.sum(dim=1).cpu()
                packed = nn.utils.rnn.pack_padded_sequence(
                    self.emb(ids), lengths, batch_first=True, enforce_sorted=False
                )
                out, _ = rnn(packed)
                out, _ = nn.utils.rnn.pad_packed_sequence(
                    out, batch_first=True, total_length=ids.size(1)
                )
                return out

            def summary(self, states, mask):
                """Masked max-pool over time: the padded positions never win."""
                neg = torch.finfo(states.dtype).min
                masked = states.masked_fill(~mask.unsqueeze(-1), neg)
                return masked.max(dim=1).values

            def forward(self, q_ids, q_mask, c_ids, c_mask, return_attention=False):
                h_q = None
                if use_question:
                    q_states = self.encode(self.q_rnn, q_ids, q_mask)
                    h_q = self.summary(q_states, q_mask)
                alpha = None
                c_vec = None
                if use_context:
                    c_states = self.encode(self.c_rnn, c_ids, c_mask)
                    if use_attention and use_question:
                        scores = torch.einsum("bd,btd->bt", self.att(h_q), c_states)
                        scores = scores.masked_fill(
                            ~c_mask, torch.finfo(scores.dtype).min
                        )
                        alpha = torch.softmax(scores, dim=1)
                        c_vec = torch.einsum("bt,btd->bd", alpha, c_states)
                    else:
                        c_vec = self.summary(c_states, c_mask)

                if use_question and use_context:
                    feats = torch.cat(
                        [h_q, c_vec, (h_q - c_vec).abs(), h_q * c_vec], dim=-1
                    )
                elif use_question:
                    feats = h_q
                else:
                    feats = c_vec
                logit = self.head(feats).squeeze(-1)
                return (logit, alpha) if return_attention else logit

        return Net()

    def _batches(self, df: pd.DataFrame, shuffle: bool, generator=None):
        import torch

        n = len(df)
        order = np.arange(n)
        if shuffle:
            rng = np.random.default_rng(self.seed)
            order = rng.permutation(n)
        questions = df["question"].tolist()
        contexts = df["context"].tolist()
        labels = df["answerable"].to_numpy(dtype=np.float32)
        for start in range(0, n, self.batch_size):
            idx = order[start : start + self.batch_size]
            q = [_encode(questions[i], self.vocab, self.max_question_len) for i in idx]
            c = [_encode(contexts[i], self.vocab, self.max_context_len) for i in idx]
            q_ids, q_mask = _pad(q)
            c_ids, c_mask = _pad(c)
            yield (
                torch.tensor(q_ids),
                torch.tensor(q_mask),
                torch.tensor(c_ids),
                torch.tensor(c_mask),
                torch.tensor(labels[idx]),
                idx,
            )

    # -- training ----------------------------------------------------------
    def fit(
        self, train: pd.DataFrame, dev: Optional[pd.DataFrame] = None
    ) -> "RecurrentAnswerabilityClassifier":
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        np.random.seed(self.seed % (2**32))
        self.model_ = self._build()
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=1e-4)
        # Unanswerable is the rare class; weight it up by its inverse frequency
        # so the model does not collapse onto the majority label the way the
        # Week 36 majority baseline does by construction.
        pos = float(train["answerable"].mean())
        # BCEWithLogits scales the loss of the POSITIVE label, and our positive
        # label is "answerable" (the common one). Weighting it by
        # (1 - pos) / pos therefore up-weights the rare unanswerable class by
        # its inverse frequency, which is what `class_weight="balanced"` does
        # for the feature model.
        weight = (1.0 - pos) / pos if 0 < pos < 1 else 1.0
        self.pos_weight_ = weight  # recorded so the appendix and the tests can read it
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight))

        best_state, best_score, best_epoch = None, -np.inf, -1
        start = time.time()
        for epoch in range(1, self.epochs + 1):
            self.model_.train()
            epoch_loss, n_seen = 0.0, 0
            for q_ids, q_mask, c_ids, c_mask, y, _ in self._batches(train, shuffle=True):
                opt.zero_grad()
                logit = self.model_(q_ids, q_mask, c_ids, c_mask)
                loss = loss_fn(logit, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                opt.step()
                epoch_loss += float(loss.detach()) * len(y)
                n_seen += len(y)
            record = {"epoch": epoch, "train_loss": epoch_loss / max(n_seen, 1)}
            if dev is not None:
                from .evaluation import answerability_metrics

                p = self.predict_proba_answerable(dev)
                m = answerability_metrics(
                    dev["answerable"].to_numpy(dtype=bool), p >= 0.5
                )
                record["dev_macro_f1"] = m["macro_f1"]
                if m["macro_f1"] > best_score:
                    best_score, best_epoch = m["macro_f1"], epoch
                    best_state = {
                        k: v.detach().clone() for k, v in self.model_.state_dict().items()
                    }
            self.history_.append(record)
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.best_epoch_ = best_epoch
        self.train_seconds_ = time.time() - start
        self.n_train_examples_ = len(train)
        return self

    # -- inference ---------------------------------------------------------
    def predict_proba_answerable(self, df: pd.DataFrame) -> np.ndarray:
        import torch

        if self.model_ is None:
            raise RuntimeError("call fit() first")
        self.model_.eval()
        out = np.zeros(len(df), dtype=float)
        with torch.no_grad():
            for q_ids, q_mask, c_ids, c_mask, _, idx in self._batches(df, shuffle=False):
                logit = self.model_(q_ids, q_mask, c_ids, c_mask)
                out[idx] = torch.sigmoid(logit).numpy()
        return out

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return self.predict_proba_answerable(df) >= threshold

    def attention_for(self, question: str, context: str, top: int = 8):
        """The context tokens the model attends to, for the error analysis."""
        import torch

        q = [_encode(question, self.vocab, self.max_question_len)]
        c = [_encode(context, self.vocab, self.max_context_len)]
        q_ids, q_mask = _pad(q)
        c_ids, c_mask = _pad(c)
        self.model_.eval()
        with torch.no_grad():
            logit, alpha = self.model_(
                torch.tensor(q_ids),
                torch.tensor(q_mask),
                torch.tensor(c_ids),
                torch.tensor(c_mask),
                return_attention=True,
            )
        if alpha is None:
            return float(torch.sigmoid(logit)), []
        weights = alpha[0].numpy()
        tokens = _tokens(context)[: self.max_context_len]
        order = np.argsort(weights)[::-1][:top]
        return float(torch.sigmoid(logit)), [
            (tokens[i], float(weights[i])) for i in order if i < len(tokens)
        ]

    def describe(self) -> str:
        return (
            f"BiGRU({self.hidden_dim}) question + BiGRU({self.hidden_dim}) context"
            f"{' + dot-product attention' if self.use_attention else ' (max-pooled, no attention)'}"
            f", |V|={len(self.vocab)}, d={self.embedding_dim}, seed={self.seed}"
        )


def _pad(seqs: Sequence[Sequence[int]]) -> Tuple[np.ndarray, np.ndarray]:
    """Pad with the padding id (0 here) and return the matching boolean mask."""
    width = max(len(s) for s in seqs)
    ids = np.zeros((len(seqs), width), dtype=np.int64)
    mask = np.zeros((len(seqs), width), dtype=bool)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = s
        mask[i, : len(s)] = True
    return ids, mask
