"""Week 43: global versus language-specific validation-only abstention.

The encoder, span head and raw scores stay fixed. Unknown languages use the
global threshold. Both methods use the shared tuner, objective and tie rule.
"""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from . import evaluation


@dataclass
class AbstentionPolicy:
    global_threshold: float
    by_language: dict[str, float]

    @classmethod
    def fit(cls, calibration: pd.DataFrame, language_specific: bool = False):
        if not len(calibration):
            raise ValueError('empty calibration split')
        scores = calibration.no_answer_score.to_numpy(float)
        if not np.isfinite(scores).all():
            raise ValueError('calibration scores must be finite')
        cut = evaluation.tune_threshold(scores, calibration.answerable, False)['threshold']
        local = {}
        if language_specific:
            for lang, group in calibration.groupby('lang', sort=True):
                local[str(lang)] = float(evaluation.tune_threshold(
                    group.no_answer_score, group.answerable, False)['threshold'])
        return cls(float(cut), local)

    def apply(self, raw: pd.DataFrame):
        out = raw.copy()
        cuts = out.lang.map(self.by_language).fillna(self.global_threshold).to_numpy(float)
        out['threshold'] = cuts
        out['answered'] = out.no_answer_score.to_numpy(float) < cuts
        out['pred_text'] = np.where(out.answered, out.span_text, '')
        out['pred_start'] = np.where(out.answered, out.span_start, -1)
        return out

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        return cls(float(values['global_threshold']), {k: float(v) for k,v in values['by_language'].items()})
