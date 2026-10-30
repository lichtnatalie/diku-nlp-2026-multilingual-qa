"""Tests for decision boundaries, held-out isolation and paired scoring."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.abstention import AbstentionPolicy
from src.challenge_evaluation import paired_results


def raw(scores,langs=None,labels=None):
    n=len(scores)
    return pd.DataFrame(dict(no_answer_score=scores,lang=langs or ['ar']*n,
                             answerable=labels or [False]*n,span_text=['answer']*n,span_start=[2]*n))


def test_threshold_equality_abstains_and_unseen_language_has_fixed_fallback():
    policy=AbstentionPolicy(0.0,{'ar':2.0})
    pred=policy.apply(raw([2,1,1,-1],['ar','ar','xx','xx']))
    assert pred.answered.tolist()==[False,True,False,True]
    assert pred.pred_text.tolist()==['','answer','','answer']
    assert pred.pred_start.tolist()==[-1,2,-1,2]


def test_fitting_does_not_read_test_labels_and_application_does_not_use_gold():
    calibration=raw([-4,-3,3,4],labels=[True,True,False,False])
    policy=AbstentionPolicy.fit(calibration,True)
    test=raw([-1,1],labels=[True,False])
    before=policy.apply(test).answered.to_numpy()
    test['answerable']=~test.answerable
    test['answer']=['wrong gold','different gold']
    assert np.array_equal(before,policy.apply(test).answered)
    with pytest.raises(ValueError):
        AbstentionPolicy.fit(calibration.iloc[:0])


def test_grouped_folds_keep_repeated_questions_together_and_cover_each_example_once():
    path=Path(__file__).resolve().parents[1]/'results/96_week43_fold_assignments.csv'
    if not path.exists():
        pytest.skip('run the extension to check the actual grouped folds')
    f=pd.read_csv(path)
    assert len(f)==1155 and f.example_id.is_unique
    assert f.groupby(['lang','question']).fold.nunique().max()==1
    assert set(f.fold)==set(range(5))


def test_pair_success_requires_both_items_not_average_accuracy():
    p=pd.DataFrame(dict(model=['m']*4,pair_id=['one','one','two','two'],lang=['ar']*4,
                        answerable=[True,False,True,False],answered=[True,True,True,False],
                        em=[1,0,0,1],pred_text=['a','a','wrong',''],has_span_output=[True]*4))
    r=paired_results(p)
    assert r.both_answerability_correct.tolist()==[False,True]
    assert r.both_exact.tolist()==[False,False]
    with pytest.raises(ValueError):
        paired_results(p.iloc[:1])
