"""Gold-subset reporting must preserve class precision and span semantics."""
import pandas as pd
import pytest
from src.challenge_evaluation import subset_metrics


def fixtures():
    context='Mercury and Mercury.'
    records=[dict(id=f'{lang}-{part}',pair_id=lang,lang=lang,
                  question=part,context=context,answerable=part=='a',
                  answer='Mercury' if part=='a' else '',
                  answer_start=0 if part=='a' else -1)
             for lang in ['ar','ko'] for part in ['a','u']]
    return records


def test_class_precision_retains_false_positives_from_other_gold_subset():
    records=fixtures()
    p=pd.DataFrame(records).assign(model='rule',answered=True,
                                   has_span_output=False)
    p['gold_answer']=p.answer
    result=subset_metrics(p,records)
    a=result.query("subset=='overall' and gold_subset=='answerable'").iloc[0]
    u=result.query("subset=='overall' and gold_subset=='unanswerable'").iloc[0]
    assert a.n==u.n==2
    assert a.class_precision==0.5
    assert a.class_recall==1
    assert a.class_f1==pytest.approx(2/3)
    assert u.class_precision==u.class_recall==u.class_f1==0


def test_position_exact_bio_and_normalized_qa_metrics_stay_distinct():
    records=fixtures()
    p=pd.DataFrame(records).assign(model='labeller',answered=True,
                                   has_span_output=True,pred_start='12.0',
                                   pred_text='Mercury')
    p['gold_answer']=p.answer
    # One unsupported question is declined. Both supported predictions select
    # the second occurrence of the correct word, at the wrong position.
    p.loc[p.id.eq('ko-u'),['answered','pred_start','pred_text']]=[False,'-1.0','']
    result=subset_metrics(p,records)
    a=result.query("subset=='overall' and gold_subset=='answerable'").iloc[0]
    u=result.query("subset=='overall' and gold_subset=='unanswerable'").iloc[0]
    assert a.character_em==a.bio_token_f1==0
    assert a.squad_em==a.squad_f1==1
    assert u.character_em==u.squad_em==u.squad_f1==0.5
    assert u.bio_token_f1==0
    assert a.token_fp==a.token_fn==2
    assert u.token_fp==1
    assert result.groupby('subset').n.sum().to_dict()=={'overall':4,'ar':2,'ko':2}
    broken=p.copy()
    broken.loc[0,'pred_start']=13
    broken.loc[0,'pred_text']='ercury'
    with pytest.raises(ValueError,match='token-aligned'):
        subset_metrics(broken,records)


def test_saved_outputs_cannot_silently_change_the_frozen_records():
    records=fixtures()
    p=pd.DataFrame(records).assign(model='rule',answered=True,has_span_output=False)
    p['gold_answer']=p.answer
    with pytest.raises(ValueError,match='Duplicate'):
        subset_metrics(pd.concat([p,p.iloc[:1]]),records)
    with pytest.raises(ValueError,match='cover'):
        subset_metrics(p.iloc[:2],records)
    p.loc[0,'context']='different passage'
    with pytest.raises(ValueError,match='changed context'):
        subset_metrics(p,records)
