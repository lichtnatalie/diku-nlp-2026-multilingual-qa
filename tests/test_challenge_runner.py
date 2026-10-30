"""Offline end-to-end QA checkpoint loading, freezing and result merging.

The tiny random model is used ONLY in a temporary test directory. These
outputs never enter the project's experimental results.
"""
import hashlib
import json
import sys
from pathlib import Path
import pandas as pd
from src import config
from src.qa_model import QAConfig,tiny_test_model
from scripts import evaluate_challenge,merge_challenge_results


def test_saved_checkpoint_runner_and_cpu_result_merge(tmp_path,monkeypatch):
    root=Path(__file__).resolve().parents[1]
    records=json.loads((root/'challenge.json').read_text())
    results=tmp_path/'results'
    results.mkdir()
    challenge=tmp_path/'challenge.json'
    challenge.write_text(json.dumps(records,ensure_ascii=False))
    definition=dict(challenge_sha256=hashlib.sha256(challenge.read_bytes()).hexdigest())
    (results/'97_week43_challenge_definition.json').write_text(json.dumps(definition))
    frozen=dict(policies={str(config.SEED):{
        'global':dict(global_threshold=0.0,by_language={}),
        'language_specific':dict(global_threshold=0.0,by_language={'ar':0.2,'ko':-0.2,'te':-0.5})}})
    (results/'94_week43_frozen_policies.json').write_text(json.dumps(frozen))
    tokenizer,model=tiny_test_model([r['context'] for r in records]+[r['question'] for r in records],vocab_size=400)
    checkpoint=tmp_path/'tiny_test_only'
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    cfg=QAConfig(model_name='TEST ONLY',max_length=96,stride=16,fp16=False)
    (checkpoint/'decision_rule.json').write_text(json.dumps(dict(seed=config.SEED,threshold=0.0,config=cfg.to_dict())))
    monkeypatch.setattr(config,'PROJECT_ROOT',tmp_path)
    monkeypatch.setattr(config,'RESULTS_DIR',results)
    monkeypatch.setattr(sys,'argv',['evaluate_challenge','--qa-only','--qa-checkpoint',str(checkpoint)])
    evaluate_challenge.main()
    p=pd.read_csv(results/'challenge/predictions_qa.csv')
    assert len(p)==20 and set(p.model)=={'integrated QA','language thresholds'}
    provenance=json.loads((results/'challenge/provenance_qa.json').read_text())
    assert provenance['threshold_tuning_performed'] is False
    # Controlled synthetic CPU outputs exercise the merger, not model quality.
    original=pd.DataFrame(records)
    originals=[]
    for name in ['anchor rule','BiGRU','labeller']:
        g=original.copy()
        g['model']=name
        g['example_id']=g.id
        g['answered']=g.answerable
        g['pred_text']=g.answer
        g['gold_answer']=g.answer
        g['has_span_output']=name=='labeller'
        g['em']=1.0
        g['f1']=1.0
        originals.append(g)
    out=results/'challenge'
    pd.concat(originals).to_csv(out/'predictions_cpu_interim.csv',index=False)
    cpu=dict(provenance,status='INCOMPLETE: CPU models only')
    (out/'provenance_cpu_interim.json').write_text(json.dumps(cpu))
    merge_challenge_results.main()
    merged=pd.read_csv(out/'predictions.csv')
    assert len(merged)==50 and not merged.duplicated(['model','id']).any()
    assert len(pd.read_csv(out/'paired_results.csv'))==25
