#!/usr/bin/env python3
"""Evaluate frozen models on challenge.json, without fitting or tuning.

Full run: python -m scripts.evaluate_challenge --qa-checkpoint PATH
CPU-only interim run: python -m scripts.evaluate_challenge --cpu-only
The latter is explicitly incomplete and cannot produce a final QA result.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from datetime import datetime,timezone
import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src import config,evaluation,spans,span_models
from src import answerability_models as am
from src.abstention import AbstentionPolicy
from src.challenge_evaluation import paired_results
from student_starter.validate_challenge import validate_records


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qa-checkpoint',type=Path)
    parser.add_argument('--cpu-only',action='store_true')
    parser.add_argument('--qa-only',action='store_true',help='Colab: evaluate only saved QA and its extension')
    args=parser.parse_args()
    if not args.cpu_only and not args.qa_checkpoint:
        raise SystemExit('A saved fine-tuned QA checkpoint is required. --cpu-only creates labelled interim results only.')
    if args.cpu_only and args.qa_only:
        raise SystemExit('--cpu-only and --qa-only are mutually exclusive')
    torch.set_num_threads(2)
    root=config.PROJECT_ROOT
    out=config.RESULTS_DIR/'challenge'
    out.mkdir(exist_ok=True)
    frozen=json.loads((config.RESULTS_DIR/'94_week43_frozen_policies.json').read_text())
    policies=frozen['policies'][str(config.SEED)]
    cpu=root/'models/challenge_cpu'
    if not args.qa_only:
        freeze=json.loads((cpu/'freeze.json').read_text())
        for name,expected in freeze['files'].items():
            assert digest(cpu/name)==expected,f'Frozen CPU file changed: {name}'
    decision=None
    recovery=None
    if args.qa_checkpoint:
        decision=json.loads((args.qa_checkpoint/'decision_rule.json').read_text())
        assert decision['seed']==config.SEED,'Challenge comparison uses the prespecified first seed'
        expected=policies['global']['global_threshold']
        if abs(decision['threshold']-expected)>=1e-9 or not np.isfinite(decision['threshold']):
            raise ValueError(
                f"Checkpoint threshold {decision['threshold']} differs from the recorded "
                f"threshold {expected}. Saved training epochs: {decision['config'].get('epochs')}. "
                "Recover the matching original checkpoint. Do not edit either threshold."
            )
        recovery_path=args.qa_checkpoint/'checkpoint_recovery.json'
        if recovery_path.is_file():
            recovery=json.loads(recovery_path.read_text())
            for name,record in recovery['files'].items():
                if Path(name).name!=name:
                    raise ValueError('Invalid recovery manifest filename')
                assert digest(args.qa_checkpoint/name)==record['sha256'],f'Recovered file changed: {name}'
    records=json.loads((root/'challenge.json').read_text())
    errors=validate_records(records)
    if errors:
        raise SystemExit('\n'.join(errors))
    definition=json.loads((config.RESULTS_DIR/'97_week43_challenge_definition.json').read_text())
    assert digest(root/'challenge.json')==definition['challenge_sha256'],'Challenge changed after it was recorded'
    df=pd.DataFrame(records)
    df['example_id']=df.id
    rows=[]

    def collect(name,decided,has_span):
        decided=decided.copy()
        for col in ['pair_id','id','lang','question','context','answerable']:
            decided[col]=df[col].to_numpy()
        decided['model']=name
        decided['has_span_output']=has_span
        if has_span:
            decided=evaluation.qa_example_scores(decided)
        else:
            decided['em']=np.nan
            decided['f1']=np.nan
        rows.append(decided)

    if not args.qa_only:
        rule=joblib.load(cpu/'rule.joblib')
        collect('anchor rule',pd.DataFrame(dict(answered=rule.predict(df),pred_text='',gold_answer=df.answer)),False)
        meta=json.loads((cpu/'rnn_config.json').read_text())
        rnn=am.RecurrentAnswerabilityClassifier(meta['vocab'],**meta['config'])
        rnn.model_=rnn._build()
        rnn.model_.load_state_dict(torch.load(cpu/'rnn.pt',map_location='cpu',weights_only=True))
        p=rnn.predict_proba_answerable(df)
        collect('BiGRU',pd.DataFrame(dict(answered=p>=meta['threshold'],p_answerable=p,
                                        threshold=meta['threshold'],pred_text='',gold_answer=df.answer)),False)
        model=joblib.load(cpu/'span.joblib')
        encoded=spans.encode_split(df)
        lp=model.token_log_probs(encoded)
        labels=model.predict(encoded,decoding='beam',log_probs=lp)
        scores=model.no_answer_scores(encoded,labels,lp)
        threshold=json.loads((cpu/'span_decision.json').read_text())['threshold']
        gated=span_models.apply_abstention(labels,scores,threshold)
        span_rows=[]
        for e,pred in zip(encoded,gated):
            start,text=spans.bio_to_char_span(e['context'],e['offsets'],pred)
            span_rows.append(dict(answered=bool(text),pred_start=-1 if start is None else start,
                                  pred_text=text,gold_answer=e['answer'],answerable=e['answerable']))
        collect('labeller',pd.DataFrame(span_rows),True)
        evaluation.span_report(encoded,gated,'labeller').to_csv(out/'span_metrics.csv',index=False)
    if args.qa_checkpoint:
        from transformers import AutoTokenizer,AutoModelForQuestionAnswering
        from src.qa_model import QAConfig,IntegratedQAModel
        tokenizer=AutoTokenizer.from_pretrained(args.qa_checkpoint,local_files_only=True)
        model=AutoModelForQuestionAnswering.from_pretrained(args.qa_checkpoint,local_files_only=True)
        cfg=QAConfig(**decision['config'])
        cfg.eval_batch_size=2
        qa=IntegratedQAModel(cfg,tokenizer=tokenizer,model=model,
                             device='cuda' if torch.cuda.is_available() else 'cpu')
        raw=qa.predict_scores(df)
        raw.to_csv(out/'qa_raw_scores.csv',index=False)
        for name,key in [('integrated QA','global'),('language thresholds','language_specific')]:
            policy=AbstentionPolicy.from_dict(policies[key])
            collect(name,policy.apply(raw),True)
    pred=pd.concat(rows,ignore_index=True)
    suffix='_cpu_interim' if args.cpu_only else '_qa' if args.qa_only else ''
    pred.to_csv(out/f'predictions{suffix}.csv',index=False)
    paired=paired_results(pred)
    paired.to_csv(out/f'paired_results{suffix}.csv',index=False)
    reports=[]
    for model,g in pred.groupby('model',sort=False):
        if g.has_span_output.all():
            report=evaluation.qa_report(g,model)
        else:
            report=evaluation.answerability_report(g,g.answered,model)
        reports.append(report)
    pd.concat(reports,ignore_index=True).to_csv(out/f'metrics{suffix}.csv',index=False)
    provenance=dict(created_utc=datetime.now(timezone.utc).isoformat(),seed=config.SEED,
                    status='INCOMPLETE: CPU models only' if args.cpu_only else 'QA checkpoint only; merge with CPU results' if args.qa_only else 'all required models evaluated',
                    challenge_sha256=digest(root/'challenge.json'),extension_policies_sha256=digest(config.RESULTS_DIR/'94_week43_frozen_policies.json'),
                    model_names=list(pred.model.unique()),threshold_tuning_performed=False,
                    qa_checkpoint=str(args.qa_checkpoint) if args.qa_checkpoint else None,
                    qa_saved_decision_rule=decision)
    if recovery is not None:
        provenance['qa_checkpoint_recovery']=recovery
    (out/f'provenance{suffix}.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(pd.concat(reports,ignore_index=True).query("subset=='overall'").to_string(index=False))
    print(paired.to_string(index=False))


if __name__=='__main__':
    main()
