#!/usr/bin/env python3
"""Reconstruct and freeze the CPU core models before challenge inference.

Only train/validation are read. Existing results are preserved. Reconstructed
models have their own validation scores, thresholds and environment record.
"""
import json
import hashlib
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src import config, data, spans, span_models, baselines, evaluation
from src import answerability_models as am
from scripts.run_week37 import tune_abstention
from scripts.run_week39 import split_dev


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    out = config.PROJECT_ROOT/'models/challenge_cpu'
    out.mkdir(parents=True,exist_ok=True)
    splits = data.load_project_splits()
    train,valid = splits['train'],splits['validation']
    records=[]
    if not (out/'rule.joblib').exists():
        rule=baselines.CrossLingualAnchorRule().fit(train)
        joblib.dump(rule,out/'rule.joblib')
    if not (out/'span.joblib').exists():
        print('Fitting the original Week 37 labeller (first seed).',flush=True)
        start=time.perf_counter()
        enc_train,enc_valid=spans.encode_split(train),spans.encode_split(valid)
        model=span_models.QuestionConditionedLabeller(beam_size=4,seed=config.SEED).fit(enc_train)
        lp=model.token_log_probs(enc_valid)
        base=model.predict(enc_valid,decoding='beam',log_probs=lp)
        cut,em,_=tune_abstention(model,enc_valid,base,lp)
        gated=span_models.apply_abstention(base,model.no_answer_scores(enc_valid,base,lp),cut)
        metrics=evaluation.span_metrics(enc_valid,gated)
        joblib.dump(model,out/'span.joblib')
        (out/'span_decision.json').write_text(json.dumps(dict(seed=config.SEED,threshold=cut,
                    selected_on='validation exact character-span match',validation=metrics,
                    train_seconds=time.perf_counter()-start,n_train=len(train)),indent=2)+'\n')
        print('Span model frozen:',metrics,flush=True)
        del enc_train,enc_valid,model,lp,base
    if not (out/'rnn.pt').exists():
        print('Reconstructing the Week 39 BiGRU with its existing chosen settings.',flush=True)
        fit,dev=split_dev(train)
        original=json.loads((config.RESULTS_DIR/'environment_week39.json').read_text())['recurrent']
        keys=['hidden_dim','embedding_dim','dropout','lr','epochs','max_context_len']
        cfg={k:original[k] for k in keys}
        cfg['seed']=config.SEED
        vocab=am.build_vocabulary(fit)
        model=am.RecurrentAnswerabilityClassifier(vocab,**cfg).fit(fit,dev)
        p=model.predict_proba_answerable(valid)
        cut=evaluation.tune_threshold(p,valid.answerable,True)['threshold']
        metrics=evaluation.answerability_metrics(valid.answerable,p>=cut)
        torch.save(model.model_.state_dict(),out/'rnn.pt')
        (out/'rnn_config.json').write_text(json.dumps(dict(config=cfg,vocab=vocab,threshold=cut,
                    best_epoch=model.best_epoch_,history=model.history_,validation=metrics,
                    train_seconds=model.train_seconds_,n_train=len(fit)),indent=2)+'\n')
        print('BiGRU frozen:',metrics,flush=True)
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),seed=config.SEED,
                  policy='reconstructed core configurations; thresholds selected on validation only',
                  challenge_accessed=False,python=sys.version,platform=platform.platform(),
                  torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,
                  files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='freeze.json'})
    (out/'freeze.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('CPU models and thresholds frozen without reading challenge data.',flush=True)


if __name__=='__main__':
    main()
