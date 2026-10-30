#!/usr/bin/env python3
"""Run the recorded Week 43 hypothesis from saved QA predictions.

No training and no challenge access. Run: python3 -m scripts.run_week43
"""
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, evaluation, reporting
from src.abstention import AbstentionPolicy


def main():
    root = config.PROJECT_ROOT
    protocol = root / 'EXTENSION-PROTOCOL.md'
    if not protocol.exists():
        raise SystemExit('Record the extension hypothesis before running it.')
    source = config.RESULTS_DIR / '66_week40_validation_predictions.csv'
    raw = pd.read_csv(source, keep_default_na=False)
    for col in ('answerable','answered'):
        raw[col] = raw[col].astype(str).eq('True')
    seeds = sorted(raw.seed.unique())
    assert seeds == list(config.SEEDS[:3])
    reference = raw[raw.seed.eq(seeds[0])].sort_values('example_id').reset_index(drop=True)
    assert len(reference) == 1155 and reference.example_id.is_unique
    folds = np.full(len(reference), -1, dtype=int)
    groups = reference.lang.astype(str) + '\x00' + reference.question.astype(str)
    strata = reference.lang.astype(str) + ':' + reference.answerable.astype(int).astype(str)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    for fold, (cal_idx,test_idx) in enumerate(splitter.split(reference,strata,groups)):
        assert set(groups.iloc[cal_idx]).isdisjoint(set(groups.iloc[test_idx]))
        folds[test_idx] = fold
    assert (folds >= 0).all()
    metrics, all_oof, thresholds, policies = [], [], [], {}
    for seed in seeds:
        frame = raw[raw.seed.eq(seed)].sort_values('example_id').reset_index(drop=True)
        for col in ('example_id','lang','question','answerable','gold_answer','context'):
            assert frame[col].equals(reference[col]), col
        seed_policies = {}
        for name, per_lang in [('global',False),('language_specific',True)]:
            parts = []
            for fold in range(5):
                cal, test = frame.loc[folds != fold].copy(), frame.loc[folds == fold].copy()
                assert set(cal.example_id).isdisjoint(test.example_id)
                policy = AbstentionPolicy.fit(cal,per_lang)
                decided = policy.apply(test)
                decided['fold'] = fold
                parts.append(decided)
                thresholds.append(dict(seed=int(seed),method=name,fold=fold,n_calibration=len(cal),
                                       n_test=len(test),**policy.to_dict()))
            oof = pd.concat(parts).sort_values('example_id').reset_index(drop=True)
            assert oof.example_id.equals(frame.example_id)
            report = evaluation.qa_report(oof,name)
            report['seed'] = int(seed)
            metrics.append(report)
            oof['method'] = name
            all_oof.append(evaluation.qa_example_scores(oof))
            fitted = AbstentionPolicy.fit(frame,per_lang)
            seed_policies[name] = fitted.to_dict()
            if not per_lang:
                assert abs(fitted.global_threshold - float(frame.threshold.iloc[0])) < 1e-9
        policies[str(seed)] = seed_policies
    metrics = pd.concat(metrics,ignore_index=True)
    reporting.save_table(metrics,'90_week43_oof_metrics')
    summary = metrics.groupby(['model','subset'],sort=False).agg(
        n=('n','first'),macro_f1=('macro_f1','mean'),macro_f1_sd=('macro_f1','std'),
        exact_match=('exact_match','mean'),exact_match_sd=('exact_match','std'),
        f1=('f1','mean'),f1_sd=('f1','std'),unanswerable_recall=('unanswerable_recall','mean')).reset_index()
    reporting.save_table(summary,'91_week43_oof_summary')
    threshold_df = pd.DataFrame(thresholds)
    threshold_df['by_language'] = threshold_df.by_language.map(lambda x:json.dumps(x,sort_keys=True))
    reporting.save_table(threshold_df,'92_week43_calibration_thresholds')
    pd.concat(all_oof,ignore_index=True).to_csv(config.RESULTS_DIR/'93_week43_oof_predictions.csv',index=False)
    deltas = []
    for (seed,subset), group in metrics.groupby(['seed','subset'],sort=False):
        g = group.set_index('model')
        d = dict(seed=int(seed),subset=subset,n=int(g.loc['global','n']))
        for metric in ['macro_f1','exact_match','f1','unanswerable_recall']:
            d[metric+'_delta'] = float(g.loc['language_specific',metric]-g.loc['global',metric])
        deltas.append(d)
    reporting.save_table(pd.DataFrame(deltas),'95_week43_paired_deltas')
    names = ['66_week40_validation_predictions.csv']
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),hypothesis_record='EXTENSION-PROTOCOL.md',
                    protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
                    prediction_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    answer_iff='no_answer_score < threshold',fallback_language='global threshold',
                    frozen_before_challenge_inference=True,seeds=[int(x) for x in seeds],
                    challenge_seed=int(config.SEED),folds=5,grouping='exact question and language',
                    policies=policies,python=sys.version,platform=platform.platform())
    (config.RESULTS_DIR/'94_week43_frozen_policies.json').write_text(json.dumps(manifest,indent=2)+'\n')
    pd.DataFrame(dict(example_id=reference.example_id,fold=folds,lang=reference.lang,
                      answerable=reference.answerable,question=reference.question)).to_csv(
                          config.RESULTS_DIR/'96_week43_fold_assignments.csv',index=False)
    print(summary.to_string(index=False))
    print('\nPaired overall differences (language-specific minus global):')
    print(pd.DataFrame(deltas).query("subset == 'overall'").to_string(index=False))
    print('\nPolicies frozen. No challenge data were read.')


if __name__ == '__main__':
    main()
