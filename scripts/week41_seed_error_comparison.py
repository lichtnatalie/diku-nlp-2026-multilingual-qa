#!/usr/bin/env python3
"""Offline comparison of persistent and exactly-one-seed QA errors.

Run from the project root: python3 scripts/week41_seed_error_comparison.py
Uses only the standard library and saved predictions; does not train models.
Original taxonomy judgements are retained. OpenAI Codex rechecked the 19 new
judgements against the supplied passages and the shared precedence rule.
"""
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
SEED = 20260901
CATEGORIES = ('polar question', 'unrelated passage', 'topic present, fact missing',
              'questionable gold label', 'predicate paraphrase', 'answer type mismatch',
              'right type, wrong referent', 'span boundary')


def read(name):
    with (RESULTS / name).open(newline='') as f:
        return list(csv.DictReader(f))


def classify(r):
    a, b = r['answerable'] == 'True', r['answered'] == 'True'
    if a and not b:
        return 'wrongly abstains'
    if not a and b:
        return 'wrongly answers'
    if a and b and int(r['em']) == 0:
        return 'wrong span'
    return 'correct'


def relation(r):
    if r['error_type'] != 'wrong span':
        return ''
    gs, ps = int(r['gold_start']), int(r['pred_start'])
    ge, pe = gs + len(r['gold_answer']), ps + len(r['pred_text'])
    if ps <= gs and pe >= ge:
        return 'pred contains gold'
    if ps >= gs and pe <= ge:
        return 'gold contains pred'
    if pe <= gs or ps >= ge:
        return 'disjoint'
    return 'partial overlap'


def stratum(r):
    geometry = ('disjoint' if r['span_relation'] == 'disjoint' else 'overlap') if r['error_type'] == 'wrong span' else ''
    return r['error_type'], r['lang'], geometry


def write(name, rows):
    assert rows
    with (RESULTS / (name + '.csv')).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    qa = read('66_week40_validation_predictions.csv')
    groups = defaultdict(list)
    for r in qa:
        assert r['answerable'] in ('True', 'False') and r['answered'] in ('True', 'False')
        r['error_type'] = classify(r)
        r['span_relation'] = relation(r)
        groups[r['example_id']].append(r)
    assert len(qa) == 3465 and len(groups) == 1155
    for group in groups.values():
        assert {r['seed'] for r in group} == {'20260901', '20260902', '20260903'}
        for field in ('lang', 'question', 'context', 'gold_start', 'gold_answer', 'answerable'):
            assert len({r[field] for r in group}) == 1, field
    multiplicity = Counter(sum(r['error_type'] != 'correct' for r in g) for g in groups.values())
    persistent = [g[0] for g in groups.values() if len({r['error_type'] for r in g}) == 1 and g[0]['error_type'] != 'correct']
    single = [r for g in groups.values() if sum(r['error_type'] != 'correct' for r in g) == 1 for r in g if r['error_type'] != 'correct']
    original = read('73_week41_qa_error_sample.csv')
    original_tax = {r['example_id']: r for r in read('74_week41_qa_error_taxonomy.csv')}
    new_tax = json.loads((RESULTS / '79_week41_single_seed_judgements.json').read_text())['judgements']
    rng = random.Random(SEED)
    pairs, strata = [], []
    for key in sorted({stratum(r) for r in original}):
        a = sorted([r for r in original if stratum(r) == key], key=lambda r: r['example_id'])
        b = sorted([r for r in single if stratum(r) == key], key=lambda r: r['example_id'])
        n = min(len(a), len(b))
        strata.append(dict(error_type=key[0], lang=key[1], geometry=key[2], original_available=len(a), single_available=len(b), matched_per_group=n))
        for ar, br in zip(rng.sample(a, n), rng.sample(b, n)):
            ar = next(r for r in groups[ar['example_id']] if r['seed'] == str(SEED))
            assert all(r['error_type'] == ar['error_type'] for r in groups[ar['example_id']])
            pairs.append((ar, br))
    sample = []
    for pair_id, pair in enumerate(pairs, 1):
        assert stratum(pair[0]) == stratum(pair[1])
        for label, r in zip(('persistent', 'one_seed'), pair):
            t = original_tax[r['example_id']] if label == 'persistent' else new_tax[r['example_id']]
            assert t['category'] in CATEGORIES
            sample.append(dict(pair_id=pair_id, group=label, example_id=r['example_id'], seed=r['seed'],
                               lang=r['lang'], error_type=r['error_type'], span_relation=r['span_relation'],
                               category=t['category'], question_gloss=t['question_gloss'],
                               secondary=t.get('secondary', ''), note=t['note'],
                               judgement_status='original author-reviewed taxonomy' if label == 'persistent' else 'AI-assisted judgement rechecked by OpenAI Codex on 2026-10-21',
                               question=r['question'], gold_answer=r['gold_answer'], pred_text=r['pred_text'],
                               span_text=r['span_text'], context=r['context'],
                               all_seed_outputs=json.dumps([{k:x[k] for k in ('seed','error_type','pred_text','no_answer_score','threshold')} for x in groups[r['example_id']]], ensure_ascii=False)))
    assert len(pairs) == 19 and len({r['example_id'] for r in sample}) == 38
    assert {r['example_id'] for r in sample if r['group'] == 'one_seed'} == set(new_tax)
    # A questionable benchmark label is a property of the pair, regardless of
    # which model made the error. Enforce precedence across all report samples.
    categories_by_id = defaultdict(set)
    for r in read('50_week39_error_taxonomy.csv') + list(original_tax.values()) + sample:
        categories_by_id[r['example_id']].add(r['category'])
    for example_id, labels in categories_by_id.items():
        assert 'questionable gold label' not in labels or labels == {'questionable gold label'}, (
            f'Inconsistent questionable-label precedence for {example_id}: {labels}')
    counts = Counter((r['group'], r['category']) for r in sample)
    categories = [dict(category=c, persistent=counts['persistent',c], one_seed=counts['one_seed',c]) for c in CATEGORIES]
    assert sum(r['persistent'] for r in categories) == sum(r['one_seed'] for r in categories) == 19
    write('78_week41_seed_comparison_sample', sample)
    write('80_week41_seed_comparison_strata', strata)
    write('81_week41_seed_comparison_categories', categories)
    inventory = [dict(n_error_seeds=n, examples=multiplicity[n]) for n in range(4)]
    write('82_week41_error_multiplicity', inventory)
    manifest = dict(seed=SEED, python_random_algorithm='random.Random.sample on sorted IDs and strata',
                    inputs={name:hashlib.sha256((RESULTS/name).read_bytes()).hexdigest() for name in ('66_week40_validation_predictions.csv','73_week41_qa_error_sample.csv','74_week41_qa_error_taxonomy.csv','50_week39_error_taxonomy.csv','79_week41_single_seed_judgements.json')},
                    examples=len(groups), multiplicity=dict(sorted(multiplicity.items())), same_type_all_three=len(persistent),
                    all_three_different_types=multiplicity[3]-len(persistent), pairs=len(pairs),
                    single_seed_sample_by_seed=dict(Counter(r['seed'] for r in sample if r['group']=='one_seed')))
    (RESULTS/'83_week41_seed_comparison_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    labels = {'polar question':'yes/no or alternative question'}
    lines = [r'\begin{table}[t]',r'\centering',r'\footnotesize',r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{lrr}',r'\toprule',r'Category & All three & One seed \\',r'\midrule']
    for r in categories:
        lines.append(f"{labels.get(r['category'],r['category'])} & {r['persistent']} & {r['one_seed']} " + r'\\')
    lines += [r'\midrule',r'Total & 19 & 19 \\',r'\bottomrule',r'\end{tabular}',r'\caption{Matched exploratory sample of integrated-QA errors: 19 per group, matched by language, error type and overlap/disjoint status for wrong spans. Counts describe the sample, not prevalence.}',r'\label{tab:seederrors}',r'\end{table}']
    (ROOT.parent/'report/generated/tab_seed_error_comparison.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps(manifest,indent=2))
    print('\nCATEGORY COUNTS')
    for r in categories:
        print(r)


if __name__ == '__main__':
    main()
