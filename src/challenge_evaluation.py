"""Paired challenge scoring, distinct from aggregate row accuracy."""
import pandas as pd
from . import evaluation, spans


def subset_metrics(predictions: pd.DataFrame, records: list[dict]) -> pd.DataFrame:
    """Score frozen outputs by gold answerability, pooled and per language.

    Class P/R/F1 is computed on the full population, then assigned to its
    class row. Computing precision within a pure-gold subset would hide
    false positives. Extraction metrics instead score each gold subset.
    No model, decision rule or threshold is loaded or modified.
    """
    reference = {r['id']: r for r in records}
    if len(reference) != len(records) or predictions.duplicated(['model', 'id']).any():
        raise ValueError('Duplicate challenge IDs or model/record outputs')
    for col in ['answerable', 'answered', 'has_span_output']:
        if not pd.api.types.is_bool_dtype(predictions[col]):
            raise ValueError(f'{col} must contain parsed booleans')
    rows = []
    for model, group in predictions.groupby('model', sort=False):
        if set(group.id) != set(reference):
            raise ValueError(f'{model}: outputs do not cover the frozen challenge')
        for row in group.itertuples():
            gold = reference[row.id]
            for key in ['context', 'question', 'lang', 'answerable']:
                if getattr(row, key) != gold[key]:
                    raise ValueError(f'{model}/{row.id}: changed {key}')
            if row.gold_answer != gold['answer']:
                raise ValueError(f'{model}/{row.id}: changed gold answer')
        has_span = bool(group.has_span_output.all())
        if group.has_span_output.any() != has_span:
            raise ValueError(f'{model}: inconsistent span-output availability')
        scopes = [('overall', group)] + [
            (lang, group[group.lang.eq(lang)])
            for lang in dict.fromkeys(r['lang'] for r in records)
        ]
        for scope, population in scopes:
            if population.empty:
                continue
            classes = evaluation.answerability_metrics(
                population.answerable.to_numpy(), population.answered.to_numpy())
            for name, answerable in [('answerable', True), ('unanswerable', False)]:
                sub = population[population.answerable.eq(answerable)]
                if sub.empty:
                    continue
                result = dict(model=model, subset=scope, gold_subset=name, n=len(sub),
                              class_precision=classes[name+'_precision'],
                              class_recall=classes[name+'_recall'],
                              class_f1=classes[name+'_f1'])
                if has_span:
                    scored = evaluation.qa_example_scores(sub)
                    result.update(squad_em=float(scored.em.mean()),
                                  squad_f1=float(scored.f1.mean()))
                if model == 'labeller':
                    encoded, predicted = [], []
                    for row in sub.itertuples():
                        gold = reference[row.id]
                        info = spans.char_span_to_bio(
                            gold['context'], answerable,
                            gold['answer_start'], gold['answer'])
                        info.update(context=gold['context'], answerable=answerable)
                        # Mixed classifier/span CSV columns store offsets as
                        # strings such as "8.0". Accept only integral offsets.
                        saved_start = float(row.pred_start)
                        if not saved_start.is_integer():
                            raise ValueError(f'{model}/{row.id}: nonintegral offset')
                        saved_start = int(saved_start)
                        pred = spans.char_span_to_bio(
                            gold['context'], bool(row.answered),
                            saved_start, row.pred_text)
                        start, text = spans.bio_to_char_span(
                            gold['context'], pred['offsets'], pred['labels'])
                        if (start, text) != ((saved_start, row.pred_text)
                                             if row.answered else (None, '')):
                            raise ValueError(f'{model}/{row.id}: span is not token-aligned')
                        if not row.answered and row.pred_text:
                            raise ValueError(f'{model}/{row.id}: nonempty abstention')
                        encoded.append(info)
                        predicted.append(pred['labels'])
                    span = evaluation.span_metrics(encoded, predicted)
                    result.update(character_em=span['exact_match'],
                                  bio_token_f1=span['token_f1'],
                                  token_tp=span['token_tp'], token_fp=span['token_fp'],
                                  token_fn=span['token_fn'])
                rows.append(result)
    return pd.DataFrame(rows)


def paired_results(predictions: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for (model,pair_id), group in predictions.groupby(['model','pair_id'],sort=False):
        if len(group)!=2 or set(group.answerable)!={False,True}:
            raise ValueError(f'{model}/{pair_id} is not a balanced two-record pair')
        a=group[group.answerable].iloc[0]
        u=group[~group.answerable].iloc[0]
        ac,uc=bool(a.answered),not bool(u.answered)
        span_available=bool(group['has_span_output'].all())
        rows.append(dict(model=model,pair_id=pair_id,lang=a.lang,
                         answerable_correct=ac,unanswerable_correct=uc,
                         both_answerability_correct=ac and uc,
                         answerable_em=float(a.em) if span_available else None,
                         unanswerable_em=float(u.em) if span_available else None,
                         both_exact=bool(a.em and u.em) if span_available else None,
                         answerable_prediction=a.pred_text if span_available else '',
                         unanswerable_prediction=u.pred_text if span_available else ''))
    return pd.DataFrame(rows)
