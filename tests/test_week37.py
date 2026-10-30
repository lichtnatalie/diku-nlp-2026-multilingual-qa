"""Tests for Week 37: character offsets <-> BIO labels, span metrics, models.

The synthetic fixture in make_fixture.py is used rather than the real data,
because these tests have to check FAILURE paths - an off-by-one offset, an
unanswerable row carrying a substantive answer string, a duplicate answer in the
context - and the real data contains none of them. A suite run only against real
data could not tell a working check from one that always returns zero.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import evaluation, span_models, spans  # noqa: E402
from src.tokenisation import unicode_word_tokenise  # noqa: E402
from tests.make_fixture import make_raw  # noqa: E402


# ----------------------------------------------------------- tokenisation ---

def test_context_tokeniser_is_week36_plus_punctuation():
    """The two tokenisers must differ only by the punctuation tokens.

    The context side keeps punctuation so that answer spans ending on a bracket
    can be represented; the question side does not, because dropping combining
    marks would shatter Telugu. Dropping the punctuation tokens has to give back
    Week 36's tokens exactly, or the two halves of the project are counting
    different things.
    """
    text = "Ada Lovelace wrote the first algorithm, in 1843."
    tokens, offsets = spans.tokenise_with_offsets(text)
    words = [t for t in tokens if not (len(t) == 1 and not t.isalnum())]
    assert words == unicode_word_tokenise(text)
    assert len(tokens) == len(offsets)


def test_punctuation_is_its_own_token():
    """One token per punctuation character, so a span can end on any of them."""
    tokens, _ = spans.tokenise_with_offsets("Corporation (BBC), 1922.")
    assert tokens == ["Corporation", "(", "BBC", ")", ",", "1922", "."]


def test_offsets_point_at_their_own_token():
    """Slicing the context with a token's offsets must return that token."""
    text = "పరిపాలన కేంద్రం Hyderabad, 1956."
    tokens, offsets = spans.tokenise_with_offsets(text)
    for token, (start, end) in zip(tokens, offsets):
        assert text[start:end] == token


def test_empty_text_gives_no_tokens():
    """Degenerate input must not raise."""
    assert spans.tokenise_with_offsets("") == ([], [])


# ------------------------------------------------------------- conversion ---

def test_section3_conversion_checks_all_pass():
    """The Section 3 automatic checks are themselves a test."""
    checks = spans.conversion_checks()
    assert checks["passed"].all(), checks[~checks["passed"]].to_string()
    # The brief names four cases. Assert they are present, not merely that the
    # cases that happen to exist pass - otherwise deleting one would "fix" a
    # failure.
    assert int(checks["required_by_brief"].sum()) == 4
    named = set(checks[checks["required_by_brief"]]["check"])
    assert named == {
        "answer at character 0",
        "multi-token answer",
        "punctuation adjacent to the answer",
        "unanswerable example",
    }


def test_unanswerable_is_all_O_even_with_a_substantive_answer_string():
    """The trap Week 36 found: every unanswerable row has a non-empty answer.

    Gating on `answer == ""` instead of on `answerable` would label a span here.
    """
    context = "Nigeria is its largest country by population."
    info = spans.char_span_to_bio(context, answerable=False,
                                  answer_start=-1, answer_text="Nigeria")
    assert set(info["labels"]) == {spans.OUTSIDE}
    assert info["status"] == spans.STATUS_UNANSWERABLE
    assert spans.bio_to_char_span(context, info["offsets"], info["labels"]) == (None, "")


def test_conversion_uses_the_offset_not_a_string_search():
    """With the answer twice in the context, the stored offset must decide."""
    context = "Paris is in France. Paris is large."
    second = context.index("Paris", 1)
    info = spans.char_span_to_bio(context, True, second, "Paris")
    start, text = spans.bio_to_char_span(context, info["offsets"], info["labels"])
    assert (start, text) == (second, "Paris")
    assert start != context.index("Paris")


def test_inconsistent_offset_raises():
    """A stored offset that does not match the stored text is corrupt data."""
    with pytest.raises(ValueError):
        spans.char_span_to_bio("Ada Lovelace wrote it.", True, 0, "Lovelace")


def test_span_outside_context_raises():
    """Guard against an offset past the end of the context."""
    with pytest.raises(ValueError):
        spans.char_span_to_bio("short", True, 3, "much longer answer")


def test_span_ending_in_punctuation_round_trips_exactly():
    """The reason the context tokeniser keeps punctuation.

    Under a tokeniser that discards it there is no token carrying the closing
    bracket, so this span could only ever come back as "Corporation (BBC" and
    every row like it was a permanent exact-match miss.
    """
    context = "The British Broadcasting Corporation (BBC) is a broadcaster."
    answer = "Corporation (BBC)"
    start = context.index(answer)
    info = spans.char_span_to_bio(context, True, start, answer)
    assert info["status"] == spans.STATUS_EXACT
    assert spans.bio_to_char_span(context, info["offsets"], info["labels"]) == (start, answer)
    assert info["gold_char_span"] == (start, answer)


def test_widened_status_for_a_mid_token_span():
    """A gold span cutting a token in half is flagged, not silently accepted."""
    context = "Jeremy Bentham founded utilitarianism."
    info = spans.char_span_to_bio(context, True, 1, "eremy Bentham")
    assert info["status"] == spans.STATUS_WIDENED


# ------------------------------------------------------------ span logic ----

def test_find_spans_promotes_an_orphan_inside_label():
    """We follow lab_2's convention, not the lecture slides' get_spans."""
    assert spans.find_spans([spans.OUTSIDE, spans.INSIDE, spans.OUTSIDE]) == [(1, 2)]
    assert spans.find_spans([spans.BEGIN, spans.INSIDE]) == [(0, 2)]
    assert spans.find_spans([spans.BEGIN, spans.BEGIN]) == [(0, 1), (1, 2)]
    assert spans.find_spans([spans.OUTSIDE, spans.OUTSIDE]) == []


def test_valid_bio_transition_matches_iob2():
    """I-ANS may only continue a span, and may never start a sequence."""
    assert not spans.valid_bio_transition(None, spans.INSIDE)
    assert not spans.valid_bio_transition(spans.OUTSIDE, spans.INSIDE)
    assert spans.valid_bio_transition(spans.BEGIN, spans.INSIDE)
    assert spans.valid_bio_transition(spans.INSIDE, spans.INSIDE)
    assert spans.valid_bio_transition(None, spans.BEGIN)


def test_assert_scheme_rejects_iob1_style_data():
    """The check that would have caught the lab's collapse from 0.78 to 0.011."""
    spans.assert_scheme([[spans.BEGIN, spans.INSIDE, spans.OUTSIDE]])
    with pytest.raises(AssertionError):
        spans.assert_scheme([[spans.OUTSIDE, spans.INSIDE]])


def test_mismatched_lengths_raise():
    """Offsets and labels must describe the same token sequence."""
    with pytest.raises(ValueError):
        spans.bio_to_char_span("abc def", [(0, 3)], [spans.OUTSIDE, spans.OUTSIDE])


# --------------------------------------------------------------- metrics ----

def _encoded_fixture():
    """Encode the fixture's validation split, dropping its deliberately broken row."""
    raw = make_raw()
    df = raw["validation"].copy()
    df["example_id"] = [f"v-{i:03d}" for i in range(len(df))]
    keep = []
    for row in df.itertuples():
        try:
            spans.char_span_to_bio(row.context, bool(row.answerable),
                                   int(row.answer_start) if row.answerable else None,
                                   row.answer if row.answerable else "")
            keep.append(True)
        except ValueError:
            keep.append(False)
    return spans.encode_split(df[keep].reset_index(drop=True))


def test_perfect_prediction_scores_one():
    """Feeding the gold labels back in must score 1.0 on both metrics."""
    enc = _encoded_fixture()
    gold = [e["labels"] for e in enc]
    m = evaluation.span_metrics(enc, gold)
    assert m["token_f1"] == pytest.approx(1.0)
    # Exact match is against the raw gold span, so any trimmed row cannot reach 1.
    statuses = {e["status"] for e in enc}
    if statuses <= {spans.STATUS_EXACT, spans.STATUS_UNANSWERABLE}:
        assert m["exact_match"] == pytest.approx(1.0)


def test_all_O_scores_zero_token_f1_but_hits_every_unanswerable():
    """The always-O trap, stated as a test: two metrics, opposite verdicts."""
    enc = _encoded_fixture()
    allO = [[spans.OUTSIDE] * len(e["tokens"]) for e in enc]
    m = evaluation.span_metrics(enc, allO)
    assert m["token_f1"] == 0.0
    assert m["exact_match_unanswerable"] == pytest.approx(1.0)
    assert m["exact_match_answerable"] == 0.0
    assert m["n_predicted_span"] == 0


def test_unanswerable_rows_can_only_lower_token_precision():
    """Stated in Section 1 and asserted here: they contribute no tp and no fn."""
    enc = [e for e in _encoded_fixture() if not e["answerable"]]
    assert enc, "fixture has no unanswerable rows"
    predict_everything = [[spans.BEGIN] + [spans.INSIDE] * (len(e["tokens"]) - 1) for e in enc]
    m = evaluation.span_metrics(enc, predict_everything)
    assert m["token_tp"] == 0
    assert m["token_fn"] == 0
    assert m["token_fp"] > 0


def test_span_report_has_one_row_per_language_plus_overall():
    """The report layout Section 1 asks for, with N on every row."""
    enc = _encoded_fixture()
    allO = [[spans.OUTSIDE] * len(e["tokens"]) for e in enc]
    table = evaluation.span_report(enc, allO, "test-model")
    assert table.iloc[0]["subset"] == "overall"
    assert (table["n"] > 0).all()
    assert table["model"].nunique() == 1


def test_prediction_length_mismatch_raises():
    """A wrong number of predictions must fail loudly, not silently zip short."""
    enc = _encoded_fixture()
    with pytest.raises(ValueError):
        evaluation.span_metrics(enc, [[spans.OUTSIDE]] * (len(enc) - 1))


# ---------------------------------------------------------------- models ----

def test_empty_baseline_abstains_everywhere():
    """The Section 3 baseline, by definition."""
    enc = _encoded_fixture()
    preds = span_models.EmptySpanBaseline().predict(enc)
    assert all(set(p) <= {spans.OUTSIDE} for p in preds)


def test_overlap_baseline_predicts_a_contiguous_run_or_nothing():
    """Whatever it predicts must be a single contiguous span."""
    enc = _encoded_fixture()
    preds = span_models.LexicalOverlapSpanBaseline().predict(enc)
    for labels in preds:
        found = spans.find_spans(labels)
        assert len(found) <= 1


def test_overlap_baseline_ignores_the_gold_answer():
    """Section 3 forbids using gold text or offsets. Changing them must not
    change the prediction."""
    enc = _encoded_fixture()
    before = span_models.LexicalOverlapSpanBaseline().predict(enc)
    for e in enc:
        e["gold_char_span"] = (0, "TAMPERED")
        e["labels"] = [spans.OUTSIDE] * len(e["tokens"])
    after = span_models.LexicalOverlapSpanBaseline().predict(enc)
    assert before == after


def test_question_features_are_present_and_removable():
    """The ablation switch must actually drop every question-derived feature."""
    qf = span_models.question_features("무엇인가 Seoul 1953", "ko")
    tokens = ["Seoul", "is", "big"]
    full = span_models.token_features(tokens, 0, qf, use_question=True)
    ablated = span_models.token_features(tokens, 0, qf, use_question=False)
    dropped = set(full) - set(ablated)
    assert dropped, "ablation dropped nothing"
    assert all(k.startswith("q_") for k in dropped)
    assert not any(k.startswith("q_") for k in ablated)


def test_question_features_differ_between_questions():
    """The point of Section 3: the same context token must be able to get a
    different feature vector under a different question."""
    tokens = ["Seoul", "is", "big"]
    a = span_models.token_features(tokens, 0, span_models.question_features("어디인가", "ko"))
    b = span_models.token_features(tokens, 0, span_models.question_features("무엇인가", "ko"))
    assert a != b


def test_beam_decoding_never_emits_an_illegal_transition():
    """The constraint must hold on the output, not just in the candidate loop."""
    enc = _encoded_fixture()
    model = span_models.QuestionConditionedLabeller(beam_size=4)
    model.fit(enc)
    for labels in model.predict(enc, decoding="beam"):
        spans.assert_scheme([labels])


def test_constrained_decoding_cannot_emit_two_spans():
    """The task allows at most one answer, so the decoder must allow at most one.

    Checked on both constrained decodings, and on argmax after its post-hoc
    reduction, so the guarantee is the same whichever row of the results table
    the reader is looking at.
    """
    enc = _encoded_fixture()
    model = span_models.QuestionConditionedLabeller(beam_size=4)
    model.fit(enc)
    for decoding in ("argmax", "greedy", "beam"):
        for labels in model.predict(enc, decoding=decoding):
            assert len(spans.find_spans(labels)) <= 1, decoding


def test_the_state_machine_is_exactly_O_star_B_I_star_O_star():
    """The constraint itself, independent of any fitted model.

    Plain IOB2 validity would accept B O B (two spans); this must not.
    """
    def walk(sequence):
        state = span_models._BEFORE
        for label in sequence:
            state = span_models._next_state(state, label)
            if state is None:
                return False
        return True

    O, B, I = spans.OUTSIDE, spans.BEGIN, spans.INSIDE
    assert walk([O, B, I, I, O])
    assert walk([O, O, O])          # abstention is always legal
    assert walk([B, I])
    assert not walk([B, O, B])      # two spans
    assert not walk([O, I])         # I cannot open a span
    assert not walk([B, O, I])


def test_abstention_blanks_predictions_above_the_threshold():
    """A high no-answer score means NO ANSWER - the direction Section 1 wants
    stated."""
    preds = [[spans.BEGIN, spans.INSIDE], [spans.BEGIN, spans.OUTSIDE]]
    scores = [1.0, -1.0]
    gated = span_models.apply_abstention(preds, scores, threshold=0.0)
    assert set(gated[0]) == {spans.OUTSIDE}
    assert gated[1] == preds[1]


# ----------------------------------------------------------------- audit ----

def test_alignment_audit_counts_add_up():
    """The audit must account for every answerable row exactly once."""
    enc = _encoded_fixture()
    table = spans.alignment_audit(enc, "fixture")
    overall = table[table["subset"] == "overall"].iloc[0]
    counted = (overall["exact"] + overall["trimmed"]
               + overall["widened"] + overall["no_word_tokens"])
    assert counted == overall["n_answerable"]
    assert overall["n"] == overall["n_answerable"] + overall["n_unanswerable"]


def test_unanswerable_check_finds_no_spans_in_the_fixture():
    """The seventh Section 3 check, run against the fixture."""
    enc = _encoded_fixture()
    result = spans.unanswerable_check(enc)
    assert result["n_with_non_O_label"] == 0


# ------------------------------------------------------- error taxonomy -----

def test_error_kinds_are_defined_on_positions_not_strings():
    """The extent categories must mean what the report says they mean.

    Context: "St. Peter's Basilica is in Rome. St Mark's is in Venice."
    Gold span is "St. Peter's Basilica" at 0. A predicted "St" at 33 is a
    substring of the gold text but lies elsewhere, so it is a wrong location,
    not a too-narrow span.
    """
    from scripts.run_week37 import error_kind

    gold_start, gold = 0, "St. Peter's Basilica"
    assert error_kind(0, "St. Peter's", gold_start, gold) == "boundary too narrow"
    assert error_kind(4, "Peter's Basilica", gold_start, gold) == "boundary too narrow"
    assert error_kind(33, "St", gold_start, gold) == "wrong location"
    assert error_kind(0, "St. Peter's Basilica is", gold_start, gold) == "boundary too wide"
    # "Basilica is in" starts inside the gold span and ends outside it.
    assert error_kind(12, "Basilica is in", gold_start, gold) == "partial overlap"
    # Same string at a different offset (a repeated answer).
    assert error_kind(40, gold, gold_start, gold) == "right text, wrong occurrence"


def test_error_kinds_partition_the_overlap_cases():
    """Every relation between two non-identical spans gets exactly one kind."""
    from scripts.run_week37 import ANSWERABLE_ERROR_KINDS, error_kind

    context = "abcdefghijklmnopqrstuvwxyz"
    gold_start, gold_end = 8, 14
    gold = context[gold_start:gold_end]
    seen = set()
    for ps in range(len(context)):
        for pe in range(ps + 1, len(context) + 1):
            if (ps, pe) == (gold_start, gold_end):
                continue
            kind = error_kind(ps, context[ps:pe], gold_start, gold)
            assert kind in ANSWERABLE_ERROR_KINDS
            seen.add(kind)
    # All positional kinds occur; "wrong occurrence" needs a repeated string,
    # which this alphabet does not have.
    assert seen == set(ANSWERABLE_ERROR_KINDS) - {
        "abstained on answerable", "right text, wrong occurrence"}
