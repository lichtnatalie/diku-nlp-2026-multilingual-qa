"""Week 41: offline tests for the paired comparison, the error typing and the
question-form detector. Run: python -m pytest tests/test_week41.py"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.week41_paired import mcnemar  # noqa: E402
from scripts.week41_qa_error_sample import error_type, span_relation  # noqa: E402
from scripts.week41_question_form import is_polar  # noqa: E402


def test_mcnemar_counts_and_symmetry():
    a = np.array([1, 1, 0, 0, 1, 0], bool)
    b = np.array([1, 0, 1, 0, 0, 0], bool)
    r = mcnemar(a, b)
    assert (r["both_correct"], r["only_a_correct"], r["only_b_correct"], r["both_wrong"]) == (1, 2, 1, 2)
    assert r["n"] == 6
    assert abs(r["mcnemar_exact_p"] - mcnemar(b, a)["mcnemar_exact_p"]) < 1e-12
    assert mcnemar(a, a)["mcnemar_exact_p"] == 1.0  # no discordant pairs


def test_error_types():
    r = SimpleNamespace(answerable=True, answered=False, em=0)
    assert error_type(r) == "wrongly abstains"
    assert error_type(SimpleNamespace(answerable=False, answered=True, em=0)) == "wrongly answers"
    assert error_type(SimpleNamespace(answerable=True, answered=True, em=0)) == "wrong span"
    assert error_type(SimpleNamespace(answerable=True, answered=True, em=1)) == "correct"
    assert error_type(SimpleNamespace(answerable=False, answered=False, em=1)) == "correct"


def _rel(gs, gold, ps, pred):
    return span_relation(SimpleNamespace(error_type="wrong span", gold_start=gs, gold_answer=gold,
                                         pred_start=ps, pred_text=pred))


def test_span_relation():
    ctx = "Zabid in the Tihama region of Yemen"
    g = ctx.index("the Tihama")
    assert _rel(g, "the Tihama region", 0, "Zabid in the Tihama region") == "pred contains gold"
    assert _rel(0, "Zabid in the Tihama", 0, "Zabid") == "gold contains pred"
    assert _rel(0, "Zabid", ctx.index("Yemen"), "Yemen") == "disjoint"
    assert _rel(0, "Zabid in", 6, "in the") == "partial overlap"


def test_polar_detector():
    assert is_polar("ar", "هل بنى النبيّ سليمان القدس؟")
    assert not is_polar("ar", "كم تزن ساعة بيج بن؟")
    assert is_polar("ko", "미국은 주마다 법이 다른가?")
    assert not is_polar("ko", "제3공화국 당시 지도자는 누구였는가?")
    assert not is_polar("ko", "청나라 패망일은 언젠가요?")
    assert is_polar("te", "ఏది?") is None
