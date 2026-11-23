import numpy as np
import pytest

from docextract.confidence import (
    FEATURES,
    ConfidenceModel,
    agreement,
    assess,
    choose_threshold,
    features,
    field_logprobs,
    fit_logistic,
    value_spans,
)
from docextract.schema import LineItem, Receipt

OUTPUT = ('```json\n{"merchant": null, "line_items": [{"name": "Egg \\"Tart\\"", '
          '"price": "13,000"}], "total": "13,000"}\n```')


def test_value_spans_point_at_values():
    spans = value_spans(OUTPUT)
    assert set(spans) == {"merchant", "line_items.0.name", "line_items.0.price", "total"}
    s, e = spans["line_items.0.price"]
    assert OUTPUT[s:e] == '"13,000"'
    assert value_spans('{"total": "13,0') == {}


def test_field_logprobs_take_the_weakest_token():
    text = '{"total": "13,000"}'
    tokens = [('{"', 0.0), ("total", 0.0), ('": "', -0.1), ("13", -0.2), (",", -0.05),
              ("000", -2.5), ('"}', 0.0)]
    # '": "' straddles the opening quote of the value, so it counts too
    assert field_logprobs(text, tokens) == {"total": -2.5}
    assert field_logprobs(text, []) == {}


def test_check_features_only_touch_involved_fields():
    checks = {"items_sum": True, "total_math": False, "line_math": None}
    f_total = dict(zip(FEATURES, features("total", -0.1, None, checks), strict=True))
    f_name = dict(zip(FEATURES, features("line_items.0.name", -0.1, None, checks), strict=True))
    assert f_total["check_fail"] == 1 and f_total["check_pass"] == 0
    assert f_name["check_fail"] == 0 and f_name["check_pass"] == 0 and f_name["is_item"] == 1


def test_agreement_pairs_items_by_content():
    a = Receipt(line_items=[LineItem(name="Tea", price=5), LineItem(name="Coffee", price=7)],
                total=12)
    b = Receipt(line_items=[LineItem(name="Coffee", price=7), LineItem(name="Tea", price=6)],
                total=12)
    agree = agreement(a, b)
    assert agree["total"] and agree["line_items.1.price"]
    assert agree["line_items.0.name"] and not agree["line_items.0.price"]


def test_fit_logistic_separates_and_is_calibrated_on_easy_data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 1))
    y = (rng.random(2000) < 1 / (1 + np.exp(-3 * x[:, 0]))).astype(float)
    w = fit_logistic(x, y, l2=0.0)
    assert w[1] == pytest.approx(3, abs=0.3) and w[0] == pytest.approx(0, abs=0.2)


def test_choose_threshold_maximises_coverage_at_target():
    scores = [0.99, 0.95, 0.90, 0.50]
    acc = [(10, 10), (10, 10), (9, 10), (5, 10)]
    assert choose_threshold(scores, acc, target=0.95) == 0.90   # 29/30 = 96.7% accepted
    assert choose_threshold(scores, acc, target=1.0) == 0.95
    assert choose_threshold(scores, [(0, 1)] * 4, target=0.99) > 1.0  # accept nothing


def test_decision_uses_the_weakest_field():
    model = ConfidenceModel([0.0, 1.0, 0, 0, 0, 0, 0], threshold=0.5)  # P = sigmoid(logprob)
    r = Receipt(total=10, subtotal=10)
    sure = assess(r, {"total": 5.0, "subtotal": 5.0}, model)
    unsure = assess(r, {"total": 5.0, "subtotal": -5.0}, model)
    assert sure["accept"] and not unsure["accept"]
    assert unsure["score"] == pytest.approx(1 / (1 + np.exp(5)))
    assert not assess(None, {}, model)["accept"]  # invalid output always goes to review


def test_accept_policy_can_ignore_item_lines():
    model = ConfidenceModel([0.0, 1.0, 0, 0, 0, 0, 0], threshold=0.5, accept_on=["total"])
    r = Receipt(total=10, line_items=[LineItem(name="Tea", price=10)])
    a = assess(r, {"total": 5.0, "line_items.0.name": -5.0, "line_items.0.price": 5.0}, model)
    assert a["accept"] and a["field_confidence"]["line_items.0.name"] < 0.5
    assert not assess(Receipt(line_items=r.line_items), {}, model)["accept"]  # no total, no accept
