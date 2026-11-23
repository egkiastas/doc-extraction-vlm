import pytest

from docextract.schema import LineItem, Receipt, match_items
from evaluation.metrics import field_accuracy, levenshtein, prf, slots, ted_accuracy

FIELDS = ["subtotal", "tax", "total", "line_items"]

GT = Receipt(
    line_items=[LineItem(name="Egg Tart", quantity=1, price=13000),
                LineItem(name="Pizza Toast", quantity=1, price=16000)],
    subtotal=29000, total=29000,
)


def test_perfect_prediction_in_any_item_order():
    pred = GT.model_copy(update={"line_items": GT.line_items[::-1]})
    s = slots(pred, GT, FIELDS)
    assert prf(s)["f1"] == 1.0 and field_accuracy(s) == 1.0


def test_known_counts():
    pred = Receipt(
        line_items=[LineItem(name="PIZZA TOAST", quantity=1, price=16000),  # case-insensitive
                    LineItem(name="Cash", price=50000)],                    # spurious item
        subtotal=29000, tax=2900, total=31900,
    )
    s = prf(slots(pred, GT, FIELDS))
    # TP: subtotal, pizza name/qty/price = 4
    # FP: tax (not on receipt), total (wrong), cash name+price = 4
    # FN: total (wrong), egg tart name/qty/price = 4
    assert (s["tp"], s["fp"], s["fn"]) == (4, 4, 4)
    assert s["precision"] == pytest.approx(0.5) and s["recall"] == pytest.approx(0.5)


def test_invalid_output_scores_as_all_missed():
    s = prf(slots(None, GT, FIELDS))
    assert s["tp"] == 0 and s["fp"] == 0 and s["fn"] == 8


def test_match_items_refuses_unrelated_pairs():
    assert match_items([LineItem(name="Coffee", price=1)], [LineItem(name="Tea", price=2)]) == []


def test_levenshtein():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3


def test_ted_accuracy():
    gt = {"menu": [{"nm": "EGG TART", "price": "13,000"}], "total": {"total_price": "13,000"}}
    assert ted_accuracy(gt, gt) == 1.0
    assert ted_accuracy({}, gt) == 0.0
    one_char = {"menu": [{"nm": "EGG TART", "price": "13,000"}], "total": {"total_price": "18,000"}}
    # Building gt from an empty tree costs 7 inner nodes + 20 leaf characters = 27;
    # the prediction is one character substitution away.
    assert ted_accuracy(one_char, gt) == pytest.approx(1 - 1 / 27)
