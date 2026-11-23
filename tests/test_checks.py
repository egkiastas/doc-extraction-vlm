from docextract.checks import items_sum, line_math, run_checks, total_math
from docextract.schema import LineItem, Receipt


def items(*prices):
    return [LineItem(name=f"i{n}", price=p) for n, p in enumerate(prices)]


def test_items_sum():
    assert items_sum(Receipt(line_items=items(24000, 4000), subtotal=28000))
    assert items_sum(Receipt(line_items=items(24000), subtotal=28000)) is False  # dropped item
    # tax-inclusive item prices: items add up to the total, not the pre-tax subtotal
    assert items_sum(Receipt(line_items=items(20000), subtotal=18181, tax=1818, total=20000))
    assert items_sum(Receipt(line_items=items(20000))) is None


def test_total_math():
    assert total_math(Receipt(subtotal=100909, tax=10091, total=111000))
    assert total_math(Receipt(subtotal=194000, discount=19400, total=174600))
    assert total_math(Receipt(subtotal=69000, service_charge=3795, tax=7280, total=80075))
    # tax printed for information only (already included)
    assert total_math(Receipt(subtotal=28182, tax=2818, total=28182))
    assert total_math(Receipt(subtotal=28182, tax=2818, total=13100)) is False
    assert total_math(Receipt(total=5)) is None


def test_tolerance_is_relative():
    assert total_math(Receipt(subtotal=100000, tax=10000, total=110400))  # 0.4%: rounding
    assert total_math(Receipt(subtotal=10.0, tax=1.0, total=11.1)) is False


def test_line_math():
    ok = LineItem(name="a", quantity=2, unit_price=12000, price=24000)
    bad = LineItem(name="b", quantity=2, unit_price=12000, price=12000)
    assert line_math(Receipt(line_items=[ok]))
    assert line_math(Receipt(line_items=[ok, bad])) is False
    assert run_checks(Receipt())["line_math"] is None
