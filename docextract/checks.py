"""Arithmetic consistency checks. Each returns True (pass), False (fail) or None (not enough
fields to check). They catch dropped/duplicated items and misread digits, not semantic errors."""

from docextract.schema import Receipt


def close(a: float, b: float, rel: float = 0.005, abs_: float = 0.01) -> bool:
    return abs(a - b) <= max(rel * abs(b), abs_)


def items_sum(r: Receipt) -> bool | None:
    """Sum of item prices ~ subtotal, or ~ total (tax-inclusive item prices are common)."""
    prices = [it.price for it in r.line_items if it.price is not None]
    targets = [t for t in (r.subtotal, r.total) if t is not None]
    if not prices or not targets:
        return None
    return any(close(sum(prices), t) for t in targets)


def total_math(r: Receipt) -> bool | None:
    """subtotal + tax + service - discount ~ total. Many receipts print tax-inclusive
    prices (tax shown for information only), so the variant without tax also passes."""
    if r.subtotal is None or r.total is None:
        return None
    base = r.subtotal + (r.service_charge or 0) - (r.discount or 0)
    return close(base + (r.tax or 0), r.total) or close(base, r.total)


def line_math(r: Receipt) -> bool | None:
    """quantity x unit_price ~ price for every item that has all three."""
    triples = [(it.quantity, it.unit_price, it.price) for it in r.line_items
               if None not in (it.quantity, it.unit_price, it.price)]
    if not triples:
        return None
    return all(close(q * u, p) for q, u, p in triples)


CHECKS = {"items_sum": items_sum, "total_math": total_math, "line_math": line_math}

# Which fields each check vouches for; used as a confidence feature.
CHECK_FIELDS = {
    "items_sum": ["subtotal", "line_items.price"],
    "total_math": ["subtotal", "tax", "service_charge", "discount", "total"],
    "line_math": ["line_items.quantity", "line_items.unit_price", "line_items.price"],
}


def run_checks(r: Receipt) -> dict[str, bool | None]:
    return {name: fn(r) for name, fn in CHECKS.items()}
