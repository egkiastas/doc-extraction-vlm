"""Receipt schema, amount/date normalisation and the mapping from CORD labels.

Models are asked to copy amounts exactly as printed ("46.000", "12,50"). Turning those
strings into numbers is done here, deterministically, rather than trusted to the model:
CORD is Indonesian (dot or comma as thousands separator), SROIE is Malaysian (dot decimals).
"""

import json
import re
from datetime import date
from difflib import SequenceMatcher
from typing import Annotated, Any

import numpy as np
from pydantic import BaseModel, BeforeValidator, ValidationError, field_validator
from scipy.optimize import linear_sum_assignment


def parse_amount(value: Any) -> float | None:
    """'60.000' -> 60000, '12,50' -> 12.5, 'Rp 1.250.000,00' -> 1250000, '-5.455' -> -5455.

    A separator followed by exactly three digits is a thousands separator; one or two
    digits after the last separator make it the decimal point.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    s = str(value).strip()
    negative = bool(re.match(r"^[^\d]*-", s)) or (s.startswith("(") and s.endswith(")"))
    s = re.sub(r"[^\d.,]", "", s).strip(".,")
    if not re.search(r"\d", s):
        return None
    head, sep, tail = s.rpartition(max((c for c in ".," if c in s), key=s.rfind, default="."))
    if sep and len(tail) != 3:
        number = float(re.sub(r"[.,]", "", head) + "." + tail)
    else:
        number = float(re.sub(r"[.,]", "", s))
    return -number if negative else number


_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}


def parse_date(value: Any) -> str | None:
    """Normalise to ISO yyyy-mm-dd; day-first when ambiguous (both datasets are day-first).
    Unparseable strings are returned stripped, so they still compare exactly."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    low = s.lower()
    patterns = [
        (r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", ("y", "m", "d")),
        (r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})", ("d", "m", "y")),
        (r"(\d{1,2})[\s-]*([a-z]{3})[a-z]*[\s,-]*(\d{2,4})", ("d", "mon", "y")),
        (r"([a-z]{3})[a-z]*[\s-]*(\d{1,2})[\s,-]+(\d{2,4})", ("mon", "d", "y")),
    ]
    for pattern, order in patterns:
        m = re.search(pattern, low)
        if not m:
            continue
        parts = dict(zip(order, m.groups(), strict=True))
        try:
            month = _MONTHS[parts["mon"][:3]] if "mon" in parts else int(parts["m"])
            year = int(parts["y"])
            year += 2000 if year < 100 else 0
            return date(year, month, int(parts["d"])).isoformat()
        except (KeyError, ValueError):
            continue
    return s


Amount = Annotated[float | None, BeforeValidator(parse_amount)]


class LineItem(BaseModel):
    name: str | None = None
    quantity: Amount = None
    unit_price: Amount = None
    price: Amount = None


class Receipt(BaseModel):
    merchant: str | None = None
    date: str | None = None
    currency: str | None = None
    line_items: list[LineItem] = []
    subtotal: Amount = None
    tax: Amount = None
    service_charge: Amount = None
    discount: Amount = None
    total: Amount = None

    @field_validator("date", mode="before")
    @classmethod
    def _date(cls, v: Any) -> str | None:
        return parse_date(v)

    @field_validator("discount", mode="after")
    @classmethod
    def _discount(cls, v: float | None) -> float | None:
        # Receipts print discounts as "-19,400" or "19,400"; store the magnitude.
        return abs(v) if v is not None else None

    @field_validator("line_items", mode="before")
    @classmethod
    def _items(cls, v: Any) -> Any:
        return [] if v is None else v


SCALAR_FIELDS = ["merchant", "date", "currency", "subtotal", "tax", "service_charge",
                 "discount", "total"]
ITEM_FIELDS = ["name", "quantity", "unit_price", "price"]

# What the models are constrained to (Ollama `format`) or asked for (VLM prompt).
# Amounts are strings on purpose: the model transcribes, parse_amount interprets.
_str = {"type": ["string", "null"]}
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "merchant": _str,
        "date": _str,
        "currency": _str,
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {k: _str for k in ITEM_FIELDS},
                "required": ITEM_FIELDS,
            },
        },
        **{k: _str for k in ["subtotal", "tax", "service_charge", "discount", "total"]},
    },
    "required": SCALAR_FIELDS[:3] + ["line_items"] + SCALAR_FIELDS[3:],
}

PROMPT = """Extract the data from this receipt as a JSON object with these keys:
- merchant: store or company name as printed, or null
- date: purchase date as printed, or null
- currency: ISO 4217 code (IDR, MYR, USD...) only if a currency symbol or code is printed, else null
- line_items: the purchased products, each with
    name: the item description only, without quantity or price
    quantity: the count as printed ("2", "1x"), or null
    unit_price: the per-unit price only when printed separately (often after "@"), or null
    price: the amount printed on the item's line
- subtotal, tax, service_charge, discount, total: the amounts printed next to those labels, or null

Rules:
- Copy amounts exactly as printed ("46.000", "12,50"). Never convert, round or compute.
- line_items are products only: no payment, cash, change, card, total or tax lines, and no unpriced notes such as "less ice".
- Use null for anything that is not printed on the receipt.
Return only the JSON object."""  # noqa: E501


def parse_output(text: str) -> tuple[Receipt | None, str | None]:
    """Model output -> (Receipt, None) or (None, error). Tolerates code fences and chatter
    around the outermost JSON object, nothing else."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None, "no json object"
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        return None, f"invalid json: {e.msg}"
    if not isinstance(data, dict):
        return None, "not an object"
    try:
        return Receipt.model_validate(data), None
    except ValidationError as e:
        return None, f"schema: {e.errors()[0]['loc']} {e.errors()[0]['msg']}"


# --- comparing two receipts (evaluation, and agreement between extractors) ---

def norm_text(s: str | None) -> str:
    return " ".join(re.sub(r"[^0-9A-Z ]", " ", (s or "").upper()).split())


def values_match(field: str, a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) < 0.005
    if field in ("merchant", "name"):
        return norm_text(a) == norm_text(b)
    return str(a).strip().upper() == str(b).strip().upper()


def match_items(pred: list, gt: list) -> list[tuple[int, int]]:
    """Hungarian matching of predicted to ground-truth items. A pair only counts if at least
    one field agrees or the names are similar; otherwise both sides stay unmatched."""
    if not pred or not gt:
        return []
    score = np.zeros((len(pred), len(gt)))
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            exact = sum(values_match(k, getattr(p, k), getattr(g, k)) for k in ITEM_FIELDS
                        if getattr(g, k) is not None)
            sim = SequenceMatcher(None, norm_text(p.name), norm_text(g.name)).ratio()
            score[i, j] = exact + sim
    rows, cols = linear_sum_assignment(-score)
    return [(i, j) for i, j in zip(rows, cols, strict=True) if score[i, j] >= 1.0]


def _first(value: Any) -> str | None:
    # A handful of CORD labels are lists of repeated strings (["46.636", "46.636"]); Donut's
    # predicted trees can nest anything anywhere.
    if isinstance(value, list):
        return next((v for v in value if isinstance(v, str)), None)
    return value if isinstance(value, str) else None


def cord_to_raw(gt_parse: dict) -> dict:
    """CORD gt_parse -> our schema with values as printed. Used for the training targets
    of the fine-tuned model and, via Receipt.model_validate, for evaluation.

    Priced sub-items (menu.sub, e.g. an extra topping) become line items of their own,
    because they are printed and summed like items; modifiers without a price or at 0
    ("less sugar", "Sedang") are dropped, as are item-level discounts and payment lines
    (cash, change, card).
    CORD-v2 has no merchant/date labels.
    """
    menu = gt_parse.get("menu", [])
    menu = [menu] if isinstance(menu, dict) else menu if isinstance(menu, list) else []
    rows = []
    for m in (m for m in menu if isinstance(m, dict)):
        rows.append(m)
        sub = m.get("sub", [])
        rows += [s for s in ([sub] if isinstance(sub, dict) else sub)
                 if isinstance(s, dict) and parse_amount(_first(s.get("price")))]
    items = []
    for m in rows:
        name = m.get("nm")
        if isinstance(name, list):
            name = " ".join(x for x in name if isinstance(x, str) and x.strip() not in {"", "-"})
        name = name if isinstance(name, str) else None
        items.append({
            "name": name,
            "quantity": _first(m.get("cnt")),
            "unit_price": _first(m.get("unitprice")),
            "price": _first(m.get("price")),
        })
    sub = gt_parse.get("sub_total") or {}
    tot = gt_parse.get("total") or {}
    sub = sub if isinstance(sub, dict) else {}
    tot = tot if isinstance(tot, dict) else {}
    return {
        "merchant": None,
        "date": None,
        "currency": None,
        "line_items": items,
        "subtotal": _first(sub.get("subtotal_price")),
        "tax": _first(sub.get("tax_price")),
        "service_charge": _first(sub.get("service_price")),
        "discount": _first(sub.get("discount_price")),
        "total": _first(tot.get("total_price")),
    }


def from_cord(gt_parse: dict) -> Receipt:
    return Receipt.model_validate(cord_to_raw(gt_parse))


def format_amount(x: float | None) -> str | None:
    if x is None:
        return None
    return str(int(x)) if x == int(x) else f"{x:.2f}"


def to_cord(r: Receipt) -> dict:
    """Receipt -> CORD-style tree with canonical values, for tree-edit-distance scoring."""
    menu = [
        {k2: v for k2, v in [("nm", it.name), ("cnt", format_amount(it.quantity)),
                             ("unitprice", format_amount(it.unit_price)),
                             ("price", format_amount(it.price))] if v}
        for it in r.line_items
    ]
    sub = {k: format_amount(v) for k, v in [("subtotal_price", r.subtotal), ("tax_price", r.tax),
                                           ("service_price", r.service_charge),
                                           ("discount_price", r.discount)] if v is not None}
    out: dict = {"menu": [m for m in menu if m]}
    if sub:
        out["sub_total"] = sub
    if r.total is not None:
        out["total"] = {"total_price": format_amount(r.total)}
    return out
