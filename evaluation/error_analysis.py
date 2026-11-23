"""Sort every wrong field (an unmatched line item counts once) into a small taxonomy, per
approach. The rules are heuristics I checked against ~100 hand-read failures (see
error_analysis.md); `--sample N` prints random failures with their category for that check.

    python -m evaluation.error_analysis [--sample 30]
"""

import argparse
import collections
import json
import random
import re

from docextract.checks import close
from docextract.data import EVAL_FIELDS, dataset_of
from docextract.schema import format_amount, norm_text, parse_amount
from evaluation.metrics import Slot, levenshtein, slots
from evaluation.report import APPROACHES, RESULTS, examples, load, receipt

CATEGORIES = {
    "invalid_output": "output was not valid JSON / schema, every field lost",
    "ocr_lost": "value never made it into the OCR text (approach A only)",
    "misread": "a character or two off: 16.500 vs 18.500, LUCONUT vs COCONUT",
    "number_format": "right digits, wrong magnitude: 60.000 read as 60",
    "rounding": "total before the 5-sen rounding line instead of after it (SROIE)",
    "computed_total": "subtotal + tax returned as total, though the printed total included tax",
    "wrong_field": "a value from elsewhere on the receipt: cash as total, unit price as price",
    "missed_field": "printed and labelled, but returned null",
    "unlabelled_extra": "returned a value the labels don't have (tax 0, a printed 'discount 70%')",
    "item_split_merge": "one item split in two, or two merged (name continues on the next line)",
    "item_missed": "a whole line item missing",
    "item_extra": "an extra line item (payment, modifier or total line listed as an item)",
    "qty_in_name": "quantity column folded into the item name: '2 JAMUR' for JAMUR",
    "item_name": "item matched but its name differs (partial name, extra size/variant text)",
    "merchant_variant": "a different line chosen as merchant (brand vs legal entity, branch)",
    "date_format": "same date components, different order",
    "other": "anything else",
}

QTY = re.compile(r"^\d+\s+|\s+\d+$")


def _digits(x) -> str:
    return "".join(c for c in str(format_amount(x) if isinstance(x, float) else x) if c.isalnum())


def categorise(s: Slot, rec: dict, ex, all_slots: list[Slot]) -> str:
    field = s.path.split(".")[-1] if s.path.startswith("line_items") else s.path
    if not rec["valid"]:
        return "invalid_output"
    prefix = s.path.rsplit(".", 1)[0]
    missed = ".gt" in s.path
    matched = s.path.startswith("line_items") and not missed and any(
        x.path.startswith(prefix + ".") and x.gt is not None for x in all_slots)
    if s.path.startswith("line_items") and not matched:
        # Unmatched item: compare its name with the other side's item names.
        names = [norm_text(x.pred if missed else x.gt) for x in all_slots
                 if x.path.endswith(".name") and (x.pred if missed else x.gt)]
        name = next((x.gt if missed else x.pred for x in all_slots
                     if x.path == prefix + ".name"), None)
        own = norm_text(name)
        if own and any(own != n and (own in n or n in own) for n in names):
            return "item_split_merge"
        if missed:
            if rec.get("ocr_text") is not None and own and own not in norm_text(rec["ocr_text"]):
                return "ocr_lost"
            return "item_missed"
        return "item_extra"
    if s.pred is None:
        if rec.get("ocr_text") is not None and _digits(s.gt) not in _digits(rec["ocr_text"]):
            return "ocr_lost"
        return "missed_field"
    if s.gt is None:
        return "unlabelled_extra"
    if isinstance(s.pred, float) and isinstance(s.gt, float):
        if s.gt and s.pred and any(abs(s.pred * 10**k - s.gt) < 0.01 for k in (-3, -2, -1, 1, 2)):
            return "number_format"
        if field == "total" and abs(s.pred - s.gt) <= 0.05:
            return "rounding"
        r = receipt(rec)
        if field == "total" and r.subtotal is not None and r.tax and close(
                s.pred, r.subtotal + r.tax + (r.service_charge or 0) - (r.discount or 0)):
            return "computed_total"
        others = {x.gt for x in all_slots if x is not s and isinstance(x.gt, float)}
        printed = _cord_amounts(ex.cord) if ex.cord else set()
        if rec.get("ocr_text"):
            printed |= {v for v in map(parse_amount, rec["ocr_text"].split()) if v}
        if s.pred in others or s.pred in printed:
            return "wrong_field"
        if levenshtein(_digits(s.pred), _digits(s.gt)) <= 2:
            if rec.get("ocr_text") is not None and _digits(s.gt) not in _digits(rec["ocr_text"]):
                return "ocr_lost"
            return "misread"
        return "other"
    if field == "date":
        if sorted(str(s.pred).split("-")) == sorted(str(s.gt).split("-")):
            return "date_format"
        return "misread" if levenshtein(str(s.pred), str(s.gt)) <= 2 else "other"
    a, b = norm_text(str(s.pred)), norm_text(str(s.gt))
    if field == "name" and QTY.sub("", a) == b:
        return "qty_in_name"
    if levenshtein(a, b) <= max(2, len(b) // 10):
        return "misread"
    if field == "merchant":
        return "merchant_variant"
    if field == "name":
        return "item_name"
    return "other"


def _cord_amounts(cord: dict) -> set[float]:
    """Every amount printed in a CORD label tree, including payment lines and unit prices."""
    out: set[float] = set()

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str):
            v = parse_amount(o)
            if v is not None:
                out.add(v)

    walk(cord)
    return out


def analyse(approach: str, split: str) -> list[tuple[str, Slot, str]]:
    recs = load(approach, split)
    out = []
    for i, ex in examples(split).items():
        rec = recs.get(i)
        if rec is None:
            continue
        s_all = slots(receipt(rec), ex.gt, EVAL_FIELDS[dataset_of(split)])
        seen = set()
        for s in s_all:
            if s.correct:
                continue
            c = categorise(s, rec, ex, s_all)
            # an unmatched item is one error, not one per field
            key = (s.path.rsplit(".", 1)[0], c)
            if c.startswith("item_") and c != "item_name" or c == "ocr_lost" and ".gt" in s.path:
                if key in seen:
                    continue
                seen.add(key)
            out.append((i, s, c))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()
    table = {}
    for split in ["cord-test", "sroie-test"]:
        for a in APPROACHES:
            if load(a, split):
                errs = analyse(a, split)
                table[f"{a}/{split}"] = collections.Counter(c for *_, c in errs)
                if args.sample:
                    for i, s, c in random.Random(0).sample(errs, min(args.sample, len(errs))):
                        print(f"{a:12s} {i:18s} {c:16s} {s.path:24s} pred={s.pred!r} gt={s.gt!r}")
    (RESULTS / "error_taxonomy.json").write_text(json.dumps(table, indent=1))
    cols = list(table)
    print("| category | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for cat in CATEGORIES:
        if any(table[c][cat] for c in cols):
            print(f"| {cat} | " + " | ".join(str(table[c][cat]) for c in cols) + " |")
    print("| **total wrong fields** | " + " | ".join(str(sum(table[c].values())) for c in cols)
          + " |")


if __name__ == "__main__":
    main()
