"""Field-level P/R/F1, line-item matching and CORD's tree-edit-distance accuracy.

Everything is built on "slots": one per (field, document) where the prediction or the ground
truth has a value. A wrong value counts as both a false positive and a false negative, which is
the usual convention for key-information extraction.
"""

from dataclasses import dataclass

from docextract.schema import ITEM_FIELDS, Receipt, match_items, values_match


@dataclass
class Slot:
    path: str  # "total", "line_items.3.price" (pred index) or "line_items.gt2.price" (missed)
    pred: object
    gt: object
    correct: bool


def slots(pred: Receipt | None, gt: Receipt, fields: list[str]) -> list[Slot]:
    pred = pred or Receipt()
    out = []
    for f in fields:
        if f == "line_items":
            continue
        p, g = getattr(pred, f), getattr(gt, f)
        if p is not None or g is not None:
            out.append(Slot(f, p, g, values_match(f, p, g)))
    if "line_items" not in fields:
        return out
    pairs = match_items(pred.line_items, gt.line_items)
    matched_p, matched_g = {i for i, _ in pairs}, {j for _, j in pairs}
    for i, j in pairs:
        p, g = pred.line_items[i], gt.line_items[j]
        for k in ITEM_FIELDS:
            pv, gv = getattr(p, k), getattr(g, k)
            if pv is not None or gv is not None:
                out.append(Slot(f"line_items.{i}.{k}", pv, gv, values_match(k, pv, gv)))
    for i, p in enumerate(pred.line_items):
        if i not in matched_p:
            out += [Slot(f"line_items.{i}.{k}", getattr(p, k), None, False)
                    for k in ITEM_FIELDS if getattr(p, k) is not None]
    for j, g in enumerate(gt.line_items):
        if j not in matched_g:
            out += [Slot(f"line_items.gt{j}.{k}", None, getattr(g, k), False)
                    for k in ITEM_FIELDS if getattr(g, k) is not None]
    return out


def prf(all_slots: list[Slot]) -> dict:
    tp = sum(s.correct and s.pred is not None for s in all_slots)
    fp = sum(s.pred is not None and not s.correct for s in all_slots)
    fn = sum(s.gt is not None and not s.correct for s in all_slots)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "tp": tp, "fp": fp, "fn": fn}


def field_accuracy(all_slots: list[Slot]) -> float:
    return sum(s.correct for s in all_slots) / len(all_slots) if all_slots else 1.0


# --- CORD tree-edit-distance accuracy (re-implementation of Donut's JSONParseEvaluator) ---

def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _normalize(data):
    if not data:
        return {}
    if isinstance(data, dict):
        out = {}
        for key in sorted(data, key=lambda k: (len(k), k)):
            value = _normalize(data[key])
            if value:
                out[key] = value if isinstance(value, list) else [value]
        return out
    if isinstance(data, list):
        if all(isinstance(x, dict) for x in data):
            return [n for n in (_normalize(x) for x in data) if n]
        return [str(x).strip() for x in data if isinstance(x, str | int | float) and str(x).strip()]
    return [str(data).strip()]


def _tree(data, name="<root>"):
    import zss

    node = zss.Node(name)
    if isinstance(data, dict):
        for k, v in data.items():
            node.addkid(_tree(v, k))
    elif isinstance(data, list):
        if all(isinstance(x, dict) for x in data):
            for x in data:
                node.addkid(_tree(x, "<subtree>"))
        else:
            for x in data:
                node.addkid(zss.Node(f"<leaf>{x}"))
    return node


def _ins_del(node) -> int:
    return len(node.label.replace("<leaf>", "")) if "<leaf>" in node.label else 1


def _update(a, b) -> int:
    la, lb = "<leaf>" in a.label, "<leaf>" in b.label
    if la and lb:
        return levenshtein(a.label.replace("<leaf>", ""), b.label.replace("<leaf>", ""))
    if la != lb:
        return 1 + len((a.label if la else b.label).replace("<leaf>", ""))
    return int(a.label != b.label)


def ted_accuracy(pred: dict, gt: dict) -> float:
    """1 - TED(pred, gt) / TED(empty, gt), floored at 0; as reported on CORD by Donut."""
    import zss

    def dist(x, y):
        return zss.distance(x, y, get_children=zss.Node.get_children, insert_cost=_ins_del,
                            remove_cost=_ins_del, update_cost=_update)

    g = _tree(_normalize(gt))
    denom = dist(_tree(_normalize({})), g)
    return max(0.0, 1 - dist(_tree(_normalize(pred)), g) / denom) if denom else 1.0
