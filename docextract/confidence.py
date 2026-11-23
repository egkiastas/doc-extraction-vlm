"""Per-field confidence and the auto-accept decision.

Three signals per extracted field:
  - the lowest token log-probability inside the field's value (from the decoder),
  - whether a second, independent extractor produced the same value (optional),
  - whether the arithmetic checks that involve the field pass or fail.
A logistic regression fitted on the validation split maps them to P(field correct). The
document score is the minimum over its fields: one wrong amount makes the document wrong.
"""

import json
import math
from dataclasses import dataclass

import numpy as np

from docextract.checks import CHECK_FIELDS, run_checks
from docextract.schema import ITEM_FIELDS, SCALAR_FIELDS, Receipt, match_items, values_match

FEATURES = ["min_logprob", "agree", "disagree", "check_pass", "check_fail", "is_item"]


def value_spans(text: str) -> dict[str, tuple[int, int]]:
    """Character span of every scalar value in the outermost JSON object of `text`, keyed by
    path ("total", "line_items.2.price"). Returns {} for text that is not valid JSON."""
    spans: dict[str, tuple[int, int]] = {}
    dec = json.JSONDecoder()

    def ws(i: int) -> int:
        while text[i] in " \t\r\n":
            i += 1
        return i

    def parse(i: int, path: str) -> int:
        i = ws(i)
        if text[i] in "{[":
            close, is_obj, idx = ("}" if text[i] == "{" else "]"), text[i] == "{", 0
            i = ws(i + 1)
            if text[i] == close:
                return i + 1
            while True:
                if is_obj:
                    key, i = json.decoder.scanstring(text, ws(i) + 1)
                    i = ws(i)
                    assert text[i] == ":"
                    i = parse(i + 1, f"{path}.{key}" if path else key)
                else:
                    i = parse(i, f"{path}.{idx}")
                    idx += 1
                i = ws(i)
                if text[i] == close:
                    return i + 1
                assert text[i] == ","
                i += 1
        _, end = dec.raw_decode(text, i)
        spans[path] = (i, end)
        return end

    start = text.find("{")
    try:
        if start >= 0:
            parse(start, "")
    except (AssertionError, IndexError, ValueError):
        return {}
    return spans


def field_logprobs(text: str, tokens: list[tuple[str, float]]) -> dict[str, float]:
    """Minimum token log-prob over each value span. `tokens` are (text piece, logprob) and must
    concatenate to `text`. Tokens that straddle a span boundary count for the span."""
    if not tokens:
        return {}
    offsets, pos = [], 0
    for piece, lp in tokens:
        offsets.append((pos, pos + len(piece), lp))
        pos += len(piece)
    out = {}
    for path, (s, e) in value_spans(text).items():
        lps = [lp for a, b, lp in offsets if a < e and b > s]
        if lps:
            out[path] = min(lps)
    return out


def generic_path(path: str) -> str:
    """line_items.3.price -> line_items.price"""
    parts = path.split(".")
    return f"{parts[0]}.{parts[-1]}" if parts[0] == "line_items" else path


def features(path: str, logprob: float | None, agree: bool | None,
             checks: dict[str, bool | None]) -> list[float]:
    involved = [v for name, v in checks.items()
                if v is not None and generic_path(path) in CHECK_FIELDS[name]]
    return [
        max(logprob, -20.0) if logprob is not None else 0.0,
        float(agree is True),
        float(agree is False),
        float(any(involved)),
        float(any(v is False for v in involved)),
        float(path.startswith("line_items")),
    ]


@dataclass
class ConfidenceModel:
    weights: list[float]  # bias first, then one per FEATURES entry
    threshold: float = 1.0  # document score needed to auto-accept
    accept_on: list[str] | None = None  # fields the document score covers; None means all

    def predict(self, x: list[float]) -> float:
        z = self.weights[0] + sum(w * v for w, v in zip(self.weights[1:], x, strict=True))
        return 1 / (1 + math.exp(-z))

    def to_json(self) -> dict:
        return {"features": FEATURES, "weights": self.weights, "threshold": self.threshold,
                "accept_on": self.accept_on}

    @classmethod
    def from_json(cls, d: dict) -> "ConfidenceModel":
        return cls(d["weights"], d["threshold"], d.get("accept_on"))


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 50) -> list[float]:
    """L2-regularised logistic regression by Newton's method; small and dependency-free."""
    X = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(X.shape[1])
    reg = l2 * np.eye(X.shape[1])
    reg[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        grad = X.T @ (p - y) + reg @ w
        hess = X.T @ (X * (p * (1 - p))[:, None]) + reg
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w.tolist()


def document_score(field_conf: dict[str, float], fields: list[str] | None = None) -> float:
    """The weakest field decides: one wrong total makes a wrong document."""
    vals = [c for p, c in field_conf.items() if fields is None or p.split(".")[0] in fields]
    return min(vals) if vals else 0.0


def choose_threshold(doc_scores: list[float], doc_field_acc: list[tuple[int, int]],
                     target: float = 0.99) -> float:
    """Lowest score threshold whose accepted documents reach `target` field accuracy.
    doc_field_acc holds (correct slots, total slots) per document."""
    order = np.argsort(doc_scores)[::-1]
    best, correct, total = 1.0 + 1e-9, 0, 0
    for k, i in enumerate(order):
        correct += doc_field_acc[i][0]
        total += doc_field_acc[i][1]
        nxt = doc_scores[order[k + 1]] if k + 1 < len(order) else -1.0
        if nxt == doc_scores[i]:
            continue  # can't split documents with equal scores
        if total and correct / total >= target:
            best = doc_scores[i]
    return best


def predicted_paths(r: Receipt, fields: list[str] | None = None) -> list[str]:
    """Paths of the non-null values in a prediction, optionally restricted to some fields."""
    fields = fields or SCALAR_FIELDS + ["line_items"]
    paths = [f for f in SCALAR_FIELDS if f in fields and getattr(r, f) is not None]
    if "line_items" in fields:
        paths += [f"line_items.{i}.{k}" for i, it in enumerate(r.line_items) for k in ITEM_FIELDS
                  if getattr(it, k) is not None]
    return paths


def agreement(r: Receipt, other: Receipt | None) -> dict[str, bool]:
    """Does a second extractor give the same value? Items are paired by Hungarian matching."""
    if other is None:
        return {}
    out = {f: values_match(f, getattr(r, f), getattr(other, f)) for f in SCALAR_FIELDS
           if getattr(r, f) is not None}
    pairs = dict(match_items(r.line_items, other.line_items))
    for i, it in enumerate(r.line_items):
        for k in ITEM_FIELDS:
            if getattr(it, k) is not None:
                o = other.line_items[pairs[i]] if i in pairs else None
                out[f"line_items.{i}.{k}"] = o is not None and values_match(
                    k, getattr(it, k), getattr(o, k))
    return out


def assess(r: Receipt | None, field_lps: dict[str, float], model: ConfidenceModel,
           other: Receipt | None = None, fields: list[str] | None = None) -> dict:
    """Checks, per-field confidence, document score and the accept/review decision."""
    if r is None:
        return {"checks": {}, "field_confidence": {}, "score": 0.0, "accept": False}
    checks = run_checks(r)
    agree = agreement(r, other)
    conf = {p: model.predict(features(p, field_lps.get(p), agree.get(p), checks))
            for p in predicted_paths(r, fields)}
    score = document_score(conf, model.accept_on)
    return {"checks": checks, "field_confidence": conf, "score": score,
            "accept": score >= model.threshold}
