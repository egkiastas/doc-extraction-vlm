"""Rebuild tests/fixtures from real prediction files: a few CORD test images (CC BY 4.0), the
model outputs recorded for them, their labels, and the decisions the service should make.
Run after changing parsing, checks or the confidence model, then review the diff.

    python -m tests.make_fixtures --approach qwen3vl cord-test-004 cord-test-033 ...
"""

import argparse
import json
from pathlib import Path

from docextract.confidence import ConfidenceModel, assess
from docextract.data import load_split
from docextract.schema import Receipt
from evaluation.metrics import prf, slots
from evaluation.report import RESULTS, load

FIX = Path(__file__).parent / "fixtures"
FIELDS = ["subtotal", "tax", "service_charge", "discount", "total", "line_items"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--approach", default="qwen3vl")
    ap.add_argument("ids", nargs="+")
    args = ap.parse_args()

    FIX.mkdir(exist_ok=True)
    models = json.loads((RESULTS / "confidence_model.json").read_text())
    (FIX / "confidence_model.json").write_text(json.dumps(
        {args.approach: models[args.approach]}, indent=1))
    model = ConfidenceModel.from_json(models[args.approach])
    recs = load(args.approach, "cord-test")
    examples = {e.id: e for e in load_split("cord-test")}
    recorded, expected = [], {}
    for i in args.ids:
        e, rec = examples[i], recs[i]
        (FIX / f"{i}.png").write_bytes(e.image_bytes)
        recorded.append({k: rec[k] for k in ("key", "approach", "raw", "field_logprobs")})
        r = Receipt.model_validate(rec["receipt"]) if rec["valid"] else None
        a = assess(r, rec["field_logprobs"], model)
        expected[f"{i}.png"] = {"gt": e.cord, "accept": a["accept"],
                                "f1": prf(slots(r, e.gt, FIELDS))["f1"]}
    (FIX / "recorded.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recorded))
    (FIX / "expected.json").write_text(json.dumps(expected, indent=1, ensure_ascii=False))
