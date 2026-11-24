"""Score every prediction file under results/predictions, fit the confidence model on CORD
validation, and write results/metrics.json, results/summary.md, results/confidence_model.json
and the figures. Needs no model, only the prediction files and the datasets.

    python -m evaluation.report
"""

import json
import statistics
import sys
from pathlib import Path

import numpy as np

from docextract.checks import run_checks
from docextract.confidence import (
    ConfidenceModel,
    agreement,
    choose_threshold,
    document_score,
    features,
    fit_logistic,
    predicted_paths,
)
from docextract.config import ROOT, load_config
from docextract.data import EVAL_FIELDS, dataset_of, load_split
from docextract.schema import Receipt, from_cord, to_cord
from evaluation.metrics import Slot, field_accuracy, prf, slots, ted_accuracy
from evaluation.robustness import CORRUPTIONS

RESULTS = ROOT / "results"
PRED_DIR = RESULTS / "predictions"
APPROACHES = {
    "ocr_llm": "A. docTR OCR + Qwen3-4B (Ollama, Q4)",
    "qwen3vl": "B. Qwen3-VL-2B zero-shot",
    "qwen3vl_lora": "C. Qwen3-VL-2B + LoRA on CORD",
    "donut": "Donut (CORD fine-tuned, reference)",
    "api": "D. API model",
}
# Second opinion used for the agreement feature.
PARTNER = {"qwen3vl": "ocr_llm", "ocr_llm": "qwen3vl", "qwen3vl_lora": "ocr_llm",
           "donut": "qwen3vl", "api": "qwen3vl"}
ROBUST_N = 40
_examples: dict[str, dict] = {}


def examples(split: str) -> dict:
    if split not in _examples:
        _examples[split] = {e.id: e for e in load_split(split)}
    return _examples[split]


def load(approach: str, name: str) -> dict[str, dict]:
    path = PRED_DIR / approach / f"{name}.jsonl"
    if not path.exists():
        return {}
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}


def receipt(rec: dict | None) -> Receipt | None:
    return Receipt.model_validate(rec["receipt"]) if rec and rec["valid"] else None


def score(recs: dict, split: str, ids: list[str] | None = None) -> dict:
    exs = examples(split)
    ids = ids or sorted(recs)
    fields = EVAL_FIELDS[dataset_of(split)]
    all_slots: list[Slot] = []
    ted, checks_ok, checks_n, exact = [], 0, 0, 0
    for i in ids:
        e, r = exs[i], receipt(recs.get(i))
        s = slots(r, e.gt, fields)
        all_slots += s
        exact += all(x.correct for x in s)
        if e.cord is not None:
            # Schema-level TED: both sides go through the same mapping and normalisation.
            ted.append(ted_accuracy(to_cord(r) if r else {}, to_cord(from_cord(e.cord))))
        if r:
            results = [v for v in run_checks(r).values() if v is not None]
            if results:
                checks_n += 1
                checks_ok += all(results)
    items = [x for x in all_slots if x.path.startswith("line_items")]
    header = [x for x in all_slots if not x.path.startswith("line_items")]
    lat = [recs[i]["latency_s"] for i in ids if i in recs]
    out = {
        "n": len(ids),
        "f1": prf(all_slots)["f1"],
        "header_f1": prf(header)["f1"],
        "items_f1": prf(items)["f1"] if items else None,
        "field_accuracy": field_accuracy(all_slots),
        "doc_exact": exact / len(ids),
        "json_valid": sum(recs[i]["valid"] for i in ids if i in recs) / len(ids),
        "checks_pass": checks_ok / checks_n if checks_n else None,
        "latency_mean_s": statistics.mean(lat) if lat else None,
        "latency_p95_s": float(np.percentile(lat, 95)) if lat else None,
        "device": next(iter(recs.values()))["device"] if recs else None,
        "per_field": {f: prf([x for x in all_slots if x.path.split(".")[0] == f])["f1"]
                      for f in fields},
    }
    if ted:
        out["ted_acc"] = statistics.mean(ted)
    if any("cord" in recs.get(i, {}) for i in ids):  # Donut: official score on its own tree
        out["ted_acc_official"] = statistics.mean(
            ted_accuracy(recs[i].get("cord", {}), exs[i].cord) for i in ids)
    return out


def cost_per_1k(cfg, s: dict | None, recs: dict) -> float | None:
    if not s or s["latency_mean_s"] is None:
        return None
    if s["device"] == "api":
        tin = statistics.mean(r["usage"]["input_tokens"] for r in recs.values())
        tout = statistics.mean(r["usage"]["output_tokens"] for r in recs.values())
        return 1000 * (tin * cfg.api_usd_per_mtok_in + tout * cfg.api_usd_per_mtok_out) / 1e6
    rate = getattr(cfg, f"cost_usd_per_hour_{s['device']}")
    return s["latency_mean_s"] * 1000 / 3600 * rate


# --- confidence ---------------------------------------------------------------------------

def field_rows(approach: str, split: str, use_agreement: bool):
    """Per predicted field: (doc id, path, features, correct)."""
    recs, partner = load(approach, split), load(PARTNER[approach], split)
    fields = EVAL_FIELDS[dataset_of(split)]
    rows = []
    for i, e in examples(split).items():
        r = receipt(recs.get(i))
        if r is None:
            continue
        correct = {s.path: s.correct for s in slots(r, e.gt, fields) if s.pred is not None}
        agree = agreement(r, receipt(partner.get(i))) if use_agreement else {}
        checks = run_checks(r)
        for p in predicted_paths(r, fields):
            x = features(p, recs[i]["field_logprobs"].get(p), agree.get(p), checks)
            rows.append((i, p, x, correct[p]))
    return rows


# What auto-accept promises: every field right, or only the document-level amounts (what gets
# booked). Item names are matched strictly, so the first policy is the harder one.
POLICIES = {"all fields": None,
            "amounts": ["subtotal", "tax", "service_charge", "discount", "total"]}
SIGNALS = {"logprob+checks": False, "logprob+checks+agreement": True}


def doc_curve(approach: str, split: str, conf: dict[str, dict[str, float]],
              fields: list[str] | None):
    """Per document: (score, correct slots, total slots) over `fields`, invalid outputs scored 0."""
    recs = load(approach, split)
    fields = fields or EVAL_FIELDS[dataset_of(split)]
    out = []
    for i, e in examples(split).items():
        s = slots(receipt(recs.get(i)), e.gt, fields)
        out.append((document_score(conf.get(i, {}), fields), sum(x.correct for x in s), len(s)))
    return out


def coverage_curve(docs):
    docs = sorted(docs, key=lambda d: -d[0])
    cov, acc, correct, total = [], [], 0, 0
    for k, (_, c, t) in enumerate(docs, 1):
        correct, total = correct + c, total + t
        cov.append(k / len(docs))
        acc.append(correct / total if total else 1.0)
    return cov, acc


def at_threshold(docs, threshold):
    acc = [(c, t) for s, c, t in docs if s >= threshold]
    return {"coverage": len(acc) / len(docs),
            "field_accuracy": sum(c for c, _ in acc) / sum(t for _, t in acc) if acc else None}


def confidence_analysis(approach: str) -> dict | None:
    """Fit on validation, pick each policy's threshold on validation, report on test."""
    val = load(approach, "cord-validation")
    if not val or not load(approach, "cord-test"):
        return None
    if not any(r.get("field_logprobs") for r in val.values()):
        return None  # Donut's decoder scores aren't recorded, so there is nothing to calibrate
    out: dict = {"reliability": {}, "policies": {p: {} for p in POLICIES}}
    for signals, use_agree in SIGNALS.items():
        if use_agree and not load(PARTNER[approach], "cord-validation"):
            continue
        train = field_rows(approach, "cord-validation", use_agree)
        weights = fit_logistic(np.array([x for *_, x, _ in train]),
                               np.array([c for *_, c in train], dtype=float))
        model = ConfidenceModel(weights)
        conf_v: dict[str, dict[str, float]] = {}
        for i, p, x, _ in train:
            conf_v.setdefault(i, {})[p] = model.predict(x)
        conf_t: dict[str, dict[str, float]] = {}
        probs = []
        for i, p, x, c in field_rows(approach, "cord-test", use_agree):
            conf_t.setdefault(i, {})[p] = model.predict(x)
            probs.append((conf_t[i][p], c))
        out["reliability"][signals] = probs
        for policy, fields in POLICIES.items():
            val = doc_curve(approach, "cord-validation", conf_v, fields)
            threshold = choose_threshold([d[0] for d in val], [d[1:] for d in val], target=0.99)
            m = ConfidenceModel(weights, threshold, fields)
            docs = doc_curve(approach, "cord-test", conf_t, fields)
            out["policies"][policy][signals] = {"model": m.to_json(), "curve": coverage_curve(docs),
                                                "test": at_threshold(docs, threshold)}
    # Uncalibrated baseline: raw min token probability per document.
    raw: dict[str, dict[str, float]] = {}
    for i, p, x, _ in field_rows(approach, "cord-test", False):
        raw.setdefault(i, {})[p] = float(np.exp(x[0]))
    for policy, fields in POLICIES.items():
        out["policies"][policy]["raw logprob"] = {
            "curve": coverage_curve(doc_curve(approach, "cord-test", raw, fields))}
    return out


# --- figures ------------------------------------------------------------------------------

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def _axes(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def plot_coverage(conf: dict, approach: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150, sharey=True)
    for ax, (policy, curves) in zip(axes, conf["policies"].items(), strict=True):
        for color, name in zip(SERIES, ["raw logprob", *SIGNALS], strict=True):
            if name not in curves:
                continue
            cov, acc = curves[name]["curve"]
            ax.plot(cov, acc, color=color, linewidth=2, label=name)
            t = curves[name].get("test")
            if t and t["coverage"]:
                ax.plot([t["coverage"]], [t["field_accuracy"]], "o", color=color, markersize=8,
                        markeredgecolor="white", markeredgewidth=1.5)
        ax.axhline(0.99, color=MUTED, linewidth=1, linestyle="--")
        ax.text(0.99, 0.992, "99%", color=MUTED, fontsize=8, va="bottom", ha="right")
        _axes(ax, "Share of documents auto-accepted (most confident first)",
              "Field accuracy of accepted documents" if ax is axes[0] else "")
        ax.set_title(f"Accept on {policy}", color=INK, fontsize=10, loc="left")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    axes[0].set_ylim(min(0.5, axes[0].get_ylim()[0]), 1.01)
    fig.suptitle(f"Coverage vs accuracy on CORD test, {APPROACHES[approach].split('. ')[-1]}. "
                 "Dots: threshold picked on validation", color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_reliability(conf: dict, approach: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=150)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--")
    for color, (name, rows) in zip(SERIES[1:], conf["reliability"].items(), strict=False):
        p, c = map(np.array, zip(*rows, strict=True))
        bins = np.clip((p * 10).astype(int), 0, 9)
        xs, ys, ns = [], [], []
        for b in range(10):
            m = bins == b
            if m.sum() >= 5:
                xs.append(p[m].mean())
                ys.append(c[m].mean())
                ns.append(m.sum())
        ax.plot(xs, ys, "-o", color=color, linewidth=2, markersize=5, label=name)
    _axes(ax, "Predicted P(field correct)", "Observed share correct")
    ax.set_title("Field-level calibration, CORD test", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --- tables -------------------------------------------------------------------------------

def pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}"


def main() -> None:
    cfg = load_config()
    metrics: dict = {}
    for a in APPROACHES:
        test = load(a, "cord-test")
        if not test:
            continue
        m = metrics[a] = {"cord-test": score(test, "cord-test")}
        # Latency per hardware: the main run plus the timing-only reruns (-cpu, -cuda).
        m["latency"] = {}
        for name in ["cord-test", "cord-test-cpu", "cord-test-cuda"]:
            recs = load(a, name)
            if recs:
                t = score(recs, "cord-test")
                m["latency"][t["device"]] = {"mean_s": t["latency_mean_s"], "n": t["n"],
                                             "p95_s": t["latency_p95_s"],
                                             "cost_per_1k_usd": cost_per_1k(cfg, t, recs)}
        for split in ["cord-validation", "sroie-test"]:
            if load(a, split):
                m[split] = score(load(a, split), split)
        for split in ["cord-validation", "cord-test", "sroie-test"]:
            if 0 < len(load(a, split)) < len(examples(split)):
                print(f"warning: {a}/{split} is incomplete, rerun run_eval", file=sys.stderr)
        ids = sorted(examples("cord-test"))[:ROBUST_N]
        m["robustness"] = {"clean": score(test, "cord-test", ids)["f1"]}
        for c in CORRUPTIONS:
            recs = load(a, f"cord-test-{c}")
            if len(recs) >= ROBUST_N:
                m["robustness"][c] = score(recs, "cord-test", ids)["f1"]

    conf = {a: confidence_analysis(a) for a in metrics}
    conf = {a: c for a, c in conf.items() if c}
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    for a, c in conf.items():
        plot_coverage(c, a, RESULTS / "figures" / f"coverage_{a}.png")
        plot_reliability(c, a, RESULTS / "figures" / f"calibration_{a}.png")
        metrics[a]["confidence"] = {
            policy: {k: v["test"] | {"threshold": v["model"]["threshold"]}
                     for k, v in curves.items() if "model" in v}
            for policy, curves in c["policies"].items()}
    # The service runs one model per document, so it ships the single-pass signals, and the
    # amounts policy, the only one that accepts a useful share at 99% on validation.
    (RESULTS / "confidence_model.json").write_text(json.dumps(
        {a: c["policies"]["amounts"]["logprob+checks"]["model"] for a, c in conf.items()},
        indent=1))
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (RESULTS / "summary.md").write_text(summary(metrics))
    print((RESULTS / "summary.md").read_text())


def _sec(lat: dict, device: str, n_full: int) -> str:
    if device not in lat:
        return "-"
    n = "" if lat[device]["n"] == n_full else f" (n={lat[device]['n']})"
    return f"{lat[device]['mean_s']:.1f}{n}"


DEVICES = {"mps": "s/page M4 (MPS)", "cuda": "s/page T4", "cpu": "s/page CPU"}


def summary(metrics: dict) -> str:
    # a hardware column only when some approach was timed on it
    devs = [d for d in DEVICES if any(d in m["latency"] for m in metrics.values())]
    cols = ["F1", "Line-item F1", "Amounts F1", "TED acc", "JSON valid", "Checks pass",
            *(DEVICES[d] for d in devs), "$/1k pages"]
    lines = ["## CORD test (100 receipts)", "",
             "| Approach | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for a, name in APPROACHES.items():
        if a not in metrics:
            lines.append(f"| {name} | not run yet |" + "|" * (len(cols) - 1))
            continue
        t, lat = metrics[a]["cord-test"], metrics[a]["latency"]
        ted_s = pct(t.get("ted_acc")) + (f" ({pct(t['ted_acc_official'])} official)"
                                         if "ted_acc_official" in t else "")
        cost = lat[t["device"]]["cost_per_1k_usd"]
        lines.append(
            f"| {name} | {pct(t['f1'])} | {pct(t['items_f1'])} | {pct(t['header_f1'])} | {ted_s} "
            f"| {pct(t['json_valid'])} | {pct(t['checks_pass'])} "
            + "".join(f"| {_sec(lat, d, t['n'])} " for d in devs)
            + f"| {'-' if cost is None else f'{cost:.2f}'} |")
    lines += ["", f"## Robustness: F1 on the first {ROBUST_N} CORD test receipts", "",
              "| Approach | clean | " + " | ".join(CORRUPTIONS) + " |",
              "|---|---|" + "---|" * len(CORRUPTIONS)]
    for a, name in APPROACHES.items():
        if a in metrics:
            r = metrics[a]["robustness"]
            lines.append(f"| {name} | {pct(r['clean'])} | "
                         + " | ".join(pct(r.get(c)) for c in CORRUPTIONS) + " |")
    lines += ["", "## Out of distribution: SROIE (100 Malaysian receipts, merchant/date/total)",
              "", "| Approach | F1 | merchant | date | total | JSON valid |",
              "|---|---|---|---|---|---|"]
    for a, name in APPROACHES.items():
        s = metrics.get(a, {}).get("sroie-test")
        if s:
            pf = s["per_field"]
            lines.append(f"| {name} | {pct(s['f1'])} | {pct(pf['merchant'])} | {pct(pf['date'])} "
                         f"| {pct(pf['total'])} | {pct(s['json_valid'])} |")
    lines += ["", "## Auto-accept at a threshold picked on validation for 99% field accuracy",
              "", "| Approach | Accept on | Signals | Threshold | Test coverage "
              "| Test field accuracy |", "|---|---|---|---|---|---|"]
    for a, name in APPROACHES.items():
        for policy, variants in metrics.get(a, {}).get("confidence", {}).items():
            for signals, v in variants.items():
                thr = "never accepts" if v["threshold"] > 1 else f"{v['threshold']:.3f}"
                lines.append(f"| {name} | {policy} | {signals} | {thr} | {pct(v['coverage'])} "
                             f"| {pct(v['field_accuracy'])} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
