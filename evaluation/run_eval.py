"""Run one approach over one split and append prediction records to
results/predictions/<approach>/<split>[-<corruption>][-<device>].jsonl. The device suffix is
only added when --device is given, e.g. CPU latency runs or the T4 runs in the notebook.
Resumes where it stopped.

    python -m evaluation.run_eval --approach ocr_llm --split cord-test
    python -m evaluation.run_eval --approach qwen3vl --split cord-test --device cpu --limit 5
"""

import argparse
import json
import platform
from pathlib import Path

from docextract.config import ROOT, load_config
from docextract.data import load_split
from docextract.pipeline import image_key, load_extractor, predict
from evaluation.robustness import corrupt

PRED_DIR = ROOT / "results" / "predictions"


def output_path(approach: str, split: str, corruption: str | None, device: str | None) -> Path:
    name = split + (f"-{corruption}" if corruption else "") + (f"-{device}" if device else "")
    return PRED_DIR / approach / f"{name}.jsonl"


def run(approach: str, split: str, limit: int | None = None, corruption: str | None = None,
        device: str | None = None) -> Path:
    cfg = load_config()
    out = output_path(approach, split, corruption, device)
    done = set()
    if out.exists():
        done = {json.loads(line)["id"] for line in out.read_text().splitlines()}
    todo = [e for e in load_split(split, limit=limit, seed=cfg.seed) if e.id not in done]
    if not todo:
        print(f"{out.relative_to(ROOT)}: nothing to do")
        return out

    extract = load_extractor(approach, cfg, device)
    if extract is None:
        print(f"{approach}: not configured (no API key / adapter), skipped")
        return out
    from docextract.vlm import pick_device

    device = device or ("api" if approach == "api" else pick_device())
    out.parent.mkdir(parents=True, exist_ok=True)
    # Warm-up call so model loading and kernel compilation don't land in the first latency.
    if approach != "api":
        extract(todo[0].image)
    with out.open("a") as f:
        for n, e in enumerate(todo, 1):
            image = e.image if corruption is None else corrupt(e.image, corruption, e.id)
            rec = predict(extract, image)
            rec = {"id": e.id, "key": image_key(image), "approach": approach, "split": split,
                   "corruption": corruption, "device": device, "host": platform.machine(), **rec}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{n}/{len(todo)}] {e.id} valid={rec['valid']} {rec['latency_s']:.1f}s",
                  flush=True)
    if approach == "ocr_llm":
        from docextract.ocr_llm import unload_ollama

        unload_ollama(cfg)  # leave memory free for whichever model runs next
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--approach", required=True)
    ap.add_argument("--split", default="cord-test")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--corruption")
    ap.add_argument("--device", help="cpu | mps | cuda (default: best available)")
    args = ap.parse_args()
    run(args.approach, args.split, args.limit, args.corruption, args.device)
