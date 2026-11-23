"""Approach A: OCR (docTR, Tesseract fallback) -> text laid out line by line -> local LLM via
Ollama, constrained to EXTRACTION_SCHEMA."""

import time
from dataclasses import dataclass

import httpx
import numpy as np
from PIL import Image

from docextract.config import Config
from docextract.schema import EXTRACTION_SCHEMA, PROMPT


@dataclass
class Word:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float  # relative coordinates, 0..1


def load_ocr(device: str = "cpu"):
    """Returns image -> list[Word]. docTR when installed, else Tesseract (used in the slim
    Docker image, where pulling in PyTorch for OCR isn't worth 1 GB)."""
    try:
        from doctr.models import ocr_predictor
    except ImportError:
        return _tesseract

    import torch

    # Picked on 20 validation receipts: finds 67/70 printed item names/prices vs 65/70 for the
    # default fast_base+crnn. Page straightening doubled the time without helping upright scans.
    predictor = ocr_predictor(det_arch="db_resnet50", reco_arch="parseq", pretrained=True)
    if device != "cpu":
        predictor = predictor.to(torch.device(device))

    def run(image: Image.Image) -> list[Word]:
        page = predictor([np.asarray(image.convert("RGB"))]).pages[0]
        return [Word(w.value, *w.geometry[0], *w.geometry[1])
                for b in page.blocks for line in b.lines for w in line.words]

    return run


def _tesseract(image: Image.Image) -> list[Word]:
    import pytesseract

    w, h = image.size
    d = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    return [Word(t, x / w, y / h, (x + bw) / w, (y + bh) / h)
            for t, x, y, bw, bh, c in zip(d["text"], d["left"], d["top"], d["width"], d["height"],
                                          d["conf"], strict=True)
            if t.strip() and float(c) >= 0]


def layout_lines(words: list[Word]) -> str:
    """Rebuild printed lines from word boxes. OCR engines group text into blocks/columns, which
    on a receipt puts item names and their prices far apart; joining words whose vertical
    centres are close keeps "Coffee  2  30.000" on one line."""
    lines: list[dict] = []
    for w in sorted(words, key=lambda w: (w.y0 + w.y1) / 2):
        yc, h = (w.y0 + w.y1) / 2, w.y1 - w.y0
        if lines and abs(lines[-1]["yc"] - yc) < 0.5 * min(h, lines[-1]["h"]):
            lines[-1]["words"].append(w)
        else:
            lines.append({"yc": yc, "h": h, "words": [w]})
    return "\n".join(" ".join(w.text for w in sorted(ln["words"], key=lambda w: w.x0))
                     for ln in lines)


def call_ollama(cfg: Config, text: str, cpu_only: bool = False) -> tuple[str, list]:
    """Returns the JSON text and its tokens as (piece, logprob)."""
    prompt = f"{PROMPT}\n\nReceipt text (OCR, one printed line per line):\n{text}"
    options = {"temperature": 0, "seed": cfg.seed, "num_ctx": cfg.ollama_num_ctx}
    if cpu_only:
        options["num_gpu"] = 0
    r = httpx.post(f"{cfg.ollama_url}/api/chat", timeout=600, json={
        "model": cfg.ollama_model,
        "messages": [{"role": "user", "content": prompt}],
        "format": EXTRACTION_SCHEMA,
        "stream": False,
        "logprobs": True,
        "options": options,
    })
    r.raise_for_status()
    body = r.json()
    tokens = [(t["token"], t["logprob"]) for t in body.get("logprobs") or []]
    return body["message"]["content"], tokens


def unload_ollama(cfg: Config) -> None:
    httpx.post(f"{cfg.ollama_url}/api/generate", json={"model": cfg.ollama_model, "keep_alive": 0},
               timeout=60)


def load(cfg: Config, device: str = "cpu"):
    ocr = load_ocr(device)

    def extract(image: Image.Image) -> dict:
        t0 = time.perf_counter()
        text = layout_lines(ocr(image))
        t1 = time.perf_counter()
        raw, tokens = call_ollama(cfg, text, cpu_only=device == "cpu")
        return {"raw": raw, "tokens": tokens, "ocr_text": text,
                "timing": {"ocr_s": t1 - t0, "llm_s": time.perf_counter() - t1}}

    return extract
