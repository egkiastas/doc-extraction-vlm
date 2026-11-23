"""One entry point for every approach: image -> prediction record. Used by the evals and the
service, so both see exactly the same parsing and confidence inputs."""

import hashlib
import json
import time
from pathlib import Path

from PIL import Image

from docextract.confidence import field_logprobs
from docextract.config import Config
from docextract.schema import parse_output


def load_extractor(name: str, cfg: Config, device: str | None = None):
    """Returns image -> dict(raw, tokens, ...), or None when the approach can't run here
    (API without a key, fine-tuned VLM without an adapter)."""
    if name == "ocr_llm":
        from docextract import ocr_llm
        from docextract.vlm import pick_device

        return ocr_llm.load(cfg, device or pick_device())
    if name in ("qwen3vl", "qwen3vl_lora"):
        from docextract.vlm import load_qwen

        if name == "qwen3vl":
            cfg.vlm_adapter = ""
        elif not cfg.vlm_adapter:
            return None
        return load_qwen(cfg, device)
    if name == "donut":
        from docextract.vlm import load_donut

        return load_donut(cfg, device)
    if name == "api":
        from docextract.vlm import load_api

        return load_api(cfg)
    if name == "replay":
        return load_replay(Path(cfg.service_replay_file))
    raise ValueError(f"unknown approach {name}")


def image_key(image: Image.Image) -> str:
    """Hash of the decoded pixels, so a re-saved PNG still finds its recording."""
    rgb = image.convert("RGB")
    return hashlib.sha256(f"{rgb.size}".encode() + rgb.tobytes()).hexdigest()


def load_replay(path: Path):
    """Serves recorded model outputs keyed by image: CI and the Docker demo exercise the whole
    service path without a model in memory. Unknown images raise KeyError."""
    recorded = {}
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        recorded[rec["key"]] = rec

    def extract(image: Image.Image) -> dict:
        rec = recorded[image_key(image)]
        return {"raw": rec["raw"], "field_logprobs": rec["field_logprobs"]}

    return extract


def predict(extract, image: Image.Image) -> dict:
    t0 = time.perf_counter()
    out = extract(image)
    latency = time.perf_counter() - t0
    receipt, error = parse_output(out["raw"])
    lps = out.pop("field_logprobs", None)
    if lps is None:
        lps = field_logprobs(out["raw"], out.pop("tokens", []))
    out.pop("tokens", None)
    return {
        **out,
        "valid": receipt is not None,
        "error": error,
        "receipt": receipt.model_dump() if receipt else None,
        "field_logprobs": lps,
        "latency_s": latency,
    }
