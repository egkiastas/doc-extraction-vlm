"""Image-in approaches: Qwen3-VL (zero-shot or with a LoRA adapter), Donut (the classic
CORD baseline) and an optional OpenAI-compatible API model."""

import base64
import io
import json
import math
import re

import httpx
from PIL import Image

from docextract.config import Config
from docextract.schema import EXTRACTION_SCHEMA, PROMPT, cord_to_raw


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _dtype(device: str):
    """bf16, except on CUDA GPUs older than Ampere (the T4), which get fp16. Don't use
    torch.cuda.is_bf16_supported() here: it is True on a T4 because bf16 can be emulated there,
    and emulated bf16 made T4 training and inference run about 10x slower than fp16."""
    import torch

    if device == "cuda" and torch.cuda.get_device_capability()[0] < 8:
        return torch.float16
    return torch.bfloat16


def shrink(image: Image.Image, max_pixels: int) -> Image.Image:
    """Cap the pixel count (and so the number of visual tokens) keeping the aspect ratio.
    Some CORD scans are 12 MP; at 32x32 px per token that would not fit a 4k context."""
    w, h = image.size
    if w * h <= max_pixels:
        return image
    s = math.sqrt(max_pixels / (w * h))
    return image.resize((max(32, int(w * s)), max(32, int(h * s))), Image.Resampling.BICUBIC)


def token_pieces(tokenizer, ids: list[int]) -> list[str]:
    """Text contributed by each generated token, such that the pieces join to the decoded
    text. Decoding prefixes handles byte-level tokens that split a UTF-8 character."""
    pieces, prev = [], ""
    for i in range(len(ids)):
        cur = tokenizer.decode(ids[: i + 1], skip_special_tokens=True)
        if cur.endswith("�"):
            pieces.append("")
            continue
        pieces.append(cur[len(prev):])
        prev = cur
    return pieces


def load_qwen(cfg: Config, device: str | None = None):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = device or pick_device()
    processor = AutoProcessor.from_pretrained(cfg.vlm_model)
    model = AutoModelForImageTextToText.from_pretrained(cfg.vlm_model, dtype=_dtype(device))
    if cfg.vlm_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, cfg.vlm_adapter).merge_and_unload()
    model = model.to(device).eval()

    def extract(image: Image.Image) -> dict:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": shrink(image, cfg.vlm_max_pixels)},
            {"type": "text", "text": PROMPT},
        ]}]
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt").to(device)
        n_in = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=cfg.vlm_max_new_tokens, do_sample=False,
                                 output_logits=True, return_dict_in_generate=True)
        ids = out.sequences[0, n_in:].tolist()
        logprobs = [torch.log_softmax(step[0].float(), -1)[t].item()
                    for step, t in zip(out.logits, ids, strict=True)]
        pieces = token_pieces(processor.tokenizer, ids)
        keep = [(p, lp) for p, lp in zip(pieces, logprobs, strict=True) if p]
        return {"raw": "".join(p for p, _ in keep), "tokens": keep,
                "usage": {"input_tokens": n_in, "output_tokens": len(ids)}}

    return extract


def load_donut(cfg: Config, device: str | None = None):
    import torch
    from transformers import DonutProcessor, VisionEncoderDecoderModel

    device = device or pick_device()
    processor = DonutProcessor.from_pretrained(cfg.donut_model)
    model = VisionEncoderDecoderModel.from_pretrained(cfg.donut_model).to(device).eval()
    tok = processor.tokenizer
    prompt_ids = tok("<s_cord-v2>", add_special_tokens=False, return_tensors="pt").input_ids

    def extract(image: Image.Image) -> dict:
        pixels = processor(image, return_tensors="pt").pixel_values.to(device)
        with torch.inference_mode():
            seq = model.generate(pixels, decoder_input_ids=prompt_ids.to(device),
                                 max_length=model.decoder.config.max_position_embeddings,
                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                                 bad_words_ids=[[tok.unk_token_id]], use_cache=True)
        text = tok.batch_decode(seq)[0].replace(tok.eos_token, "").replace(tok.pad_token, "")
        cord = processor.token2json(re.sub(r"<.*?>", "", text, count=1).strip())
        cord = cord if isinstance(cord, dict) else {}
        # Donut emits CORD's own label tree; map it through the same function as the labels.
        return {"raw": json.dumps(cord_to_raw(cord), ensure_ascii=False), "tokens": [],
                "cord": cord}

    return extract


def load_api(cfg: Config):
    """Approach D. Returns None when no key is configured, so callers can skip it."""
    if not cfg.api_key:
        return None

    def extract(image: Image.Image) -> dict:
        buf = io.BytesIO()
        shrink(image.convert("RGB"), cfg.vlm_max_pixels).save(buf, format="JPEG", quality=90)
        url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        r = httpx.post(f"{cfg.api_base_url}/chat/completions", timeout=300,
                       headers={"Authorization": f"Bearer {cfg.api_key}"},
                       json={"model": cfg.api_model,
                             "messages": [{"role": "user", "content": [
                                 {"type": "text", "text": PROMPT},
                                 {"type": "image_url", "image_url": {"url": url}}]}],
                             "response_format": {"type": "json_schema", "json_schema": {
                                 "name": "receipt", "schema": EXTRACTION_SCHEMA}}})
        r.raise_for_status()
        body = r.json()
        usage = body.get("usage", {})
        return {"raw": body["choices"][0]["message"]["content"], "tokens": [],
                "usage": {"input_tokens": usage.get("prompt_tokens", 0),
                          "output_tokens": usage.get("completion_tokens", 0)}}

    return extract

