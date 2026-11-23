"""POST /extract: image or PDF -> extraction -> schema validation -> consistency checks ->
confidence -> auto-accept or review queue. The review UI is mounted at /review.

    uvicorn service.app:app --port 8000
"""

import io
import json
from functools import cache
from pathlib import Path

import gradio as gr
from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

from docextract.confidence import FEATURES, ConfidenceModel, assess
from docextract.config import ROOT, load_config
from docextract.pipeline import load_extractor, predict
from docextract.schema import Receipt
from service import store
from service.review_ui import build_ui

DOCS = Counter("docextract_documents_total", "Processed documents", ["decision"])
INVALID = Counter("docextract_invalid_output_total", "Model outputs that failed schema validation")
LATENCY = Histogram("docextract_extract_seconds", "Model time per document",
                    buckets=[0.5, 1, 2, 5, 10, 20, 40, 80])
QUEUE = Gauge("docextract_review_queue_size", "Documents waiting for review")
CORRECTIONS = Counter("docextract_corrections_total", "Reviewed documents saved as examples")

cfg = load_config()
conn = store.connect(cfg.service_db_path)
QUEUE.set(len(store.pending(conn)))


def approach_name() -> str:
    if cfg.service_extractor != "replay":
        return cfg.service_extractor
    first = Path(ROOT / cfg.service_replay_file).read_text().splitlines()[0]
    return json.loads(first)["approach"]


@cache
def extractor():
    if cfg.service_extractor == "replay":
        cfg.service_replay_file = str(ROOT / cfg.service_replay_file)
    ext = load_extractor(cfg.service_extractor, cfg)
    if ext is None:
        raise RuntimeError(f"{cfg.service_extractor} can't run here: set VLM_ADAPTER or API_KEY")
    return ext


@cache
def confidence_model() -> ConfidenceModel:
    """Fitted by evaluation/report.py on CORD validation. Without it nothing is auto-accepted."""
    path = ROOT / cfg.service_confidence_file
    models = json.loads(path.read_text()) if path.exists() else {}
    m = models.get(approach_name())
    if m is None:
        return ConfidenceModel([0.0] * (len(FEATURES) + 1), threshold=1.1)
    return ConfidenceModel.from_json(m)


def load_document(data: bytes) -> Image.Image:
    """First page of a PDF (rendered at 2x, ~144 dpi) or any image Pillow can read."""
    if data[:5] == b"%PDF-":
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(data)
        return pdf[0].render(scale=2).to_pil().convert("RGB")
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(415, f"not an image or PDF: {e}") from e


app = FastAPI(title="docextract")
app.mount("/metrics", make_asgi_app())


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "extractor": cfg.service_extractor}


@app.post("/extract")
async def extract(file: UploadFile) -> dict:
    image = load_document(await file.read())
    try:
        rec = predict(extractor(), image)
    except KeyError as e:  # replay mode, unknown image
        raise HTTPException(422, "no recorded output for this image (replay mode)") from e
    LATENCY.observe(rec["latency_s"])
    if not rec["valid"]:
        INVALID.inc()
    receipt = Receipt.model_validate(rec["receipt"]) if rec["valid"] else None
    a = assess(receipt, rec["field_logprobs"], confidence_model())
    png = io.BytesIO()
    image.save(png, format="PNG")
    doc_id = store.add(conn, png.getvalue(), rec, a)
    decision = "accepted" if a["accept"] else "review"
    DOCS.labels(decision).inc()
    QUEUE.set(len(store.pending(conn)))
    return {
        "id": doc_id,
        "status": decision,
        "receipt": rec["receipt"],
        "valid": rec["valid"],
        "error": rec["error"],
        "checks": a["checks"],
        "field_confidence": a["field_confidence"],
        "score": a["score"],
        "latency_s": rec["latency_s"],
    }


@app.get("/documents/{doc_id}")
def document(doc_id: str) -> dict:
    doc = store.get(conn, doc_id)
    if doc is None:
        raise HTTPException(404, "unknown document")
    doc.pop("image")
    return doc


def _on_correction() -> None:
    CORRECTIONS.inc()
    QUEUE.set(len(store.pending(conn)))


app = gr.mount_gradio_app(app, build_ui(conn, cfg.service_corrections_dir, _on_correction),
                          path="/review")
