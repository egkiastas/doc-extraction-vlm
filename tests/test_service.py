"""API contract and an end-to-end smoke test: fixture images go through the real service path
(decode -> extractor -> validation -> checks -> confidence -> queue) with recorded model
outputs, so CI needs no model download or GPU."""

import io
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from docextract.schema import Receipt, from_cord
from evaluation.metrics import prf, slots

FIX = Path(__file__).parent / "fixtures"
EXPECTED = json.loads((FIX / "expected.json").read_text())
FIELDS = ["subtotal", "tax", "service_charge", "discount", "total", "line_items"]


@pytest.fixture(scope="module")
def api(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("service")
    env = {
        "SERVICE_EXTRACTOR": "replay",
        "SERVICE_REPLAY_FILE": str(FIX / "recorded.jsonl"),
        "SERVICE_CONFIDENCE_FILE": str(FIX / "confidence_model.json"),
        "SERVICE_DB_PATH": str(tmp / "review.db"),
        "SERVICE_CORRECTIONS_DIR": str(tmp / "corrections"),
    }
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    from service import app as app_module  # reads config at import

    yield TestClient(app_module.app), app_module
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def post(client, name: str, data: bytes | None = None, mime: str = "image/png"):
    data = data if data is not None else (FIX / name).read_bytes()
    return client.post("/extract", files={"file": (name, data, mime)})


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_extract_matches_recorded_decision(api, name):
    client, _ = api
    r = post(client, name)
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"id", "status", "receipt", "checks", "field_confidence", "score"}
    assert all(0.0 <= c <= 1.0 for c in body["field_confidence"].values())
    exp = EXPECTED[name]
    assert body["status"] == ("accepted" if exp["accept"] else "review")
    receipt = Receipt.model_validate(body["receipt"]) if body["valid"] else None
    f1 = prf(slots(receipt, from_cord(exp["gt"]), FIELDS))["f1"]
    assert f1 == pytest.approx(exp["f1"])


def test_pdf_is_rendered_and_reaches_the_extractor(api):
    client, app_module = api
    image = Image.open(FIX / sorted(EXPECTED)[0])
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PDF")
    page = app_module.load_document(buf.getvalue())
    assert page.mode == "RGB" and page.width > image.width  # rendered at 2x
    # Replay mode has no recording for the re-rendered page: a clean 422, not a crash.
    assert post(client, "r.pdf", buf.getvalue(), "application/pdf").status_code == 422


def test_rejects_non_documents(api):
    client, _ = api
    assert post(client, "notes.txt", b"hello", "text/plain").status_code == 415


def test_review_correction_becomes_a_training_example(api):
    client, app_module = api
    from service import store

    doc_id = post(client, sorted(EXPECTED)[0]).json()["id"]
    assert client.get(f"/documents/{doc_id}").json()["status"] in {"accepted", "review"}
    corrected = {"merchant": None, "date": None, "currency": None, "subtotal": "28,000",
                 "tax": None, "service_charge": None, "discount": None, "total": "28,000",
                 "line_items": [{"name": "JASMINE MT", "quantity": "1", "unit_price": None,
                                 "price": "28,000"}]}
    path = store.save_correction(app_module.conn, doc_id, corrected,
                                 app_module.cfg.service_corrections_dir)
    example = json.loads(path.read_text())
    assert (path.parent / example["image"]).exists()
    assert Receipt.model_validate(example["target"]).total == 28000
    doc = client.get(f"/documents/{doc_id}").json()
    assert doc["status"] == "corrected" and doc["correction"] == corrected
    assert doc_id not in store.pending(app_module.conn)


def test_metrics_endpoint(api):
    client, _ = api
    text = client.get("/metrics/").text
    assert "docextract_documents_total" in text and "docextract_review_queue_size" in text
