"""Gradio review UI: the document next to editable fields, lowest-confidence fields listed first.
Fields are prefilled from the model's raw output (values as printed), so a saved correction has
exactly the shape of a training target."""

import io
import json

import gradio as gr
from PIL import Image

from docextract.schema import ITEM_FIELDS, SCALAR_FIELDS
from service import store


def _raw_dict(raw: str) -> dict:
    try:
        d = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        return d if isinstance(d, dict) else {}
    except json.JSONDecodeError:
        return {}


def _summary(doc: dict, n_pending: int) -> str:
    a = doc["assessment"]
    checks = ", ".join(f"{k}: {'pass' if v else 'FAIL'}" for k, v in a["checks"].items()
                       if v is not None) or "none applicable"
    low = sorted(a["field_confidence"].items(), key=lambda kv: kv[1])[:6]
    lows = "\n".join(f"- `{p}` {c:.0%}" for p, c in low)
    error = f"\n\n**Model output invalid:** {doc['prediction']['error']}" if doc["prediction"][
        "error"] else ""
    return (f"**{doc['id']}** ({n_pending} waiting) | document score {a['score']:.2f} | "
            f"checks: {checks}{error}\n\nLeast confident fields:\n{lows}")


def build_ui(conn, corrections_dir: str, on_correction=lambda: None) -> gr.Blocks:
    def show(doc_id: str | None):
        doc = store.get(conn, doc_id) if doc_id else None
        if doc is None:
            empty = [gr.update(value=None)] * (len(SCALAR_FIELDS) + 2)
            return [None, "Nothing to review.", *empty]
        raw = _raw_dict(doc["prediction"]["raw"])
        items = [[str(it.get(k) or "") for k in ITEM_FIELDS] for it in raw.get("line_items") or []
                 if isinstance(it, dict)]
        image = Image.open(io.BytesIO(doc["image"]))
        return [doc_id, _summary(doc, len(store.pending(conn))), image,
                *[str(raw.get(f) or "") for f in SCALAR_FIELDS], items or [[""] * len(ITEM_FIELDS)]]

    def first_pending():
        q = store.pending(conn)
        return show(q[0] if q else None)

    def skip(current: str | None):
        q = store.pending(conn)
        if not q:
            return show(None)
        return show(q[(q.index(current) + 1) % len(q)] if current in q else q[0])

    def save(doc_id, *values):
        if doc_id:
            *scalars, items = values
            receipt = {f: (v.strip() or None) for f, v in zip(SCALAR_FIELDS, scalars, strict=True)}
            receipt["line_items"] = [
                {k: (str(v).strip() or None) for k, v in zip(ITEM_FIELDS, row, strict=True)}
                for row in (items or []) if any(str(v).strip() for v in row)]
            store.save_correction(conn, doc_id, receipt, corrections_dir)
            on_correction()
        return first_pending()

    with gr.Blocks(title="Receipt review") as ui:
        doc_id = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="pil", label="Document", height=780, interactive=False)
            with gr.Column(scale=1):
                summary = gr.Markdown()
                with gr.Row():
                    fields = [gr.Textbox(label=f) for f in SCALAR_FIELDS[:3]]
                with gr.Row():
                    fields += [gr.Textbox(label=f, min_width=90) for f in SCALAR_FIELDS[3:]]
                items = gr.Dataframe(headers=ITEM_FIELDS, type="array", interactive=True,
                                     label="Line items (values as printed)")
                with gr.Row():
                    save_btn = gr.Button("Save correction", variant="primary")
                    refresh = gr.Button("Skip")
        outputs = [doc_id, summary, image, *fields, items]
        ui.load(first_pending, outputs=outputs)
        refresh.click(skip, inputs=[doc_id], outputs=outputs)
        save_btn.click(save, inputs=[doc_id, *fields, items], outputs=outputs)
    return ui
