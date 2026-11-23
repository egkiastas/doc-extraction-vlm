"""Image corruptions for the robustness evals, sized relative to the image so a 600 px and a
3000 px scan are degraded comparably. Deterministic per document id.

    python -m evaluation.robustness --approach ocr_llm --limit 40   # all corruptions
"""

import argparse
import io
import random
import zlib

from PIL import Image, ImageFilter

CORRUPTIONS = ["rotate", "blur", "lowres", "jpeg"]


def corrupt(image: Image.Image, name: str, doc_id: str) -> Image.Image:
    rng = random.Random(zlib.crc32(f"{name}:{doc_id}".encode()))
    w, h = image.size
    if name == "rotate":
        # A phone photo held slightly askew, not a sideways scan.
        angle = rng.choice([-1, 1]) * rng.uniform(5, 10)
        return image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True,
                            fillcolor=(255, 255, 255))
    if name == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=max(w, h) / 600))
    if name == "lowres":
        small = image.resize((max(1, w // 3), max(1, h // 3)), Image.Resampling.BILINEAR)
        return small.resize((w, h), Image.Resampling.BILINEAR)
    if name == "jpeg":
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=10)
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    raise ValueError(name)


if __name__ == "__main__":
    from evaluation.run_eval import run

    ap = argparse.ArgumentParser()
    ap.add_argument("--approach", required=True)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--device")
    args = ap.parse_args()
    for c in CORRUPTIONS:
        run(args.approach, "cord-test", limit=args.limit, corruption=c, device=args.device)
