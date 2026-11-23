"""Download and load CORD-v2 and SROIE. Data lives in data/ and is never committed.

    python -m docextract.data            # CORD validation+test, SROIE test (~670 MB)
    python -m docextract.data --train    # also CORD train (~1.8 GB), only needed for fine-tuning
"""

import argparse
import io
import json
import random
from dataclasses import dataclass

from PIL import Image

from docextract.config import ROOT
from docextract.schema import Receipt, from_cord

DATA = ROOT / "data"

FILES = {
    "cord-train": ("naver-clova-ix/cord-v2", [f"data/train-0000{i}-of-00004-{h}.parquet" for i, h in
                   enumerate(["b4aaeceff1d90ecb", "7dbbe248962764c5", "688fe1305a55e5cc",
                              "2d0cd200555ed7fd"])]),
    "cord-validation": ("naver-clova-ix/cord-v2",
                        ["data/validation-00000-of-00001-cc3c5779fe22e8ca.parquet"]),
    "cord-test": ("naver-clova-ix/cord-v2", ["data/test-00000-of-00001-9c204eb3f4e11791.parquet"]),
    "sroie-test": ("jsdnrs/ICDAR2019-SROIE", ["data/test-00000-of-00001.parquet"]),
}

# Fields that each dataset actually labels; everything else is ignored when scoring.
EVAL_FIELDS = {
    "cord": ["subtotal", "tax", "service_charge", "discount", "total", "line_items"],
    "sroie": ["merchant", "date", "total"],
}
SROIE_SAMPLE = 100  # of 361 test receipts, fixed seed; enough to see the OOD gap on a laptop


@dataclass
class Example:
    id: str
    image_bytes: bytes
    gt: Receipt
    cord: dict | None = None  # raw CORD gt_parse, for the official tree-edit-distance score

    @property
    def image(self) -> Image.Image:
        return Image.open(io.BytesIO(self.image_bytes)).convert("RGB")


def dataset_of(split: str) -> str:
    return split.split("-")[0]


def download(splits: list[str]) -> None:
    from huggingface_hub import hf_hub_download

    for split in splits:
        repo, files = FILES[split]
        for f in files:
            hf_hub_download(repo, f, repo_type="dataset", local_dir=DATA / dataset_of(split))


def load_split(split: str, limit: int | None = None, seed: int = 0) -> list[Example]:
    import pyarrow.parquet as pq

    repo, files = FILES[split]
    rows = []
    for f in files:
        path = DATA / dataset_of(split) / f
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run `python -m docextract.data` first")
        rows += pq.read_table(path).to_pylist()

    examples = []
    if dataset_of(split) == "cord":
        for i, row in enumerate(rows):
            gt = json.loads(row["ground_truth"])["gt_parse"]
            examples.append(Example(f"{split}-{i:03d}", row["image"]["bytes"], from_cord(gt), gt))
    else:
        rows = random.Random(seed).sample(rows, SROIE_SAMPLE)
        for row in rows:
            e = row["entities"]
            gt = Receipt(merchant=e["company"], date=e["date"], total=e["total"])
            examples.append(Example(row["key"], row["image"]["bytes"], gt))
    return examples[:limit]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    args = ap.parse_args()
    splits = ["cord-validation", "cord-test", "sroie-test"]
    download(splits + (["cord-train"] if args.train else []))
