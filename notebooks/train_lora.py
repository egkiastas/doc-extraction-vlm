"""LoRA fine-tuning of Qwen3-VL-2B on the CORD training split plus any review-queue corrections.
The notebook launches it with torchrun, one process per GPU (DDP). It also runs as a single
process, and on CPU with --dry-run (tiny random model, 2 steps) to check the code path.

    torchrun --nproc_per_node 2 notebooks/train_lora.py --work /kaggle/working
"""

import argparse
import io
import json
import os
import time
from pathlib import Path

# set before torch touches CUDA: long receipts vary in length, which fragments the allocator
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint  # noqa: E402

from docextract.config import ROOT, load_config  # noqa: E402
from docextract.data import load_split  # noqa: E402
from docextract.schema import PROMPT, cord_to_raw  # noqa: E402
from docextract.vlm import _dtype, shrink  # noqa: E402

TINY = "trl-internal-testing/tiny-Qwen3VLForConditionalGeneration"
# CORD has no merchant/date/currency labels. Training on "null" for them would teach the model
# that those fields never exist (and wreck them on other receipts), so their tokens get no loss.
UNLABELLED = '"merchant": null, "date": null, "currency": null, '


def encode(processor, image_bytes: bytes, target: dict, max_pixels: int) -> dict:
    """Prompt + image + target answer as one sequence; loss only on the answer tokens."""
    tok = processor.tokenizer
    image = shrink(Image.open(io.BytesIO(image_bytes)).convert("RGB"), max_pixels)
    msgs = [{"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text", "text": PROMPT}]}]
    enc = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                        return_dict=True, return_tensors="pt")
    answer = json.dumps(target, ensure_ascii=False) + "<|im_end|>\n"
    ans = tok(answer, add_special_tokens=False, return_offsets_mapping=True)
    masked = (1, 1 + len(UNLABELLED)) if answer.startswith("{" + UNLABELLED) else (0, 0)
    labels = [-100 if masked[0] <= s and e <= masked[1] else t
              for t, (s, e) in zip(ans.input_ids, ans.offset_mapping, strict=True)]
    n_prompt, n_ans = enc["input_ids"].shape[1], len(ans.input_ids)
    batch = dict(enc)
    batch["input_ids"] = torch.cat([enc["input_ids"], torch.tensor([ans.input_ids])], 1)
    batch["attention_mask"] = torch.ones_like(batch["input_ids"])
    if "mm_token_type_ids" in enc:
        batch["mm_token_type_ids"] = torch.cat(
            [enc["mm_token_type_ids"], torch.zeros(1, n_ans, dtype=torch.long)], 1)
    batch["labels"] = torch.tensor([[-100] * n_prompt + labels])
    return batch


class Receipts(torch.utils.data.Dataset):
    def __init__(self, processor, examples: list[tuple[bytes, dict]], max_pixels: int):
        self.processor, self.examples, self.max_pixels = processor, examples, max_pixels

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return encode(self.processor, *self.examples[i], self.max_pixels)


def training_examples(dry_run: bool) -> list[tuple[bytes, dict]]:
    rows = load_split("cord-test" if dry_run else "cord-train", limit=4 if dry_run else None)
    out = [(e.image_bytes, cord_to_raw(e.cord)) for e in rows]
    # Corrections saved by the review UI are labelled examples too.
    for f in sorted((ROOT / "data" / "corrections").glob("*.json")):
        d = json.loads(f.read_text())
        out.append(((f.parent / d["image"]).read_bytes(), d["target"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True, help="checkpoints and adapter go here")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=8, help="effective batch, summed over GPUs")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--log-dir", type=Path, default=ROOT / "results" / "finetune")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    world = int(os.environ.get("WORLD_SIZE", "1"))
    main_process = int(os.environ.get("RANK", "0")) == 0
    ckpt, adapter = args.work / "ckpt", args.work / "adapter"
    cfg = load_config()
    model_id = TINY if args.dry_run else cfg.vlm_model
    processor = AutoProcessor.from_pretrained(model_id)
    train = training_examples(args.dry_run)
    data = Receipts(processor, train, cfg.vlm_max_pixels)
    if main_process:
        b = data[0]
        print(len(train), "training examples |", {k: tuple(v.shape) for k, v in b.items()},
              "| trained answer tokens:", int((b["labels"] != -100).sum()), flush=True)

    cuda = torch.cuda.is_available() and not args.dry_run
    # The frozen base is loaded in half precision, the dtype evaluation uses on the same GPU
    # (fp16 on a T4, which has no bf16 tensor cores). peft keeps the LoRA weights in fp32. An
    # fp32 base ran out of memory on a 15 GB T4: 8.6 GB of weights plus the fp16 copies autocast
    # caches. fp16 base: ~9 GB peak on the longest training receipt (2,327 tokens).
    half = _dtype("cuda") if cuda else torch.float32
    bf16 = half == torch.bfloat16
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=half)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
        # language model only; the vision tower stays frozen
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"))
    if main_process:
        model.print_trainable_parameters()
    targs = TrainingArguments(
        output_dir=str(ckpt), per_device_train_batch_size=1,
        gradient_accumulation_steps=max(1, args.batch // world),
        num_train_epochs=args.epochs, max_steps=2 if args.dry_run else -1, learning_rate=args.lr,
        lr_scheduler_type="cosine", warmup_steps=0.03, weight_decay=0.0,
        fp16=cuda and not bf16, bf16=bf16,
        activation_checkpointing=cuda, activation_checkpointing_kwargs={"use_reentrant": False},
        # every LoRA weight gets a gradient each step; the frozen vision tower isn't in DDP at all
        ddp_find_unused_parameters=False,
        logging_steps=1 if args.dry_run else 5, save_steps=25, save_total_limit=2,
        report_to="none", remove_unused_columns=False,
        dataloader_num_workers=0 if args.dry_run else 2, seed=0, use_cpu=args.dry_run)
    trainer = Trainer(model=model, args=targs, train_dataset=data,
                      data_collator=lambda rows: rows[0])
    t0 = time.time()
    trainer.train(resume_from_checkpoint=get_last_checkpoint(str(ckpt)) if ckpt.exists() else None)
    if trainer.is_world_process_zero():
        model.save_pretrained(adapter)
        args.log_dir.mkdir(parents=True, exist_ok=True)
        (args.log_dir / "qwen3vl_lora_log.json").write_text(json.dumps(
            {"train_seconds": time.time() - t0, "gpus": world, "epochs": args.epochs,
             "lr": args.lr, "effective_batch": world * targs.gradient_accumulation_steps,
             "lora_r": args.lora_r, "examples": len(train),
             "log_history": trainer.state.log_history}, indent=1))
        print(f"training took {(time.time() - t0) / 60:.1f} min on {world} process(es)")


if __name__ == "__main__":
    main()
