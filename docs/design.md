# Design decisions

The handful of choices that shaped the results, and what they cost.

## Data: CORD for line items and amounts, SROIE as the out-of-distribution set

CORD-v2 (CC BY 4.0, 800/100/100) is the only public receipt set with line-item structure, but
its v2 labels contain no merchant, date or currency: just menu items, subtotal/tax/service/discount
and payment lines. So CORD scores line items and amounts, and the out-of-distribution set has to
cover the header fields. I picked SROIE (ICDAR 2019, Malaysian receipts, labels for company, date,
address and total) over FATURA. FATURA is CC BY 4.0 and has more invoice fields, but it is
template-generated: 50 layouts rendered 200 times each. Real scanned receipts from another
country, with dot decimals instead of Indonesian dot thousands, are the more honest test of
whether this generalises. SROIE's terms come from the competition site; the Hugging Face mirror
I download from (`jsdnrs/ICDAR2019-SROIE`) states CC BY 4.0 with attribution to the organisers.
I score a fixed-seed sample of 100 of its 361 test receipts on merchant, date and total. Nothing
from either dataset is committed except three CORD test images used as CI fixtures.

Two mapping details matter more than they look. CORD lists add-ons ("+ coconut jelly 4,000") as
sub-items of a menu line. They are printed and summed like items, so priced sub-items become line
items of their own; unpriced modifiers ("less sugar") are dropped. Without this, 7-8% of the
ground-truth receipts fail the items-sum check on their own labels; with it, 3-4%.

## Output contract: transcribe, then normalise in code

Every approach returns the same JSON with amounts as the printed strings ("46.000", "12,50"),
and `parse_amount` turns them into numbers. Indonesian receipts write 46.000 for forty-six
thousand; asking a model to emit a JSON number invites it to write 46.0. The rule (a separator
followed by exactly three digits is a thousands separator) is deterministic, tested, and wrong in
exactly one known case (a three-decimal amount, which no receipt here has).

For the OCR+LLM approach Ollama constrains decoding to the JSON schema. One surprise: Ollama
reorders the schema's properties alphabetically, so the model has to write `currency` before it
has seen any line item, and in early runs it put the total there. Field definitions in the prompt
fixed that. On 25 validation receipts, schema-constrained output scored F1 0.773 against 0.764
for free JSON mode, so I kept the constraint. The VLM gets the same prompt without grammar
constraints. Transformers has no built-in JSON grammar and adding one (outlines/xgrammar) on MPS
was more moving parts than the problem needed. Its JSON validity column shows what that cost.

## OCR engine: docTR, Tesseract as fallback

docTR is plain PyTorch, so it shares the install with the VLM, runs on MPS, and has arm64 wheels.
PaddleOCR would add a second deep-learning framework to a 16 GB laptop. On 20 validation receipts
`db_resnet50 + parseq` put 67 of 70 printed item names and prices into the OCR text, against 65
for docTR's default `fast_base + crnn`, at 0.43 s/page on MPS. Page straightening doubled the time
without helping upright scans, so it is off, and the rotation robustness numbers show what that
costs. docTR returns text grouped into blocks, which on a receipt separates item names from their
prices, so `layout_lines` rebuilds printed lines from word boxes before the text reaches the LLM.
Tesseract only runs in the slim Docker image, where pulling in PyTorch for OCR isn't worth 1 GB.

## Models: Qwen3-4B (Q4) for text, Qwen3-VL-2B for images

The limits were at most 4B parameters, one model in memory, 16 GB unified memory and a free T4 for
fine-tuning. For the text LLM, `qwen3:4b-instruct-2507-q4_K_M` is the strongest instruct model at
that size I could run under Ollama (3.2 GB loaded, ~30 tokens/s). For the VLM I compared the
current small options on paper and picked **Qwen3-VL-2B-Instruct**. Qwen3.5-2B is newer and
natively multimodal, but three of every four of its layers are Gated DeltaNet linear attention,
whose fast kernels are Triton/CUDA. On MPS it falls back to a slow PyTorch path, and fp16 training
on a T4 is a risk. SmolVLM2-2.2B is the other common pick at this size. I didn't test it, since the
disk budget allowed one VLM checkpoint and Qwen3-VL has the larger fine-tuning ecosystem.
PaddleOCR-VL is an excellent OCR model but outputs text/markdown, not fields, so it would only
replace docTR in approach A. Qwen3-VL-2B in bf16 takes 5.2 GB on MPS and decodes ~16 tokens/s.
Images are capped at 1.3 MP (1,280 visual tokens) so prompt + image + output fit in 4k context.

The service ships Qwen3-VL zero-shot (the fine-tuned version is covered below). Donut is more
accurate on CORD (F1 90.5 vs 81.8), but its output vocabulary is CORD's, with no merchant or
date, and it scores 14 F1 on SROIE. OCR+LLM is faster per page but 9 points worse on CORD and
falls apart on rotated scans.

Fine-tuning (approach C) trains a LoRA on the language model only, vision tower frozen, with the
same prompt and the same transcribe-as-printed targets. CORD has no merchant/date/currency
labels, and training on `"merchant": null` 800 times would teach the model those fields never
exist, so the loss skips those tokens (`UNLABELLED` in the notebook). Corrections from the review
queue are picked up as extra training examples if present.

The run took 200 steps on two Kaggle T4s with DDP. On a 15 GB T4 the frozen base has to be in
half precision: an fp32 base plus autocast's fp16 weight copies ran out of memory, while in fp16
the longest training receipt peaks at ~9 GB (measured on MPS). It has to be fp16, not bf16. A T4
reports bf16 as supported because it can emulate it, and emulated bf16 made the first run take
10 hours.

Why C isn't the shipped model: on CORD it is clearly the best (F1 94.1 vs 81.8 for B and 90.5
for Donut) and it holds up best under corruptions except JPEG. On SROIE it is level with B (88.1
vs 89.8, within noise), and most of its extra merchant errors are registration numbers copied
along with the name. So on accuracy alone I'd pick C for receipts like CORD's. B stays the
default for two reasons. First, weights aren't committed, so a fresh clone can run B but not C.
Second, C's auto-accept threshold doesn't hold. Picked on validation for 99%, it accepts 89% of
test receipts on amounts, and those are 95.5% correct. C's validation amounts are easier than its
test amounts (F1 97.8 vs 95.3), so the threshold lands too low. B's threshold held on test: 17%
accepted, none wrong.

Shipping C would mean publishing the adapter (on the Hugging Face Hub, say), setting its threshold
on a larger held-out set or with a safety margin, and re-recording the CI fixtures from its
outputs.

## Metrics

Every (document, field) where either side has a value is a slot. A wrong value counts as a false
positive and a false negative, so P/R/F1 punish both hallucinated and missed fields. Line items
are paired with Hungarian matching on how many of their fields agree plus name similarity.
Unrelated pairs are refused, so a missing item costs recall instead of being matched to garbage.
Text is compared after upper-casing and stripping punctuation, amounts after parsing. That is
strict: "S-Ovaltine" vs "S-Ovaltine 50%" is wrong.

I also report CORD's tree-edit-distance accuracy in two forms. The *official* one runs Donut's
evaluator on Donut's raw output tree: 90.2 on the 100 test receipts, against about 91 in the Donut
paper, which checks my re-implementation. The *schema-level* one maps both
sides through `to_cord(Receipt)`, so every approach is scored on the same normalised tree. Cost
per 1,000 pages is measured latency times an assumed hourly price for the hardware (in
`config.toml`), stated as an assumption, not a quote.

## Confidence and the review policy

Three per-field signals:

- the lowest token log-probability inside the field's value (from Ollama or `generate` logits),
- whether the arithmetic checks that involve the field pass or fail,
- optionally, whether a second, independent extractor agrees.

A logistic regression fitted on the CORD validation split maps them to P(field correct). The
document score is the minimum over its fields, because a document with one wrong total is a wrong
document. The auto-accept threshold is the lowest document score at which the validation
documents above it reach 99% field accuracy, and the README reports what that threshold does on
the untouched test split. Choosing it on test would make the headline number a fit, not a result.

What an accepted document promises was the harder decision. With every field in scope, no
threshold reaches 99% on validation for A or B on its own. Even B's validation documents whose
checks all pass are only ~85% field-correct, and the misses are mostly unit price (CORD
labels it only when printed) and item-name spelling under strict matching. So the service accepts
on the document amounts (subtotal, tax, service charge, discount, total), the numbers that get
booked, and an accepted document's item lines are not guaranteed. The policy is stored with the
model (`accept_on` in `confidence_model.json`). The report shows both policies.

The service ships the single-model variant (log-prob + checks). Agreement needs a second model per
document, which doubles cost and, on this laptop, breaks the one-model-in-memory rule; the report
shows both. Merchant/date confidences in the service use the same model, but CORD has no labels
for them, so they are uncalibrated.
