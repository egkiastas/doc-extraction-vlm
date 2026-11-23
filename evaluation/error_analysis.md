# Error analysis

Every wrong field on CORD test (100 receipts) and the SROIE sample (100 receipts) is sorted into
one category by `evaluation/error_analysis.py`. An unmatched line item counts once, not once per
field. The rules are heuristics. I hand-read 120 random failures (20 per approach and dataset,
`--sample 20`), added the three categories the rules were hiding (`qty_in_name`,
`computed_total`, `rounding`), and after that disagreed with about 1 label in 20. The leftover
disagreements are mostly column shifts filed as `item_name` (`'9.000'` read as an item name) and
a tax line ("Pb1") listed as an item.

```
uv run python -m evaluation.error_analysis [--sample 20]
```

| category | what it means | A: OCR+LLM CORD | B: Qwen3-VL CORD | Donut CORD | A: SROIE | B: SROIE | Donut SROIE |
|---|---|---|---|---|---|---|---|
| invalid_output | not valid JSON/schema, every field lost | 0 | 0 | 0 | 0 | 3 | 18 |
| ocr_lost | value never reached the OCR text | 3 | - | - | 0 | - | - |
| misread | a character or two off | 12 | 24 | 24 | 11 | 8 | 9 |
| number_format | right digits, wrong magnitude | 2 | 0 | 1 | 0 | 0 | 1 |
| rounding | total before the 5-sen rounding line | 0 | 0 | 0 | 0 | 2 | 2 |
| computed_total | subtotal + tax returned as total, tax was already included | 0 | 2 | 0 | 0 | 5 | 0 |
| wrong_field | a value from elsewhere on the receipt | 30 | 33 | 6 | 2 | 0 | 0 |
| missed_field | printed and labelled, returned null | 78 | 58 | 11 | 2 | 0 | 217 |
| unlabelled_extra | a value the labels don't have | 101 | 45 | 17 | 0 | 0 | 0 |
| item_split_merge | one item split in two, or two merged | 14 | 1 | 6 | 0 | 0 | 0 |
| item_missed | whole line item missing | 12 | 6 | 8 | 0 | 0 | 0 |
| item_extra | extra line item (modifier, payment or tax line) | 37 | 26 | 12 | 0 | 0 | 0 |
| qty_in_name | quantity folded into the name: "2 JAMUR" | 14 | 11 | 0 | 0 | 0 | 0 |
| item_name | matched item, different name | 57 | 22 | 25 | 0 | 0 | 0 |
| merchant_variant | another line chosen as merchant | 0 | 0 | 0 | 24 | 10 | 0 |
| other | | 2 | 1 | 1 | 2 | 4 | 28 |
| **wrong fields** | | **362** | **229** | **111** | **41** | **32** | **275** |

## What the counts say

**Unit price is a labelling convention, and the two general models break it in opposite
directions.** CORD labels `unitprice` only when it is printed ("2 x 10.000"). The OCR+LLM pipeline
fills it in for single-quantity items anyway, copying the line price, against a prompt rule that
says not to: 72 of its 101 `unlabelled_extra` are that. Qwen3-VL does the reverse and leaves unit
price empty even when it is printed: 47 of its 58 `missed_field`. Together that is 97 of 362
errors for A and 52 of 229 for B. Neither error would hurt a booking system, but both are real
departures from the contract. A fine-tuned model learns the convention from 800 examples, and
Donut, fine-tuned on CORD, has 13 unit-price errors in total.

**OCR+LLM loses most on item structure.** Item names (57), items split or merged across lines
(14) and extra items (37) come from the text layout: once the receipt is flattened to lines, a
name that wraps onto a second line or a modifier printed under an item looks like a new item.
`ocr_lost` is only 3, so docTR reads the characters. The errors come after OCR.

**Qwen3-VL's errors are reading errors more than layout errors.** It has almost none of A's
split/merge errors (1 vs 14) and fewer than half its name errors, but more `misread` (24, mostly names: "Subur" for
"Bubur") and `wrong_field` (33: a unit price as the line price, tax and service charge swapped).

**The 2B VLM computes when it should copy.** On SROIE, 5 of Qwen3-VL's 14 wrong totals are
subtotal + GST where the receipt says "GST @6% included in total". The model added tax that was
already in the price. The arithmetic check can't catch this, because the made-up total really is
subtotal + tax. CORD receipts add tax on top, so nothing in-distribution warns about it. The
OCR+LLM pipeline never does it. Its SROIE total F1 is 94.9 against 85.4 for
the VLM.

**Quantities leak into names for both general models** (14 and 11). Many CORD receipts print
"2 JAMUR 10.000" with no column gap. Donut, trained on these layouts, never does this.

**Donut is the best CORD model and useless out of distribution.** It has the fewest errors on
CORD (111), but on SROIE it cannot output a merchant or date at all, because CORD's label space
has neither (188 of its 217 missed fields). Six outputs don't parse and 29 have no total. It is an
in-distribution reference, not a candidate for a service that sees new layouts.

**SROIE merchant labels are ambiguous.** 24 of OCR+LLM's 41 SROIE errors and 10 of Qwen3-VL's 32
are `merchant_variant`. On reading them, most are defensible answers: the brand on top ("CALTEX")
vs the operating company (BIRU PETROLEUM), or the shop name vs the "SDN BHD" entity. Some labels
are wrong: delivery notes from a market labelled with the bakery that supplies it, and a label
typo ("KEDA PAPAN YEW CHJAN"). Merchant F1 on SROIE should be read as "agrees with one annotator's
convention".

## Label noise found on the way

Strict text matching counts these as errors although the model read the receipt correctly:
"REDBEAN BRE/D" (label typo), "Hot Chocolate Marshmallo" (label truncated), "[REG] BLACK
SAKURA" / "(TA) KWETIAW" (bracket prefixes kept in some labels, dropped in others). I did not
correct any labels: the numbers are against the published annotations.
