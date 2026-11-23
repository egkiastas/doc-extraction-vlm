## CORD test (100 receipts)

| Approach | F1 | Line-item F1 | Amounts F1 | TED acc | JSON valid | Checks pass | s/page M4 (MPS) | s/page T4 | s/page CPU | $/1k pages |
|---|---|---|---|---|---|---|---|---|---|---|
| A. docTR OCR + Qwen3-4B (Ollama, Q4) | 73.0 | 70.5 | 82.7 | 69.1 | 100.0 | 54.1 | 7.5 | - | 12.5 (n=10) | 0.42 |
| B. Qwen3-VL-2B zero-shot | 81.8 | 81.9 | 81.4 | 82.8 | 100.0 | 69.0 | 10.9 | - | 205.9 (n=5) | 0.61 |
| C. Qwen3-VL-2B + LoRA on CORD | not run yet ||||||||||
| Donut (CORD fine-tuned, reference) | 90.5 | 89.1 | 95.5 | 91.5 (90.2 official) | 100.0 | 82.0 | 1.1 | - | 1.5 (n=10) | 0.06 |
| D. API model | not run yet ||||||||||

## Robustness: F1 on the first 40 CORD test receipts

| Approach | clean | rotate | blur | lowres | jpeg |
|---|---|---|---|---|---|
| A. docTR OCR + Qwen3-4B (Ollama, Q4) | 70.2 | 40.0 | 58.4 | 63.0 | 58.9 |
| B. Qwen3-VL-2B zero-shot | 80.9 | 76.2 | 79.2 | 77.8 | 71.3 |
| Donut (CORD fine-tuned, reference) | 89.2 | 86.7 | 73.4 | 79.1 | 80.6 |

## Out of distribution: SROIE (100 Malaysian receipts, merchant/date/total)

| Approach | F1 | merchant | date | total | JSON valid |
|---|---|---|---|---|---|
| A. docTR OCR + Qwen3-4B (Ollama, Q4) | 86.6 | 69.0 | 96.0 | 94.9 | 100.0 |
| B. Qwen3-VL-2B zero-shot | 89.8 | 85.4 | 98.5 | 85.4 | 99.0 |
| Donut (CORD fine-tuned, reference) | 13.7 | 0.0 | 0.0 | 30.3 | 94.0 |

## Auto-accept at a threshold picked on validation for 99% field accuracy

| Approach | Accept on | Signals | Threshold | Test coverage | Test field accuracy |
|---|---|---|---|---|---|
| A. docTR OCR + Qwen3-4B (Ollama, Q4) | all fields | logprob+checks | never accepts | 0.0 | - |
| A. docTR OCR + Qwen3-4B (Ollama, Q4) | all fields | logprob+checks+agreement | never accepts | 0.0 | - |
| A. docTR OCR + Qwen3-4B (Ollama, Q4) | amounts | logprob+checks | never accepts | 0.0 | - |
| A. docTR OCR + Qwen3-4B (Ollama, Q4) | amounts | logprob+checks+agreement | 0.972 | 18.0 | 95.8 |
| B. Qwen3-VL-2B zero-shot | all fields | logprob+checks | never accepts | 0.0 | - |
| B. Qwen3-VL-2B zero-shot | all fields | logprob+checks+agreement | 0.854 | 7.0 | 93.8 |
| B. Qwen3-VL-2B zero-shot | amounts | logprob+checks | 0.956 | 17.0 | 100.0 |
| B. Qwen3-VL-2B zero-shot | amounts | logprob+checks+agreement | 0.969 | 11.0 | 100.0 |
