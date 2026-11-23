RUN = uv run python -m

setup:
	uv sync --all-extras

data:
	$(RUN) docextract.data

# Everything that runs on the laptop, one model at a time (needs `ollama serve`).
# Resumable: rerunning skips documents that already have a prediction.
eval:
	for a in ocr_llm qwen3vl donut; do \
	  for s in cord-validation cord-test sroie-test; do $(RUN) evaluation.run_eval --approach $$a --split $$s; done; \
	  $(RUN) evaluation.robustness --approach $$a --limit 40; \
	done
	$(RUN) evaluation.run_eval --approach donut --device cpu --limit 10
	$(RUN) evaluation.run_eval --approach ocr_llm --device cpu --limit 10
	$(RUN) evaluation.run_eval --approach qwen3vl --device cpu --limit 5

report:
	$(RUN) evaluation.report
	$(RUN) evaluation.error_analysis

test:
	uv run ruff check .
	uv run pytest -q

serve:
	uv run uvicorn service.app:app --port 8000

demo:
	SERVICE_EXTRACTOR=replay docker compose up --build

.PHONY: setup data eval report test serve demo
