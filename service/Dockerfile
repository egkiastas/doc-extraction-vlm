# Slim image: no PyTorch. OCR falls back to Tesseract and the LLM is Ollama on the host,
# or the service replays recorded outputs (SERVICE_EXTRACTOR=replay) for a demo with no model loaded.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra service --no-install-project
COPY docextract docextract
COPY service service
COPY config.toml ./
COPY results/confidence_model.json results/
COPY tests/fixtures tests/fixtures
RUN uv sync --frozen --no-dev --extra service

ENV PATH="/app/.venv/bin:$PATH" GRADIO_ANALYTICS_ENABLED=False
EXPOSE 8000
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
