FROM python:3.11-slim

WORKDIR /app

# Cache models inside the image so the first request isn't slow
ENV FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    HF_HOME=/app/.cache/hf \
    PORT=7860

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download embedding + rerank models at build time (needs internet).
# Non-fatal: if it fails, models download on first request instead.
RUN python -c "import config; config.dense_embedder(); config.sparse_embedder(); config.reranker()" || true

EXPOSE 7860
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
