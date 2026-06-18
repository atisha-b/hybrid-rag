# Greenbook Hybrid RAG (Groq + Qdrant + FastAPI)

Hybrid (dense + sparse) retrieval over the PG&E Greenbook PDF, with diagrams
retrieved semantically by caption. Answers come from a swappable Groq model.
Embeddings + reranking run on CPU via `fastembed` (no torch, no GPU).

```
config.py     settings, model allowlist, lazy model/Qdrant/Groq singletons
ingest.py     parse PDF -> chunk + pair diagrams (logo filtered) -> embed -> Qdrant
retrieval.py  vector_search / hybrid_search (RRF) + cross-encoder rerank + figures
llm.py        Groq call, model validation, verbatim/synthesis prompts, token usage
rag.py        answer_query() -> full RAGResponse dict (answer, sources, images, metadata)
app.py        FastAPI: POST /query, GET /health  (final schema)
red_team.py   basic safety smoke test for the chat API
Dockerfile    container for deploy (model warmup baked in)
```

## Accounts / keys you need (all free)

| What | Where | Why |
|------|-------|-----|
| Groq API key | https://console.groq.com/keys | the LLM |
| Qdrant Cloud cluster | https://cloud.qdrant.io (free 1 GB) | the vector DB, persisted |
| Hugging Face account | https://huggingface.co | hosting the API (Docker Space) |

`fastembed` models are public — no key needed.

---

## Part A — Build it locally

### 1. Environment
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```
Put your `GROQ_API_KEY` in `.env`. For local Qdrant, leave `QDRANT_URL` as
`http://localhost:6333`.

### 2. Start Qdrant
```bash
docker compose up -d          # localhost:6333, data persisted in a volume
```

### 3. Ingest the PDF  ← this is the "where to upload" step
There is no upload UI — ingestion is a local script you point at the file:
```bash
python ingest.py greenbook-manual-full.pdf --reset
```
First run downloads the embedding/rerank models (one-time). Expect a few
minutes for 296 pages. Extracted diagrams land in `images/`; only their paths
go into Qdrant. The repeating PG&E header logo is filtered automatically.

### 4. Test retrieval end to end
```bash
python rag.py                 # runs a sample query, prints answer + metadata
```

### 5. Run the API
```bash
uvicorn app:app --reload --port 8000
```
- Swagger UI: http://localhost:8000/docs  (click "Try it out")
- Health: http://localhost:8000/health
- Query:
```bash
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"query":"What are the clearance requirements for transformers?","model":"llama-3.3-70b-versatile","ragapproach":"hybrid"}'
```

`ragapproach` accepts `"vector_search"` (dense only) or `"hybrid"` (dense +
sparse + rerank — better for exact codes/clearances in this manual). The
approach used is echoed back in `metadata.retrievalmethod`.

### 6. Red-team smoke test
```bash
# with the API running on :8000
python red_team.py
```
Checks the bot refuses out-of-scope questions, ignores prompt-injection /
jailbreak attempts, resists fabrication, and survives junk input.

---

## Part B — Deploy (free, persistent)

The trick for "no cost + persistent": the document never changes, so we host
the DB managed and bake the images into the image.

### 1. Move the vector DB to Qdrant Cloud
Create a free cluster, copy its URL + API key into your **local** `.env`
(`QDRANT_URL`, `QDRANT_API_KEY`), then ingest **once** against the cloud:
```bash
python ingest.py greenbook-manual-full.pdf --reset
```
Vectors now live in Qdrant Cloud permanently. The deployed API only reads them.

### 2. Commit the project (including `images/`)
The extracted diagrams in `images/` must ship with the app (the container
filesystem is ephemeral). Commit them. Do **not** commit `.env`.

### 3. Deploy the API to a Hugging Face Docker Space
Why HF Spaces: the free CPU tier gives ~16 GB RAM, enough for the embedding
+ rerank models (Render's free 512 MB is too small for them).

1. Create a new Space -> SDK: **Docker**.
2. Push this folder (the `Dockerfile` is ready; it listens on port 7860).
3. In Space **Settings -> Secrets**, add: `GROQ_API_KEY`, `QDRANT_URL`,
   `QDRANT_API_KEY`, and optionally `DOC_URL` (public PDF link for source URLs).
4. The Space builds, warms the models, and exposes a public URL.

Your frontend calls `https://<your-space>.hf.space/query`. Lock CORS in
`app.py` to your frontend origin before going live.

Alternatives: Render / Railway / Fly.io work the same way (Dockerfile + env
vars) — just confirm the instance has >=1.5 GB RAM for the models.

---

## Caveat
Vector-drawn diagrams (not raster images) aren't caught by image extraction.
This manual's figures are raster (verified), so you're fine — but if a future
doc has vector diagrams, the fix is rasterizing whole pages.
