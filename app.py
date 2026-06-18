"""FastAPI app implementing the final RAG schema."""
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
import rag

app = FastAPI(title="Greenbook Hybrid RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # lock to your frontend origin before prod
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
def root():
    return FileResponse("static/index.html")


# --- Schema (final) -------------------------------------------------------
class RAGRequest(BaseModel):
    query: str
    model: str
    ragapproach: str = "vector_search"

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the clearance requirements for transformers?",
                "model": "llama-3.3-70b-versatile",
                "ragapproach": "vector_search",
            }
        }


class Source(BaseModel):
    title: str
    url: str
    pageno: str


class ImageResult(BaseModel):
    image_base64: str


class Metadata(BaseModel):
    retrievaltimems: int
    generationtimems: int
    totaltimems: int
    generatedat: str
    inputtokens: int
    outputtokens: int
    totaltokens: int
    modelused: str
    retrievalmethod: str


class RAGResponse(BaseModel):
    status: str
    query: str
    answer: str
    sources: List[Source]
    images: List[ImageResult]
    metadata: Metadata


# --- Endpoints ------------------------------------------------------------
@app.get("/health")
def health():
    client = config.qdrant_client()
    return {
        "status": "ok",
        "chunks_ready": client.collection_exists(config.CHUNKS_COLLECTION),
        "figures_ready": client.collection_exists(config.FIGURES_COLLECTION),
        "models": [config.DEFAULT_MODEL],
    }


@app.post("/query", response_model=RAGResponse)
def query(req: RAGRequest):
    if not req.query or not req.query.strip():
        return rag.answer_query("", model=req.model, ragapproach=req.ragapproach)
    if len(req.query) > 1000:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Query too long (max 1000 characters).")
    if req.ragapproach not in ("vector_search", "hybrid", "hybrid_search"):
        req.ragapproach = "hybrid"
    return rag.answer_query(req.query, model=req.model, ragapproach=req.ragapproach)
