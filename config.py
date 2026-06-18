"""Central configuration and shared singletons.

Models load lazily so importing this module is cheap. First use of each
getter downloads the ONNX model once and caches it.
"""
import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

# --- Settings -------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
IMAGE_DIR = os.getenv("IMAGE_DIR", "images")

# Public URL of the source PDF; used to build clickable source links.
# Leave blank and the frontend gets "#page=N" relative anchors.
DOC_URL = os.getenv("DOC_URL", "")

# --- Collections ----------------------------------------------------------
CHUNKS_COLLECTION = "chunks"     # text chunks, dense + sparse vectors
FIGURES_COLLECTION = "figures"   # diagram captions, dense vector only

# --- Embedding / rerank models (fastembed, ONNX, CPU) ---------------------
DENSE_MODEL = "BAAI/bge-small-en-v1.5"   # 384-dim, fast, solid quality
DENSE_DIM = 384
SPARSE_MODEL = "Qdrant/bm25"             # keyword / exact-term matching
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"  # -> BAAI/bge-reranker-base for higher quality

# --- Groq models: allowlist of full model strings -------------------------
# The frontend sends a full model string (per schema). We validate against
# this set. Verify current names at https://console.groq.com/docs/models.
ALLOWED_MODELS = {
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
}
DEFAULT_MODEL = "llama-3.3-70b-versatile"


# --- Lazy singletons ------------------------------------------------------
@lru_cache(maxsize=1)
def dense_embedder():
    from fastembed import TextEmbedding
    return TextEmbedding(DENSE_MODEL)


@lru_cache(maxsize=1)
def sparse_embedder():
    from fastembed import SparseTextEmbedding
    return SparseTextEmbedding(SPARSE_MODEL)


@lru_cache(maxsize=1)
def reranker():
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    return TextCrossEncoder(RERANK_MODEL)


@lru_cache(maxsize=1)
def qdrant_client():
    from qdrant_client import QdrantClient
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


@lru_cache(maxsize=1)
def groq_client():
    from groq import Groq
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")
    return Groq(api_key=GROQ_API_KEY)
