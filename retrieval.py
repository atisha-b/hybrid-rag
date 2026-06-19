"""Retrieval: vector search and hybrid search, both reranked, plus figures.

Queries only embed the incoming text; the corpus was embedded at ingest.
"""
import re
import config
from qdrant_client import models

CANDIDATES = 20
VISUAL_KEYWORDS = ("diagram", "figure", "table", "layout", "structure",
                   "design", "setup", "illustration", "schematic", "drawing",
                   "clearance", "dimension")
VISUAL_INTENT = ("show me", "look like", "looks like", "what does", "display")

# Page-matched figures: lower bar (the page context confirms relevance).
# Non-page-matched figures: higher bar (raw semantic similarity must be strong).
FIGURE_THRESHOLD = 0.45
NON_BOOSTED_THRESHOLD = 0.72

PAGE_BOOST = 0.15
RELEVANCE_THRESHOLD = -2.0  # cross-encoder score below this = off-topic query

METHOD_LABELS = {
    "vector_search": "dense_vector",
    "hybrid": "hybrid_dense_sparse_rrf",
    "hybrid_search": "hybrid_dense_sparse_rrf",
}

# ── Query cleaning ────────────────────────────────────────────────────────────
# "show me a table for conduit fill requirements" → "conduit fill requirements"
_SHOW_ME_RE = re.compile(
    r"^(?:show\s+me|give\s+me|find\s+me|display)\s+(?:a\s+|the\s+)?"
    r"(?:figure|diagram|table|image|picture|layout|drawing|schematic|illustration)"
    r"(?:\s+(?:for|of|about|on|showing))?\s+(.+)$",
    re.IGNORECASE,
)
# "what does the transfer switch diagram look like" → "transfer switch"
_WHAT_DOES_RE = re.compile(
    r"^what\s+does?\s+(?:the\s+|a\s+)?(.+?)\s+"
    r"(?:figure|diagram|table|image|layout|drawing|schematic|illustration)"
    r"(?:\s+(?:look\s+like|show|display))?.*$",
    re.IGNORECASE,
)
# "show me overhead service drop attachment" → "overhead service drop attachment"
_SHOW_ME_PLAIN_RE = re.compile(
    r"^(?:show\s+me|give\s+me|find\s+me|display)\s+(?:(?:a|an|the)\s+)?(.+)$",
    re.IGNORECASE,
)


def semantic_query(query: str) -> str:
    """Strip visual-intent phrasing so embedding focuses on the topic, not the request style."""
    q = query.strip()
    m = _SHOW_ME_RE.match(q)      # "show me a table for X" → "X"
    if m:
        return m.group(1).strip()
    m = _WHAT_DOES_RE.match(q)    # "what does the X diagram look like" → "X"
    if m:
        return m.group(1).strip()
    m = _SHOW_ME_PLAIN_RE.match(q)  # "show me X" (no visual noun) → "X"
    if m:
        return m.group(1).strip()
    return q


def is_visual_query(query: str) -> bool:
    q = query.lower()
    return any(w in q for w in VISUAL_KEYWORDS) or any(p in q for p in VISUAL_INTENT)


# ── Retrieval ─────────────────────────────────────────────────────────────────

def _rerank(sq, points, top_k):
    if not points:
        return [], -999.0
    docs = [p.payload["text"] for p in points]
    scores = list(config.reranker().rerank(sq, docs))
    order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
    top_score = scores[order[0]]
    return [points[i].payload for i in order[:top_k]], top_score


def vector_search(query, top_k=5):
    """Dense-only semantic retrieval, then rerank."""
    sq = semantic_query(query)
    client = config.qdrant_client()
    dense = next(config.dense_embedder().query_embed(sq))
    points = client.query_points(
        collection_name=config.CHUNKS_COLLECTION,
        query=dense.tolist(), using="dense",
        limit=CANDIDATES, with_payload=True,
    ).points
    return _rerank(sq, points, top_k)


def hybrid_search(query, top_k=5):
    """Dense + sparse fused with RRF, then rerank."""
    sq = semantic_query(query)
    client = config.qdrant_client()
    dense = next(config.dense_embedder().query_embed(sq))
    sparse = next(config.sparse_embedder().query_embed(sq))
    points = client.query_points(
        collection_name=config.CHUNKS_COLLECTION,
        prefetch=[
            models.Prefetch(query=dense.tolist(), using="dense", limit=CANDIDATES),
            models.Prefetch(
                query=models.SparseVector(indices=sparse.indices.tolist(),
                                          values=sparse.values.tolist()),
                using="bm25", limit=CANDIDATES),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=CANDIDATES, with_payload=True,
    ).points
    return _rerank(sq, points, top_k)


def retrieve(query, approach="vector_search", top_k=5):
    """Route to the requested approach. Returns (results, method_label, top_score)."""
    if approach in ("hybrid", "hybrid_search"):
        results, top_score = hybrid_search(query, top_k)
        return results, METHOD_LABELS["hybrid"], top_score
    results, top_score = vector_search(query, top_k)
    return results, METHOD_LABELS["vector_search"], top_score


def find_figures(query, text_results, max_images=2):
    client = config.qdrant_client()
    if not client.collection_exists(config.FIGURES_COLLECTION):
        return []
    sq = semantic_query(query)
    dense = next(config.dense_embedder().query_embed(sq))
    hits = client.query_points(
        collection_name=config.FIGURES_COLLECTION,
        query=dense.tolist(), using="dense",
        limit=max(10, max_images * 3), with_payload=True,
    ).points

    pages = {r["page"] for r in text_results}

    # Apply dual threshold: relaxed when figure page matches a retrieved text page,
    # strict otherwise so unrelated figures from different sections don't slip through.
    scored = []
    for h in hits:
        page = h.payload.get("page")
        in_pages = page in pages
        effective_score = h.score + (PAGE_BOOST if in_pages else 0.0)
        threshold = FIGURE_THRESHOLD if in_pages else NON_BOOSTED_THRESHOLD
        if effective_score >= threshold:
            scored.append((effective_score, h.payload))

    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return []

    top = scored[0][0]
    selected = []
    for s, p in scored:
        if s >= top * 0.95:
            selected.append(p)
        if len(selected) >= max_images:
            break
    return selected
