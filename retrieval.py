"""Retrieval: vector search and hybrid search, both reranked, plus figures.

Queries only embed the incoming text; the corpus was embedded at ingest.
"""
import config
from qdrant_client import models

CANDIDATES = 20
VISUAL_KEYWORDS = ("diagram", "figure", "table", "layout", "structure",
                   "design", "setup", "illustration", "schematic", "drawing",
                   "clearance", "dimension")
FIGURE_THRESHOLD = 0.45
PAGE_BOOST = 0.15
RELEVANCE_THRESHOLD = -2.0  # cross-encoder score below this = off-topic query

# ragapproach -> human-readable label stored in response metadata
METHOD_LABELS = {
    "vector_search": "dense_vector",
    "hybrid": "hybrid_dense_sparse_rrf",
    "hybrid_search": "hybrid_dense_sparse_rrf",
}


def _rerank(query, points, top_k):
    if not points:
        return [], -999.0
    docs = [p.payload["text"] for p in points]
    scores = list(config.reranker().rerank(query, docs))
    order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
    top_score = scores[order[0]]
    return [points[i].payload for i in order[:top_k]], top_score


def vector_search(query, top_k=5):
    """Dense-only semantic retrieval, then rerank."""
    client = config.qdrant_client()
    dense = next(config.dense_embedder().query_embed(query))
    points = client.query_points(
        collection_name=config.CHUNKS_COLLECTION,
        query=dense.tolist(), using="dense",
        limit=CANDIDATES, with_payload=True,
    ).points
    return _rerank(query, points, top_k)


def hybrid_search(query, top_k=5):
    """Dense + sparse fused with RRF, then rerank."""
    client = config.qdrant_client()
    dense = next(config.dense_embedder().query_embed(query))
    sparse = next(config.sparse_embedder().query_embed(query))
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
    return _rerank(query, points, top_k)


def retrieve(query, approach="vector_search", top_k=5):
    """Route to the requested approach. Returns (results, method_label, top_score)."""
    if approach in ("hybrid", "hybrid_search"):
        results, top_score = hybrid_search(query, top_k)
        return results, METHOD_LABELS["hybrid"], top_score
    results, top_score = vector_search(query, top_k)
    return results, METHOD_LABELS["vector_search"], top_score


def is_visual_query(query):
    q = query.lower()
    return any(w in q for w in VISUAL_KEYWORDS)


def find_figures(query, text_results, max_images=2):
    client = config.qdrant_client()
    if not client.collection_exists(config.FIGURES_COLLECTION):
        return []
    dense = next(config.dense_embedder().query_embed(query))
    hits = client.query_points(
        collection_name=config.FIGURES_COLLECTION,
        query=dense.tolist(), using="dense",
        limit=max(10, max_images * 3), with_payload=True,
    ).points

    pages = {r["page"] for r in text_results}
    scored = [(h.score + (PAGE_BOOST if h.payload.get("page") in pages else 0.0),
               h.payload) for h in hits]
    scored.sort(key=lambda x: x[0], reverse=True)

    strong = [(s, p) for s, p in scored if s >= FIGURE_THRESHOLD]
    if not strong:
        return []
    top = strong[0][0]
    selected = []
    for s, p in strong:
        if s >= top * 0.95:
            selected.append(p)
        if len(selected) >= max_images:
            break
    return selected
