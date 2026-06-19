"""Show actual figure scores so we can tune the threshold."""
import sys
sys.path.insert(0, ".")
import config, retrieval

queries = [
    "overhead service drop attachment",
    "transfer switch",
    "pad-mounted equipment",
    "riser pole",
    "conduit fill requirements",
    "what is a transformer",  # should NOT get image
]

client = config.qdrant_client()

for q in queries:
    sq = retrieval.semantic_query(q)
    # get text results first (needed for page boost)
    results, _, _ = retrieval.retrieve(q, approach="hybrid", top_k=5)
    pages = {r["page"] for r in results}

    dense = next(config.dense_embedder().query_embed(sq))
    hits = client.query_points(
        collection_name=config.FIGURES_COLLECTION,
        query=dense.tolist(), using="dense",
        limit=6, with_payload=True,
    ).points

    print(f"\n{'='*65}")
    print(f"QUERY: {q}")
    print(f"SEMQ : {sq}")
    print(f"TEXT PAGES: {sorted(pages)}")
    print(f"TOP FIGURE SCORES (threshold={retrieval.FIGURE_THRESHOLD}):")
    for h in hits[:5]:
        page = h.payload.get("page", "?")
        boosted = h.score + (retrieval.PAGE_BOOST if page in pages else 0)
        caption = str(h.payload.get("caption", ""))[:70]
        flag = "PASS" if boosted >= retrieval.FIGURE_THRESHOLD else "fail"
        print(f"  {flag}  raw={h.score:.3f}  boosted={boosted:.3f}  p.{page if page=='?' else page+1}  {caption}")
