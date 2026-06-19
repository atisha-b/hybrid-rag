"""Diagnostic: show exactly what chunks and figures are retrieved for a query."""
import sys
sys.path.insert(0, ".")
import config, retrieval

queries = [
    "conduit fill requirements",
    "conduit size requirements",
    "service conduit specifications",
    "conduit installation requirements underground",
    "overhead service drop attachment",
    "riser pole installation",
]

for q in queries:
    sq = retrieval.semantic_query(q)
    results, method, top_score = retrieval.retrieve(q, approach="hybrid", top_k=5)
    figs = retrieval.find_figures(q, results, max_images=2) if retrieval.is_visual_query(q) else []
    print(f"\n{'='*70}")
    print(f"QUERY : {q}")
    print(f"SEMQ  : {sq}")
    print(f"SCORE : {top_score:.3f}  METHOD: {method}  PASS_THRESHOLD: {top_score >= retrieval.RELEVANCE_THRESHOLD}")
    print(f"CHUNKS: {len(results)}  FIGURES: {len(figs)}")
    for i, r in enumerate(results[:3]):
        print(f"  [{i}] p.{r['page']+1}  {r['text'][:120].replace(chr(10),' ')}")
    if figs:
        for f in figs:
            print(f"  FIG: p.{f.get('page','?')+1}  {str(f.get('caption',''))[:80]}")
