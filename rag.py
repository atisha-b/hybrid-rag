"""RAG orchestration -> returns a dict matching the RAGResponse schema."""
import base64
import time
from datetime import datetime, timezone

import config
import llm
import retrieval


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _encode_image(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except OSError:
        return None


def _source_url(page):
    base = config.DOC_URL
    return f"{base}#page={page + 1}" if base else f"#page={page + 1}"


def _empty_metadata(model):
    return {"retrievaltimems": 0, "generationtimems": 0, "totaltimems": 0,
            "generatedat": _now_iso(), "inputtokens": 0, "outputtokens": 0,
            "totaltokens": 0, "modelused": model, "retrievalmethod": ""}


def answer_query(query, model, ragapproach="vector_search", mode="synthesis",
                 top_k=5, max_images=2):
    model = llm.validate_model(model)
    t0 = time.perf_counter()

    if not query or not query.strip():
        return {"status": "error", "query": query or "", "answer": "Empty query.",
                "sources": [], "images": [], "metadata": _empty_metadata(model)}

    try:
        t_r = time.perf_counter()
        results, method = retrieval.retrieve(query, approach=ragapproach, top_k=top_k)
        retrieval_ms = int((time.perf_counter() - t_r) * 1000)

        if not results:
            meta = _empty_metadata(model)
            meta["retrievaltimems"] = retrieval_ms
            meta["totaltimems"] = int((time.perf_counter() - t0) * 1000)
            meta["retrievalmethod"] = method
            return {"status": "success", "query": query,
                    "answer": "Not explicitly defined.", "sources": [],
                    "images": [], "metadata": meta}

        context = "\n\n".join(f"[page {r['page'] + 1}] {r['text']}" for r in results)

        t_g = time.perf_counter()
        gen = llm.generate(query, context, model, mode=mode)
        generation_ms = int((time.perf_counter() - t_g) * 1000)

        images = []
        for fig in retrieval.find_figures(query, results, max_images=max_images):
            b64 = fig.get("image_b64") or _encode_image(fig.get("image_path", ""))
            if b64:
                images.append({"image_base64": b64})

        sources = [{"title": r.get("title", ""), "url": _source_url(r["page"]),
                    "pageno": str(r["page"] + 1)} for r in results]

        total_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "status": "success",
            "query": query,
            "answer": gen["content"],
            "sources": sources,
            "images": images,
            "metadata": {
                "retrievaltimems": retrieval_ms,
                "generationtimems": generation_ms,
                "totaltimems": total_ms,
                "generatedat": _now_iso(),
                "inputtokens": gen["input_tokens"],
                "outputtokens": gen["output_tokens"],
                "totaltokens": gen["total_tokens"],
                "modelused": gen["model"],
                "retrievalmethod": method,
            },
        }
    except Exception as e:  # keep the contract stable for the frontend
        meta = _empty_metadata(model)
        meta["totaltimems"] = int((time.perf_counter() - t0) * 1000)
        return {"status": "error", "query": query, "answer": f"Error: {e}",
                "sources": [], "images": [], "metadata": meta}


if __name__ == "__main__":
    import json
    out = answer_query("What are the clearance requirements for transformers?",
                       model=config.DEFAULT_MODEL, ragapproach="hybrid")
    print(json.dumps({"status": out["status"], "answer": out["answer"][:300],
                      "n_sources": len(out["sources"]), "n_images": len(out["images"]),
                      "metadata": out["metadata"]}, indent=2))
