"""RAG orchestration -> returns a dict matching the RAGResponse schema."""
import base64
import re
import time
from datetime import datetime, timezone

import config
import llm
import retrieval
from groq import RateLimitError

_CONVERSATIONAL_RE = re.compile(
    r"^\s*("
    # greetings
    r"hi+(\s+there)?|hello+|hey+(\s+there)?|howdy|sup|yo|greetings|"
    r"good\s+(morning|afternoon|evening|day)|what[\'\s]*s\s+up|"
    # what/who is PGE/this
    r"what\s+is\s+(pg&?e|pacific\s+gas|this(\s+app)?(\s+about)?)|"
    r"who\s+is\s+(pg&?e|pacific\s+gas)|"
    r"tell\s+me\s+about\s+(pg&?e|pacific\s+gas|yourself|this)|"
    # capability questions
    r"what\s+(can\s+you\s+(do|help(\s+me)?(\s+with)?)|are\s+you|is\s+this(\s+about)?|do\s+you\s+(do|know|cover))|"
    r"who\s+are\s+you|"
    r"what\s+topics?(\s+(do\s+you\s+cover|can\s+you\s+help(\s+(me\s+)?with)?))?|"
    r"how\s+can\s+(i\s+use\s+(you|this)|you\s+help(\s+me)?)|"
    # filler
    r"help(\s+me)?|thanks?(\s+you)?|ok(ay)?|cool|great|awesome"
    r")\s*[!?.]*\s*$",
    re.IGNORECASE,
)

_GREETING_RESPONSE = (
    "Hi! I'm the PG&E Electric Rule Book (Greenbook) assistant. "
    "PG&E (Pacific Gas and Electric Company) is California's largest utility, "
    "and the Greenbook is their technical manual for electrical construction standards.\n\n"
    "I can help you with:\n"
    "- Electrical clearance requirements\n"
    "- Equipment specifications (transformers, conduits, conductors)\n"
    "- Construction standards and material codes\n"
    "- Mandrel sizes, bolt patterns, and installation dimensions\n\n"
    "What would you like to know?"
)

_OFF_TOPIC_RESPONSE = (
    "That topic isn't covered in the PG&E Greenbook. "
    "I can help with electrical clearances, equipment specifications, "
    "conduit sizing, conductor ratings, and construction standards."
)


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

    if _CONVERSATIONAL_RE.match(query.strip()):
        meta = _empty_metadata(model)
        meta["totaltimems"] = int((time.perf_counter() - t0) * 1000)
        return {"status": "success", "query": query,
                "answer": _GREETING_RESPONSE, "sources": [], "images": [], "metadata": meta}

    try:
        t_r = time.perf_counter()
        results, method, top_score = retrieval.retrieve(query, approach=ragapproach, top_k=top_k)
        retrieval_ms = int((time.perf_counter() - t_r) * 1000)

        if not results or top_score < retrieval.RELEVANCE_THRESHOLD:
            meta = _empty_metadata(model)
            meta["retrievaltimems"] = retrieval_ms
            meta["totaltimems"] = int((time.perf_counter() - t0) * 1000)
            meta["retrievalmethod"] = method
            return {"status": "success", "query": query,
                    "answer": _OFF_TOPIC_RESPONSE, "sources": [],
                    "images": [], "metadata": meta}

        context = "\n\n".join(f"[page {r['page'] + 1}] {r['text']}" for r in results)

        t_g = time.perf_counter()
        gen = llm.generate(query, context, model, mode=mode)
        generation_ms = int((time.perf_counter() - t_g) * 1000)

        answer_text = gen["content"]

        # LLM found nothing useful in the context — suppress citations and images
        llm_found_nothing = "not in the Greenbook" in answer_text

        images = []
        if not llm_found_nothing and retrieval.is_visual_query(query):
            for fig in retrieval.find_figures(query, results, max_images=max_images):
                b64 = fig.get("image_b64") or _encode_image(fig.get("image_path", ""))
                if b64:
                    images.append({"image_base64": b64})

        seen_pages: set = set()
        sources = []
        if not llm_found_nothing:
            for r in results:
                p = r["page"]
                if p not in seen_pages:
                    seen_pages.add(p)
                    sources.append({"title": r.get("title", ""), "url": _source_url(p),
                                    "pageno": str(p + 1)})
                if len(sources) >= 3:
                    break

        total_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "status": "success",
            "query": query,
            "answer": answer_text,
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
    except RateLimitError:
        meta = _empty_metadata(model)
        meta["totaltimems"] = int((time.perf_counter() - t0) * 1000)
        return {"status": "error", "query": query,
                "answer": "The AI model is temporarily rate-limited. Please try again in a few minutes.",
                "sources": [], "images": [], "metadata": meta}
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
