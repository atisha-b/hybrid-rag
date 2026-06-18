"""Ingestion: PDF -> text chunks + diagram/caption pairs -> Qdrant.

Run once per document:
    python ingest.py path/to/document.pdf --reset

Tuned for the Greenbook manual:
- Filters out the repeating logo/header banner (boilerplate) so it never
  shows up as a "figure".
- Pairs each real diagram to its nearest caption by position.
- Stores images on disk; only paths/captions go in Qdrant.
- Chunks along paragraph boundaries so procedures stay intact.
"""
import argparse
import base64
import os
import re
import uuid
from collections import Counter

import fitz  # PyMuPDF
from qdrant_client import models

import config

CAPTION_RE = re.compile(r"\b(?:figure|fig\.?|table|exhibit|diagram)\s*\d", re.IGNORECASE)
TARGET_WORDS = 250
OVERLAP_WORDS = 40

# Image filtering (drops logos, icons, bullets, transparency masks)
MIN_IMAGE_BYTES = 5000
MIN_WIDTH = 160
MIN_HEIGHT = 120
BOILERPLATE_MIN_PAGES = 8   # an image on more pages than this is a header/logo


def _center(bbox):
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2, (y0 + y1) / 2)


def _caption_blocks(page):
    blocks = []
    for b in page.get_text("blocks"):
        if len(b) >= 7 and b[6] != 0:
            continue
        text = b[4].strip()
        if text and CAPTION_RE.search(text):
            blocks.append({"bbox": b[:4], "text": " ".join(text.split())})
    return blocks


def _nearest_caption(img_bbox, captions):
    if not captions:
        return None
    ix, iy = _center(img_bbox)
    below = [c for c in captions if c["bbox"][1] >= img_bbox[3] - 5]
    pool = below if below else captions

    def dist(c):
        cx, cy = _center(c["bbox"])
        return (cx - ix) ** 2 + (cy - iy) ** 2

    return min(pool, key=dist)["text"]


def _boilerplate_xrefs(doc):
    """xrefs that appear on many pages = logos/headers/footers -> skip."""
    counts = Counter()
    for page in doc:
        for info in page.get_image_info(xrefs=True):
            x = info.get("xref", 0)
            if x:
                counts[x] += 1
    return {x for x, c in counts.items() if c > BOILERPLATE_MIN_PAGES}


def _chunk_page(text):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current, count = [], [], 0
    for para in paragraphs:
        words = para.split()
        if count + len(words) > TARGET_WORDS and current:
            chunks.append(" ".join(current))
            tail = " ".join(current).split()[-OVERLAP_WORDS:]
            current, count = tail.copy(), len(tail)
        current.extend(words)
        count += len(words)
    if current:
        chunks.append(" ".join(current))
    return chunks or ([text.strip()] if text.strip() else [])


def parse_pdf(pdf_path, image_dir):
    os.makedirs(image_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    source = os.path.basename(pdf_path)
    title = doc.metadata.get("title") or os.path.splitext(source)[0]
    skip_xrefs = _boilerplate_xrefs(doc)

    chunks, figures = [], []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        captions = _caption_blocks(page)

        for chunk_text in _chunk_page(text):
            chunks.append({"text": chunk_text, "page": page_num,
                           "source": source, "title": title})

        seen = set()
        for info in page.get_image_info(xrefs=True):
            xref = info.get("xref", 0)
            if not xref or xref in seen or xref in skip_xrefs:
                continue
            seen.add(xref)
            if info["width"] < MIN_WIDTH or info["height"] < MIN_HEIGHT:
                continue

            base = doc.extract_image(xref)
            img_bytes = base["image"]
            if len(img_bytes) < MIN_IMAGE_BYTES:
                continue

            ext = base["ext"]
            pix = fitz.Pixmap(doc, xref)
            if pix.n - pix.alpha > 3:                 # CMYK/other -> RGB PNG
                pix = fitz.Pixmap(fitz.csRGB, pix)
                img_bytes = pix.tobytes("png")
                ext = "png"

            fname = f"{os.path.splitext(source)[0]}_p{page_num}_{xref}.{ext}"
            fpath = os.path.join(image_dir, fname)
            with open(fpath, "wb") as f:
                f.write(img_bytes)

            caption = _nearest_caption(info["bbox"], captions) or text[:200].strip()
            image_b64 = base64.b64encode(img_bytes).decode()
            figures.append({"caption": caption, "image_path": fpath,
                            "image_b64": image_b64,
                            "page": page_num, "source": source, "title": title})

    doc.close()
    return chunks, figures


def _recreate_collections(client):
    for name in (config.CHUNKS_COLLECTION, config.FIGURES_COLLECTION):
        if client.collection_exists(name):
            client.delete_collection(name)
    client.create_collection(
        collection_name=config.CHUNKS_COLLECTION,
        vectors_config={"dense": models.VectorParams(size=config.DENSE_DIM,
                                                     distance=models.Distance.COSINE)},
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    client.create_collection(
        collection_name=config.FIGURES_COLLECTION,
        vectors_config={"dense": models.VectorParams(size=config.DENSE_DIM,
                                                     distance=models.Distance.COSINE)},
    )


def _batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def index_chunks(client, chunks, batch=128):
    texts = [c["text"] for c in chunks]
    dense = list(config.dense_embedder().embed(texts, batch_size=1, parallel=0))
    sparse = list(config.sparse_embedder().embed(texts, batch_size=1, parallel=0))
    points = [
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector={"dense": dense[i].tolist(),
                    "bm25": models.SparseVector(indices=sparse[i].indices.tolist(),
                                                values=sparse[i].values.tolist())},
            payload=chunks[i],
        ) for i in range(len(chunks))
    ]
    for b in _batched(points, batch):
        client.upsert(config.CHUNKS_COLLECTION, points=b)


def index_figures(client, figures, batch=128):
    if not figures:
        return
    captions = [f["caption"] for f in figures]
    dense = list(config.dense_embedder().embed(captions, batch_size=1, parallel=0))
    points = [
        models.PointStruct(id=str(uuid.uuid4()),
                           vector={"dense": dense[i].tolist()},
                           payload=figures[i])
        for i in range(len(figures))
    ]
    for b in _batched(points, batch):
        client.upsert(config.FIGURES_COLLECTION, points=b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf_path")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    client = config.qdrant_client()
    if args.reset or not client.collection_exists(config.CHUNKS_COLLECTION):
        _recreate_collections(client)

    print(f"Parsing {args.pdf_path} ...")
    chunks, figures = parse_pdf(args.pdf_path, config.IMAGE_DIR)
    print(f"  {len(chunks)} chunks, {len(figures)} figures")
    print("Embedding + indexing (first run downloads models) ...")
    index_chunks(client, chunks)
    index_figures(client, figures)
    print("Done.")


if __name__ == "__main__":
    main()
