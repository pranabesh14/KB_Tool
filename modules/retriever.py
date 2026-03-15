"""
retriever.py — Retrieval helpers: reranking and neighbour-chunk expansion.

Two rerankers are supported:
  1. CrossEncoder (ms-marco-MiniLM-L-6-v2) – preferred, GPU/CPU
  2. Cosine fallback                        – uses the same embedding model

Public API
----------
rerank(query, docs, embeddings, top_k)          -> List[Document]
expand_with_neighbors(docs, doc_store, window)  -> List[Document]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from langchain.docstore.document import Document
from sklearn.metrics.pairwise import cosine_similarity

from modules.config import get_logger

logger = get_logger("retriever")

DocStore = Dict[str, List[Dict[str, Any]]]

# ─── CrossEncoder (lazy-loaded) ───────────────────────────────────────────────

_CROSS_ENCODER: Optional[Any] = None
_USE_CROSS_ENCODER: bool = True


def _get_cross_encoder():
    global _CROSS_ENCODER, _USE_CROSS_ENCODER
    if _CROSS_ENCODER is not None:
        return _CROSS_ENCODER
    try:
        from sentence_transformers import CrossEncoder
        _CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("CrossEncoder loaded.")
    except Exception as exc:
        logger.warning("CrossEncoder unavailable (%s). Falling back to cosine.", exc)
        _USE_CROSS_ENCODER = False
    return _CROSS_ENCODER


# ─── Reranking ────────────────────────────────────────────────────────────────

def _cosine_rerank(
    query: str,
    docs: List[Document],
    embeddings,
    top_k: int,
) -> List[Document]:
    if not docs:
        return []
    q_emb  = np.array(embeddings.embed_query(query)).reshape(1, -1)
    d_embs = np.array(embeddings.embed_documents([d.page_content for d in docs]))
    sims   = cosine_similarity(q_emb, d_embs)[0]
    ranked = sorted(zip(docs, sims), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_k]]


def _cross_rerank(
    query: str,
    docs: List[Document],
    top_k: int,
    score_threshold: float = 0.3,
) -> List[Document]:
    encoder = _get_cross_encoder()
    if encoder is None:
        return docs[:top_k]
    pairs  = [[query, d.page_content] for d in docs]
    scores = encoder.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    result = [d for d, s in ranked if s > score_threshold]
    return result[:top_k] if result else [d for d, _ in ranked[:top_k]]


def rerank(
    query: str,
    docs: List[Document],
    embeddings,
    top_k: int = 6,
) -> List[Document]:
    """Rerank docs. Uses CrossEncoder when available, else cosine similarity."""
    if not docs:
        return []
    _get_cross_encoder()  # trigger lazy load / availability check
    if _USE_CROSS_ENCODER and _CROSS_ENCODER is not None:
        return _cross_rerank(query, docs, top_k)
    return _cosine_rerank(query, docs, embeddings, top_k)


# ─── Neighbour expansion ──────────────────────────────────────────────────────

def expand_with_neighbors(
    sel_docs: List[Document],
    doc_store: DocStore,
    window: int = 1,
) -> List[Document]:
    """
    For each selected Document, merge its surrounding chunks from doc_store.
    'window' = how many chunks to include on each side (default 1).
    Returns a deduplicated list of merged Documents.
    """
    if not sel_docs or not doc_store:
        return sel_docs

    seen: set = set()
    result: List[Document] = []

    for doc in sel_docs:
        md  = doc.metadata or {}
        src = md.get("source", "")

        src_chunks = doc_store.get(src, [])
        if not src_chunks:
            key = (src, doc.page_content[:200])
            if key not in seen:
                seen.add(key)
                result.append(doc)
            continue

        chunk_idx = _resolve_chunk_index(doc, src_chunks, md)

        if chunk_idx is None:
            key = (src, doc.page_content[:200])
            if key not in seen:
                seen.add(key)
                result.append(doc)
            continue

        lo  = max(0, chunk_idx - window)
        hi  = min(len(src_chunks) - 1, chunk_idx + window)
        key = (src, lo, hi)
        if key in seen:
            continue
        seen.add(key)

        merged_content = "\n".join(src_chunks[i]["content"] for i in range(lo, hi + 1)).strip()
        merged_md      = _build_merged_metadata(src_chunks, lo, hi, md)
        result.append(Document(page_content=merged_content, metadata=merged_md))

    return result


def _resolve_chunk_index(
    doc: Document,
    src_chunks: List[Dict[str, Any]],
    md: Dict[str, Any],
) -> Optional[int]:
    """Try to find the chunk_index for a document within its source's chunk list."""
    idx = md.get("chunk_index")
    if isinstance(idx, int):
        return idx

    # Video: match by start_time
    if md.get("source_type") == "video" and "start_time" in md:
        target = float(md.get("start_time") or 0.0)
        for i, item in enumerate(src_chunks):
            imd = item.get("metadata", {})
            if "start_time" in imd and abs(float(imd["start_time"]) - target) < 0.01:
                return i

    # Fallback: match by content
    target_content = doc.page_content.strip()
    for i, item in enumerate(src_chunks):
        if item.get("content", "").strip() == target_content:
            return i

    return None


def _build_merged_metadata(
    src_chunks: List[Dict[str, Any]],
    lo: int,
    hi: int,
    original_md: Dict[str, Any],
) -> Dict[str, Any]:
    """Build metadata for a merged chunk, merging time ranges if present."""
    merged: Dict[str, Any] = {"source": original_md.get("source", "")}
    if "source_type" in original_md:
        merged["source_type"] = original_md["source_type"]
    merged["chunk_index"] = lo

    m_start: Optional[float] = None
    m_end:   Optional[float] = None

    for i in range(lo, hi + 1):
        imd = src_chunks[i].get("metadata", {}) or {}
        if "start_time" in imd:
            st = float(imd["start_time"] or 0.0)
            if m_start is None or st < m_start:
                m_start = st
        if "end_time" in imd:
            et = float(imd["end_time"] or 0.0)
            if m_end is None or et > m_end:
                m_end = et

    if m_start is not None:
        merged["start_time"] = m_start
    if m_end is not None:
        merged["end_time"] = m_end

    return merged