from **future** import annotations
“””
query_handler.py - High-level query orchestration for video and KB sources.

Each public function:

1. Checks long-term QA memory (returns cached answer immediately if found)
1. Retrieves from FAISS -> reranks -> expands neighbours
1. Calls answer_engine which skips the LLM for whichever source has no context
1. Saves the answer back to QA memory
1. Returns a structured dict

## Public API

get_video_answer(query, video_index_tuple, llm, embeddings, entity_memory, **kw) -> dict
get_kb_answer   (query, pdf_index_tuple,   llm, embeddings, entity_memory, **kw) -> dict
get_combined_answer(query, video_tuple, pdf_tuple, llm, embeddings, entity_memory, **kw) -> dict
“””

import os
import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(**file**)))
if _PROJECT_ROOT not in sys.path:
sys.path.insert(0, _PROJECT_ROOT)

from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from modules.config import TOP_K_RETRIEVAL, RERANK_K, NEIGHBOR_WINDOW, get_logger
from modules.retriever import rerank, expand_with_neighbors
from modules.qa_memory import search_qa_memory, add_to_qa_index
from modules.answer_engine import answer_from_docs
from modules.transcription import seconds_to_hhmmss

logger = get_logger(“query_handler”)

DocStore = Dict[str, Any]

_NO_VIDEO_ANS = “The answer is not present in the videos.”
_NO_KB_ANS    = “The answer is not present in the pdfs.”

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _unpack(index_tuple) -> Tuple[Any, DocStore]:
if isinstance(index_tuple, tuple) and len(index_tuple) == 2:
return index_tuple
return index_tuple, {}

def _retrieve_and_expand(
query: str,
index,
doc_store: DocStore,
top_k: int,
rerank_k: int,
neighbor_window: int,
embeddings,
) -> List[Document]:
“”“Retrieve -> rerank -> expand neighbours. Returns [] if index is None.”””
if index is None:
return []
docs = index.similarity_search(query, k=min(top_k, 30))
if not docs:
logger.info(“No docs retrieved from FAISS for query: %.80s”, query)
return []
logger.debug(“Retrieved %d docs from FAISS, reranking to top %d.”, len(docs), rerank_k)
best = rerank(query, docs, embeddings, top_k=rerank_k)
if not best:
logger.info(“Reranker returned no results for query: %.80s”, query)
return []
expanded = expand_with_neighbors(best, doc_store, window=neighbor_window)
logger.debug(“Expanded to %d docs after neighbour window=%d.”, len(expanded), neighbor_window)
return expanded

def _timestamp_summary(expanded_docs: List[Document]) -> Tuple[str, str]:
“”“Build timestamp string and source count string from expanded video docs.”””
source_info: Dict[str, set] = {}
for d in expanded_docs:
md  = d.metadata or {}
src = md.get(“source”, “unknown”)
if “end_time” in md:
source_info.setdefault(src, set()).add(seconds_to_hhmmss(md[“end_time”]))
ts_parts = [
f”{src} (at: {’, ’.join(sorted(ts))})”
for src, ts in source_info.items()
]
return “; “.join(ts_parts) or “N/A”, f”{len(source_info)} source(s)”

def _pdf_source_names(expanded_docs: List[Document]) -> List[str]:
seen, names = set(), []
for d in expanded_docs:
src = (d.metadata or {}).get(“source”, “”)
if src and src not in seen:
seen.add(src)
names.append(src)
return names

# ─── Public API ───────────────────────────────────────────────────────────────

def get_video_answer(
query: str,
video_index_tuple,
llm,
embeddings,
entity_memory,
top_k: int = TOP_K_RETRIEVAL,
rerank_k: int = RERANK_K,
neighbor_window: int = NEIGHBOR_WINDOW,
) -> Dict[str, Any]:
“””
Returns: {“snippet”: str, “timestamp”: str, “source”: str}
“””
out = {“snippet”: _NO_VIDEO_ANS, “timestamp”: “”, “source”: “”}

```
cached = search_qa_memory(query, "video")
if cached:
    logger.info("Video QA cache hit for query: %.80s", query)
    return {"snippet": cached, "timestamp": "(from memory)", "source": "QA Memory"}

video_index, doc_store = _unpack(video_index_tuple)
expanded = _retrieve_and_expand(query, video_index, doc_store, top_k, rerank_k, neighbor_window, embeddings)
if not expanded:
    logger.info("No video context found for query: %.80s", query)
    return out

logger.info("Generating video answer from %d expanded docs.", len(expanded))
ans_dict = answer_from_docs(query, expanded, llm, entity_memory, embeddings)
snippet  = ans_dict["video"]
if snippet == _NO_VIDEO_ANS:
    logger.info("LLM found no relevant video answer for query: %.80s", query)
    return out

logger.info("Video answer generated successfully.")
add_to_qa_index(query, snippet, "video", embeddings)
timestamp, source = _timestamp_summary(expanded)
return {"snippet": snippet, "timestamp": timestamp, "source": source}
```

def get_kb_answer(
query: str,
pdf_index_tuple,
llm,
embeddings,
entity_memory,
top_k: int = TOP_K_RETRIEVAL,
rerank_k: int = RERANK_K,
neighbor_window: int = NEIGHBOR_WINDOW,
) -> Dict[str, Any]:
“””
Returns: {“answer”: str, “sources”: list[str]}
“””
out = {“answer”: _NO_KB_ANS, “sources”: []}

```
cached = search_qa_memory(query, "pdf")
if cached:
    logger.info("KB QA cache hit for query: %.80s", query)
    return {"answer": cached, "sources": ["QA Memory"]}

pdf_index, doc_store = _unpack(pdf_index_tuple)
expanded = _retrieve_and_expand(query, pdf_index, doc_store, top_k, rerank_k, neighbor_window, embeddings)
if not expanded:
    logger.info("No KB context found for query: %.80s", query)
    return out

logger.info("Generating KB answer from %d expanded docs.", len(expanded))
ans_dict = answer_from_docs(query, expanded, llm, entity_memory, embeddings)
ans = ans_dict["kb"]
if ans == _NO_KB_ANS:
    logger.info("LLM found no relevant KB answer for query: %.80s", query)
    return out

logger.info("KB answer generated successfully.")
add_to_qa_index(query, ans, "pdf", embeddings)
return {"answer": ans, "sources": _pdf_source_names(expanded)}
```

def get_combined_answer(
query: str,
video_index_tuple,
pdf_index_tuple,
llm,
embeddings,
entity_memory,
confluence_index_tuple=None,
sharepoint_index_tuple=None,
top_k: int = TOP_K_RETRIEVAL,
rerank_k: int = RERANK_K,
neighbor_window: int = NEIGHBOR_WINDOW,
) -> Dict[str, Any]:
“””
Retrieves from ALL available indexes (video, PDF, Confluence, SharePoint)
in one shot and generates answers per source type.

```
Returns:
    {
      "video_answer":       str,
      "video_timestamp":    str,
      "video_source":       str,
      "kb_answer":          str,
      "kb_sources":         list[str],
      "confluence_answer":  str,
      "confluence_sources": list[str],
      "sharepoint_answer":  str,
      "sharepoint_sources": list[str],
      "has_video":          bool,
      "has_kb":             bool,
      "has_confluence":     bool,
      "has_sharepoint":     bool,
    }
"""
logger.info("Combined query: %.80s", query)

# QA memory check
video_cached = search_qa_memory(query, "video")
kb_cached    = search_qa_memory(query, "pdf")

# Retrieve from all indexes
video_index, vid_store  = _unpack(video_index_tuple)
pdf_index,   pdf_store  = _unpack(pdf_index_tuple)
conf_index,  conf_store = _unpack(confluence_index_tuple) if confluence_index_tuple else (None, {})
sp_index,    sp_store   = _unpack(sharepoint_index_tuple) if sharepoint_index_tuple else (None, {})

vid_expanded  = _retrieve_and_expand(query, video_index, vid_store,  top_k, rerank_k, neighbor_window, embeddings)
pdf_expanded  = _retrieve_and_expand(query, pdf_index,   pdf_store,  top_k, rerank_k, neighbor_window, embeddings)
conf_expanded = _retrieve_and_expand(query, conf_index,  conf_store, top_k, rerank_k, neighbor_window, embeddings)
sp_expanded   = _retrieve_and_expand(query, sp_index,    sp_store,   top_k, rerank_k, neighbor_window, embeddings)

logger.info(
    "Retrieval: video=%d, pdf=%d, confluence=%d, sharepoint=%d",
    len(vid_expanded), len(pdf_expanded), len(conf_expanded), len(sp_expanded)
)

# Single LLM call with all docs together
all_docs = vid_expanded + pdf_expanded + conf_expanded + sp_expanded
if all_docs and not (video_cached and kb_cached):
    logger.info("Calling LLM with %d total docs.", len(all_docs))
    ans_dict = answer_from_docs(query, all_docs, llm, entity_memory, embeddings)
else:
    ans_dict = {"video": _NO_VIDEO_ANS, "kb": _NO_KB_ANS,
                "confluence": _NO_KB_ANS, "sharepoint": _NO_KB_ANS}

# Video
video_answer = video_cached or ans_dict.get("video", _NO_VIDEO_ANS)
has_video    = video_answer != _NO_VIDEO_ANS
if has_video and not video_cached:
    add_to_qa_index(query, video_answer, "video", embeddings)
timestamp, vid_source = _timestamp_summary(vid_expanded) if vid_expanded else ("N/A", "N/A")

# PDF KB
kb_answer = kb_cached or ans_dict.get("kb", _NO_KB_ANS)
has_kb    = kb_answer != _NO_KB_ANS
if has_kb and not kb_cached:
    add_to_qa_index(query, kb_answer, "pdf", embeddings)

# Confluence
conf_answer = ans_dict.get("confluence", _NO_KB_ANS)
has_conf    = conf_answer != _NO_KB_ANS
conf_sources = _pdf_source_names(conf_expanded)

# SharePoint
sp_answer = ans_dict.get("sharepoint", _NO_KB_ANS)
has_sp    = sp_answer != _NO_KB_ANS
sp_sources = _pdf_source_names(sp_expanded)

return {
    "video_answer":       video_answer,
    "video_timestamp":    timestamp,
    "video_source":       vid_source,
    "kb_answer":          kb_answer,
    "kb_sources":         _pdf_source_names(pdf_expanded),
    "confluence_answer":  conf_answer,
    "confluence_sources": conf_sources,
    "sharepoint_answer":  sp_answer,
    "sharepoint_sources": sp_sources,
    "has_video":          has_video,
    "has_kb":             has_kb,
    "has_confluence":     has_conf,
    "has_sharepoint":     has_sp,
}
```