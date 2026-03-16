“””
qa_memory.py — Long-term QA memory backed by FAISS.

Q&A pairs that were answered successfully are stored in a persistent FAISS
index (one each for video and KB).  On subsequent similar queries the cached
answer is surfaced first, reducing LLM calls and improving consistency.

## Public API

load_qa_indexes(embeddings)
save_qa_indexes()
add_to_qa_index(query, answer, source_type, embeddings)
search_qa_memory(query, source_type, k) -> str | None
“””

import os
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from modules.config import VIDEO_QA_PATH, KB_QA_PATH, get_logger

logger = get_logger(“qa_memory”)

# Module-level index singletons

_video_qa_index: Optional[FAISS] = None
_kb_qa_index:    Optional[FAISS] = None

_SKIP_ANSWERS = {
“the answer is not present in the videos.”,
“the answer is not present in the pdfs.”,
“the answer is not present in the documents.”,
}

# ─── Persistence ──────────────────────────────────────────────────────────────

def load_qa_indexes(embeddings) -> None:
global _video_qa_index, _kb_qa_index
for path, attr_name in [(VIDEO_QA_PATH, “_video_qa_index”), (KB_QA_PATH, “_kb_qa_index”)]:
if os.path.exists(path):
try:
idx = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
globals()[attr_name] = idx
logger.info(“Loaded QA index: %s”, path)
except Exception as exc:
logger.warning(“Could not load QA index %s: %s”, path, exc)

def save_qa_indexes() -> None:
if _video_qa_index:
try:
_video_qa_index.save_local(VIDEO_QA_PATH)
except Exception as exc:
logger.warning(“Failed to save video QA index: %s”, exc)
if _kb_qa_index:
try:
_kb_qa_index.save_local(KB_QA_PATH)
except Exception as exc:
logger.warning(“Failed to save KB QA index: %s”, exc)

# ─── CRUD ─────────────────────────────────────────────────────────────────────

def add_to_qa_index(query: str, answer: str, source_type: str, embeddings) -> None:
“”“Store a Q&A pair in the appropriate long-term memory index.”””
global _video_qa_index, _kb_qa_index
if not answer or answer.strip().lower() in _SKIP_ANSWERS:
return

```
doc = Document(
    page_content=f"Q: {query}\nA: {answer}",
    metadata={"source_type": source_type},
)
if source_type == "video":
    if _video_qa_index is None:
        _video_qa_index = FAISS.from_documents([doc], embeddings)
    else:
        _video_qa_index.add_documents([doc])
else:  # "pdf" / "kb"
    if _kb_qa_index is None:
        _kb_qa_index = FAISS.from_documents([doc], embeddings)
    else:
        _kb_qa_index.add_documents([doc])

save_qa_indexes()
```

def search_qa_memory(query: str, source_type: str, k: int = 1) -> Optional[str]:
“””
Returns the top cached Q&A text for the query, or None if not found.
source_type: “video” | “pdf”
“””
index = _video_qa_index if source_type == “video” else _kb_qa_index
if index is None:
return None
try:
hits = index.similarity_search(query, k=k)
if hits:
return hits[0].page_content
except Exception as exc:
logger.warning(“QA memory search failed: %s”, exc)
return None