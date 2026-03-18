“””
document_loader.py - Load and chunk documents into LangChain Documents.

Supported inputs:

- PDF  (.pdf)  - extracted with PyPDF2, chunked with RecursiveCharacterTextSplitter
- Text (.txt)  - read directly, same chunker
- Video JSON   - transcript produced by transcription.py; grouped into ~1000-char chunks
  with timing metadata (start_time / end_time per chunk)

Every returned Document carries:
metadata = {
“source”:      <filename or video_url>,
“source_type”: “pdf” | “video”,
“chunk_index”: <int>,
# video only:
“start_time”:  <float seconds>,
“end_time”:    <float seconds>,
}
“””

import os
import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(**file**)))
if _PROJECT_ROOT not in sys.path:
sys.path.insert(0, _PROJECT_ROOT)

import json
import os
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from modules.config import CHUNK_SIZE_PDF, CHUNK_OVERLAP, CHUNK_SIZE_VIDEO, get_logger

logger = get_logger(“document_loader”)

# ─── Internal loaders ─────────────────────────────────────────────────────────

def _load_pdf(path: str) -> str:
from PyPDF2 import PdfReader
reader = PdfReader(path)
pages  = [page.extract_text() or “” for page in reader.pages]
return “\n”.join(pages)

def _load_text(path: str) -> str:
with open(path, “r”, encoding=“utf-8”, errors=“ignore”) as fh:
return fh.read()

def _chunk_text_doc(text: str, source_name: str) -> List[Document]:
splitter = RecursiveCharacterTextSplitter(
chunk_size=CHUNK_SIZE_PDF, chunk_overlap=CHUNK_OVERLAP
)
raw_docs = splitter.create_documents(
[text],
metadatas=[{“source”: source_name, “source_type”: “pdf”}],
)
docs = []
for i, d in enumerate(raw_docs):
md = dict(d.metadata)
md[“chunk_index”] = i
docs.append(Document(page_content=d.page_content, metadata=md))
return docs

def _load_video_json(path: str) -> List[Document]:
with open(path, “r”, encoding=“utf-8”) as fh:
data = json.load(fh)

```
# support both old key ("video_url") and new key ("source")
video_url = data.get("source") or data.get("video_url", "")
segments  = data.get("segments", [])

docs: List[Document] = []
buf: List[str] = []
cur_len   = 0
cur_start = None
cur_end   = None
chunk_idx = 0

for seg in segments:
    text = (seg.get("text") or "").strip()
    if not text:
        continue
    s_start = float(seg.get("start", 0.0))
    s_end   = float(seg.get("end", s_start))

    if cur_start is None:
        cur_start = s_start

    buf.append(text)
    cur_end  = s_end
    cur_len += len(text)

    if cur_len >= CHUNK_SIZE_VIDEO:
        content = "\n".join(buf).strip()
        if content:
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source":      video_url,
                        "source_type": "video",
                        "start_time":  cur_start,
                        "end_time":    cur_end,
                        "chunk_index": chunk_idx,
                    },
                )
            )
            chunk_idx += 1
        # one-segment overlap
        buf      = buf[-1:]
        cur_len  = sum(len(x) for x in buf)
        cur_start = s_start  # keep last segment's start as new chunk start

# flush remaining
if buf:
    content = "\n".join(buf).strip()
    if content:
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source":      video_url,
                    "source_type": "video",
                    "start_time":  cur_start if cur_start is not None else 0.0,
                    "end_time":    cur_end   if cur_end   is not None else 0.0,
                    "chunk_index": chunk_idx,
                },
            )
        )

return docs
```

# ─── Public API ───────────────────────────────────────────────────────────────

def load_and_split_document(path: str) -> List[Document]:
“””
Load a single file and return a list of chunked LangChain Documents.
Logs the error and returns [] on failure so the caller can continue
processing other files - but the error is visible in the log.
“””
if not os.path.isfile(path):
logger.error(“File not found - skipping: %s”, path)
return []

```
try:
    if path.endswith(".json"):
        docs = _load_video_json(path)
    else:
        source_name = os.path.basename(path)
        if path.lower().endswith(".pdf"):
            text = _load_pdf(path)
        else:
            text = _load_text(path)

        if not text.strip():
            logger.warning("File produced no text - is it empty or scanned? %s", path)
            return []

        docs = _chunk_text_doc(text, source_name)

    if not docs:
        logger.warning("No chunks produced for: %s", path)
    else:
        logger.info("Loaded %d chunks from: %s", len(docs), os.path.basename(path))
    return docs

except Exception as exc:
    logger.error("Failed to load %s: %s", os.path.basename(path), exc, exc_info=True)
    return []


def expected_source_for_file(path: str) -> str:
“””
Return the ‘source’ key that load_and_split_document will embed in metadata.
Used by the incremental indexer to check whether a file is already indexed.
“””
if path.endswith(”.json”):
try:
with open(path, “r”, encoding=“utf-8”) as fh:
data = json.load(fh)
return data.get(“source”) or data.get(“video_url”, “”) or os.path.basename(path)
except Exception:
return os.path.basename(path)
return os.path.basename(path)