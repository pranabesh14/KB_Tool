“””
vector_store.py - FAISS index management (build / load / incremental update).

Each index directory contains:

- FAISS binary files  (index.faiss, index.pkl)
- doc_store.json      - ordered mapping: source -> [{“content”: …, “metadata”: …}]
  used for neighbour-chunk expansion at query time.

## Public API

build_index_from_files(files, embeddings, index_dir)  -> (vs, doc_store)
load_index(index_dir, embeddings)                      -> (vs | None, doc_store)
update_index_incremental(files, embeddings, index_dir) -> (vs, doc_store, added_sources)
“””

import os
import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(**file**)))
if _PROJECT_ROOT not in sys.path:
sys.path.insert(0, _PROJECT_ROOT)

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from modules.document_loader import load_and_split_document, expected_source_for_file
from modules.config import get_logger

logger = get_logger(“vector_store”)

DocStore = Dict[str, List[Dict[str, Any]]]

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _doc_store_path(index_dir: str) -> str:
return os.path.join(index_dir, “doc_store.json”)

def _save_doc_store(doc_store: DocStore, index_dir: str) -> None:
try:
with open(_doc_store_path(index_dir), “w”, encoding=“utf-8”) as fh:
json.dump(doc_store, fh, ensure_ascii=False, indent=2)
except Exception as exc:
logger.warning(“Failed to save doc_store at %s: %s”, index_dir, exc)

def _load_doc_store(index_dir: str) -> DocStore:
path = _doc_store_path(index_dir)
if os.path.exists(path):
try:
with open(path, “r”, encoding=“utf-8”) as fh:
return json.load(fh)
except Exception as exc:
logger.warning(“Failed to load doc_store at %s: %s”, path, exc)
return {}

def _populate_doc_store(docs: List[Document], doc_store: DocStore, fallback_name: str) -> None:
“”“Append documents to the in-memory doc_store dict (mutates in place).”””
for d in docs:
src = (d.metadata or {}).get(“source”) or fallback_name
doc_store.setdefault(src, []).append(
{“content”: d.page_content, “metadata”: d.metadata or {}}
)

# ─── Public API ───────────────────────────────────────────────────────────────

def build_index_from_files(
files: List[str],
embeddings,
index_dir: str,
) -> Tuple[Optional[FAISS], DocStore]:
“”“Build a fresh FAISS index from a list of file paths. Overwrites existing index.”””
all_docs: List[Document] = []
doc_store: DocStore = {}

```
for path in files:
    file_docs = load_and_split_document(path)
    all_docs.extend(file_docs)
    _populate_doc_store(file_docs, doc_store, os.path.basename(path))

if not all_docs:
    logger.warning("No documents found for index at %s", index_dir)
    return None, {}

os.makedirs(index_dir, exist_ok=True)
vs = FAISS.from_documents(all_docs, embeddings)
vs.save_local(index_dir)
_save_doc_store(doc_store, index_dir)
logger.info("Built index at %s with %d chunks.", index_dir, len(all_docs))
return vs, doc_store
```

def load_index(
index_dir: str,
embeddings,
) -> Tuple[Optional[FAISS], DocStore]:
“””
Load an existing FAISS index + doc_store from disk.
Returns (None, {}) safely on first run when no index exists yet.
“””
# Both files must exist for a valid FAISS index
index_faiss = os.path.join(index_dir, “index.faiss”)
index_pkl   = os.path.join(index_dir, “index.pkl”)

```
if not os.path.isdir(index_dir):
    logger.info("First run - index directory not created yet: %s", index_dir)
    return None, {}

if not os.path.isfile(index_faiss) or not os.path.isfile(index_pkl):
    logger.info("First run - index files not built yet in: %s", index_dir)
    return None, {}

try:
    vs = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    doc_store = _load_doc_store(index_dir)
    logger.info("Loaded FAISS index from %s (%d sources).", index_dir, len(doc_store))
    return vs, doc_store
except Exception as exc:
    logger.warning("Failed to load index at %s: %s", index_dir, exc)
    return None, {}
```

def update_index_incremental(
files: List[str],
embeddings,
index_dir: str,
) -> Tuple[Optional[FAISS], DocStore, List[str]]:
“””
Append only NEW files (by source key) to an existing index.
Returns (vs, doc_store, list_of_newly_added_sources).
Raises RuntimeError if embeddings fail so the caller can show the user.
“””
vs_existing, doc_store = load_index(index_dir, embeddings)
existing_sources = set(doc_store.keys())

```
# Filter to genuinely new files
new_files = [
    f for f in files
    if expected_source_for_file(f) not in existing_sources
]
added_sources: List[str] = [expected_source_for_file(f) for f in new_files]

if not new_files:
    logger.info("No new files to index at %s.", index_dir)
    return vs_existing, doc_store, []

# Load and chunk documents
new_docs: List[Document] = []
failed_files: List[str] = []
for path in new_files:
    file_docs = load_and_split_document(path)
    if file_docs:
        new_docs.extend(file_docs)
        _populate_doc_store(file_docs, doc_store, os.path.basename(path))
    else:
        failed_files.append(os.path.basename(path))

if failed_files:
    logger.warning("These files produced no chunks: %s", ", ".join(failed_files))

if not new_docs:
    logger.warning("No documents to index - all files failed to load for %s.", index_dir)
    return vs_existing, doc_store, []

# Build FAISS index - let this raise so the caller sees embedding errors
logger.info("Building FAISS index for %d chunks in %s ...", len(new_docs), index_dir)
try:
    vs_new = FAISS.from_documents(new_docs, embeddings)
except Exception as exc:
    logger.error("FAISS index creation failed: %s", exc, exc_info=True)
    raise RuntimeError(
        f"Failed to build FAISS index at {index_dir}.\n"
        f"Most likely cause: embeddings are not working correctly.\n"
        f"Error: {exc}"
    ) from exc

if vs_existing is None:
    vs_existing = vs_new
else:
    vs_existing.merge_from(vs_new)

os.makedirs(index_dir, exist_ok=True)
vs_existing.save_local(index_dir)
_save_doc_store(doc_store, index_dir)
logger.info(
    "Index updated at %s - added %d source(s), %d chunks.",
    index_dir, len(added_sources), len(new_docs)
)
return vs_existing, doc_store, added_sources
