“””
meta_store.py - Persistent JSON registry of uploaded and indexed files.

Tracks:

- pdfs          - all PDF/TXT file paths ever added
- videos        - all transcript JSON file paths ever added
- indexed_pdfs  - subset that is already in the FAISS PDF index
- indexed_videos- subset that is already in the FAISS video index

## Public API

load_meta()             -> dict
save_meta(meta)
scan_data_dir(meta)     -> meta  (auto-discover files already in DATA_DIR)
“””

import os
import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(**file**)))
if _PROJECT_ROOT not in sys.path:
sys.path.insert(0, _PROJECT_ROOT)

import json
import os
from typing import Any, Dict

from modules.config import DATA_DIR, META_FILE, get_logger

logger = get_logger(“meta_store”)

_DEFAULT: Dict[str, Any] = {
“pdfs”:           [],
“videos”:         [],
“indexed_pdfs”:   [],
“indexed_videos”: [],
}

def load_meta() -> Dict[str, Any]:
if os.path.exists(META_FILE):
try:
with open(META_FILE, “r”, encoding=“utf-8”) as fh:
meta = json.load(fh)
except Exception as exc:
logger.warning(“Could not read meta file: %s”, exc)
meta = {}
else:
meta = {}

```
# Ensure all keys exist
for k, default in _DEFAULT.items():
    meta.setdefault(k, list(default))

# Prune non-existent paths
for key in ("pdfs", "videos", "indexed_pdfs", "indexed_videos"):
    meta[key] = [p for p in meta[key] if os.path.exists(p)]

return meta
```

def save_meta(meta: Dict[str, Any]) -> None:
try:
with open(META_FILE, “w”, encoding=“utf-8”) as fh:
json.dump(meta, fh, ensure_ascii=False, indent=2)
except Exception as exc:
logger.warning(“Could not write meta file: %s”, exc)

def scan_data_dir(meta: Dict[str, Any]) -> Dict[str, Any]:
“””
Auto-discover PDF/TXT and JSON files already sitting in DATA_DIR and
merge them into the meta registry.  Deduplicates.
“””
existing_pdfs = [
os.path.join(DATA_DIR, f)
for f in os.listdir(DATA_DIR)
if f.lower().endswith((”.pdf”, “.txt”))
]
existing_vids = [
os.path.join(DATA_DIR, f)
for f in os.listdir(DATA_DIR)
if f.lower().endswith(”.json”) and f != os.path.basename(META_FILE)
]

meta["pdfs"]   = sorted(set(meta["pdfs"]   + existing_pdfs))
meta["videos"] = sorted(set(meta["videos"] + existing_vids))
return meta