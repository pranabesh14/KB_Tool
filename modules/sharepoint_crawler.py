“””
sharepoint_crawler.py - SharePoint On-Premise content crawler.

Fetches documents and videos from SharePoint document libraries
via the SharePoint REST API, authenticated via Azure AD (same App
Registration as Confluence).

Videos are downloaded to TEMP_DIR and transcribed with Whisper
(same pipeline as local video files).

PDF/Word documents are extracted and chunked same as Confluence attachments.

## Public API

crawl_sharepoint_url(url) -> List[Document]
Crawl a SharePoint site/library URL and return indexable Documents.
“””

import os
import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(**file**)))
if _PROJECT_ROOT not in sys.path:
sys.path.insert(0, _PROJECT_ROOT)

import io
import uuid
from typing import List, Optional
from urllib.parse import urlparse, quote

import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from modules.config import (
SHAREPOINT_BASE_URL, SHAREPOINT_CACHE_DIR, TEMP_DIR,
CHUNK_SIZE_PDF, CHUNK_OVERLAP, get_logger,
)
from modules.confluence_auth import get_headers

logger = get_logger(“sharepoint_crawler”)

_REQUEST_TIMEOUT = 30
_DOC_EXTS        = {”.pdf”, “.doc”, “.docx”, “.txt”}
_VIDEO_EXTS      = {”.mp4”, “.mkv”, “.avi”, “.mov”, “.webm”, “.m4v”, “.wmv”}
_AUDIO_EXTS      = {”.mp3”, “.wav”, “.aac”, “.flac”, “.m4a”}
_MEDIA_EXTS      = _VIDEO_EXTS | _AUDIO_EXTS

# — Helpers —————————————————————––

def _chunk(text: str, metadata: dict) -> List[Document]:
if not text.strip():
return []
splitter = RecursiveCharacterTextSplitter(
chunk_size=CHUNK_SIZE_PDF, chunk_overlap=CHUNK_OVERLAP
)
raw = splitter.create_documents([text], metadatas=[metadata])
return [
Document(page_content=d.page_content, metadata={**d.metadata, “chunk_index”: i})
for i, d in enumerate(raw)
]

def _extract_site_and_path(url: str):
“””
Split a SharePoint URL into (site_url, relative_path).
e.g. https://sp.company.com/sites/MyTeam/Shared%20Documents/file.pdf
-> site_url  = https://sp.company.com/sites/MyTeam
-> rel_path  = /Shared Documents/file.pdf
“””
parsed   = urlparse(url)
parts    = parsed.path.strip(”/”).split(”/”)
# Find the site collection boundary (/sites/Name or /teams/Name)
site_end = 2  # default: take first two path segments
for i, p in enumerate(parts):
if p.lower() in (“sites”, “teams”, “personal”):
site_end = i + 2
break
site_path = “/” + “/”.join(parts[:site_end])
rel_path  = “/” + “/”.join(parts[site_end:]) if len(parts) > site_end else “”
site_url  = f”{parsed.scheme}://{parsed.netloc}{site_path}”
return site_url, rel_path

def _sp_api(site_url: str, endpoint: str) -> str:
“”“Build a SharePoint REST API URL.”””
return f”{site_url.rstrip(’/’)}/_api/{endpoint.lstrip(’/’)}”

# — SharePoint REST API calls ———————————————––

def _list_folder(site_url: str, folder_rel_path: str) -> List[dict]:
“””
List all files in a SharePoint folder (non-recursive).
Returns list of file metadata dicts.
“””
encoded = quote(folder_rel_path)
api_url = _sp_api(
site_url,
f”web/GetFolderByServerRelativeUrl(’{encoded}’)/Files”
“?$select=Name,ServerRelativeUrl,Length,TimeLastModified&$top=500”
)
try:
resp = requests.get(api_url, headers=get_headers(“sharepoint”),
timeout=_REQUEST_TIMEOUT)
resp.raise_for_status()
return resp.json().get(“value”, [])
except Exception as exc:
logger.error(“Error listing folder %s: %s”, folder_rel_path, exc)
return []

def _list_subfolders(site_url: str, folder_rel_path: str) -> List[str]:
“”“Return server-relative paths of immediate subfolders.”””
encoded = quote(folder_rel_path)
api_url = _sp_api(
site_url,
f”web/GetFolderByServerRelativeUrl(’{encoded}’)/Folders”
“?$select=ServerRelativeUrl&$top=200”
)
try:
resp = requests.get(api_url, headers=get_headers(“sharepoint”),
timeout=_REQUEST_TIMEOUT)
resp.raise_for_status()
return [f[“ServerRelativeUrl”] for f in resp.json().get(“value”, [])]
except Exception as exc:
logger.warning(“Error listing subfolders %s: %s”, folder_rel_path, exc)
return []

def _stream_file_to_disk(
site_url: str, server_rel_url: str, dest_path: str,
chunk_size: int = 8 * 1024 * 1024,
) -> bool:
“””
Stream a SharePoint file directly to disk in chunks.
Uses constant RAM (chunk_size per chunk) regardless of file size.
A 2 GB video uses only 8 MB of ram during download.


Parameters
----------
chunk_size : int
    Bytes per chunk (default 8 MB).

Returns True on success, False on failure.
"""
encoded = quote(server_rel_url)
api_url = _sp_api(
    site_url,
    f"web/GetFileByServerRelativeUrl('{encoded}')/$value"
)
try:
    resp = requests.get(
        api_url,
        headers=get_headers("sharepoint"),
        timeout=300,    # large files need more time
        stream=True,    # do NOT load into memory
    )
    resp.raise_for_status()

    downloaded = 0
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:   # filter out keep-alive empty chunks
                f.write(chunk)
                downloaded += len(chunk)

    mb = downloaded / (1024 * 1024)
    logger.info("Streamed %.1f MB to disk: %s", mb, os.path.basename(dest_path))
    return True

except Exception as exc:
    logger.error("Error streaming %s: %s", server_rel_url, exc)
    # Remove partial file if download failed
    if os.path.exists(dest_path):
        try:
            os.remove(dest_path)
        except Exception:
            pass
    return False


def _get_single_file_metadata(site_url: str, server_rel_url: str) -> Optional[dict]:
“”“Get metadata for a single file by its server-relative URL.”””
encoded = quote(server_rel_url)
api_url = _sp_api(
site_url,
f”web/GetFileByServerRelativeUrl(’{encoded}’)”
“?$select=Name,ServerRelativeUrl,Length,TimeLastModified”
)
try:
resp = requests.get(api_url, headers=get_headers(“sharepoint”),
timeout=_REQUEST_TIMEOUT)
resp.raise_for_status()
return resp.json()
except Exception as exc:
logger.error(“Error getting file metadata %s: %s”, server_rel_url, exc)
return None

# — Document extraction —————————————————––

def _extract_doc_text(content: bytes, ext: str, filename: str) -> str:
try:
if ext == “.pdf”:
from PyPDF2 import PdfReader
return “\n”.join(
p.extract_text() or “”
for p in PdfReader(io.BytesIO(content)).pages
)
if ext in (”.doc”, “.docx”):
import mammoth
return mammoth.extract_raw_text(io.BytesIO(content)).value
if ext == “.txt”:
return content.decode(“utf-8”, errors=“ignore”)
except Exception as exc:
logger.warning(“Could not extract text from %s: %s”, filename, exc)
return “”

# — Video transcription —————————————————––

def _transcribe_sharepoint_video(file_path: str, filename: str, file_url: str) -> List[Document]:
“””
Transcribe a SharePoint video/audio file that has already been streamed to disk.
File is deleted after transcription.


Parameters
----------
file_path : str  Path to the video/audio file on disk (in TEMP_DIR).
filename  : str  Original filename (used for logging and metadata).
file_url  : str  SharePoint URL (used as source in metadata and cache key).

Returns
-------
List[Document] with source_type="sharepoint_video".
"""
from modules.transcription import _convert_to_numpy, _get_whisper
import json as _json
import hashlib
from modules.config import DATA_DIR
from modules.document_loader import _load_video_json

# Check transcript cache first - avoids re-transcribing if already done
cache_key  = "sp_transcript_" + hashlib.md5(file_url.encode()).hexdigest()[:12]
cache_path = os.path.normpath(os.path.join(DATA_DIR, cache_key + ".json"))

if os.path.exists(cache_path):
    logger.info("Transcript cache hit for: %s", filename)
    docs = _load_video_json(cache_path)
    for d in docs:
        d.metadata["source_type"] = "sharepoint_video"
        d.metadata["filename"]    = filename
    return docs

try:
    logger.info("Transcribing SharePoint video: %s (%.1f MB)",
                filename, os.path.getsize(file_path) / (1024 * 1024))

    # Convert to numpy (moviepy extracts audio, no system ffmpeg needed)
    audio_np = _convert_to_numpy(file_path)

    # Transcribe locally with Whisper
    result = _get_whisper().transcribe(audio_np)

    # Save transcript JSON (same format as yt-dlp + local video transcripts)
    payload = {
        "source":   file_url,
        "text":     result.get("text", ""),
        "segments": [
            {
                "start": float(s.get("start", 0.0)),
                "end":   float(s.get("end",   0.0)),
                "text":  (s.get("text") or "").strip(),
            }
            for s in (result.get("segments") or [])
        ],
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        _json.dump(payload, f, ensure_ascii=False, indent=2)

    docs = _load_video_json(cache_path)
    for d in docs:
        d.metadata["source_type"] = "sharepoint_video"
        d.metadata["filename"]    = filename

    logger.info("SharePoint video '%s' -> %d chunks", filename, len(docs))
    return docs

except Exception as exc:
    logger.error("Transcription failed for %s: %s", filename, exc)
    return []
finally:
    # Always delete the temp video file - transcript JSON is kept as cache
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.debug("Deleted temp video: %s", file_path)
        except Exception as del_exc:
            logger.warning("Could not delete temp file %s: %s", file_path, del_exc)


# — Public API ––––––––––––––––––––––––––––––––

def crawl_sharepoint_url(url: str) -> List[Document]:
“””
Crawl a SharePoint URL and return indexable Documents.


Supported URL types:
  - Single file  : https://sp.co/sites/Team/Shared Documents/report.pdf
  - Folder/library: https://sp.co/sites/Team/Shared Documents/MyFolder
  - Site root    : https://sp.co/sites/Team  (crawls default Shared Documents)

Documents: extracted as text, chunked same as PDFs.
Videos/Audio: downloaded and transcribed with Whisper.

Returns
-------
List[Document] with source_type "sharepoint", "sharepoint_video",
or "sharepoint_attachment".
"""
if not SHAREPOINT_BASE_URL:
    raise RuntimeError(
        "SHAREPOINT_BASE_URL is not set in .env\n"
        "Example: https://sharepoint.yourcompany.com"
    )

site_url, rel_path = _extract_site_and_path(url)
logger.info("SharePoint crawl: site=%s  path=%s", site_url, rel_path)

# Determine if URL is a single file or a folder
ext = os.path.splitext(rel_path)[1].lower()

if ext:
    # Single file
    return _process_file(site_url, rel_path, url)
else:
    # Folder - crawl recursively (one level deep for safety)
    return _crawl_folder(site_url, rel_path or "/Shared Documents", url)


def _crawl_folder(
site_url: str, folder_rel_path: str, base_url: str
) -> List[Document]:
“”“Crawl all files in a folder and immediate subfolders.”””
all_docs: List[Document] = []


# Files in current folder
files = _list_folder(site_url, folder_rel_path)
for file_meta in files:
    server_rel = file_meta.get("ServerRelativeUrl", "")
    file_url   = site_url + server_rel
    docs = _process_file(site_url, server_rel, file_url)
    all_docs.extend(docs)

# Immediate subfolders (one level)
for subfolder in _list_subfolders(site_url, folder_rel_path):
    sub_files = _list_folder(site_url, subfolder)
    for file_meta in sub_files:
        server_rel = file_meta.get("ServerRelativeUrl", "")
        file_url   = site_url + server_rel
        docs = _process_file(site_url, server_rel, file_url)
        all_docs.extend(docs)

logger.info(
    "SharePoint folder crawl done: %d chunks from %s",
    len(all_docs), folder_rel_path
)
return all_docs


def _process_file(
site_url: str, server_rel_url: str, file_url: str
) -> List[Document]:
“””
Stream and process a single SharePoint file.

Videos/audio: streamed directly to TEMP_DIR (constant RAM usage),
              then transcribed with Whisper, temp file deleted after.
Documents:    streamed to TEMP_DIR, text extracted, temp file deleted.

This approach uses only chunk_size RAM (8 MB default) regardless of
how large the file is - safe for multi-GB video files.
"""


filename = os.path.basename(server_rel_url)
ext      = os.path.splitext(filename)[1].lower()

if ext not in (_DOC_EXTS | _MEDIA_EXTS):
    logger.debug("Skipping unsupported file: %s", filename)
    return []

# Stream file to a temp path in TEMP_DIR
tmp_path = os.path.normpath(
    os.path.join(TEMP_DIR, f"sp_{uuid.uuid4().hex}{ext}")
)
logger.info("Streaming SharePoint file to disk: %s", filename)
ok = _stream_file_to_disk(site_url, server_rel_url, tmp_path)
if not ok:
    return []

try:
    # Video / audio -> transcribe with Whisper
    # _transcribe_sharepoint_video deletes tmp_path in its finally block
    if ext in _MEDIA_EXTS:
        return _transcribe_sharepoint_video(tmp_path, filename, file_url)

    # Document -> read bytes from disk, extract text, chunk
    with open(tmp_path, "rb") as f:
        raw_bytes = f.read()
    text = _extract_doc_text(raw_bytes, ext, filename)
    if not text:
        return []

    meta = {
        "source":      file_url,
        "source_type": "sharepoint",
        "filename":    filename,
        "site_url":    site_url,
    }
    docs = _chunk(text, meta)
    logger.info("SharePoint file '%s' -> %d chunks", filename, len(docs))
    return docs

except Exception as exc:
    logger.error("Error processing %s: %s", filename, exc)
    return []
finally:
    # Clean up temp file for documents
    # (videos are cleaned up inside _transcribe_sharepoint_video)
    if ext in _DOC_EXTS and os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except Exception:
            pass
