“””
confluence_crawler.py - Confluence page crawler and content extractor.

Fetches Confluence pages via REST API, extracts clean text, follows
internal links up to CONFLUENCE_CRAWL_DEPTH levels, and downloads
PDF/doc attachments.

Returns LangChain Documents (same format as PDFs) ready for FAISS indexing.

## Public API

crawl_confluence_url(url, max_depth) -> List[Document]
“””

import os
import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(**file**)))
if _PROJECT_ROOT not in sys.path:
sys.path.insert(0, _PROJECT_ROOT)

import io
import json
import re
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from modules.config import (
CONFLUENCE_BASE_URL, CONFLUENCE_CACHE_DIR,
CONFLUENCE_CRAWL_DEPTH, CHUNK_SIZE_PDF, CHUNK_OVERLAP,
get_logger,
)
from modules.confluence_auth import get_headers

logger = get_logger(“confluence_crawler”)

_REST_API          = “/rest/api/content”
_ATTACH_EXTS       = {”.pdf”, “.doc”, “.docx”, “.txt”}
_REQUEST_TIMEOUT   = 30

# — Helpers —————————————————————––

def *cache_path(page_id: str) -> str:
return os.path.join(CONFLUENCE_CACHE_DIR, f”page*{page_id}.json”)

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

def _html_to_text(html: str) -> str:
soup = BeautifulSoup(html, “html.parser”)
for tag in soup([“script”, “style”, “nav”, “header”, “footer”]):
tag.decompose()
text = soup.get_text(separator=”\n”)
return re.sub(r”\n{3,}”, “\n\n”, text).strip()

def _extract_internal_links(html: str, base_url: str) -> List[str]:
soup  = BeautifulSoup(html, “html.parser”)
links, seen = [], set()
confluence_patterns = [”/pages/”, “/display/”, “/wiki/spaces/”, “pageId=”]

```
for tag in soup.find_all("a", href=True):
    href = tag["href"]
    if href.startswith("/"):
        href = urljoin(base_url, href)
    parsed = urlparse(href)
    if parsed.netloc and parsed.netloc != urlparse(base_url).netloc:
        continue
    if any(p in href for p in confluence_patterns) and href not in seen:
        seen.add(href)
        links.append(href)
logger.debug("Found %d internal Confluence links", len(links))
return links
```

def _url_to_page_id(url: str) -> Optional[str]:
for pattern in [r”pageId=(\d+)”, r”/pages/(\d+)[/?\s]”, r”/pages/(\d+)$”]:
m = re.search(pattern, url)
if m:
return m.group(1)
return None

def _search_by_title(space_key: str, title: str) -> Optional[str]:
search_url = (
f”{CONFLUENCE_BASE_URL.rstrip(’/’)}{_REST_API}”
f”?spaceKey={space_key}&title={requests.utils.quote(title)}&expand=version”
)
try:
resp = requests.get(search_url, headers=get_headers(“confluence”),
timeout=_REQUEST_TIMEOUT)
resp.raise_for_status()
results = resp.json().get(“results”, [])
if results:
return str(results[0][“id”])
except Exception as exc:
logger.warning(“Title search failed ‘%s’/’%s’: %s”, space_key, title, exc)
return None

# — REST API calls ————————————————————

def _fetch_page(page_id: str) -> Optional[dict]:
“”“Fetch page by ID, using local cache to avoid redundant API calls.”””
cache = _cache_path(page_id)
if os.path.exists(cache):
try:
with open(cache, “r”, encoding=“utf-8”) as f:
data = json.load(f)
logger.info(“Cache hit for page %s”, page_id)
return data
except Exception as exc:
logger.warning(“Cache read failed for %s: %s”, page_id, exc)

```
api_url = (
    f"{CONFLUENCE_BASE_URL.rstrip('/')}{_REST_API}/{page_id}"
    "?expand=body.storage,metadata.labels,space,ancestors"
)
logger.info("Fetching Confluence page: %s", api_url)
try:
    headers = get_headers("confluence")
    resp    = requests.get(api_url, headers=headers, timeout=_REQUEST_TIMEOUT)
    logger.info("Confluence API response: HTTP %d for page %s", resp.status_code, page_id)

    if resp.status_code == 401:
        raise RuntimeError(
            "HTTP 401 Unauthorized - invalid token or missing permissions. "
            "Check AZURE_AD_CLIENT_SECRET and API permissions in Azure Portal."
        )
    if resp.status_code == 403:
        raise RuntimeError(
            "HTTP 403 Forbidden - App Registration lacks permission for page "
            + str(page_id) + ". Check API permissions and admin consent."
        )
    if resp.status_code == 404:
        raise RuntimeError(
            "HTTP 404 Not Found for page " + str(page_id)
            + ". API URL: " + api_url
            + ". Check CONFLUENCE_BASE_URL in .env."
        )

    resp.raise_for_status()
    data = resp.json()

    with open(cache, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info("Fetched page %s: title='%s'", page_id, data.get("title", "?"))
    return data

except RuntimeError:
    raise
except requests.exceptions.ConnectionError as exc:
    raise RuntimeError(
        "Cannot connect to Confluence at " + CONFLUENCE_BASE_URL
        + ". Check CONFLUENCE_BASE_URL in .env and network connectivity."
    ) from exc
except Exception as exc:
    logger.error("Error fetching page %s: %s", page_id, exc, exc_info=True)
    return None
```

def _fetch_page_from_url(url: str) -> Optional[dict]:
“”“Resolve a Confluence page URL to page data via REST API.”””
logger.info(“Resolving Confluence URL: %s”, url)
page_id = _url_to_page_id(url)

```
if page_id:
    logger.info("Extracted page ID from URL: %s", page_id)
else:
    # Try /display/SPACE/Title format
    parsed = urlparse(url)
    parts  = [p for p in parsed.path.split("/") if p]
    logger.debug("URL path parts: %s", parts)

    if "display" in parts:
        idx   = parts.index("display")
        space = parts[idx + 1] if len(parts) > idx + 1 else None
        title = parts[idx + 2].replace("+", " ") if len(parts) > idx + 2 else None
        logger.info("Trying space/title lookup: space=%s title=%s", space, title)
        if space and title:
            page_id = _search_by_title(space, title)

    # Try /wiki/spaces/SPACE/pages/ID/Title format
    if not page_id and "spaces" in parts and "pages" in parts:
        pages_idx = parts.index("pages")
        if len(parts) > pages_idx + 1:
            candidate = parts[pages_idx + 1]
            if candidate.isdigit():
                page_id = candidate
                logger.info("Extracted page ID from /wiki/spaces/ URL: %s", page_id)

if not page_id:
    logger.error(
        "Could not extract page ID from URL: %s "
        "(supported formats: ?pageId=ID, /display/SPACE/Title, "
        "/wiki/spaces/SPACE/pages/ID/Title)",
        url
    )
    return None

return _fetch_page(page_id)
```

def _fetch_attachment_docs(page_id: str, page_url: str) -> List[Document]:
“”“Download and extract text from PDF/doc attachments on a page.”””
docs = []
url  = (
f”{CONFLUENCE_BASE_URL.rstrip(’/’)}{_REST_API}/{page_id}”
“/child/attachment?expand=version,metadata&limit=50”
)
try:
resp = requests.get(url, headers=get_headers(“confluence”),
timeout=_REQUEST_TIMEOUT)
resp.raise_for_status()
attachments = resp.json().get(“results”, [])
except Exception as exc:
logger.warning(“Could not list attachments for page %s: %s”, page_id, exc)
return docs

```
for att in attachments:
    filename = att.get("title", "")
    ext      = os.path.splitext(filename)[1].lower()
    if ext not in _ATTACH_EXTS:
        continue

    dl_path = att.get("_links", {}).get("download", "")
    if not dl_path:
        continue
    dl_url = CONFLUENCE_BASE_URL.rstrip("/") + dl_path

    try:
        ar = requests.get(dl_url, headers=get_headers("confluence"),
                          timeout=60, stream=True)
        ar.raise_for_status()
        text = _extract_file_text(ar.content, ext, filename)
        if text:
            meta = {
                "source":      f"{page_url}/attachments/{filename}",
                "source_type": "confluence_attachment",
                "page_id":     page_id,
                "filename":    filename,
            }
            att_docs = _chunk(text, meta)
            docs.extend(att_docs)
            logger.info("Attachment '%s' -> %d chunks", filename, len(att_docs))
    except Exception as exc:
        logger.warning("Could not process attachment %s: %s", filename, exc)
return docs
```

def _extract_file_text(content: bytes, ext: str, filename: str) -> str:
try:
if ext == “.pdf”:
from PyPDF2 import PdfReader
return “\n”.join(
p.extract_text() or “” for p in PdfReader(io.BytesIO(content)).pages
)
if ext in (”.doc”, “.docx”):
import mammoth
return mammoth.extract_raw_text(io.BytesIO(content)).value
if ext == “.txt”:
return content.decode(“utf-8”, errors=“ignore”)
except Exception as exc:
logger.warning(“Could not extract text from %s: %s”, filename, exc)
return “”

# — Public API ––––––––––––––––––––––––––––––––

def crawl_confluence_url(
url: str,
max_depth: int = CONFLUENCE_CRAWL_DEPTH,
) -> List[Document]:
“””
Crawl a Confluence page and all internal links up to max_depth levels.

```
Parameters
----------
url       : Starting Confluence page URL.
max_depth : Link follow depth (0 = starting page only, 2 = default).

Returns
-------
List[Document] chunked and labelled with source_type="confluence"
or "confluence_attachment".
"""
if not CONFLUENCE_BASE_URL:
    raise RuntimeError(
        "CONFLUENCE_BASE_URL is not set in .env\n"
        "Example: https://confluence.yourcompany.com"
    )

all_docs:     List[Document] = []
visited_ids:  Set[str]       = set()
queue: List[tuple]           = [(url, 0)]   # (url, depth)

logger.info("Confluence crawl: %s (max_depth=%d)", url, max_depth)

while queue:
    current_url, depth = queue.pop(0)
    if depth > max_depth:
        continue

    try:
        page_data = _fetch_page_from_url(current_url)
    except RuntimeError as exc:
        # Descriptive errors (auth, 404 etc.) - raise so UI shows them
        raise
    except Exception as exc:
        logger.error("Unexpected error fetching %s: %s", current_url, exc, exc_info=True)
        continue

    if not page_data:
        logger.warning("No data returned for URL: %s", current_url)
        continue

    page_id = str(page_data.get("id", ""))
    if page_id in visited_ids:
        continue
    visited_ids.add(page_id)

    title     = page_data.get("title", "Untitled")
    space_key = page_data.get("space", {}).get("key", "")
    page_url  = (
        CONFLUENCE_BASE_URL.rstrip("/")
        + page_data.get("_links", {}).get("webui", f"/pages/{page_id}")
    )
    body_html = (
        page_data.get("body", {}).get("storage", {}).get("value", "")
    )
    logger.info(
        "Page '%s': body length=%d chars", title, len(body_html)
    )

    # Page body
    if body_html:
        body_text = _html_to_text(body_html)
        logger.info("Page '%s': extracted text length=%d chars", title, len(body_text))
        if body_text:
            meta = {
                "source":      page_url,
                "source_type": "confluence",
                "page_id":     page_id,
                "title":       title,
                "space_key":   space_key,
                "crawl_depth": depth,
            }
            page_docs = _chunk(body_text, meta)
            all_docs.extend(page_docs)
            logger.info(
                "Page '%s' (depth %d) -> %d chunks", title, depth, len(page_docs)
            )

    # Attachments
    all_docs.extend(_fetch_attachment_docs(page_id, page_url))

    # Queue linked pages
    if depth < max_depth and body_html:
        for link in _extract_internal_links(body_html, CONFLUENCE_BASE_URL):
            if link not in [u for u, _ in queue]:
                queue.append((link, depth + 1))

logger.info(
    "Confluence crawl done: %d pages, %d chunks", len(visited_ids), len(all_docs)
)
return all_docs
