“””
app.py - Streamlit front-end for KB Tool.

Responsibilities (UI layer only):

- File upload & video URL input
- “Process” button -> delegate to vector_store helpers
- Chat interface -> delegate to query_handler
- Sidebar: New Chat + recent chat history

All heavy logic lives in the modules/ package.
“””

import os
import sys
import warnings

# Suppress LangChain deprecation warnings (ConversationEntityMemory migration noise)

warnings.filterwarnings(“ignore”, category=DeprecationWarning, module=“langchain”)
warnings.filterwarnings(“ignore”, message=”.*migrating_memory.*”)

# ── Ensure the project root is on sys.path so ‘modules’ is always importable

# regardless of the directory Streamlit is launched from.

_PROJECT_ROOT = os.path.dirname(os.path.abspath(**file**))
if _PROJECT_ROOT not in sys.path:
sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
try:
# LangChain >= 0.2 moved memory classes to langchain-classic
from langchain_classic.memory import ConversationEntityMemory
except ImportError:
# Fallback for older langchain installations
from langchain.memory import ConversationEntityMemory  # type: ignore

# ── Internal modules ──────────────────────────────────────────────────────────

from modules.config import (
DATA_DIR, INDEX_DIR_PDF, INDEX_DIR_VIDEO, INDEX_DIR_CONFLUENCE,
TOP_K_RETRIEVAL, RERANK_K, NEIGHBOR_WINDOW,
get_logger,
)

logger = get_logger(“app”)
from modules.llm_client import init_llm_and_embeddings
from modules.meta_store import load_meta, save_meta, scan_data_dir
from modules.transcription import transcribe, is_supported_video, _get_ffmpeg_dir_for_ytdlp

# Pre-check bundled ffmpeg at startup so any download happens early

@st.cache_resource
def _preload_ffmpeg():
ffmpeg_dir = _get_ffmpeg_dir_for_ytdlp()
if ffmpeg_dir:
logger.info(“Bundled ffmpeg ready at startup: %s”, ffmpeg_dir)
else:
logger.warning(“Bundled ffmpeg not available at startup - URL transcription may fail”)
return ffmpeg_dir

_ffmpeg_dir = _preload_ffmpeg()
from modules.vector_store import load_index, update_index_incremental
from modules.document_loader import expected_source_for_file
from modules.query_handler import get_combined_answer
from modules.confluence_crawler import crawl_confluence_url
from modules.sharepoint_crawler import crawl_sharepoint_url
from modules.confluence_auth import clear_token_cache
from modules.confluence_crawler import crawl_confluence_url
from modules.confluence_auth import clear_token_cache

# ─────────────────────────────────────────────────────────────────────────────

# Page config  (must be first Streamlit call)

# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title=“🧠 Knowledge Explorer”, layout=“wide”)

# ─────────────────────────────────────────────────────────────────────────────

# Shared resources  (cached across reruns)

# ─────────────────────────────────────────────────────────────────────────────

llm, embeddings = init_llm_and_embeddings()

# ─────────────────────────────────────────────────────────────────────────────

# Session-state initialisation

# ─────────────────────────────────────────────────────────────────────────────

def _init_session():
defaults = {
“messages”:             [],
“chat_history”:         [],
“new_chat_count”:       0,
“pdf_files”:            [],
“video_jsons”:          [],
“confluence_urls”:      [],
“current_chat_index”:   None,
“vs_pdf”:               None,
“vs_pdf_store”:         {},
“vs_video”:             None,
“vs_video_store”:       {},
“vs_confluence”:        None,
“vs_confluence_store”:  {},
}
for k, v in defaults.items():
if k not in st.session_state:
st.session_state[k] = v

```
if "entity_memory" not in st.session_state:
    st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=10)
```

_init_session()

# ─────────────────────────────────────────────────────────────────────────────

# Startup: load meta + indexes  (only once per process)

# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def _startup_load():
try:
meta = load_meta()
meta = scan_data_dir(meta)
save_meta(meta)
vs_pdf,   pdf_store  = load_index(INDEX_DIR_PDF,         embeddings)
vs_video, vid_store  = load_index(INDEX_DIR_VIDEO,       embeddings)
vs_conf,  conf_store = load_index(INDEX_DIR_CONFLUENCE,  embeddings)
vs_sp,    sp_store   = load_index(INDEX_DIR_SHAREPOINT,  embeddings)
logger.info(
“Startup: PDF=%s, Video=%s, Confluence=%s, SharePoint=%s”,
“loaded” if vs_pdf   else “empty”,
“loaded” if vs_video else “empty”,
“loaded” if vs_conf  else “empty”,
“loaded” if vs_sp    else “empty”,
)
return meta, vs_pdf, pdf_store, vs_video, vid_store, vs_conf, conf_store, vs_sp, sp_store
except Exception as exc:
logger.error(“Startup load failed: %s”, exc)
return {}, None, {}, None, {}, None, {}, None, {}

_meta, _vs_pdf, _pdf_store, _vs_video, _vid_store, _vs_conf, _conf_store, _vs_sp, _sp_store = _startup_load()

if st.session_state.vs_pdf is None:
st.session_state.vs_pdf              = _vs_pdf
st.session_state.vs_pdf_store        = _pdf_store
if st.session_state.vs_video is None:
st.session_state.vs_video            = _vs_video
st.session_state.vs_video_store      = _vid_store
if st.session_state.vs_confluence is None:
st.session_state.vs_confluence       = _vs_conf
st.session_state.vs_confluence_store = _conf_store
if st.session_state.vs_sharepoint is None:
st.session_state.vs_sharepoint       = _vs_sp
st.session_state.vs_sharepoint_store = _sp_store

if not st.session_state.pdf_files:
st.session_state.pdf_files       = list(_meta.get(“pdfs”,            []))
if not st.session_state.video_jsons:
st.session_state.video_jsons     = list(_meta.get(“videos”,          []))
if not st.session_state.confluence_urls:
st.session_state.confluence_urls = list(_meta.get(“confluence_urls”, []))
if not st.session_state.sharepoint_urls:
st.session_state.sharepoint_urls = list(_meta.get(“sharepoint_urls”, []))

# ─────────────────────────────────────────────────────────────────────────────

# Helpers

# ─────────────────────────────────────────────────────────────────────────────

def _new_entity_memory():
return ConversationEntityMemory(llm=llm, k=10)

def _crawl_and_index(
urls_str: str, crawl_fn, index_dir: str,
session_key: str, store_key: str,
meta_key: str, label: str, meta: dict,
) -> None:
“””
Shared helper: crawl a list of URLs, index into FAISS, update session state.
Used for both Confluence and SharePoint.
“””
import json
from langchain_community.vectorstores import FAISS as FAISSStore

```
urls      = [u.strip() for u in urls_str.split(",") if u.strip()]
all_docs  = []

with st.spinner(f"Fetching {len(urls)} {label} URL(s)..."):
    for url in urls:
        try:
            docs = crawl_fn(url)
            all_docs.extend(docs)
            if url not in st.session_state.get(f"{label.lower()}_urls", []):
                st.session_state.setdefault(f"{label.lower()}_urls", []).append(url)
            meta.setdefault(meta_key, [])
            if url not in meta[meta_key]:
                meta[meta_key].append(url)
            st.success(f"✅ {label}: fetched {len(docs)} chunks from {url}")
            logger.info("%s crawl: %d chunks from %s", label, len(docs), url)
        except Exception as exc:
            logger.error("%s crawl failed for %s: %s", label, url, exc, exc_info=True)
            st.error(f"❌ {label} failed: {url}\n{exc}")

if not all_docs:
    return

with st.spinner(f"Indexing {label} content ({len(all_docs)} chunks)..."):
    try:
        vs_existing, doc_store = load_index(index_dir, embeddings)

        # Build doc_store entries
        for d in all_docs:
            src = (d.metadata or {}).get("source", label.lower())
            doc_store.setdefault(src, []).append({
                "content": d.page_content, "metadata": d.metadata or {}
            })

        vs_new = FAISSStore.from_documents(all_docs, embeddings)
        if vs_existing is None:
            vs_existing = vs_new
        else:
            vs_existing.merge_from(vs_new)

        os.makedirs(index_dir, exist_ok=True)
        vs_existing.save_local(index_dir)
        with open(os.path.join(index_dir, "doc_store.json"), "w", encoding="utf-8") as f:
            json.dump(doc_store, f, ensure_ascii=False, indent=2)

        st.session_state[session_key] = vs_existing
        st.session_state[store_key]   = doc_store
        st.success(f"✅ {label} index updated with {len(all_docs)} chunks.")
        logger.info("%s index updated: %d chunks", label, len(all_docs))
    except Exception as exc:
        logger.error("%s indexing failed: %s", label, exc, exc_info=True)
        st.error(f"❌ {label} indexing failed:\n{exc}")
```

def _generate_chat_title(messages: list) -> str:
for msg in messages:
if msg[“role”] == “user” and msg[“content”].strip():
words = msg[“content”].split()
return “ “.join(words[:6]) + (”…” if len(words) > 6 else “”)
return “New Chat”

def _save_or_update_chat_history():
title = _generate_chat_title(st.session_state.messages)
idx   = st.session_state.current_chat_index

```
if idx is not None and idx < len(st.session_state.chat_history):
    entry = st.session_state.chat_history.pop(idx)
    entry["messages"] = st.session_state.messages.copy()
    entry["title"]    = title
    st.session_state.chat_history.insert(0, entry)
else:
    st.session_state.chat_history.insert(0, {
        "title":    title,
        "messages": st.session_state.messages.copy(),
    })

st.session_state.current_chat_index = 0
st.session_state.chat_history = st.session_state.chat_history[:5]
```

# ─────────────────────────────────────────────────────────────────────────────

# Sidebar

# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
if st.button(“➕ New Chat”):
st.session_state.messages          = []
st.session_state.pdf_files         = []
st.session_state.video_jsons       = []
st.session_state.confluence_urls   = []
st.session_state.sharepoint_urls   = []
st.session_state.new_chat_count   += 1
st.session_state.entity_memory     = _new_entity_memory()
st.session_state.current_chat_index = None
st.rerun()

```
st.markdown("---")
st.markdown("### 🕘 Recent Chats")

if not st.session_state.chat_history:
    st.info("No previous chats.")
else:
    for i, chat in enumerate(st.session_state.chat_history[:5]):
        if st.button(chat.get("title", f"Chat {i+1}"), key=f"load_chat_{i}"):
            st.session_state.messages           = chat["messages"].copy()
            st.session_state.current_chat_index = i
            st.session_state.entity_memory      = _new_entity_memory()
            st.rerun()
```

# ─────────────────────────────────────────────────────────────────────────────

# Header

# ─────────────────────────────────────────────────────────────────────────────

st.markdown(”””

<div style="text-align:center; font-size:48px; font-weight:bold;
            background:linear-gradient(to right,#ffffff,#ffffff,#FFB84D);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
🧠 Knowledge Explorer
</div>
<div style="text-align:center; font-size:20px; color:#FFD580; margin-top:5px;">
Upload - Ask - Discover
</div>
""", unsafe_allow_html=True)

st.markdown(”—”)

# ── First-run banner - shown when no files have been indexed yet ───────────────

if st.session_state.vs_pdf is None and st.session_state.vs_video is None:
st.info(
“👋 **Welcome!** No knowledge base exists yet.\n\n”
“1. Upload PDF / TXT files or video files below (or paste a video URL)\n”
“2. Click **⚙️ Process** to build the index\n”
“3. Then ask your questions in the chat”,
icon=“ℹ️”,
)

# ─────────────────────────────────────────────────────────────────────────────

# File upload

# ─────────────────────────────────────────────────────────────────────────────

uploaded_files = st.file_uploader(
“📄 Upload PDF / TXT files”,
type=[“pdf”, “txt”],
accept_multiple_files=True,
key=f”uploader_{st.session_state.new_chat_count}”,
)

if uploaded_files:
meta = load_meta()
for uf in uploaded_files:
save_path = os.path.join(DATA_DIR, uf.name)
with open(save_path, “wb”) as fh:
fh.write(uf.read())
if save_path not in st.session_state.pdf_files:
st.session_state.pdf_files.append(save_path)
logger.info(“PDF/TXT uploaded: %s”, uf.name)
if save_path not in meta[“pdfs”]:
meta[“pdfs”].append(save_path)
save_meta(meta)

video_files_uploaded = st.file_uploader(
“🎬 Upload video / audio files”,
type=[“mp4”, “mkv”, “avi”, “mov”, “webm”, “m4v”, “mp3”, “wav”, “aac”, “flac”, “m4a”],
accept_multiple_files=True,
key=f”video_uploader_{st.session_state.new_chat_count}”,
help=“Supported: mp4, mkv, avi, mov, webm, m4v, mp3, wav, aac, flac, m4a”,
)

if video_files_uploaded:
meta = load_meta()
for vf in video_files_uploaded:
save_path = os.path.join(DATA_DIR, vf.name)
with open(save_path, “wb”) as fh:
fh.write(vf.read())
if save_path not in st.session_state.get(“raw_video_files”, []):
st.session_state.setdefault(“raw_video_files”, []).append(save_path)
logger.info(“Video file uploaded: %s”, vf.name)
save_meta(meta)

video_urls_input = st.text_input(
“🌐 Or paste platform video URLs (Dailymotion, Vimeo, etc.) - comma-separated”,
key=f”video_urls_{st.session_state.new_chat_count}”,
help=“Supports any site yt-dlp can handle. SSL errors are retried automatically.”,
)

st.markdown(”—”)

# ── Confluence + SharePoint inputs ────────────────────────────────────────────

c1, c2, c3 = st.columns([3, 3, 1])
with c1:
confluence_urls_input = st.text_input(
“🔗 Confluence Page URLs (comma-separated)”,
key=f”conf_urls_{st.session_state.new_chat_count}”,
help=“Pages will be fetched + 2 levels of internal links followed automatically.”,
)
with c2:
sharepoint_urls_input = st.text_input(
“📁 SharePoint URLs (comma-separated)”,
key=f”sp_urls_{st.session_state.new_chat_count}”,
help=“Paste a SharePoint file, folder, or document library URL.”,
)
with c3:
st.markdown(”<br>”, unsafe_allow_html=True)
if st.button(“🔓 Re-auth”, help=“Clear cached Azure AD token and re-authenticate”):
clear_token_cache()
st.success(“Token cleared - you will be prompted to log in on next Process.”)

# ─────────────────────────────────────────────────────────────────────────────

# Process button

# ─────────────────────────────────────────────────────────────────────────────

if st.button(“⚙️ Process”):
logger.info(“Process button clicked. PDF files: %d, Video files: %d”,
len(st.session_state.pdf_files),
len(st.session_state.get(“raw_video_files”, [])))
meta = load_meta()

```
# ── Transcribe platform URLs ───────────────────────────────────────────
if video_urls_input.strip():
    urls = [u.strip() for u in video_urls_input.split(",") if u.strip()]
    with st.spinner(f"Downloading & transcribing {len(urls)} URL(s)..."):
        for url in urls:
            try:
                jpath = transcribe(url)
                if jpath not in st.session_state.video_jsons:
                    st.session_state.video_jsons.append(jpath)
                if jpath not in meta["videos"]:
                    meta["videos"].append(jpath)
                logger.info("URL transcribed: %s", url)
            except Exception as exc:
                logger.error("URL transcription failed for %s: %s", url, exc)
                st.error(f"❌ Could not transcribe {url}\n{exc}")
        save_meta(meta)

# ── Transcribe newly uploaded video files ─────────────────────────────
raw_videos = st.session_state.get("raw_video_files", [])
if raw_videos:
    with st.spinner(f"Transcribing {len(raw_videos)} uploaded video(s)..."):
        for video_path in raw_videos:
            try:
                jpath = transcribe(video_path)
                if jpath not in st.session_state.video_jsons:
                    st.session_state.video_jsons.append(jpath)
                if jpath not in meta["videos"]:
                    meta["videos"].append(jpath)
            except Exception as exc:
                logger.error("Transcription failed for %s: %s", os.path.basename(video_path), exc)
                st.error(f"❌ Failed to transcribe {os.path.basename(video_path)}: {exc}")
        st.session_state.raw_video_files = []
        save_meta(meta)

# ── Update PDF index ───────────────────────────────────────────────────
with st.spinner("Updating document index..."):
    try:
        vs_pdf, pdf_store, added_pdf = update_index_incremental(
            st.session_state.pdf_files, embeddings, INDEX_DIR_PDF
        )
        st.session_state.vs_pdf       = vs_pdf
        st.session_state.vs_pdf_store = pdf_store
        for f in st.session_state.pdf_files:
            src = expected_source_for_file(f)
            if src in added_pdf and f not in meta["indexed_pdfs"]:
                meta["indexed_pdfs"].append(f)
        save_meta(meta)
        if added_pdf:
            st.success(f"✅ Indexed {len(added_pdf)} new document(s).")
        else:
            st.info("ℹ️ No new documents to index.")
    except Exception as exc:
        logger.error("PDF indexing failed: %s", exc, exc_info=True)
        st.error(f"❌ Document indexing failed:\n{exc}")

# ── Update Video index ─────────────────────────────────────────────────
with st.spinner("Updating video index..."):
    try:
        vs_vid, vid_store, added_vid = update_index_incremental(
            st.session_state.video_jsons, embeddings, INDEX_DIR_VIDEO
        )
        st.session_state.vs_video       = vs_vid
        st.session_state.vs_video_store = vid_store
        for f in st.session_state.video_jsons:
            src = expected_source_for_file(f)
            if src in added_vid and f not in meta["indexed_videos"]:
                meta["indexed_videos"].append(f)
        save_meta(meta)
        if added_vid:
            st.success(f"✅ Indexed {len(added_vid)} new video transcript(s).")
    except Exception as exc:
        logger.error("Video indexing failed: %s", exc, exc_info=True)
        st.error(f"❌ Video indexing failed:\n{exc}")

# ── Crawl + index Confluence pages ────────────────────────────────────
if confluence_urls_input.strip():
    _crawl_and_index(
        urls_str=confluence_urls_input,
        crawl_fn=crawl_confluence_url,
        index_dir=INDEX_DIR_CONFLUENCE,
        session_key="vs_confluence",
        store_key="vs_confluence_store",
        meta_key="confluence_urls",
        label="Confluence",
        meta=meta,
    )
    save_meta(meta)

# ── Crawl + index SharePoint content ──────────────────────────────────
if sharepoint_urls_input.strip():
    _crawl_and_index(
        urls_str=sharepoint_urls_input,
        crawl_fn=crawl_sharepoint_url,
        index_dir=INDEX_DIR_SHAREPOINT,
        session_key="vs_sharepoint",
        store_key="vs_sharepoint_store",
        meta_key="sharepoint_urls",
        label="SharePoint",
        meta=meta,
    )
    save_meta(meta)
```

# ─────────────────────────────────────────────────────────────────────────────

# Chat interface

# ─────────────────────────────────────────────────────────────────────────────

st.markdown(”—”)

for msg in st.session_state.messages:
with st.chat_message(msg[“role”]):
st.markdown(msg[“content”])

user_q = st.chat_input(“Ask your question…”)

if user_q:
logger.info(“User query received: %.120s”, user_q)
st.session_state.messages.append({“role”: “user”, “content”: user_q})
with st.chat_message(“user”):
st.markdown(user_q)

```
with st.spinner("Thinking..."):
    result = get_combined_answer(
        user_q,
        (st.session_state.vs_video,      st.session_state.vs_video_store),
        (st.session_state.vs_pdf,         st.session_state.vs_pdf_store),
        llm,
        embeddings,
        st.session_state.entity_memory,
        confluence_index_tuple=(
            st.session_state.vs_confluence,
            st.session_state.vs_confluence_store
        ),
        sharepoint_index_tuple=(
            st.session_state.vs_sharepoint,
            st.session_state.vs_sharepoint_store
        ),
        top_k=TOP_K_RETRIEVAL,
        rerank_k=RERANK_K,
        neighbor_window=NEIGHBOR_WINDOW,
    )

logger.info(
    "Answer generated - video=%s, kb=%s, confluence=%s, sharepoint=%s",
    result["has_video"], result["has_kb"],
    result["has_confluence"], result["has_sharepoint"]
)

# ── Format response ────────────────────────────────────────────────────
sections = []

if result["has_video"]:
    sections.append(
        f"### 🎬 From Video\n{result['video_answer']}\n\n"
        f"> 📍 **Timestamp:** {result['video_timestamp']} &nbsp;|&nbsp; "
        f"**Source:** {result['video_source']}"
    )
else:
    sections.append("### 🎬 From Video\n_No relevant content found in uploaded videos._")

sections.append("---")

if result["has_kb"]:
    kb_src = ", ".join(result["kb_sources"]) if result["kb_sources"] else "N/A"
    sections.append(
        f"### 📄 From Knowledge Base\n{result['kb_answer']}\n\n"
        f"> 📂 **Sources:** {kb_src}"
    )
else:
    sections.append("### 📄 From Knowledge Base\n_No relevant content found in uploaded documents._")

sections.append("---")

if result["has_confluence"]:
    conf_src = ", ".join(result["confluence_sources"]) if result["confluence_sources"] else "N/A"
    sections.append(
        f"### 🔗 From Confluence\n{result['confluence_answer']}\n\n"
        f"> 📂 **Pages:** {conf_src}"
    )
else:
    sections.append("### 🔗 From Confluence\n_No relevant content found in Confluence._")

sections.append("---")

if result["has_sharepoint"]:
    sp_src = ", ".join(result["sharepoint_sources"]) if result["sharepoint_sources"] else "N/A"
    sections.append(
        f"### 📁 From SharePoint\n{result['sharepoint_answer']}\n\n"
        f"> 📂 **Files:** {sp_src}"
    )
else:
    sections.append("### 📁 From SharePoint\n_No relevant content found in SharePoint._")

answer_md = "\n\n".join(sections)

with st.chat_message("assistant"):
    st.markdown(answer_md)

st.session_state.messages.append({"role": "assistant", "content": answer_md})
_save_or_update_chat_history()
```

st.caption(“💡 Tip: Click Process only when adding new files/videos - existing knowledge is reused automatically.”)