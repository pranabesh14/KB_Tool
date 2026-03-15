"""
config.py — Central configuration and environment setup for KB Tool.
All constants, directory paths, and env vars are managed here.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

load_dotenv()

# ─── Azure OpenAI ─────────────────────────────────────────────────────────────
AZURE_OPENAI_API_KEY     = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_DEPLOYMENT_CHAT    = os.getenv("AZURE_DEPLOYMENT_CHAT", "gpt-4o")
AZURE_DEPLOYMENT_EMBED   = os.getenv("AZURE_DEPLOYMENT_EMBED", "text-embedding-ada-002")

# ─── HuggingFace (optional – for local cross-encoder / embeddings) ─────────────
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")

# ─── Directory layout ─────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR        = os.path.join(BASE_DIR, "data")
INDEX_DIR_PDF   = os.path.join(BASE_DIR, "faiss_pdf")
INDEX_DIR_VIDEO = os.path.join(BASE_DIR, "faiss_video")
QA_INDEX_DIR    = os.path.join(BASE_DIR, "qa_indexes")
VIDEO_QA_PATH   = os.path.join(QA_INDEX_DIR, "video_QA_index")
KB_QA_PATH      = os.path.join(QA_INDEX_DIR, "kb_QA_index")
META_FILE       = os.path.join(DATA_DIR, "index_meta.json")
LOG_FILE        = os.path.join(BASE_DIR, "kb_tool.log")

for _d in (DATA_DIR, INDEX_DIR_PDF, INDEX_DIR_VIDEO, QA_INDEX_DIR):
    os.makedirs(_d, exist_ok=True)

# ─── Retrieval hyperparameters ────────────────────────────────────────────────
TOP_K_RETRIEVAL  = 20
RERANK_K         = 6
NEIGHBOR_WINDOW  = 2
CHUNK_SIZE_PDF   = 800
CHUNK_OVERLAP    = 100
CHUNK_SIZE_VIDEO = 1000
ELAB_THRESHOLD   = 0.65   # cosine similarity threshold for elaboration detection

# ─── Logging ──────────────────────────────────────────────────────────────────
# Rotating log: max 5 MB per file, keep last 3 files → max 15 MB on disk
_LOG_MAX_BYTES  = 5 * 1024 * 1024   # 5 MB
_LOG_BACKUP_COUNT = 3

_FILE_FMT    = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s"
_CONSOLE_FMT = "[%(levelname)-8s] %(name)s — %(message)s"
_DATE_FMT    = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = "kb_tool") -> logging.Logger:
    """
    Return a named logger writing to:
      • kb_tool.log  (rotating, DEBUG+)  — full detail for troubleshooting
      • stderr       (WARNING+)          — only important messages in the terminal
    """
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger                   # already configured — return as-is

    logger.setLevel(logging.DEBUG)

    # ── Rotating file handler ──────────────────────────────────────────────────
    fh = RotatingFileHandler(
        LOG_FILE,
        maxBytes=_LOG_MAX_BYTES,
        backupCount=_LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_FILE_FMT, datefmt=_DATE_FMT))
    logger.addHandler(fh)

    # ── Console handler ────────────────────────────────────────────────────────
    sh = logging.StreamHandler()
    sh.setLevel(logging.WARNING)
    sh.setFormatter(logging.Formatter(_CONSOLE_FMT, datefmt=_DATE_FMT))
    logger.addHandler(sh)

    return logger