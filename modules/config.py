“””
config.py - Central configuration and environment setup for KB Tool.
All constants, directory paths, and env vars are managed here.
“””

import os
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

load_dotenv()

# — Azure OpenAI ——————————————————————

AZURE_OPENAI_API_KEY     = os.getenv(“AZURE_OPENAI_API_KEY”, “”)
AZURE_OPENAI_ENDPOINT    = os.getenv(“AZURE_OPENAI_ENDPOINT”, “”)
AZURE_OPENAI_API_VERSION = os.getenv(“AZURE_OPENAI_API_VERSION”, “2024-02-01”)
AZURE_DEPLOYMENT_CHAT    = os.getenv(“AZURE_DEPLOYMENT_CHAT”, “gpt-4o”)
AZURE_DEPLOYMENT_EMBED   = os.getenv(“AZURE_DEPLOYMENT_EMBED”, “text-embedding-ada-002”)

# — Azure AD (shared by Confluence + SharePoint OAuth 2.0) ———————–

AZURE_AD_TENANT_ID        = os.getenv(“AZURE_AD_TENANT_ID”, “”)
AZURE_AD_CLIENT_ID        = os.getenv(“AZURE_AD_CLIENT_ID”, “”)
AZURE_AD_CLIENT_SECRET    = os.getenv(“AZURE_AD_CLIENT_SECRET”, “”)
AZURE_AD_CONFLUENCE_SCOPE = os.getenv(“AZURE_AD_CONFLUENCE_SCOPE”, “”)
AZURE_AD_SHAREPOINT_SCOPE = os.getenv(“AZURE_AD_SHAREPOINT_SCOPE”, “”)

# — Confluence —————————————————————––

CONFLUENCE_BASE_URL    = os.getenv(“CONFLUENCE_BASE_URL”, “”)
CONFLUENCE_CRAWL_DEPTH = int(os.getenv(“CONFLUENCE_CRAWL_DEPTH”, “2”))

# — SharePoint —————————————————————––

SHAREPOINT_BASE_URL = os.getenv(“SHAREPOINT_BASE_URL”, “”)  # e.g. https://sharepoint.yourcompany.com

# — HuggingFace (optional) —————————————————––

HUGGINGFACE_API_KEY = os.getenv(“HUGGINGFACE_API_KEY”, “”)

# — Whisper –––––––––––––––––––––––––––––––––––

WHISPER_MODEL_SIZE = os.getenv(“WHISPER_MODEL_SIZE”, “base”)

# — Directory layout ———————————————————––

BASE_DIR              = os.path.dirname(os.path.dirname(os.path.abspath(**file**)))
DATA_DIR              = os.path.join(BASE_DIR, “data”)
TEMP_DIR              = os.path.join(BASE_DIR, “temp”)
INDEX_DIR_PDF         = os.path.join(BASE_DIR, “faiss_pdf”)
INDEX_DIR_VIDEO       = os.path.join(BASE_DIR, “faiss_video”)
INDEX_DIR_CONFLUENCE  = os.path.join(BASE_DIR, “faiss_confluence”)
INDEX_DIR_SHAREPOINT  = os.path.join(BASE_DIR, “faiss_sharepoint”)
QA_INDEX_DIR          = os.path.join(BASE_DIR, “qa_indexes”)
VIDEO_QA_PATH         = os.path.join(QA_INDEX_DIR, “video_QA_index”)
KB_QA_PATH            = os.path.join(QA_INDEX_DIR, “kb_QA_index”)
META_FILE             = os.path.join(DATA_DIR, “index_meta.json”)
LOG_FILE              = os.path.join(BASE_DIR, “kb_tool.log”)
CONFLUENCE_CACHE_DIR  = os.path.join(DATA_DIR, “confluence_cache”)
SHAREPOINT_CACHE_DIR  = os.path.join(DATA_DIR, “sharepoint_cache”)

for _d in (DATA_DIR, TEMP_DIR, INDEX_DIR_PDF, INDEX_DIR_VIDEO,
INDEX_DIR_CONFLUENCE, INDEX_DIR_SHAREPOINT,
QA_INDEX_DIR, CONFLUENCE_CACHE_DIR, SHAREPOINT_CACHE_DIR):
os.makedirs(_d, exist_ok=True)

# — Retrieval hyperparameters ––––––––––––––––––––––––––

TOP_K_RETRIEVAL  = 20
RERANK_K         = 6
NEIGHBOR_WINDOW  = 1
CHUNK_SIZE_PDF   = 800
CHUNK_OVERLAP    = 100
CHUNK_SIZE_VIDEO = 1000
ELAB_THRESHOLD   = 0.65

# — Logging –––––––––––––––––––––––––––––––––––

_LOG_MAX_BYTES    = 5 * 1024 * 1024
_LOG_BACKUP_COUNT = 3
_FILE_FMT    = “%(asctime)s [%(levelname)-8s] %(name)s - %(message)s”
_CONSOLE_FMT = “[%(levelname)-8s] %(name)s - %(message)s”
_DATE_FMT    = “%Y-%m-%d %H:%M:%S”

def get_logger(name: str = “kb_tool”) -> logging.Logger:
logger = logging.getLogger(name)
if logger.hasHandlers():
return logger
logger.setLevel(logging.DEBUG)
fh = RotatingFileHandler(LOG_FILE, maxBytes=_LOG_MAX_BYTES,
backupCount=_LOG_BACKUP_COUNT, encoding=“utf-8”)
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter(_FILE_FMT, datefmt=_DATE_FMT))
logger.addHandler(fh)
sh = logging.StreamHandler()
sh.setLevel(logging.WARNING)
sh.setFormatter(logging.Formatter(_CONSOLE_FMT, datefmt=_DATE_FMT))
logger.addHandler(sh)
return logger