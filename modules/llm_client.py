“””
llm_client.py — Factory for Azure OpenAI LLM and Embeddings.

Uses LangChain’s AzureChatOpenAI and AzureOpenAIEmbeddings so the rest of
the codebase can treat them as standard LangChain objects.  Falls back to
local HuggingFace embeddings when Azure embedding credentials are absent.
“””

import os
import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(**file**)))
if _PROJECT_ROOT not in sys.path:
sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from modules.config import (
AZURE_OPENAI_API_KEY,
AZURE_OPENAI_ENDPOINT,
AZURE_OPENAI_API_VERSION,
AZURE_DEPLOYMENT_CHAT,
AZURE_DEPLOYMENT_EMBED,
get_logger,
)

logger = get_logger(“llm_client”)

@st.cache_resource
def init_llm_and_embeddings():
“””
Returns (llm, embeddings).  Cached for the lifetime of the Streamlit process.

```
LLM   : AzureChatOpenAI  (gpt-4o by default, configured via .env)
Embed : AzureOpenAIEmbeddings  OR  HuggingFaceEmbeddings (fallback)
"""
# ── LLM ───────────────────────────────────────────────────────────────────
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_DEPLOYMENT_CHAT,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    api_key=AZURE_OPENAI_API_KEY,
    temperature=0.0,
    max_tokens=2048,
)
logger.info("Azure LLM initialised: deployment=%s", AZURE_DEPLOYMENT_CHAT)

# ── Embeddings ─────────────────────────────────────────────────────────────
embeddings = _init_embeddings()

return llm, embeddings
```

def _init_embeddings():
“””
Try Azure embeddings first; fall back to local HuggingFace model.
Validates each option with a test embed call before returning.
“””
if AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT and AZURE_DEPLOYMENT_EMBED:
try:
emb = AzureOpenAIEmbeddings(
azure_endpoint=AZURE_OPENAI_ENDPOINT,
azure_deployment=AZURE_DEPLOYMENT_EMBED,
openai_api_version=AZURE_OPENAI_API_VERSION,
api_key=AZURE_OPENAI_API_KEY,
)
# Validate with a real embed call — catches auth/deployment errors early
emb.embed_query(“test”)
logger.info(“Using Azure OpenAI embeddings: %s”, AZURE_DEPLOYMENT_EMBED)
return emb
except Exception as exc:
logger.warning(
“Azure embeddings failed (%s). Falling back to local HuggingFace model.”, exc
)

```
# Local fallback — no Azure credentials needed, runs on CPU
try:
    emb = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    emb.embed_query("test")   # validate it loaded correctly
    logger.info("Using HuggingFace embeddings: BAAI/bge-base-en-v1.5")
    return emb
except Exception as exc:
    raise RuntimeError(
        f"Both Azure and HuggingFace embeddings failed to initialise.\n"
        f"Last error: {exc}\n"
        f"Check your .env credentials and that sentence-transformers is installed."
    ) from exc
