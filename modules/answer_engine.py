from **future** import annotations
“””
answer_engine.py - LLM-based answer generation from retrieved documents.

## Key responsibilities

1. Detect whether the user’s follow-up question is an elaboration of the
   previous answer (cosine similarity gate) and amend the query accordingly.
1. Pull relevant short-term context from ConversationEntityMemory.
1. Pull long-term cached Q&A from qa_memory.
1. Build source-typed prompts (video-only / KB-only) and invoke the LLM.
1. Return a dict {“video”: str, “kb”: str}.

## Public API

answer_from_docs(query, docs, llm, entity_memory, embeddings) -> dict[str, str]
“””

import os
import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(**file**)))
if _PROJECT_ROOT not in sys.path:
sys.path.insert(0, _PROJECT_ROOT)

from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings

from modules.config import ELAB_THRESHOLD, get_logger
from modules.qa_memory import search_qa_memory

logger = get_logger(“answer_engine”)

# ─── Elaboration detection ────────────────────────────────────────────────────

def _is_elaboration(query: str, last_answer: str, embeddings, threshold: float = ELAB_THRESHOLD) -> bool:
“”“Return True when the query is semantically close to the last answer.”””
if not last_answer.strip():
return False
try:
q_emb = embeddings.embed_query(query)
a_emb = embeddings.embed_query(last_answer)
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity([q_emb], [a_emb])[0][0]
return float(sim) >= threshold
except Exception as exc:
logger.warning(“Elaboration check failed: %s”, exc)
return False

def _extract_last_turn(entity_memory) -> tuple[str, str]:
“”“Pull (last_question, last_answer) from entity memory buffer.”””
try:
buf = entity_memory.buffer
if isinstance(buf, str):
lines = buf.split(”\n”)
if len(lines) >= 2:
return (
lines[-2].replace(“Human:”, “”).strip(),
lines[-1].replace(“AI:”, “”).strip(),
)
elif isinstance(buf, list) and buf:
last = buf[-1]
if isinstance(last, dict):
return last.get(“input”, “”), last.get(“output”, “”)
if isinstance(last, (list, tuple)) and len(last) == 2:
return str(last[0]), str(last[1])
except Exception:
pass
return “”, “”

# ─── Prompts ──────────────────────────────────────────────────────────────────

_PROMPT_VIDEO_ONLY = PromptTemplate.from_template(”””
You are a helpful assistant answering ONLY from the provided video transcript chunks.

User question: {question}

Short-term memory (recent conversation):
{memory_context}

Long-term memory (previously answered Q&A from video):
{long_term}

Video context - each chunk is labelled with its source:
{video_context}

Instructions:

- Read ALL chunks carefully but answer ONLY from chunks that are DIRECTLY relevant to the question.
- If chunks from different videos are present, use ONLY the chunks that answer the question.
- Do NOT mix content from unrelated chunks just because they appear in context.
- Provide a detailed, tutorial-style explanation using only the relevant chunks.
- Do NOT add outside knowledge or mention documents.
- If no chunk is directly relevant to the question, reply EXACTLY: “The answer is not present in the videos.”
  “””)

_PROMPT_KB_ONLY = PromptTemplate.from_template(”””
You are a helpful assistant answering ONLY from the provided knowledge base (PDF/text documents).

User question: {question}

Short-term memory (recent conversation):
{memory_context}

Long-term memory (previously answered Q&A from KB):
{long_term}

Knowledge Base context (your ONLY source of truth):
{kb_context}

Instructions:

- Write a comprehensive, well-organised, factual explanation using only KB data.
- Include examples or applications if mentioned in the documents.
- Do NOT refer to videos or external knowledge.
- If the KB does not contain the answer, reply EXACTLY: “The answer is not present in the pdfs.”
  “””)

_PROMPT_CONFLUENCE = PromptTemplate.from_template(”””
You are a helpful assistant answering ONLY from the provided Confluence wiki pages.

User question: {question}

Short-term memory (recent conversation):
{memory_context}

Confluence page content - each chunk is labelled with its page title/source:
{confluence_context}

Instructions:

- Answer ONLY from the Confluence content provided above.
- Each chunk is labelled [Source: page title] - use only the chunks relevant to the question.
- Do NOT mix content from unrelated pages.
- If no Confluence page contains the answer, reply EXACTLY: “The answer is not present in the Confluence pages.”
  “””)

_PROMPT_SHAREPOINT = PromptTemplate.from_template(”””
You are a helpful assistant answering ONLY from the provided SharePoint documents and transcripts.

User question: {question}

Short-term memory (recent conversation):
{memory_context}

SharePoint content - each chunk is labelled with its filename/source:
{sharepoint_context}

Instructions:

- Answer ONLY from the SharePoint content provided above.
- Each chunk is labelled [Source: filename] - use only chunks relevant to the question.
- Do NOT add outside knowledge.
- If no SharePoint content contains the answer, reply EXACTLY: “The answer is not present in SharePoint.”
  “””)

# ─── Public API ───────────────────────────────────────────────────────────────

def answer_from_docs(
query: str,
docs: List[Document],
llm,
entity_memory,
embeddings,
) -> Dict[str, str]:
“””
Generate answers from a list of retrieved Documents.

```
Returns {"video": <str>, "kb": <str>} - one or both may be the
"not present" sentinel string.
"""
# ── Short-term memory ─────────────────────────────────────────────────────
memory_context = ""
try:
    memory_context = entity_memory.load_memory_variables({"input": query}).get("history", "")
except Exception:
    pass

# ── Elaboration detection ─────────────────────────────────────────────────
last_q, last_a = _extract_last_turn(entity_memory)
if _is_elaboration(query, last_a, embeddings) and last_q and last_a:
    query = f"{query} (follow-up to: Q: {last_q}, A: {last_a})"

# ── Long-term QA memory ───────────────────────────────────────────────────
video_mem = search_qa_memory(query, "video") or ""
kb_mem    = search_qa_memory(query, "pdf")   or ""

# ── Split retrieved docs by source type ───────────────────────────────────
# Label each chunk with its source so the LLM picks only relevant content

def _labelled_chunks(source_types: list) -> str:
    chunks = []
    for d in docs:
        st = (d.metadata or {}).get("source_type", "")
        if st in source_types:
            src      = (d.metadata or {}).get("source", "unknown")
            title    = (d.metadata or {}).get("title", "")
            src_name = title or os.path.basename(src) or src
            chunks.append(f"[Source: {src_name}]\n{d.page_content}")
    return "\n\n---\n\n".join(chunks).strip()

video_context      = _labelled_chunks(["video", "sharepoint_video"])
kb_context         = _labelled_chunks(["pdf"])
confluence_context = _labelled_chunks(["confluence", "confluence_attachment"])
sharepoint_context = _labelled_chunks(["sharepoint"])

# ── Call LLM for each source type that has context ────────────────────────
_NO_VID  = "The answer is not present in the videos."
_NO_KB   = "The answer is not present in the pdfs."
_NO_CONF = "The answer is not present in the Confluence pages."
_NO_SP   = "The answer is not present in SharePoint."

from_video = (
    _invoke_llm(llm, _PROMPT_VIDEO_ONLY.format(
        question=query, memory_context=memory_context,
        long_term=video_mem, video_context=video_context,
    )) if video_context else _NO_VID
)

from_kb = (
    _invoke_llm(llm, _PROMPT_KB_ONLY.format(
        question=query, memory_context=memory_context,
        long_term=kb_mem, kb_context=kb_context,
    )) if kb_context else _NO_KB
)

from_confluence = (
    _invoke_llm(llm, _PROMPT_CONFLUENCE.format(
        question=query, memory_context=memory_context,
        confluence_context=confluence_context,
    )) if confluence_context else _NO_CONF
)

from_sharepoint = (
    _invoke_llm(llm, _PROMPT_SHAREPOINT.format(
        question=query, memory_context=memory_context,
        sharepoint_context=sharepoint_context,
    )) if sharepoint_context else _NO_SP
)

# ── Save to short-term memory ─────────────────────────────────────────────
try:
    entity_memory.save_context(
        {"input": query},
        {"output": (
            f"Video: {from_video}\nKB: {from_kb}\n"
            f"Confluence: {from_confluence}\nSharePoint: {from_sharepoint}"
        )},
    )
except Exception as exc:
    logger.warning("Could not save to entity memory: %s", exc)

return {
    "video":       from_video,
    "kb":          from_kb,
    "confluence":  from_confluence,
    "sharepoint":  from_sharepoint,
}
```

def _invoke_llm(llm, prompt_text: str) -> str:
try:
raw = llm.invoke(prompt_text)
return getattr(raw, “content”, str(raw)).strip()
except Exception as exc:
logger.error(“LLM invocation failed: %s”, exc)
return “The answer is not present in the documents.”