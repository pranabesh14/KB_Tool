“””
answer_engine.py — LLM-based answer generation from retrieved documents.

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

from **future** import annotations

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
You are a helpful assistant answering ONLY from the provided video transcript.

User question: {question}

Short-term memory (recent conversation):
{memory_context}

Long-term memory (previously answered Q&A from video):
{long_term}

Video context (your ONLY source of truth):
{video_context}

Instructions:

- Provide a **long, detailed, tutorial-style** explanation that fully answers the question.
- Use the same tone/phrasing from the video — include examples, analogies, and context.
- Do NOT add outside knowledge or mention documents.
- If no relevant info exists, reply EXACTLY: “The answer is not present in the videos.”
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

- Write a **comprehensive, well-organised, factual** explanation using only KB data.
- Include examples or applications if mentioned in the documents.
- Do NOT refer to videos or external knowledge.
- If the KB does not contain the answer, reply EXACTLY: “The answer is not present in the pdfs.”
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
Returns {"video": <str>, "kb": <str>} — one or both may be the
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
video_context = "\n".join(
    d.page_content for d in docs if (d.metadata or {}).get("source_type") == "video"
).strip()
kb_context = "\n".join(
    d.page_content for d in docs if (d.metadata or {}).get("source_type") == "pdf"
).strip()

# ── Call LLM only when context is actually available ─────────────────────
_NO_VID = "The answer is not present in the videos."
_NO_KB  = "The answer is not present in the pdfs."

if video_context:
    from_video = _invoke_llm(
        llm,
        _PROMPT_VIDEO_ONLY.format(
            question=query,
            memory_context=memory_context,
            long_term=video_mem,
            video_context=video_context,
        ),
    )
else:
    from_video = _NO_VID

if kb_context:
    from_kb = _invoke_llm(
        llm,
        _PROMPT_KB_ONLY.format(
            question=query,
            memory_context=memory_context,
            long_term=kb_mem,
            kb_context=kb_context,
        ),
    )
else:
    from_kb = _NO_KB

# ── Save to short-term memory ─────────────────────────────────────────────
try:
    entity_memory.save_context(
        {"input": query},
        {"output": f"Video: {from_video}\nKB: {from_kb}"},
    )
except Exception as exc:
    logger.warning("Could not save to entity memory: %s", exc)

return {"video": from_video, "kb": from_kb}
```

def _invoke_llm(llm, prompt_text: str) -> str:
try:
raw = llm.invoke(prompt_text)
return getattr(raw, “content”, str(raw)).strip()
except Exception as exc:
logger.error(“LLM invocation failed: %s”, exc)
return “The answer is not present in the documents.”