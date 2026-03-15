# 🧠 Knowledge Explorer — KB Tool

A Streamlit RAG application that builds a searchable knowledge base from **PDF/TXT documents** and **YouTube(only downloaded ones) / video URLs**, then answers questions using **Azure OpenAI (GPT-4o)**.

---

## Project layout

```
kb_tool/
├── app.py                   # Streamlit UI (thin layer — no business logic)
├── modules/
│   ├── config.py            # All env vars, paths, hyperparameters, shared logger
│   ├── llm_client.py        # Azure LLM + Embeddings factory (cached)
│   ├── transcription.py     # moviepy + Whisper pipeline (local files only)
│   ├── document_loader.py   # PDF/TXT/JSON → LangChain Documents
│   ├── vector_store.py      # FAISS build / load / incremental update + doc_store
│   ├── retriever.py         # CrossEncoder reranking + neighbour-chunk expansion
│   ├── qa_memory.py         # Long-term FAISS-backed Q&A memory (video + KB)
│   ├── answer_engine.py     # Elaboration detection + LLM prompting
│   ├── query_handler.py     # End-to-end orchestration (video, KB, combined)
│   └── meta_store.py        # JSON registry of uploaded and indexed files
├── data/                    # Uploaded PDFs, TXTs, transcript JSONs + index_meta.json
├── faiss_pdf/               # FAISS index for PDF/TXT documents
├── faiss_video/             # FAISS index for video transcripts
├── qa_indexes/              # Long-term QA memory indexes
├── requirements.txt
└── .env.example
```

---

## Setup

### 1. Prerequisites

```bash
# FFmpeg must be on your PATH
sudo apt install ffmpeg          # Ubuntu/Debian
brew install ffmpeg              # macOS
```

### 2. Python environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Azure credentials

```bash
cp .env.example .env
# Edit .env and fill in your Azure OpenAI details
```

Key variables:

| Variable | Description |
|---|---|
| `AZURE_OPENAI_API_KEY` | Your Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | e.g. `https://my-resource.openai.azure.com/` |
| `AZURE_OPENAI_API_VERSION` | e.g. `2024-02-01` |
| `AZURE_DEPLOYMENT_CHAT` | Chat deployment name (e.g. `gpt-4o`) |
| `AZURE_DEPLOYMENT_EMBED` | Embedding deployment name (e.g. `text-embedding-ada-002`) |

> If Azure embedding credentials are missing, the app automatically falls back to `BAAI/bge-base-en-v1.5` (HuggingFace local model).

### 4. Run

```bash
streamlit run app.py
```

---

## How it works

```
Upload PDF/TXT ──┐
                 ├──► document_loader ──► vector_store (FAISS) ──┐
Upload Video ────┘      transcription        meta_store          │
                         (moviepy+Whisper)   (registry)          │
                                                                  ▼
User question ───────────────────────────────────────────► query_handler
                                                                  │
                                               retriever (rerank + expand)
                                                                  │
                                               answer_engine (Azure GPT-4o)
                                                                  │
                                               qa_memory (cache answer)
                                                                  │
                                                         Streamlit chat ◄──
```

### Video transcription pipeline

1. User uploads a local video or audio file via the Streamlit file uploader.
2. **moviepy** extracts the audio track (bundles its own ffmpeg via `imageio-ffmpeg` — no system install needed).
3. For `.wav`, `.mp3`, `.flac`, `.ogg` files, Whisper reads them directly (moviepy skipped for speed).
4. **Whisper** (`base` model) transcribes to text with per-segment timestamps.
5. Result is cached as JSON in `data/` — the same file is never re-processed.

> **Note:** YouTube and other platform URLs are not supported. Only local files.

### Module responsibilities

| Module | Responsibility |
|---|---|
| `config.py` | All env vars, paths, hyperparameters, shared logger |
| `llm_client.py` | Azure LLM + embedding factory (`@st.cache_resource`) |
| `transcription.py` | moviepy → Whisper → JSON cache (local files only) |
| `document_loader.py` | PDF/TXT/JSON → chunked LangChain Documents |
| `vector_store.py` | FAISS build / load / incremental update + doc_store |
| `retriever.py` | CrossEncoder reranking + neighbour-chunk expansion |
| `qa_memory.py` | Long-term FAISS-backed Q&A cache (video + KB) |
| `answer_engine.py` | Elaboration detection + LLM prompting |
| `query_handler.py` | End-to-end orchestration for video, KB, and combined queries |
| `meta_store.py` | JSON registry tracking uploaded and indexed files |
| `app.py` | Streamlit UI only — no business logic |

### Retrieval pipeline

1. FAISS similarity search (`top_k` candidates).
2. CrossEncoder reranking (falls back to cosine if unavailable).
3. Neighbour-chunk expansion (merge ±`window` adjacent chunks for richer context).
4. Azure GPT-4o generates a detailed answer from the merged context.
5. Q&A pair stored in long-term FAISS memory for future identical/similar queries.

---

## Configuration knobs (`modules/config.py`)

| Constant | Default | Description |
|---|---|---|
| `TOP_K_RETRIEVAL` | 20 | FAISS candidates before reranking |
| `RERANK_K` | 6 | Docs kept after reranking |
| `NEIGHBOR_WINDOW` | 2 | Chunks expanded on each side |
| `CHUNK_SIZE_PDF` | 800 | Characters per PDF chunk |
| `CHUNK_SIZE_VIDEO` | 1000 | Characters per video chunk |
| `ELAB_THRESHOLD` | 0.65 | Cosine sim threshold for elaboration detection |
