# RAG-QA System

A production-grade **Retrieval-Augmented Generation** system for domain-specific question answering. Upload your documents, ask questions in natural language, and get accurate answers grounded in your content with full source citations.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36-red?style=flat-square&logo=streamlit)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## What It Does

Traditional search finds documents containing your keywords. This system understands the *meaning* of your question and finds the most relevant passages — even if they use different words — then uses a large language model to synthesise a precise, cited answer.

```
Your Question
     │
     ▼
Semantic Search (FAISS + Sentence Transformers)
     │
     ▼
Top-K Relevant Chunks Retrieved
     │
     ▼
LLM generates grounded answer with [Source N] citations
     │
     ▼
Answer + Source Cards shown in UI
```

---

## Key Features

- **Semantic search** using sentence-transformer embeddings and FAISS vector retrieval
- **Multi-format ingestion** — PDF, DOCX, TXT, and Markdown files
- **Smart chunking** — recursive, sentence, paragraph, and fixed strategies with configurable overlap
- **LLM integration** — supports Google Gemini, Anthropic Claude, and OpenAI GPT
- **Conversation memory** — follow-up questions understand previous context
- **Source citations** — every answer shows exactly which document and page it came from
- **Interactive Streamlit UI** — upload documents, configure settings, and chat in real time
- **RAG evaluation** — faithfulness, answer relevance, and context recall scoring
- **Docker support** — one-command containerised deployment

---

## Architecture

```
rag-qa-system/
├── src/
│   ├── ingestion/          # Day 1 — Document loading and chunking
│   │   ├── loaders.py      # PDF, DOCX, TXT, MD readers
│   │   └── chunker.py      # Text splitting with overlap
│   ├── embeddings/         # Day 2 — Vector store
│   │   └── vector_store.py # FAISS index build, save, load, search
│   ├── retrieval/          # Day 2 — Semantic retrieval
│   │   └── retriever.py    # Top-K search with deduplication
│   ├── llm/                # Day 3 — LLM integration
│   │   ├── client.py       # Gemini / Anthropic / OpenAI wrapper
│   │   └── prompts.py      # RAG prompt templates
│   ├── pipeline/           # Day 3 — RAG orchestration
│   │   └── rag_chain.py    # Full RAG loop with memory
│   └── evaluation/         # Day 5 — Quality metrics
│       └── metrics.py      # Faithfulness, relevance, recall
├── app/
│   └── streamlit_app.py    # Day 4 — Interactive web UI
├── data/
│   ├── raw/                # Upload documents here
│   ├── processed/          # Chunked JSON output
│   └── eval_samples.json   # Evaluation question set
├── vectorstore/            # FAISS index files
├── tests/                  # 80+ unit tests
├── ingest.py               # CLI ingestion pipeline
├── build_index.py          # CLI vector index builder
├── query.py                # CLI interactive Q&A
├── evaluate.py             # CLI evaluation runner
├── config.py               # Centralised configuration
├── Dockerfile
└── requirements.txt
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Embeddings | `sentence-transformers` — all-MiniLM-L6-v2 (384-dim) |
| Vector Store | `faiss-cpu` — IndexFlatIP (cosine similarity) |
| LLM | Google Gemini 2.0 Flash / Claude 3.5 Sonnet / GPT-4o Mini |
| Document Loading | `pymupdf`, `python-docx` |
| Web UI | `streamlit` |
| Evaluation | LLM-as-judge (faithfulness, relevance, recall) |
| Testing | `pytest` — 80+ tests |
| Containerisation | Docker |

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/user-kaus2bh/rag-qa-system.git
cd rag-qa-system
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and add your API key:

```env
GEMINI_API_KEY=your_key_here
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.0-flash
```

Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com).

### 5. Add documents

Place any `.pdf`, `.txt`, `.docx`, or `.md` files into `data/raw/`.

### 6. Run the pipeline

```bash
# Step 1 — Ingest documents into chunks
python ingest.py

# Step 2 — Build the FAISS vector index
python build_index.py

# Step 3 — Launch the web UI
streamlit run app/streamlit_app.py --server.fileWatcherType none
```

Open your browser at **http://localhost:8501**

---

## CLI Usage

### Interactive Q&A

```bash
python query.py
```

### Single question

```bash
python query.py -q "What is the main topic of these documents?"
```

### Custom settings

```bash
python query.py --provider gemini --model gemini-2.0-flash --top-k 5
```

### Run evaluation

```bash
python evaluate.py --samples data/eval_samples.json
```

---

## Docker

```bash
# Build image
docker build -t rag-qa-system .

# Run (add your API key)
docker run -p 8501:8501 \
  -e GEMINI_API_KEY=your_key_here \
  -v $(pwd)/data/raw:/app/data/raw \
  rag-qa-system
```

Open **http://localhost:8501**

---

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Individual day tests
python -m pytest tests/test_ingestion.py -v
python -m pytest tests/test_retrieval.py -v
python -m pytest tests/test_pipeline.py -v
```

---

## Configuration

All settings are in `.env` (copy from `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `LLM_PROVIDER` | `gemini` | AI provider: gemini / anthropic / openai |
| `LLM_MODEL` | `gemini-2.0-flash` | Model name |
| `LLM_TEMPERATURE` | `0.1` | Response creativity (0.0–1.0) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TOP_K_RETRIEVAL` | `5` | Chunks retrieved per question |

---

## How It Works

**1. Ingestion** — Documents are loaded page by page (PDF) or section by section (DOCX/TXT). Text is cleaned and split into overlapping chunks using a recursive separator strategy that respects paragraph and sentence boundaries.

**2. Embedding** — Each chunk is encoded by `all-MiniLM-L6-v2` into a 384-dimensional vector that captures its semantic meaning. Vectors are normalised to unit length for cosine similarity.

**3. Indexing** — All chunk vectors are stored in a FAISS `IndexFlatIP` index alongside their metadata (source file, page, content). The index is persisted to disk as `index.faiss` + `metadata.json`.

**4. Retrieval** — When you ask a question, it is encoded into a vector using the same model. FAISS performs inner-product search to find the top-K most similar chunk vectors in milliseconds.

**5. Generation** — The retrieved chunks are formatted with source labels and injected into a structured prompt. The LLM reads the context and generates an answer grounded only in the retrieved information, citing sources.

**6. Memory** — Follow-up questions are condensed into standalone questions using the conversation history before retrieval, enabling coherent multi-turn conversations.

---

## Project Structure — Day by Day

| Day | What was built |
|---|---|
| Day 1 | Document ingestion pipeline — loaders, chunker, CLI, 20 tests |
| Day 2 | FAISS vector store — embeddings, build/save/load/search, 27 tests |
| Day 3 | RAG chain — LLM client, prompt templates, full pipeline, memory, 33 tests |
| Day 4 | Streamlit UI — dark theme, chat interface, file upload, source citations |
| Day 5 | Evaluation metrics, README, Dockerfile, GitHub Actions CI |

---

## License

MIT License — free to use, modify, and distribute.
