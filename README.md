# pdf-rag

Ask questions about your PDF and DOCX files using natural language — powered by Cohere embeddings, Qdrant vector search, and Claude.

Upload one or more documents, then query across all of them. Claude answers strictly from the document content and cites which file each answer came from.

---

## How it works

```
PDF / DOCX
    │
    ▼
Extract text (pdfplumber / python-docx)
    │
    ▼
Chunk text (800 chars, 150 char overlap)
    │
    ▼
Embed chunks (Cohere embed-english-light-v3.0 · 384-dim)
    │
    ▼
Store vectors in Qdrant
    │
    ▼
Query → top-5 chunks → Claude (claude-opus-4-6 + adaptive thinking)
    │
    ▼
Answer with source citations
```

---

## Features

- **Multi-document** — ingest as many PDFs or DOCX files as you want; query across all of them at once
- **Source citations** — every answer cites which document(s) it came from
- **Adaptive thinking** — uses Claude's extended thinking for complex questions
- **Web UI** — clean browser interface for uploading files and asking questions
- **CLI** — interactive terminal interface for local use
- **Flexible storage** — local Qdrant (no account needed) or Qdrant Cloud

---

## Stack

| Layer | Technology |
|---|---|
| Embeddings | Cohere `embed-english-light-v3.0` (384-dim) |
| Vector DB | Qdrant (local file or cloud) |
| LLM | Anthropic Claude `claude-opus-4-6` |
| PDF parsing | pdfplumber |
| DOCX parsing | python-docx |
| Web server | FastAPI + uvicorn |
| Frontend | Static HTML (served by FastAPI) |

---

## Setup

### Prerequisites

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/)
- A [Cohere API key](https://dashboard.cohere.com/)
- Optionally, a [Qdrant Cloud](https://cloud.qdrant.io/) cluster (or use local storage)

### Install

```bash
git clone https://github.com/AyeniOluwatosinOlawale/pdf-rag.git
cd pdf-rag
pip install -r requirements.txt
```

### Environment variables

```bash
# Required
export ANTHROPIC_API_KEY=sk-ant-...
export COHERE_API_KEY=...

# Optional — Qdrant Cloud (omit to use local ./qdrant_db)
export QDRANT_URL=https://your-cluster.qdrant.io
export QDRANT_API_KEY=...

# Optional — override local storage path
export QDRANT_DIR=./qdrant_db
```

---

## Usage

### Web interface

```bash
uvicorn server:app --reload
```

Open `http://localhost:8000`, upload a PDF or DOCX, then type your question.

### CLI

```bash
python main.py
```

Available commands:

```
> ingest path/to/file.pdf     # ingest a PDF or DOCX
> ask <your question>         # query across all ingested documents
> sources                     # list all ingested files
> quit                        # exit
```

### REST API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/ingest-file` | Upload a PDF or DOCX (`multipart/form-data`) |
| `POST` | `/api/query` | Ask a question (`{"question": "..."}`) |
| `GET` | `/api/sources` | List all ingested documents |

**Example:**

```bash
# Ingest a file
curl -X POST http://localhost:8000/api/ingest-file \
  -F "file=@report.pdf"

# Ask a question
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key findings?"}'
```

---

## Deployment

### Render (recommended)

The included `render.yaml` configures a web service with a persistent disk for Qdrant storage.

1. Connect this repo to [Render](https://render.com)
2. Add environment variables in the Render dashboard:
   - `ANTHROPIC_API_KEY`
   - `COHERE_API_KEY`
3. Deploy — Render handles the rest

### Vercel / Netlify

The repo includes `vercel.json` and `netlify.toml`. The API entrypoint for serverless is `api/index.py`.

> Note: serverless platforms have ephemeral filesystems — use Qdrant Cloud (`QDRANT_URL` + `QDRANT_API_KEY`) instead of local storage when deploying to Vercel or Netlify.

### Heroku / Railway

A `Procfile` is included:

```
web: uvicorn server:app --host 0.0.0.0 --port $PORT
```

---

## Configuration

Edit constants at the top of `rag.py` to tune behaviour:

| Constant | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between consecutive chunks |
| `TOP_K` | `5` | Number of chunks retrieved per query |
| `MODEL` | `claude-opus-4-6` | Claude model used for answers |
| `VECTOR_SIZE` | `384` | Must match the Cohere model output |
| `COHERE_MODEL` | `embed-english-light-v3.0` | Cohere embedding model |

---

## License

MIT