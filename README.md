# Enterprise Agentic RAG Knowledge Assistant

An enterprise-grade knowledge assistant that answers questions from company documents using **Retrieval-Augmented Generation (RAG)**, **LangGraph agentic workflows**, **streaming responses**, conversation memory, source citations, and a hallucination guard.

---

## Features

- **Agentic RAG pipeline** — LangGraph graph with intent classification (QA, Summary, Comparison, Out-of-scope)
- **Query rewriting** — follow-up questions are rewritten into standalone search queries using conversation history
- **Streaming responses** — token-by-token streaming via Server-Sent Events (SSE)
- **General chat mode** — works as a normal chatbot without document context
- **Document ingestion** — supports PDF, DOCX, TXT, and Markdown files
- **Vector search** — ChromaDB with local HuggingFace or OpenAI embeddings
- **Hallucination guard** — rule-based answer validation before returning to the user
- **Conversation memory** — per-session in-memory chat history
- **Audit logging** — every chat turn saved to SQLite
- **REST API** — FastAPI with API key authentication
- **Streamlit UI** — quick frontend for testing the full pipeline

---

## Project Structure

```
enterprise-agentic-rag-assistant/
├── backend/
│   ├── app/
│   │   ├── agents/
│   │   │   ├── agentic_rag_graph.py   # LangGraph workflow
│   │   │   ├── query_writer.py        # Question rewriting
│   │   │   └── rag_agent.py           # RAG + streaming answer generation
│   │   ├── api/routes/
│   │   │   ├── chat.py                # /chat and /chat/stream endpoints
│   │   │   ├── documents.py           # Upload, search, stats, reset
│   │   │   ├── embeddings.py          # Embedding test endpoints
│   │   │   ├── health.py              # Health check
│   │   │   └── logs.py                # Chat log viewer
│   │   ├── core/
│   │   │   ├── config.py              # Settings from .env
│   │   │   └── security.py            # API key verification
│   │   ├── db/
│   │   │   ├── crud.py                # Database operations
│   │   │   ├── models.py              # SQLAlchemy models
│   │   │   └── session.py             # DB engine and session
│   │   ├── rag/
│   │   │   ├── document_loader.py     # PDF, DOCX, TXT loaders
│   │   │   ├── text_splitter.py       # Chunking
│   │   │   └── vector_store.py        # ChromaDB operations
│   │   ├── schemas/
│   │   │   ├── chat_schema.py
│   │   │   ├── document_schema.py
│   │   │   └── log_schema.py
│   │   ├── services/
│   │   │   ├── embedding_service.py   # HuggingFace / OpenAI embeddings
│   │   │   ├── hallusination_guard.py # Answer validation
│   │   │   ├── llm_service.py         # Ollama / OpenAI LLM
│   │   │   └── memory_service.py      # In-memory chat history
│   │   └── main.py
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    └── streamlit_app.py               # Streamlit test UI
```

---

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) running locally **or** an OpenAI API key

### 1. Clone the repository

```bash
git clone https://github.com/your-username/enterprise-agentic-rag-assistant.git
cd enterprise-agentic-rag-assistant
```

### 2. Create a virtual environment

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt

# Streamlit (for the frontend)
pip install streamlit requests
```

### 4. Configure environment variables

Copy the example file and edit it:

```bash
cp .env.example .env
```

```env
# LLM provider: "ollama" or "openai"
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434

# Uncomment to use OpenAI instead
# LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-...
# OPENAI_MODEL=gpt-4o-mini

# Embeddings: "local" (HuggingFace) or "openai"
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# API key for the REST API
API_KEY=dev-secret-key-change-this
```

### 5. Pull a model (Ollama users only)

```bash
ollama pull llama3.2
```

---

## Running

### Backend

```bash
cd backend
uvicorn app.main:app --reload
```

API runs at `http://localhost:8000`  
Interactive docs at `http://localhost:8000/docs`

### Frontend (Streamlit)

Open a second terminal:

```bash
cd frontend
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`

---

## API Overview

All endpoints (except `/health/`) require the `X-API-Key` header.

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health/` | Health check |
| POST | `/chat/` | Non-streaming chat |
| POST | `/chat/stream` | Streaming chat (SSE) |
| DELETE | `/chat/memory/{session_id}` | Clear session memory |
| POST | `/documents/upload` | Upload and index a document |
| POST | `/documents/search` | Semantic search |
| GET | `/documents/stats` | Vector store stats |
| DELETE | `/documents/reset` | Reset vector store |
| GET | `/documents/records` | Uploaded document records |
| POST | `/embeddings/test` | Test single embedding |
| GET | `/logs/chats` | View chat audit logs |

### Chat modes

Both `/chat/` and `/chat/stream` accept a `mode` field:

```json
{ "question": "What is the leave policy?", "mode": "rag" }
```

| Mode | Behaviour |
|------|-----------|
| `rag` (default) | Retrieves relevant document chunks, runs agentic pipeline |
| `chat` | General LLM conversation, no document retrieval |

### Streaming SSE event types

```
data: {"type": "start",    "intent": "qa", "was_rewritten": false}
data: {"type": "metadata", "sources": [...], "confidence": "high"}
data: {"type": "chunk",    "content": "Hello "}
data: {"type": "done",     "guard_status": "passed"}
```

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | `ollama` or `openai` |
| `OLLAMA_MODEL` | `llama3.1` | Any model pulled in Ollama |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OPENAI_API_KEY` | — | Required when using OpenAI |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `EMBEDDING_PROVIDER` | `local` | `local` or `openai` |
| `LOCAL_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model |
| `CHROMA_DB_PATH` | `app/storage/chroma_db` | ChromaDB storage path |
| `UPLOAD_DIR` | `app/storage/uploaded_docs` | Uploaded files path |
| `DATABASE_URL` | `sqlite:///./app/storage/app.db` | SQLite database path |
| `API_KEY` | `dev-secret-key-change-this` | REST API authentication key |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| API | FastAPI, Uvicorn |
| Agentic workflow | LangGraph |
| LLM | Ollama (local) / OpenAI |
| Embeddings | HuggingFace sentence-transformers / OpenAI |
| Vector store | ChromaDB |
| Database | SQLite + SQLAlchemy |
| Frontend | Streamlit |
