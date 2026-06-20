from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.routes import health, chat, documents, embeddings, logs
from app.db.session import create_database_tables


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "An enterprise-grade Agentic RAG Knowledge Assistant "
        "that answers questions from company documents using retrieval, "
        "LLMs, source citations, memory, hallucination guards, and agentic workflows."
    )
)


@app.on_event("startup")
def on_startup():
    create_database_tables()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(health.router)
app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(embeddings.router)
app.include_router(logs.router)


@app.get("/")
def root():
    return {
        "message": "Welcome to the Enterprise Agentic RAG Knowledge Assistant API",
        "docs": "/docs",
        "health": "/health/",
        "chat": "/chat/",
        "documents_upload": "/documents/upload",
        "documents_search": "/documents/search",
        "documents_stats": "/documents/stats",
        "documents_records": "/documents/records",
        "documents_reset": "/documents/reset",
        "embeddings_test": "/embeddings/test",
        "chat_logs": "/logs/chats"
    }