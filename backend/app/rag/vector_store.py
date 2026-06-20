from typing import List, Tuple
from uuid import uuid4
from pathlib import Path
from collections import defaultdict

from langchain_core.documents import Document
from langchain_chroma import Chroma

from app.core.config import settings
from app.services.embedding_service import get_embedding_model


COLLECTION_NAME = "enterprise_knowledge_base"


def get_vector_store() -> Chroma:
    persist_path = Path(settings.CHROMA_DB_PATH)
    persist_path.mkdir(parents=True, exist_ok=True)

    embedding_model = get_embedding_model()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(persist_path)
    )

    return vector_store


def add_documents_to_vector_store(documents: List[Document]) -> int:
    if not documents:
        return 0

    vector_store = get_vector_store()

    ids = []

    for document in documents:
        file_name = document.metadata.get("file_name", "unknown_file")
        chunk_id = document.metadata.get("chunk_id", "unknown_chunk")
        unique_id = f"{file_name}_{chunk_id}_{uuid4().hex}"
        ids.append(unique_id)

    vector_store.add_documents(
        documents=documents,
        ids=ids
    )

    return len(ids)


def search_similar_documents(
    query: str,
    top_k: int = 4
) -> List[Tuple[Document, float]]:
    if not query or not query.strip():
        return []

    vector_store = get_vector_store()

    results = vector_store.similarity_search_with_score(
        query=query,
        k=top_k
    )

    return results


def format_search_results(
    results: List[Tuple[Document, float]]
) -> list[dict]:
    formatted_results = []

    for document, score in results:
        metadata = document.metadata or {}

        page_number = metadata.get("page")

        if page_number is not None:
            try:
                page_number = int(page_number) + 1
            except Exception:
                page_number = None

        preview = document.page_content[:500]

        if len(document.page_content) > 500:
            preview += "..."

        formatted_results.append(
            {
                "file_name": metadata.get("file_name", "unknown"),
                "file_type": metadata.get("file_type"),
                "page_number": page_number,
                "chunk_id": metadata.get("chunk_id"),
                "similarity_score": float(score),
                "content_preview": preview
            }
        )

    return formatted_results


def get_vector_store_stats() -> dict:
    """
    Returns collection statistics.

    Uses Chroma collection.get() to inspect stored metadata.
    """

    vector_store = get_vector_store()
    collection = vector_store._collection

    collection_data = collection.get(
        include=["metadatas"]
    )

    metadatas = collection_data.get("metadatas", []) or []

    document_counter = defaultdict(
        lambda: {
            "file_name": "unknown",
            "file_type": None,
            "chunk_count": 0
        }
    )

    for metadata in metadatas:
        metadata = metadata or {}

        file_name = metadata.get("file_name", "unknown")
        file_type = metadata.get("file_type")

        document_counter[file_name]["file_name"] = file_name
        document_counter[file_name]["file_type"] = file_type
        document_counter[file_name]["chunk_count"] += 1

    documents = list(document_counter.values())

    documents = sorted(
        documents,
        key=lambda item: item["file_name"]
    )

    return {
        "collection_name": COLLECTION_NAME,
        "total_chunks": len(metadatas),
        "unique_document_count": len(documents),
        "documents": documents
    }


def reset_vector_store() -> None:
    """
    Deletes and recreates the Chroma collection.

    This removes all indexed chunks from the vector database.
    It does not delete uploaded files from storage/uploaded_docs.
    """

    vector_store = get_vector_store()
    vector_store.reset_collection()