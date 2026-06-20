import re
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from fastapi import Depends

from app.db.session import get_db
from app.db.crud import create_document_record, get_recent_document_records

from app.core.config import settings
from app.schemas.document_schema import (
    DocumentIngestionResponse,
    DocumentSearchRequest,
    DocumentSearchResponse,
    RetrievedDocument,
    VectorStoreStatsResponse,
    IndexedDocumentSummary,
    ResetVectorStoreResponse,
    DocumentRecordListResponse,
    DocumentRecordResponse
)
from app.rag.document_loader import load_document
from app.rag.text_splitter import split_documents, get_sample_chunks
from app.rag.vector_store import (
    COLLECTION_NAME,
    add_documents_to_vector_store,
    search_similar_documents,
    format_search_results,
    get_vector_store_stats,
    reset_vector_store
)


router = APIRouter(prefix="/documents", tags=["Documents"])


ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024


def sanitize_filename(filename: str) -> str:
    filename = Path(filename).name
    filename = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)
    return filename


def validate_file_extension(filename: str) -> str:
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{extension}'. "
                f"Allowed file types are: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )
        )

    return extension

@router.post("/upload", response_model=DocumentIngestionResponse)
async def upload_and_process_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file name found.")

    file_extension = validate_file_extension(file.filename)

    file_bytes = await file.read()
    file_size = len(file_bytes)

    if file_size == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail="File is too large. Maximum allowed size is 10 MB."
        )

    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_original_name = sanitize_filename(file.filename)
    unique_file_name = f"{uuid.uuid4().hex}_{safe_original_name}"
    saved_path = upload_dir / unique_file_name

    with open(saved_path, "wb") as output_file:
        output_file.write(file_bytes)

    try:
        loaded_documents = load_document(str(saved_path))
        chunks = split_documents(loaded_documents)
        sample_chunks = get_sample_chunks(chunks)
        indexed_chunk_count = add_documents_to_vector_store(chunks)

    except Exception as error:
        if saved_path.exists():
            saved_path.unlink()

        raise HTTPException(
            status_code=500,
            detail=f"Document was uploaded but ingestion failed: {str(error)}"
        )

    if not loaded_documents:
        raise HTTPException(
            status_code=400,
            detail="Document uploaded, but no readable text was found."
        )

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="Document text was loaded, but chunking produced no chunks."
        )

    return DocumentIngestionResponse(
        status="success",
        message="Document uploaded, loaded, chunked, embedded, and indexed successfully.",
        original_file_name=safe_original_name,
        saved_file_name=unique_file_name,
        file_type=file_extension,
        file_size_bytes=file_size,
        saved_path=str(saved_path),
        loaded_document_count=len(loaded_documents),
        chunk_count=len(chunks),
        indexed_chunk_count=indexed_chunk_count,
        collection_name=COLLECTION_NAME,
        sample_chunks=sample_chunks
    )


@router.post("/search", response_model=DocumentSearchResponse)
def search_documents(request: DocumentSearchRequest):
    try:
        raw_results = search_similar_documents(
            query=request.query,
            top_k=request.top_k
        )

        formatted_results = format_search_results(raw_results)

        return DocumentSearchResponse(
            status="success",
            query=request.query,
            result_count=len(formatted_results),
            results=[
                RetrievedDocument(**result)
                for result in formatted_results
            ]
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Document search failed: {str(error)}"
        )


@router.get("/stats", response_model=VectorStoreStatsResponse)
def get_documents_stats():
    """
    Shows indexed vector database statistics.
    """

    try:
        stats = get_vector_store_stats()

        return VectorStoreStatsResponse(
            status="success",
            collection_name=stats["collection_name"],
            total_chunks=stats["total_chunks"],
            unique_document_count=stats["unique_document_count"],
            documents=[
                IndexedDocumentSummary(**document)
                for document in stats["documents"]
            ]
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get vector store stats: {str(error)}"
        )


@router.delete("/reset", response_model=ResetVectorStoreResponse)
def reset_documents_vector_store():
    """
    Resets the Chroma vector database collection.

    Warning:
    This deletes indexed chunks from ChromaDB.
    It does not delete uploaded files from local storage.
    """

    try:
        reset_vector_store()

        return ResetVectorStoreResponse(
            status="success",
            message="Vector store collection reset successfully.",
            collection_name=COLLECTION_NAME
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset vector store: {str(error)}"
        )

@router.get("/records", response_model=DocumentRecordListResponse)
def list_document_records(
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    Lists uploaded document metadata records from SQLite.
    """

    try:
        records = get_recent_document_records(
            db=db,
            limit=limit
        )

        return DocumentRecordListResponse(
            status="success",
            count=len(records),
            records=[
                DocumentRecordResponse.model_validate(record)
                for record in records
            ]
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list document records: {str(error)}"
        )