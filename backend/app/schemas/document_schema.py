from pydantic import BaseModel, Field
from typing import List, Optional


class DocumentUploadResponse(BaseModel):
    status: str
    message: str
    original_file_name: str
    saved_file_name: str
    file_type: str
    file_size_bytes: int
    saved_path: str


class DocumentIngestionResponse(BaseModel):
    status: str
    message: str
    original_file_name: str
    saved_file_name: str
    file_type: str
    file_size_bytes: int
    saved_path: str
    loaded_document_count: int
    chunk_count: int
    indexed_chunk_count: int
    collection_name: str
    sample_chunks: List[str]


class UploadedDocumentInfo(BaseModel):
    file_name: str
    file_type: str
    file_size_bytes: Optional[int] = None
    path: str


class DocumentSearchRequest(BaseModel):
    query: str = Field(..., min_length=3)
    top_k: int = Field(default=4, ge=1, le=10)


class RetrievedDocument(BaseModel):
    file_name: str
    file_type: Optional[str] = None
    page_number: Optional[int] = None
    chunk_id: Optional[int] = None
    similarity_score: Optional[float] = None
    content_preview: str


class DocumentSearchResponse(BaseModel):
    status: str
    query: str
    result_count: int
    results: List[RetrievedDocument]


class IndexedDocumentSummary(BaseModel):
    file_name: str
    file_type: Optional[str] = None
    chunk_count: int


class VectorStoreStatsResponse(BaseModel):
    status: str
    collection_name: str
    total_chunks: int
    unique_document_count: int
    documents: List[IndexedDocumentSummary]


class ResetVectorStoreResponse(BaseModel):
    status: str
    message: str
    collection_name: str

from datetime import datetime


class DocumentRecordResponse(BaseModel):
    id: int
    original_file_name: str
    saved_file_name: str
    file_type: str
    file_size_bytes: int
    saved_path: str
    loaded_document_count: int
    chunk_count: int
    indexed_chunk_count: int
    collection_name: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class DocumentRecordListResponse(BaseModel):
    status: str
    count: int
    records: List[DocumentRecordResponse]