from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        description="User question for the knowledge assistant"
    )

    session_id: Optional[str] = Field(
        default="default-session",
        description="Chat session ID for maintaining conversation context"
    )

    top_k: Optional[int] = Field(
        default=4,
        ge=1,
        le=10,
        description="Number of relevant document chunks to retrieve"
    )

    mode: Optional[Literal["rag", "chat"]] = Field(
        default="rag",
        description="'rag' uses document retrieval; 'chat' is general LLM conversation"
    )


class SourceDocument(BaseModel):
    file_name: str
    file_type: Optional[str] = None
    page_number: Optional[int] = None
    chunk_id: Optional[int] = None
    similarity_score: Optional[float] = None
    content_preview: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDocument] = []
    confidence: Optional[str] = "basic"

    intent: Optional[str] = "qa"
    route_reason: Optional[str] = None

    rewritten_question: Optional[str] = None
    was_rewritten: bool = False
    history_message_count: int = 0

    guard_status: Optional[str] = "not_checked"
    guard_reason: Optional[str] = None


class ClearMemoryResponse(BaseModel):
    status: str
    message: str
    session_id: str