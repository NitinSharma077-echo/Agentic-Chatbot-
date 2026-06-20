from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class ChatLogResponse(BaseModel):
    id: int
    session_id: str

    question: str
    rewritten_question: Optional[str] = None
    was_rewritten: bool = False

    answer: str

    intent: Optional[str] = None
    route_reason: Optional[str] = None

    confidence: Optional[str] = None
    guard_status: Optional[str] = None
    guard_reason: Optional[str] = None

    source_count: int = 0
    created_at: datetime

    class Config:
        from_attributes = True


class ChatLogListResponse(BaseModel):
    status: str
    count: int
    logs: List[ChatLogResponse]