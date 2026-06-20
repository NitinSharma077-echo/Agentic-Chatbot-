from typing import List, Optional

from sqlalchemy.orm import Session

from app.db.models import ChatLog, DocumentRecord


def create_chat_log(
    db: Session,
    session_id: str,
    question: str,
    answer: str,
    rewritten_question: Optional[str] = None,
    was_rewritten: bool = False,
    intent: Optional[str] = None,
    route_reason: Optional[str] = None,
    confidence: Optional[str] = None,
    guard_status: Optional[str] = None,
    guard_reason: Optional[str] = None,
    source_count: int = 0
) -> ChatLog:
    log = ChatLog(
        session_id=session_id,
        question=question,
        rewritten_question=rewritten_question,
        was_rewritten=was_rewritten,
        answer=answer,
        intent=intent,
        route_reason=route_reason,
        confidence=confidence,
        guard_status=guard_status,
        guard_reason=guard_reason,
        source_count=source_count
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def get_recent_chat_logs(
    db: Session,
    session_id: Optional[str] = None,
    limit: int = 20
) -> List[ChatLog]:
    query = db.query(ChatLog)
    if session_id:
        query = query.filter(ChatLog.session_id == session_id)
    return query.order_by(ChatLog.created_at.desc()).limit(limit).all()


def create_document_record(
    db: Session,
    original_file_name: str,
    saved_file_name: str,
    file_type: str,
    file_size_bytes: int,
    saved_path: str,
    loaded_document_count: int = 0,
    chunk_count: int = 0,
    indexed_chunk_count: int = 0,
    collection_name: Optional[str] = None
) -> DocumentRecord:
    record = DocumentRecord(
        original_file_name=original_file_name,
        saved_file_name=saved_file_name,
        file_type=file_type,
        file_size_bytes=file_size_bytes,
        saved_path=saved_path,
        loaded_document_count=loaded_document_count,
        chunk_count=chunk_count,
        indexed_chunk_count=indexed_chunk_count,
        collection_name=collection_name
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_recent_document_records(
    db: Session,
    limit: int = 50
) -> List[DocumentRecord]:
    return (
        db.query(DocumentRecord)
        .order_by(DocumentRecord.created_at.desc())
        .limit(limit)
        .all()
    )
