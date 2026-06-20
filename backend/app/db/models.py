from datetime import datetime

from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean

from app.db.session import Base


class DocumentRecord(Base):
    __tablename__ = "document_records"

    id = Column(Integer, primary_key=True, index=True)

    original_file_name = Column(String(255), nullable=False)
    saved_file_name = Column(String(255), nullable=False, index=True)
    file_type = Column(String(20), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    saved_path = Column(String(500), nullable=False)

    loaded_document_count = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    indexed_chunk_count = Column(Integer, default=0)
    collection_name = Column(String(255), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)


class ChatLog(Base):
    __tablename__ = "chat_logs"

    id = Column(Integer, primary_key=True, index=True)

    session_id = Column(String(255), index=True, nullable=False)

    question = Column(Text, nullable=False)
    rewritten_question = Column(Text, nullable=True)
    was_rewritten = Column(Boolean, default=False)

    answer = Column(Text, nullable=False)

    intent = Column(String(100), nullable=True)
    route_reason = Column(Text, nullable=True)

    confidence = Column(String(100), nullable=True)
    guard_status = Column(String(100), nullable=True)
    guard_reason = Column(Text, nullable=True)

    source_count = Column(Integer, default=0)

    created_at = Column(DateTime, default=datetime.utcnow)