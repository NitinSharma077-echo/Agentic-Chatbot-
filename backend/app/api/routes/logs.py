from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.crud import get_recent_chat_logs
from app.schemas.log_schema import ChatLogListResponse, ChatLogResponse


router = APIRouter(prefix="/logs", tags=["Logs"])


@router.get("/chats", response_model=ChatLogListResponse)
def list_chat_logs(
    session_id: Optional[str] = None,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """
    Lists recent chat logs from SQLite.
    Useful for debugging, audits, and demo explanation.
    """

    try:
        logs = get_recent_chat_logs(
            db=db,
            session_id=session_id,
            limit=limit
        )

        return ChatLogListResponse(
            status="success",
            count=len(logs),
            logs=[
                ChatLogResponse.model_validate(log)
                for log in logs
            ]
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list chat logs: {str(error)}"
        )