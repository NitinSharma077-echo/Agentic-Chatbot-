import json

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.schemas.chat_schema import (
    ChatRequest,
    ChatResponse,
    SourceDocument,
    ClearMemoryResponse
)
from app.agents.agentic_rag_graph import run_agentic_rag_assistant, classify_intent
from app.agents.rag_agent import (
    stream_rag_answer,
    stream_summary_answer,
    stream_comparison_answer,
    stream_general_chat,
    generate_general_chat_answer,
)
from app.agents.query_writer import rewrite_question_with_history
from app.services.memory_service import (
    clear_chat_history,
    normalize_session_id,
    get_chat_history,
    add_conversation_turn
)
from app.db.session import get_db
from app.db.crud import create_chat_log
from app.core.security import verify_api_key


router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
    dependencies=[Depends(verify_api_key)]
)


@router.post("/", response_model=ChatResponse)
def chat_with_assistant(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    try:
        if request.mode == "chat":
            normalized_sid = normalize_session_id(request.session_id)
            history = get_chat_history(normalized_sid)
            result = generate_general_chat_answer(
                question=request.question,
                chat_history=history
            )
            result["history_message_count"] = len(history)
            add_conversation_turn(normalized_sid, request.question, result["answer"])
        else:
            result = run_agentic_rag_assistant(
                question=request.question,
                session_id=request.session_id,
                top_k=request.top_k or 4
            )

        sources = [
            SourceDocument(**source)
            for source in result.get("sources", [])
        ]

        create_chat_log(
            db=db,
            session_id=normalize_session_id(request.session_id),
            question=request.question,
            rewritten_question=result.get("rewritten_question"),
            was_rewritten=result.get("was_rewritten", False),
            answer=result["answer"],
            intent=result.get("intent", "qa"),
            route_reason=result.get("route_reason"),
            confidence=result.get("confidence", "basic"),
            guard_status=result.get("guard_status", "not_checked"),
            guard_reason=result.get("guard_reason"),
            source_count=len(sources)
        )

        return ChatResponse(
            answer=result["answer"],
            sources=sources,
            confidence=result.get("confidence", "basic"),
            intent=result.get("intent", "qa"),
            route_reason=result.get("route_reason"),
            rewritten_question=result.get("rewritten_question"),
            was_rewritten=result.get("was_rewritten", False),
            history_message_count=result.get("history_message_count", 0),
            guard_status=result.get("guard_status", "not_checked"),
            guard_reason=result.get("guard_reason")
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Agentic chat generation failed: {str(error)}"
        )


@router.delete("/memory/{session_id}", response_model=ClearMemoryResponse)
def clear_session_memory(session_id: str):
    try:
        normalized_session_id = normalize_session_id(session_id)
        clear_chat_history(normalized_session_id)

        return ClearMemoryResponse(
            status="success",
            message="Chat memory cleared successfully.",
            session_id=normalized_session_id
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear chat memory: {str(error)}"
        )


@router.post("/stream")
def chat_stream(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    normalized_session_id = normalize_session_id(request.session_id)
    chat_history = get_chat_history(normalized_session_id)
    top_k = request.top_k or 4

    # ── General chat mode: skip retrieval pipeline entirely ──────────────────
    if request.mode == "chat":
        def generate_general():
            yield f"data: {json.dumps({'type': 'start', 'intent': 'chat', 'route_reason': 'General chat mode.', 'was_rewritten': False, 'rewritten_question': request.question})}\n\n"

            full_answer = ""
            for event in stream_general_chat(request.question, chat_history):
                if event["type"] == "chunk":
                    full_answer += event["content"]
                    yield f"data: {json.dumps({'type': 'chunk', 'content': event['content']})}\n\n"
                elif event["type"] == "done":
                    yield f"data: {json.dumps({'type': 'done', 'guard_status': 'not_checked', 'guard_reason': None, 'confidence': 'general_chat'})}\n\n"
                elif event["type"] == "error":
                    yield f"data: {json.dumps({'type': 'error', 'content': event['content']})}\n\n"
                    return

            add_conversation_turn(normalized_session_id, request.question, full_answer)
            create_chat_log(
                db=db,
                session_id=normalized_session_id,
                question=request.question,
                answer=full_answer,
                intent="chat",
                route_reason="General chat mode.",
                confidence="general_chat",
                guard_status="not_checked",
                source_count=0
            )

        return StreamingResponse(
            generate_general(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )

    # ── RAG mode ─────────────────────────────────────────────────────────────
    rewrite_result = rewrite_question_with_history(
        question=request.question,
        chat_history=chat_history
    )
    rewritten_question = rewrite_result["rewritten_question"]
    was_rewritten = rewrite_result["was_rewritten"]

    combined = f"{request.question}\n{rewritten_question}"
    intent, route_reason = classify_intent(combined)

    retrieval_query = rewritten_question if was_rewritten else None

    def generate():
        yield f"data: {json.dumps({'type': 'start', 'intent': intent, 'route_reason': route_reason, 'was_rewritten': was_rewritten, 'rewritten_question': rewritten_question})}\n\n"

        if intent == "out_of_scope":
            out_answer = (
                "This question appears to be outside the uploaded enterprise documents. "
                "Please ask a question related to the indexed company knowledge base."
            )
            yield f"data: {json.dumps({'type': 'chunk', 'content': out_answer})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'guard_status': 'blocked', 'guard_reason': 'Question was routed as out of scope.', 'sources': [], 'confidence': 'out_of_scope'})}\n\n"

            create_chat_log(
                db=db,
                session_id=normalized_session_id,
                question=request.question,
                rewritten_question=rewritten_question,
                was_rewritten=was_rewritten,
                answer=out_answer,
                intent=intent,
                route_reason=route_reason,
                confidence="out_of_scope",
                guard_status="blocked",
                guard_reason="Question was routed as out of scope.",
                source_count=0
            )
            return

        if intent == "summary":
            streamer = stream_summary_answer(
                question=request.question,
                top_k=max(top_k, 6),
                retrieval_query=retrieval_query,
                chat_history=chat_history
            )
        elif intent == "comparison":
            streamer = stream_comparison_answer(
                question=request.question,
                top_k=max(top_k, 8),
                retrieval_query=retrieval_query,
                chat_history=chat_history
            )
        else:
            streamer = stream_rag_answer(
                question=request.question,
                top_k=top_k,
                retrieval_query=retrieval_query,
                chat_history=chat_history
            )

        full_answer = ""
        sources = []
        confidence = "basic"
        guard_status = "not_checked"
        guard_reason = None

        for event in streamer:
            if event["type"] == "metadata":
                sources = event["sources"]
                confidence = event["confidence"]
                yield f"data: {json.dumps({'type': 'metadata', 'sources': sources, 'confidence': confidence})}\n\n"

            elif event["type"] == "chunk":
                full_answer += event["content"]
                yield f"data: {json.dumps({'type': 'chunk', 'content': event['content']})}\n\n"

            elif event["type"] == "done":
                guard_status = event["guard_status"]
                guard_reason = event["guard_reason"]
                if event.get("blocked_answer"):
                    full_answer = event["blocked_answer"]
                    yield f"data: {json.dumps({'type': 'correction', 'content': full_answer})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'guard_status': guard_status, 'guard_reason': guard_reason, 'confidence': confidence})}\n\n"

            elif event["type"] == "error":
                yield f"data: {json.dumps({'type': 'error', 'content': event['content']})}\n\n"
                return

        add_conversation_turn(
            session_id=normalized_session_id,
            user_question=request.question,
            assistant_answer=full_answer
        )
        create_chat_log(
            db=db,
            session_id=normalized_session_id,
            question=request.question,
            rewritten_question=rewritten_question,
            was_rewritten=was_rewritten,
            answer=full_answer,
            intent=intent,
            route_reason=route_reason,
            confidence=confidence,
            guard_status=guard_status,
            guard_reason=guard_reason,
            source_count=len(sources)
        )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )