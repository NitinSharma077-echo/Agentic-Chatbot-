from collections import defaultdict, deque
from threading import Lock
from typing import Dict, List, Literal, Optional


MessageRole = Literal["user", "assistant"]

MAX_MESSAGES_PER_SESSION = 12

_chat_sessions: Dict[str, deque] = defaultdict(
    lambda: deque(maxlen=MAX_MESSAGES_PER_SESSION)
)

_memory_lock = Lock()


def normalize_session_id(session_id: Optional[str]) -> str:
    """
    Creates a safe fallback session ID.
    """

    if not session_id or not session_id.strip():
        return "default-session"

    return session_id.strip()


def get_chat_history(session_id: Optional[str]) -> List[dict]:
    """
    Returns chat history for one session.

    A copy is returned so the original memory cannot be modified accidentally.
    """

    normalized_session_id = normalize_session_id(session_id)

    with _memory_lock:
        return list(_chat_sessions[normalized_session_id])


def add_message(
    session_id: Optional[str],
    role: MessageRole,
    content: str
) -> None:
    """
    Adds one message to session memory.
    """

    if role not in {"user", "assistant"}:
        raise ValueError("Role must be either 'user' or 'assistant'.")

    if not content or not content.strip():
        return

    normalized_session_id = normalize_session_id(session_id)

    with _memory_lock:
        _chat_sessions[normalized_session_id].append(
            {
                "role": role,
                "content": content.strip()
            }
        )


def add_conversation_turn(
    session_id: Optional[str],
    user_question: str,
    assistant_answer: str
) -> None:
    """
    Adds user question and assistant answer together.
    """

    add_message(
        session_id=session_id,
        role="user",
        content=user_question
    )

    add_message(
        session_id=session_id,
        role="assistant",
        content=assistant_answer
    )


def clear_chat_history(session_id: Optional[str]) -> None:
    """
    Clears memory for one session.
    """

    normalized_session_id = normalize_session_id(session_id)

    with _memory_lock:
        _chat_sessions[normalized_session_id].clear()


def format_chat_history_for_prompt(
    chat_history: List[dict],
    max_messages: int = 6
) -> str:
    """
    Converts recent chat history into readable prompt text.
    """

    if not chat_history:
        return "No previous conversation."

    recent_messages = chat_history[-max_messages:]

    formatted_messages = []

    for message in recent_messages:
        role = message.get("role", "unknown")
        content = message.get("content", "")

        if not content:
            continue

        formatted_messages.append(
            f"{role.upper()}: {content}"
        )

    if not formatted_messages:
        return "No previous conversation."

    return "\n".join(formatted_messages)