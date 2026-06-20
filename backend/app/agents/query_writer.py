import re
from typing import List

from langchain_core.prompts import ChatPromptTemplate

from app.services.llm_service import get_chat_llm
from app.services.memory_service import format_chat_history_for_prompt


FOLLOW_UP_PATTERNS = [
    r"\bit\b",
    r"\bthis\b",
    r"\bthat\b",
    r"\bthese\b",
    r"\bthose\b",
    r"\bthey\b",
    r"\bthem\b",
    r"\bhe\b",
    r"\bshe\b",
    r"\bsame\b",
    r"\babove\b",
    r"\bprevious\b",
    r"\bearlier\b",
    r"\bwhat about\b",
    r"\band\b",
    r"\balso\b",
    r"\bmore\b",
    r"\beligibility\b",
    r"\bcriteria\b",
    r"\bprocess\b",
    r"\bsteps\b"
]


REWRITE_SYSTEM_PROMPT = """
You are a query rewriting assistant for an enterprise RAG system.

Your job:
Rewrite the user's latest question into a clear standalone search query.

Rules:
1. Use the conversation history only to understand references.
2. Do not answer the question.
3. Do not add facts that are not present in the conversation.
4. Keep the rewritten question short and specific.
5. If the latest question is already standalone, return it unchanged.
6. Return only the rewritten question, no explanation.
"""


def should_rewrite_question(
    question: str,
    chat_history: List[dict]
) -> bool:
    """
    Decides whether rewriting is needed.

    If there is no history, rewriting is not needed.
    If the question has follow-up words, rewriting is useful.
    """

    if not chat_history:
        return False

    if not question or not question.strip():
        return False

    normalized_question = question.lower().strip()

    if len(normalized_question.split()) <= 5:
        return True

    for pattern in FOLLOW_UP_PATTERNS:
        if re.search(pattern, normalized_question):
            return True

    return False


def clean_rewritten_question(text: str, fallback_question: str) -> str:
    """
    Cleans LLM output.
    """

    if not text:
        return fallback_question

    cleaned = text.strip()
    cleaned = cleaned.strip('"')
    cleaned = cleaned.strip("'")
    cleaned = cleaned.replace("Rewritten question:", "").strip()

    if not cleaned:
        return fallback_question

    if len(cleaned) > 500:
        return fallback_question

    return cleaned


def rewrite_question_with_history(
    question: str,
    chat_history: List[dict]
) -> dict:
    """
    Rewrites a follow-up question into a standalone question.
    """

    if not should_rewrite_question(question, chat_history):
        return {
            "rewritten_question": question,
            "was_rewritten": False
        }

    history_text = format_chat_history_for_prompt(
        chat_history=chat_history,
        max_messages=6
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REWRITE_SYSTEM_PROMPT),
            (
                "human",
                """
Conversation History:
{chat_history}

Latest Question:
{question}

Standalone Search Query:
"""
            )
        ]
    )

    try:
        llm = get_chat_llm()
        chain = prompt | llm

        response = chain.invoke(
            {
                "chat_history": history_text,
                "question": question
            }
        )

        rewritten = getattr(response, "content", str(response))

        cleaned_rewritten = clean_rewritten_question(
            text=rewritten,
            fallback_question=question
        )

        return {
            "rewritten_question": cleaned_rewritten,
            "was_rewritten": cleaned_rewritten.strip() != question.strip()
        }

    except Exception:
        return {
            "rewritten_question": question,
            "was_rewritten": False
        }