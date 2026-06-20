from typing import List, Tuple, Dict, Any

from langchain_core.documents import Document


BLOCKED_FALLBACK_ANSWER = (
    "I could not find enough reliable information in the uploaded documents "
    "to answer this confidently."
)


RISKY_PHRASES = [
    "i think",
    "probably",
    "maybe",
    "it seems",
    "generally",
    "usually",
    "in most cases",
    "as far as i know",
    "based on my knowledge",
    "outside the provided context"
]


def has_risky_language(answer: str) -> bool:
    """
    Detects phrases that often indicate guessing.

    This is a simple rule-based guard.
    Later we can replace it with an LLM judge or LangSmith evaluation.
    """

    if not answer:
        return False

    normalized_answer = answer.lower()

    return any(phrase in normalized_answer for phrase in RISKY_PHRASES)


def answer_is_empty_or_invalid(answer: str) -> bool:
    """
    Checks whether answer is empty or unusable.
    """

    if not answer:
        return True

    cleaned = answer.strip()

    if len(cleaned) < 10:
        return True

    return False


def calculate_context_size(search_results: List[Tuple[Document, float]]) -> int:
    """
    Calculates total retrieved context size.
    """

    total_length = 0

    for document, _score in search_results:
        total_length += len(document.page_content or "")

    return total_length


def get_best_retrieval_score(
    search_results: List[Tuple[Document, float]]
) -> float | None:
    """
    Gets the best retrieval score.

    Chroma usually returns distance-like scores.
    Lower score generally means better match.
    """

    if not search_results:
        return None

    return float(search_results[0][1])


def validate_rag_answer(
    question: str,
    answer: str,
    search_results: List[Tuple[Document, float]],
    confidence: str
) -> Dict[str, Any]:
    """
    Validates the generated answer.

    Returns a validation object:
    {
        "is_valid": bool,
        "guard_status": "passed" | "warning" | "blocked",
        "guard_reason": "...",
        "final_answer": "..."
    }
    """

    if not search_results:
        return {
            "is_valid": False,
            "guard_status": "blocked",
            "guard_reason": "No retrieved document context was available.",
            "final_answer": BLOCKED_FALLBACK_ANSWER
        }

    if answer_is_empty_or_invalid(answer):
        return {
            "is_valid": False,
            "guard_status": "blocked",
            "guard_reason": "Generated answer was empty or too short.",
            "final_answer": BLOCKED_FALLBACK_ANSWER
        }

    best_score = get_best_retrieval_score(search_results)
    context_size = calculate_context_size(search_results)

    if confidence == "low" and best_score is not None and best_score > 1.8:
        return {
            "is_valid": False,
            "guard_status": "blocked",
            "guard_reason": (
                "Retrieval confidence was too low to safely answer."
            ),
            "final_answer": BLOCKED_FALLBACK_ANSWER
        }

    if context_size < 100 and len(answer) > 400:
        return {
            "is_valid": False,
            "guard_status": "blocked",
            "guard_reason": (
                "Answer was too detailed compared to the available context."
            ),
            "final_answer": BLOCKED_FALLBACK_ANSWER
        }

    if has_risky_language(answer):
        return {
            "is_valid": True,
            "guard_status": "warning",
            "guard_reason": (
                "Answer contains uncertain language. User should verify sources."
            ),
            "final_answer": (
                answer
                + "\n\nNote: Please verify this answer from the cited uploaded documents."
            )
        }

    return {
        "is_valid": True,
        "guard_status": "passed",
        "guard_reason": "Answer passed basic grounding checks.",
        "final_answer": answer
    }