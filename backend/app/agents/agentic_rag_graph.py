from typing import TypedDict, List, Dict, Any, Optional

from langgraph.graph import StateGraph, START, END

from app.agents.rag_agent import (
    generate_rag_answer,
    generate_summary_answer,
    generate_comparison_answer
)
from app.agents.query_writer import rewrite_question_with_history
from app.services.memory_service import (
    get_chat_history,
    add_conversation_turn,
    normalize_session_id
)


class AgentState(TypedDict, total=False):
    """
    Shared state passed between LangGraph nodes.
    """

    question: str
    session_id: str
    top_k: int

    chat_history: List[Dict[str, str]]

    rewritten_question: str
    was_rewritten: bool

    intent: str
    route_reason: str

    answer: str
    sources: List[Dict[str, Any]]
    confidence: str


def classify_intent(question: str) -> tuple[str, str]:
    """
    Classifies the user's question into an intent.

    First version uses deterministic routing.
    """

    normalized_question = question.lower().strip()

    summary_keywords = [
        "summarize",
        "summary",
        "brief",
        "overview",
        "short notes",
        "main points",
        "key points",
        "explain this document"
    ]

    comparison_keywords = [
        "compare",
        "difference",
        "differentiate",
        "versus",
        "vs",
        "old and new",
        "before and after",
        "between"
    ]

    out_of_scope_keywords = [
        "weather",
        "stock price",
        "share price",
        "cricket score",
        "news today",
        "current news",
        "movie ticket",
        "restaurant near me",
        "flight price",
        "bitcoin price"
    ]

    for keyword in out_of_scope_keywords:
        if keyword in normalized_question:
            return (
                "out_of_scope",
                f"Question matched out-of-scope keyword: '{keyword}'"
            )

    for keyword in comparison_keywords:
        if keyword in normalized_question:
            return (
                "comparison",
                f"Question matched comparison keyword: '{keyword}'"
            )

    for keyword in summary_keywords:
        if keyword in normalized_question:
            return (
                "summary",
                f"Question matched summary keyword: '{keyword}'"
            )

    return (
        "qa",
        "Default route selected for normal document question answering."
    )


def rewrite_node(state: AgentState) -> AgentState:
    """
    Rewrites follow-up questions into standalone questions.

    Example:
    Previous: What is the leave policy?
    Current: What about eligibility?
    Rewritten: What is the eligibility criteria for the leave policy?
    """

    result = rewrite_question_with_history(
        question=state.get("question", ""),
        chat_history=state.get("chat_history", [])
    )

    return {
        "rewritten_question": result["rewritten_question"],
        "was_rewritten": result["was_rewritten"]
    }


def router_node(state: AgentState) -> AgentState:
    """
    Decides which specialized node should handle the question.
    """

    original_question = state.get("question", "")
    rewritten_question = state.get("rewritten_question", original_question)

    combined_question_for_routing = f"{original_question}\n{rewritten_question}"

    intent, route_reason = classify_intent(combined_question_for_routing)

    return {
        "intent": intent,
        "route_reason": route_reason
    }


def route_by_intent(state: AgentState) -> str:
    """
    Conditional edge function.
    """

    intent = state.get("intent", "qa")

    if intent == "summary":
        return "summary"

    if intent == "comparison":
        return "comparison"

    if intent == "out_of_scope":
        return "out_of_scope"

    return "qa"


def qa_node(state: AgentState) -> AgentState:
    result = generate_rag_answer(
        question=state["question"],
        top_k=state.get("top_k", 4),
        retrieval_query=state.get("rewritten_question"),
        chat_history=state.get("chat_history", [])
    )

    return {
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "confidence": result.get("confidence", "basic"),
        "guard_status": result.get("guard_status", "not_checked"),
        "guard_reason": result.get("guard_reason")
    }

def summary_node(state: AgentState) -> AgentState:
    result = generate_summary_answer(
        question=state["question"],
        top_k=max(state.get("top_k", 4), 6),
        retrieval_query=state.get("rewritten_question"),
        chat_history=state.get("chat_history", [])
    )

    return {
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "confidence": result.get("confidence", "basic"),
        "guard_status": result.get("guard_status", "not_checked"),
        "guard_reason": result.get("guard_reason")
    }
def comparison_node(state: AgentState) -> AgentState:
    result = generate_comparison_answer(
        question=state["question"],
        top_k=max(state.get("top_k", 4), 8),
        retrieval_query=state.get("rewritten_question"),
        chat_history=state.get("chat_history", [])
    )

    return {
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "confidence": result.get("confidence", "basic"),
        "guard_status": result.get("guard_status", "not_checked"),
        "guard_reason": result.get("guard_reason")
    }

def out_of_scope_node(state: AgentState) -> AgentState:
    return {
        "answer": (
            "This question appears to be outside the uploaded enterprise documents. "
            "Please ask a question related to the indexed company knowledge base."
        ),
        "sources": [],
        "confidence": "out_of_scope",
        "guard_status": "blocked",
        "guard_reason": "Question was routed as out of scope."
    }

def build_agentic_rag_graph():
    """
    Builds and compiles the LangGraph workflow.

    Graph:
    START
      ↓
    rewrite
      ↓
    router
      ↓
    qa / summary / comparison / out_of_scope
      ↓
    END
    """

    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("rewrite", rewrite_node)
    graph_builder.add_node("router", router_node)
    graph_builder.add_node("qa", qa_node)
    graph_builder.add_node("summary", summary_node)
    graph_builder.add_node("comparison", comparison_node)
    graph_builder.add_node("out_of_scope", out_of_scope_node)

    graph_builder.add_edge(START, "rewrite")
    graph_builder.add_edge("rewrite", "router")

    graph_builder.add_conditional_edges(
        "router",
        route_by_intent,
        {
            "qa": "qa",
            "summary": "summary",
            "comparison": "comparison",
            "out_of_scope": "out_of_scope"
        }
    )

    graph_builder.add_edge("qa", END)
    graph_builder.add_edge("summary", END)
    graph_builder.add_edge("comparison", END)
    graph_builder.add_edge("out_of_scope", END)

    return graph_builder.compile()


agentic_rag_graph = build_agentic_rag_graph()


def run_agentic_rag_assistant(
    question: str,
    session_id: Optional[str] = "default-session",
    top_k: int = 4
) -> dict:
    """
    Public function used by the chat API.

    It:
    1. Loads previous chat history
    2. Runs LangGraph
    3. Stores the new user/assistant turn
    4. Returns answer metadata
    """

    normalized_session_id = normalize_session_id(session_id)
    chat_history = get_chat_history(normalized_session_id)

    initial_state: AgentState = {
        "question": question,
        "session_id": normalized_session_id,
        "top_k": top_k,
        "chat_history": chat_history
    }

    final_state = agentic_rag_graph.invoke(initial_state)

    answer = final_state.get("answer", "")

    add_conversation_turn(
        session_id=normalized_session_id,
        user_question=question,
        assistant_answer=answer
    )

    updated_history = get_chat_history(normalized_session_id)

    return {
        "answer": answer,
        "sources": final_state.get("sources", []),
        "confidence": final_state.get("confidence", "basic"),
        "intent": final_state.get("intent", "qa"),
        "route_reason": final_state.get("route_reason", ""),
        "rewritten_question": final_state.get("rewritten_question", question),
        "was_rewritten": final_state.get("was_rewritten", False),
        "history_message_count": len(updated_history),
        "guard_status": final_state.get("guard_status", "not_checked"),
        "guard_reason": final_state.get("guard_reason")
    }