from typing import Iterator, List, Tuple, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from app.rag.vector_store import search_similar_documents
from app.services.llm_service import get_chat_llm
from app.services.memory_service import format_chat_history_for_prompt
from app.services.hallusination_guard import validate_rag_answer


GENERAL_CHAT_SYSTEM_PROMPT = """
You are a helpful, knowledgeable AI assistant.
Answer the user's question clearly and concisely.
Use the conversation history to maintain context across turns.
If asked about documents or company knowledge, let the user know they can switch to RAG mode.
"""


BASE_GROUNDING_RULES = """
You are an Enterprise Knowledge Assistant.

Rules:
1. Use only the provided enterprise document context as the factual source.
2. If the context does not contain the answer, say:
   "I could not find this information in the uploaded documents."
3. Do not guess.
4. Do not use outside knowledge.
5. Keep the answer clear, professional, and simple.
6. Mention that the answer is based on the uploaded documents.
7. Conversation history may be used only to understand the user's intent, not as a factual source.
"""


QA_SYSTEM_PROMPT = BASE_GROUNDING_RULES + """

Task:
Answer the user's question directly using the provided context.

Formatting:
- Start with a clear answer.
- Use bullet points if helpful.
- Keep the answer concise unless the user asks for detail.
"""


SUMMARY_SYSTEM_PROMPT = BASE_GROUNDING_RULES + """

Task:
Summarize the relevant uploaded document content.

Formatting:
Use this structure:
1. Short Summary
2. Key Points
3. Important Details
4. Possible Action Items, if available in the context

Do not add action items unless they are supported by the context.
"""


COMPARISON_SYSTEM_PROMPT = BASE_GROUNDING_RULES + """

Task:
Compare the items, policies, processes, or document sections requested by the user.

Formatting:
Use this structure:
1. Quick Conclusion
2. Comparison Table
3. Key Differences
4. Recommendation or Interpretation, only if supported by context

If the context does not contain enough information for both sides of the comparison,
clearly say what is missing.
"""


def build_context_from_documents(
    search_results: List[Tuple[Document, float]]
) -> str:
    context_parts = []

    for index, result in enumerate(search_results, start=1):
        document, score = result
        metadata = document.metadata or {}

        file_name = metadata.get("file_name", "unknown_file")
        page = metadata.get("page")
        chunk_id = metadata.get("chunk_id", "unknown_chunk")

        if page is not None:
            try:
                page = int(page) + 1
            except Exception:
                page = "unknown"

        source_label = (
            f"Source {index} | "
            f"File: {file_name} | "
            f"Page: {page if page is not None else 'N/A'} | "
            f"Chunk: {chunk_id} | "
            f"Similarity Score: {round(float(score), 4)}"
        )

        context_parts.append(
            f"{source_label}\n"
            f"Content:\n{document.page_content}"
        )

    return "\n\n---\n\n".join(context_parts)


def build_source_documents(
    search_results: List[Tuple[Document, float]]
) -> list[dict]:
    sources = []

    for document, score in search_results:
        metadata = document.metadata or {}

        page_number = metadata.get("page")

        if page_number is not None:
            try:
                page_number = int(page_number) + 1
            except Exception:
                page_number = None

        preview = document.page_content[:500]

        if len(document.page_content) > 500:
            preview += "..."

        sources.append(
            {
                "file_name": metadata.get("file_name", "unknown"),
                "file_type": metadata.get("file_type"),
                "page_number": page_number,
                "chunk_id": metadata.get("chunk_id"),
                "similarity_score": float(score),
                "content_preview": preview
            }
        )

    return sources


def calculate_confidence(search_results: List[Tuple[Document, float]]) -> str:
    if not search_results:
        return "no_context"

    best_score = float(search_results[0][1])

    if best_score <= 0.6:
        return "high"

    if best_score <= 1.2:
        return "medium"

    return "low"


def generate_answer_from_context(
    question: str,
    system_prompt: str,
    top_k: int = 4,
    retrieval_query: Optional[str] = None,
    chat_history: Optional[List[dict]] = None
) -> dict:
    if not question or not question.strip():
        return {
            "answer": "Please ask a valid question.",
            "sources": [],
            "confidence": "no_question",
            "guard_status": "blocked",
            "guard_reason": "Question was empty."
        }

    search_query = retrieval_query or question

    search_results = search_similar_documents(
        query=search_query,
        top_k=top_k
    )

    if not search_results:
        return {
            "answer": (
                "I could not find this information in the uploaded documents. "
                "Please upload relevant documents first or ask a question related to the indexed knowledge base."
            ),
            "sources": [],
            "confidence": "no_context",
            "guard_status": "blocked",
            "guard_reason": "No relevant document chunks were retrieved."
        }

    context = build_context_from_documents(search_results)
    sources = build_source_documents(search_results)
    confidence = calculate_confidence(search_results)

    history_text = format_chat_history_for_prompt(
        chat_history=chat_history or [],
        max_messages=6
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                """
Conversation History:
{chat_history}

Original User Question:
{question}

Standalone Retrieval Query:
{retrieval_query}

Enterprise Document Context:
{context}

Now complete the task using only the enterprise document context above.
"""
            )
        ]
    )

    llm = get_chat_llm()
    chain = prompt | llm

    response = chain.invoke(
        {
            "chat_history": history_text,
            "question": question,
            "retrieval_query": search_query,
            "context": context
        }
    )

    raw_answer = getattr(response, "content", str(response))

    validation = validate_rag_answer(
        question=question,
        answer=raw_answer,
        search_results=search_results,
        confidence=confidence
    )

    return {
        "answer": validation["final_answer"],
        "sources": sources,
        "confidence": confidence,
        "guard_status": validation["guard_status"],
        "guard_reason": validation["guard_reason"]
    }


def generate_rag_answer(
    question: str,
    top_k: int = 4,
    retrieval_query: Optional[str] = None,
    chat_history: Optional[List[dict]] = None
) -> dict:
    return generate_answer_from_context(
        question=question,
        system_prompt=QA_SYSTEM_PROMPT,
        top_k=top_k,
        retrieval_query=retrieval_query,
        chat_history=chat_history
    )


def generate_summary_answer(
    question: str,
    top_k: int = 6,
    retrieval_query: Optional[str] = None,
    chat_history: Optional[List[dict]] = None
) -> dict:
    return generate_answer_from_context(
        question=question,
        system_prompt=SUMMARY_SYSTEM_PROMPT,
        top_k=top_k,
        retrieval_query=retrieval_query,
        chat_history=chat_history
    )


def generate_comparison_answer(
    question: str,
    top_k: int = 8,
    retrieval_query: Optional[str] = None,
    chat_history: Optional[List[dict]] = None
) -> dict:
    return generate_answer_from_context(
        question=question,
        system_prompt=COMPARISON_SYSTEM_PROMPT,
        top_k=top_k,
        retrieval_query=retrieval_query,
        chat_history=chat_history
    )


def stream_answer_from_context(
    question: str,
    system_prompt: str,
    top_k: int = 4,
    retrieval_query: Optional[str] = None,
    chat_history: Optional[List[dict]] = None
) -> Iterator[dict]:
    if not question or not question.strip():
        yield {"type": "error", "content": "Please ask a valid question."}
        return

    search_query = retrieval_query or question

    search_results = search_similar_documents(query=search_query, top_k=top_k)

    if not search_results:
        yield {
            "type": "error",
            "content": (
                "I could not find this information in the uploaded documents. "
                "Please upload relevant documents first or ask a question "
                "related to the indexed knowledge base."
            )
        }
        return

    context = build_context_from_documents(search_results)
    sources = build_source_documents(search_results)
    confidence = calculate_confidence(search_results)

    yield {"type": "metadata", "sources": sources, "confidence": confidence}

    history_text = format_chat_history_for_prompt(
        chat_history=chat_history or [],
        max_messages=6
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                """
Conversation History:
{chat_history}

Original User Question:
{question}

Standalone Retrieval Query:
{retrieval_query}

Enterprise Document Context:
{context}

Now complete the task using only the enterprise document context above.
"""
            )
        ]
    )

    llm = get_chat_llm()
    chain = prompt | llm

    full_answer = ""
    for chunk in chain.stream(
        {
            "chat_history": history_text,
            "question": question,
            "retrieval_query": search_query,
            "context": context
        }
    ):
        content = getattr(chunk, "content", "")
        if content:
            full_answer += content
            yield {"type": "chunk", "content": content}

    validation = validate_rag_answer(
        question=question,
        answer=full_answer,
        search_results=search_results,
        confidence=confidence
    )

    yield {
        "type": "done",
        "guard_status": validation["guard_status"],
        "guard_reason": validation["guard_reason"],
        "blocked_answer": (
            validation["final_answer"]
            if validation["guard_status"] == "blocked"
            else None
        )
    }


def stream_rag_answer(
    question: str,
    top_k: int = 4,
    retrieval_query: Optional[str] = None,
    chat_history: Optional[List[dict]] = None
) -> Iterator[dict]:
    return stream_answer_from_context(
        question=question,
        system_prompt=QA_SYSTEM_PROMPT,
        top_k=top_k,
        retrieval_query=retrieval_query,
        chat_history=chat_history
    )


def stream_summary_answer(
    question: str,
    top_k: int = 6,
    retrieval_query: Optional[str] = None,
    chat_history: Optional[List[dict]] = None
) -> Iterator[dict]:
    return stream_answer_from_context(
        question=question,
        system_prompt=SUMMARY_SYSTEM_PROMPT,
        top_k=top_k,
        retrieval_query=retrieval_query,
        chat_history=chat_history
    )


def stream_comparison_answer(
    question: str,
    top_k: int = 8,
    retrieval_query: Optional[str] = None,
    chat_history: Optional[List[dict]] = None
) -> Iterator[dict]:
    return stream_answer_from_context(
        question=question,
        system_prompt=COMPARISON_SYSTEM_PROMPT,
        top_k=top_k,
        retrieval_query=retrieval_query,
        chat_history=chat_history
    )


def generate_general_chat_answer(
    question: str,
    chat_history: Optional[List[dict]] = None
) -> dict:
    if not question or not question.strip():
        return {"answer": "Please ask a valid question.", "sources": [], "confidence": "no_question"}

    history_text = format_chat_history_for_prompt(chat_history or [], max_messages=6)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", GENERAL_CHAT_SYSTEM_PROMPT),
            ("human", "Conversation History:\n{chat_history}\n\nUser: {question}")
        ]
    )

    llm = get_chat_llm()
    response = (prompt | llm).invoke({"chat_history": history_text, "question": question})

    return {
        "answer": getattr(response, "content", str(response)),
        "sources": [],
        "confidence": "general_chat",
        "intent": "chat",
        "route_reason": "General chat mode selected by user.",
        "rewritten_question": question,
        "was_rewritten": False,
        "guard_status": "not_checked",
        "guard_reason": None
    }


def stream_general_chat(
    question: str,
    chat_history: Optional[List[dict]] = None
) -> Iterator[dict]:
    if not question or not question.strip():
        yield {"type": "error", "content": "Please ask a valid question."}
        return

    history_text = format_chat_history_for_prompt(chat_history or [], max_messages=6)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", GENERAL_CHAT_SYSTEM_PROMPT),
            ("human", "Conversation History:\n{chat_history}\n\nUser: {question}")
        ]
    )

    llm = get_chat_llm()
    chain = prompt | llm

    for chunk in chain.stream({"chat_history": history_text, "question": question}):
        content = getattr(chunk, "content", "")
        if content:
            yield {"type": "chunk", "content": content}

    yield {
        "type": "done",
        "guard_status": "not_checked",
        "guard_reason": "General chat mode — no retrieval.",
        "blocked_answer": None
    }