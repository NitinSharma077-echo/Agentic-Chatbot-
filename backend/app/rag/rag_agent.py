from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from app.rag.vector_store import search_similar_documents
from app.services.llm_service import get_chat_llm

RAG_SYSTEM_PROMPT = """
You are an Enterprise Knowledge Assistant.
Your job is to answer user questions using ONLY the provided enterprise document context.

Rules:
1. Use only the provided context.
2. If the context does not contain the answer, say:
   "I could not find this information in the uploaded documents."
3. Do not guess.
4. Do not use outside knowledge.
5. Keep the answer clear, professional, and simple.
6. If the answer contains steps, use bullet points.
7. Mention that the answer is based on the uploaded documents.
"""


def build_context_from_documents(
    search_results: List[Tuple[Document, float]]
) -> str:
    """
    Converts retrieved documents into a single context string.

    Each chunk includes source metadata so the LLM can understand
    where the information came from.
    """

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
    """
    Converts retrieved documents into API response source objects.
    """

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
    """
    Creates a simple confidence label based on retrieval result quality.

    Important:
    Chroma similarity_search_with_score usually returns distance-like scores.
    Lower score often means better match.
    This is a simple practical label for demo purposes.
    """

    if not search_results:
        return "no_context"

    best_score = float(search_results[0][1])

    if best_score <= 0.6:
        return "high"

    if best_score <= 1.2:
        return "medium"

    return "low"


def generate_rag_answer(question: str, top_k: int = 4) -> dict:
    """
    Main RAG answer generation function.

    Steps:
    1. Retrieve relevant chunks from ChromaDB
    2. Build context
    3. Create prompt
    4. Call LLM
    5. Return answer, sources, and confidence
    """

    if not question or not question.strip():
        return {
            "answer": "Please ask a valid question.",
            "sources": [],
            "confidence": "no_question"
        }

    search_results = search_similar_documents(
        query=question,
        top_k=top_k
    )

    if not search_results:
        return {
            "answer": (
                "I could not find this information in the uploaded documents. "
                "Please upload relevant documents first or ask a question related to the indexed knowledge base."
            ),
            "sources": [],
            "confidence": "no_context"
        }

    context = build_context_from_documents(search_results)
    sources = build_source_documents(search_results)
    confidence = calculate_confidence(search_results)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT),
            (
                "human",
                """
Question:
{question}

Enterprise Document Context:
{context}

Now answer the question using only the context above.
"""
            )
        ]
    )

    llm = get_chat_llm()

    chain = prompt | llm

    response = chain.invoke(
        {
            "question": question,
            "context": context
        }
    )

    answer = getattr(response, "content", str(response))

    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence
    }
