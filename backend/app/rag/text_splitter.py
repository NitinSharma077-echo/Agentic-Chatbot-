from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

def split_documents(documents: List[Document], chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[Document]:
    if not documents:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", ""]
    )
    chunks = splitter.split_documents(documents)
    enriched_chunks = []
    for index, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata)
        metadata["chunk_id"] = index
        metadata["chunk_size"] = len(chunk.page_content)
        enriched_chunks.append(
            Document(
                page_content=chunk.page_content,
                metadata=metadata
            )
        )
    return enriched_chunks

def get_sample_chunks(chunks: List[Document], limit: int = 3) -> List[str]:
    """
    Returns a few small chunk previews for API response.
    Useful for debugging and demo.
    """

    previews = []

    for chunk in chunks[:limit]:
        preview = chunk.page_content[:300]

        if len(chunk.page_content) > 300:
            preview += "..."

        previews.append(preview)

    return previews