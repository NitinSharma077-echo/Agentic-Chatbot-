from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from app.core.config import settings


@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Creates and caches the embedding model.

    Why cache?
    Loading embedding models repeatedly is slow.
    This function loads the model once and reuses it.
    """

    embedding_provider = settings.EMBEDDING_PROVIDER.lower().strip()

    if embedding_provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai"
            )

        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.OPENAI_API_KEY
        )

    if embedding_provider == "local":
        return HuggingFaceEmbeddings(
            model_name=settings.LOCAL_EMBEDDING_MODEL,
            model_kwargs={
                "device": "cpu"
            },
            encode_kwargs={
                "normalize_embeddings": True
            }
        )

    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER: {settings.EMBEDDING_PROVIDER}. "
        "Use either 'local' or 'openai'."
    )


def embed_text(text: str):
    """
    Converts a single text query into an embedding vector.
    Useful for testing.
    """

    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    embedding_model = get_embedding_model()
    return embedding_model.embed_query(text)


def embed_documents(texts: list[str]):
    """
    Converts multiple text chunks into embeddings.
    """

    clean_texts = [text for text in texts if text and text.strip()]

    if not clean_texts:
        return []

    embedding_model = get_embedding_model()
    return embedding_model.embed_documents(clean_texts)