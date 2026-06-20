from functools import lru_cache

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from app.core.config import settings


def get_llm_provider() -> str:
    """
    Returns the selected LLM provider.
    Supported providers:
    - ollama
    - openai
    """
    return settings.LLM_PROVIDER.lower().strip()


def get_llm_model_name() -> str:
    """
    Returns the active LLM model name based on selected provider.
    """
    provider = get_llm_provider()

    if provider == "openai":
        return settings.OPENAI_MODEL

    if provider == "ollama":
        return settings.OLLAMA_MODEL

    return settings.OLLAMA_MODEL


@lru_cache(maxsize=1)
def get_chat_llm():
    """
    Creates and caches the chat LLM.

    Why cache?
    The LLM client should not be recreated for every request.
    This improves performance and keeps the code cleaner.
    """

    provider = get_llm_provider()

    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is required when LLM_PROVIDER=openai"
            )

        return ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.2
        )

    if provider == "ollama":
        return ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0.2
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER: {settings.LLM_PROVIDER}. "
        "Use either 'ollama' or 'openai'."
    )