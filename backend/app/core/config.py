from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    APP_NAME: str = Field(default="Enterprise Agentic RAG Knowledge Assistant")
    APP_VERSION: str = Field(default="1.0.0")
    APP_ENV: str = Field(default="development")

    LLM_PROVIDER: str = Field(default="ollama")

    OPENAI_API_KEY: str = Field(default="")
    OPENAI_MODEL: str = Field(default="gpt-4o-mini")

    OLLAMA_MODEL: str = Field(default="gemma3:1b")
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434")

    EMBEDDING_PROVIDER: str = Field(default="local")
    LOCAL_EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2"
    )

    CHROMA_DB_PATH: str = Field(default="app/storage/chroma_db")
    UPLOAD_DIR: str = Field(default="app/storage/uploaded_docs")

    DATABASE_URL: str = Field(default="sqlite:///./app/storage/app.db")

    API_KEY: str = Field(default="dev-secret-key-change-this")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()