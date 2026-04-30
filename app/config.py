"""Конфигурация приложения через переменные окружения.

Все настройки централизованы здесь. Pydantic валидирует типы и значения
по умолчанию подобраны так, чтобы локальный запуск работал без правок.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки приложения.

    Загружаются из переменных окружения и (опционально) из ``.env``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    # Default = Ollama + Qwen2.5:3b — полностью локально, без секретов,
    # позволяет запустить проект одной командой `docker-compose up --build`.
    llm_provider: Literal["openai", "anthropic", "ollama"] = "ollama"
    llm_model: str = "qwen2.5:3b"
    llm_temperature: float = 0.0

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    ollama_base_url: str = "http://ollama:11434"

    # Embeddings
    embedding_model: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Vector store
    chroma_persist_dir: str = "/app/chroma_db"
    chroma_collection: str = "cdekstart_kb"

    # Retrieval
    retrieval_top_k: int = Field(default=4, ge=1, le=20)

    # Knowledge base
    kb_data_dir: str = "/app/data"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    @property
    def kb_path(self) -> Path:
        return Path(self.kb_data_dir)


# Singleton-стиль: конфиг создаётся один раз при импорте
settings = Settings()
