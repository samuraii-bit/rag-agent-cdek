"""Локальные эмбеддинги через sentence-transformers.

Используется multilingual-модель — она корректно обрабатывает
русский текст без OpenAI API.
"""
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from app.config import settings

if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def get_embedder() -> "SentenceTransformer":
    """Кэшируем загрузку модели — она тяжёлая (несколько сотен МБ)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Превратить список строк в список векторов."""
    model = get_embedder()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,  # косинусное сходство ↔ скалярное произведение
        show_progress_bar=False,
    )
    return [vec.tolist() for vec in vectors]
