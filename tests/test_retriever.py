"""Тесты для слоя индексации/ретривала.

Реальная sentence-transformers модель тяжёлая, поэтому здесь мы
патчим ``embed_texts`` псевдо-эмбеддингом на основе хеша слов.
ChromaDB при этом используется настоящая (in-process, persistent
client во временной директории).
"""
from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path

import pytest

from app.rag import retriever as retriever_module
from app.rag.retriever import KnowledgeBase

EMBED_DIM = 32


def _fake_embed(texts: list[str]) -> list[list[float]]:
    """Детерминированный псевдо-эмбеддинг.

    Каждое слово влияет на разные позиции вектора. Этого хватает для
    проверки, что (а) индексация проходит без ошибок и (б) метаданные
    корректно фильтруют результаты.
    """
    vectors: list[list[float]] = []
    for text in texts:
        vec = [0.0] * EMBED_DIM
        for token in text.lower().split():
            digest = hashlib.md5(token.encode("utf-8")).digest()
            for i, b in enumerate(digest):
                vec[i % EMBED_DIM] += b / 255.0
        # нормализация
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        vectors.append([v / norm for v in vec])
    return vectors


@pytest.fixture
def kb(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(retriever_module, "embed_texts", _fake_embed)
    persist = tmp_path / "chroma"
    data_dir = Path(__file__).resolve().parents[1] / "data"
    kb = KnowledgeBase(
        persist_dir=str(persist),
        collection_name="test_kb",
        data_dir=data_dir,
    )
    kb.rebuild_index()
    yield kb
    shutil.rmtree(persist, ignore_errors=True)


def test_index_contains_all_files(kb: KnowledgeBase) -> None:
    """Индексация должна добавить все 5 файлов."""
    assert kb.count() == 5


def test_search_returns_results(kb: KnowledgeBase) -> None:
    chunks = kb.search("стипендия", top_k=3)
    assert chunks, "ожидали хотя бы один результат"
    sources = {c.source for c in chunks}
    # Без фильтра возвращаются и страновые, и общие документы.
    assert sources, "источники не должны быть пустыми"


def test_country_filter_excludes_other_country(kb: KnowledgeBase) -> None:
    """Если задана Германия, документы по Франции не должны попадать."""
    chunks = kb.search("стипендия", country="germany", top_k=5)
    assert chunks
    countries = {c.country for c in chunks}
    assert "france" not in countries, (
        f"Франция не должна попасть в выдачу при фильтре по Германии: {countries}"
    )
    # Общие (country='none') допускаются вместе со страновыми.
    assert countries.issubset({"germany", "none"})


def test_country_filter_france(kb: KnowledgeBase) -> None:
    chunks = kb.search("ставка стипендии", country="france", top_k=5)
    countries = {c.country for c in chunks}
    assert "germany" not in countries
    assert countries.issubset({"france", "none"})


def test_ensure_indexed_idempotent(kb: KnowledgeBase) -> None:
    """Повторный вызов ensure_indexed не должен задвоить документы."""
    before = kb.count()
    kb.ensure_indexed()
    assert kb.count() == before
