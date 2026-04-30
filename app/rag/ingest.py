"""Загрузка базы знаний из файлов и индексация в ChromaDB.

Стратегия чанкинга
------------------
База знаний крошечная (5 коротких файлов), поэтому каждый файл
индексируется как один чанк. Это:

* сохраняет контекст внутри файла (не теряем связь между фактами);
* при ``top_k=3`` LLM видит 3 наиболее релевантных файла целиком;
* избавляет от рисков склейки фактов из разных стран на этапе ретривала.

Метаданные
----------
Каждому чанку присваивается ``country`` (``germany`` / ``france`` /
``none``) и ``topic``. ``country`` критичен для фильтрации: когда
страна известна, мы сужаем поиск.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger

# Сопоставление имени файла -> метаданные
FILE_METADATA: dict[str, dict[str, str]] = {
    "general_info.txt": {"country": "none", "topic": "general"},
    "deadlines.txt": {"country": "none", "topic": "deadlines"},
    "benefits.txt": {"country": "none", "topic": "benefits"},
    "germany_rules.txt": {"country": "germany", "topic": "country_rules"},
    "france_rules.txt": {"country": "france", "topic": "country_rules"},
}


@dataclass(frozen=True)
class KbDocument:
    """Документ базы знаний, готовый к индексации."""

    doc_id: str
    text: str
    metadata: dict[str, str]


def load_kb(data_dir: Path) -> list[KbDocument]:
    """Прочитать все известные файлы базы знаний."""
    docs: list[KbDocument] = []
    for filename, meta in FILE_METADATA.items():
        path = data_dir / filename
        if not path.exists():
            logger.warning("Файл {} не найден, пропускаем.", path)
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            logger.warning("Файл {} пуст, пропускаем.", path)
            continue
        docs.append(
            KbDocument(
                doc_id=filename,
                text=text,
                metadata={"source": filename, **meta},
            )
        )
    logger.info("Загружено документов из {}: {}", data_dir, len(docs))
    return docs
