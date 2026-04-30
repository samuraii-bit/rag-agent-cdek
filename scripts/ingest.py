"""CLI для ручной (пере)индексации базы знаний.

Использование (из контейнера или локально):

    python -m scripts.ingest

или для полного rebuild:

    python -m scripts.ingest --rebuild
"""
from __future__ import annotations

import argparse
import sys

from loguru import logger

from app.rag.retriever import get_default_kb


def main() -> int:
    parser = argparse.ArgumentParser(description="Индексация базы знаний CdekStart.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Полностью пересобрать индекс (стирает существующий).",
    )
    args = parser.parse_args()

    kb = get_default_kb()
    if args.rebuild:
        count = kb.rebuild_index()
    else:
        count = kb.ensure_indexed()
    logger.info("Готово. В индексе документов: {}", count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
