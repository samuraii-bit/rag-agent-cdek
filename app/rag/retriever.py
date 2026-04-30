"""Векторный ретривер на ChromaDB."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from app.config import settings
from app.rag.embeddings import embed_texts
from app.rag.ingest import KbDocument, load_kb

if TYPE_CHECKING:  # pragma: no cover
    from chromadb.api import ClientAPI
    from chromadb.api.models.Collection import Collection


@dataclass(frozen=True)
class RetrievedChunk:
    """Один найденный фрагмент."""

    text: str
    source: str
    country: str
    topic: str
    score: float

    @property
    def metadata(self) -> dict[str, str]:
        return {"source": self.source, "country": self.country, "topic": self.topic}


class KnowledgeBase:
    """Обёртка над коллекцией ChromaDB.

    Выполняет три задачи:

    * ``rebuild_index`` — стирает коллекцию и индексирует все файлы заново.
    * ``ensure_indexed`` — индексирует только при пустой коллекции.
    * ``search`` — векторный поиск с опциональным фильтром по стране.
    """

    def __init__(
        self,
        *,
        persist_dir: str | None = None,
        collection_name: str | None = None,
        data_dir: Path | None = None,
    ) -> None:
        self._persist_dir = persist_dir or settings.chroma_persist_dir
        self._collection_name = collection_name or settings.chroma_collection
        self._data_dir = data_dir or settings.kb_path
        self._client: "ClientAPI" | None = None
        self._collection: "Collection" | None = None

    # ---- low-level access ----------------------------------------------------
    def _get_client(self) -> "ClientAPI":
        if self._client is None:
            import chromadb

            Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self._persist_dir)
        return self._client

    def _get_collection(self) -> "Collection":
        if self._collection is None:
            client = self._get_client()
            # cosine distance даёт результат в [0, 2]; чем меньше — тем ближе.
            self._collection = client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    # ---- public API ----------------------------------------------------------
    def count(self) -> int:
        return self._get_collection().count()

    def ensure_indexed(self) -> int:
        """Индексирует, только если коллекция пуста. Возвращает кол-во документов."""
        if self.count() > 0:
            logger.info("Индекс уже построен ({} документов).", self.count())
            return self.count()
        return self.rebuild_index()

    def rebuild_index(self) -> int:
        """Полная переиндексация."""
        documents = load_kb(self._data_dir)
        if not documents:
            logger.warning("Нет документов для индексации.")
            return 0

        # Чтобы не дублировать чанки, чистим коллекцию.
        client = self._get_client()
        try:
            client.delete_collection(self._collection_name)
        except Exception:  # noqa: BLE001
            # коллекции может не быть — не страшно
            pass
        self._collection = None  # пересоздадим в _get_collection

        collection = self._get_collection()
        embeddings = embed_texts([d.text for d in documents])
        collection.add(
            ids=[d.doc_id for d in documents],
            documents=[d.text for d in documents],
            metadatas=[d.metadata for d in documents],
            embeddings=embeddings,
        )
        logger.info("Проиндексировано документов: {}", len(documents))
        return len(documents)

    def search(
        self,
        query: str,
        *,
        country: str | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Векторный поиск.

        Если ``country`` указана, результат сужается до документов этой
        страны и до общих (``country == 'none'``) — общая информация
        (например, дедлайны) релевантна для любого запроса.
        """
        top_k = top_k or settings.retrieval_top_k
        collection = self._get_collection()

        where: dict[str, Any] | None = None
        if country and country != "none":
            where = {"country": {"$in": [country, "none"]}}

        query_vec = embed_texts([query])[0]
        result = collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            where=where,
        )

        chunks: list[RetrievedChunk] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for _id, doc, meta, dist in zip(ids, docs, metas, distances, strict=False):
            chunks.append(
                RetrievedChunk(
                    text=doc,
                    source=str((meta or {}).get("source", _id)),
                    country=str((meta or {}).get("country", "none")),
                    topic=str((meta or {}).get("topic", "")),
                    # переводим cosine distance в подобие [0..1]
                    score=max(0.0, 1.0 - float(dist)),
                )
            )
        return chunks


_default_kb: KnowledgeBase | None = None


def get_default_kb() -> KnowledgeBase:
    """Singleton для приложения."""
    global _default_kb
    if _default_kb is None:
        _default_kb = KnowledgeBase()
    return _default_kb
