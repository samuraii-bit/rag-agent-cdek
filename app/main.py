"""FastAPI-приложение CdekStart RAG-агента.

Точка входа для uvicorn — переменная ``app``. Тесты используют
фабрику ``create_app`` с подменёнными зависимостями (LLM/KB),
поэтому lifespan-старт пропускается.
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from loguru import logger

from app import __version__
from app.config import settings
from app.memory.store import SessionMemory, get_memory_store
from app.schemas import ChatRequest, ChatResponse, HealthResponse, SourceRef

if TYPE_CHECKING:  # pragma: no cover
    from langgraph.graph.state import CompiledStateGraph

    from app.rag.retriever import KnowledgeBase


def _configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )


def create_app(
    *,
    graph: "CompiledStateGraph | None" = None,
    kb: "KnowledgeBase | None" = None,
    memory: SessionMemory | None = None,
) -> FastAPI:
    """Собрать FastAPI-приложение.

    * Если ``graph`` и ``kb`` переданы — lifespan ничего не строит
      (используется в тестах).
    * Иначе — на старте поднимаются настоящие KB и LangGraph
      (production / dev).
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _configure_logging()
        logger.info("Стартуем CdekStart RAG-агент v{}", __version__)

        if graph is None or kb is None:
            # Полноценный старт.
            from app.graph.builder import build_agent_graph
            from app.rag.retriever import get_default_kb

            real_kb = get_default_kb()
            indexed = real_kb.ensure_indexed()
            logger.info("База знаний готова, документов: {}", indexed)

            try:
                app.state.graph = build_agent_graph(kb=real_kb)
                logger.info("LangGraph скомпилирован.")
            except Exception as exc:  # noqa: BLE001
                logger.exception("Не удалось собрать граф: {}", exc)
                app.state.graph = None
            app.state.indexed_docs = indexed
        else:
            # Тестовый режим — всё уже передано фабрике.
            app.state.graph = graph
            app.state.indexed_docs = kb.count()

        app.state.memory = memory or get_memory_store()
        yield
        logger.info("Останавливаемся.")

    app = FastAPI(
        title="CdekStart RAG Agent",
        description=(
            "Контекстный RAG-чат-бот по правилам международной "
            "стажировки CdekStart на LangGraph + FastAPI."
        ),
        version=__version__,
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse, tags=["service"])
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=__version__,
            indexed_docs=getattr(app.state, "indexed_docs", 0),
        )

    @app.post("/chat", response_model=ChatResponse, tags=["chat"])
    async def chat(request: ChatRequest) -> ChatResponse:
        """Ответить на сообщение пользователя в рамках сессии."""
        graph_obj = getattr(app.state, "graph", None)
        if graph_obj is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Агент не инициализирован. Проверьте логи и переменные "
                    "окружения LLM-провайдера."
                ),
            )

        memory_store: SessionMemory = app.state.memory
        session = memory_store.get(request.session_id)

        new_messages = session.messages + [HumanMessage(content=request.message)]
        initial_state = {
            "messages": new_messages,
            "country": session.country,
        }

        try:
            final_state = graph_obj.invoke(initial_state)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ошибка в графе: {}", exc)
            raise HTTPException(
                status_code=500,
                detail="Внутренняя ошибка агента. Попробуйте позже.",
            ) from exc

        memory_store.update(
            request.session_id,
            messages=final_state.get("messages", new_messages),
            country=final_state.get("country"),
        )

        sources = [
            SourceRef(source=c.source, score=c.score)
            for c in final_state.get("retrieved", [])
        ]
        return ChatResponse(
            session_id=request.session_id,
            answer=final_state.get("answer", ""),
            type=final_state.get("answer_type", "answer"),
            sources=sources,
        )

    @app.delete("/chat/{session_id}", tags=["chat"])
    async def reset_session(session_id: str) -> dict[str, str]:
        """Сбросить сессию (на случай, если хотим начать диалог заново)."""
        app.state.memory.reset(session_id)
        return {"status": "reset", "session_id": session_id}

    return app


# Production-точка входа: ``uvicorn app.main:app``
app = create_app()
