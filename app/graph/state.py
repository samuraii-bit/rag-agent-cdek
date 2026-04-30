"""Состояние LangGraph-агента.

LangGraph хранит ``state`` как ``TypedDict``; узлы возвращают только
поля, которые меняют (а не весь стейт целиком).
"""
from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from app.rag.retriever import RetrievedChunk

AnswerType = Literal["answer", "clarification", "refusal"]


class AgentState(TypedDict, total=False):
    """Полный стейт агента.

    Поля
    ----
    messages
        История диалога (персистится между запросами в рамках сессии).
        ``add_messages`` корректно мерджит новые сообщения, не дублируя.
    query
        Последний запрос пользователя — извлекается в начале графа.
    country
        Известная страна: ``germany`` | ``france`` | ``None``.
        Может быть унаследована из предыдущих ходов.
    needs_country
        Нужен ли стране-зависимый контекст. Определяется LLM-узлом
        ``analyze``.
    retrieved
        Найденные чанки.
    answer
        Готовый ответ для пользователя.
    answer_type
        Тип ответа — попадает в JSON-ответ API:
        ``answer`` (нашли), ``clarification`` (просим уточнить),
        ``refusal`` (нет в базе).
    """

    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    country: str | None
    needs_country: bool
    retrieved: list[RetrievedChunk]
    answer: str
    answer_type: AnswerType
