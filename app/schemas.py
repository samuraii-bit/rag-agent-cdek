"""Pydantic-схемы для API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Тело запроса POST /chat.

    ``session_id`` нужен для поддержки контекста: бот помнит предыдущие
    реплики (в т.ч. упомянутую страну) внутри одной сессии.
    """

    session_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Идентификатор диалога. Любая стабильная строка.",
        examples=["user-42"],
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Сообщение пользователя.",
        examples=["Какая стипендия?"],
    )


class SourceRef(BaseModel):
    """Ссылка на использованный документ из базы знаний."""

    source: str = Field(..., description="Имя файла-источника, например 'germany_rules.txt'.")
    score: float | None = Field(None, description="Сходство (если доступно).")


class ChatResponse(BaseModel):
    """Ответ /chat."""

    session_id: str
    answer: str = Field(..., description="Ответ ассистента.")
    type: Literal["answer", "clarification", "refusal"] = Field(
        ...,
        description=(
            "answer — ответ по базе; "
            "clarification — бот задаёт уточняющий вопрос; "
            "refusal — информации нет в базе."
        ),
    )
    sources: list[SourceRef] = Field(
        default_factory=list,
        description="Источники, на которые опирался ответ. Пусто для clarification.",
    )


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    indexed_docs: int
