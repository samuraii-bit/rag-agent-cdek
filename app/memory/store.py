"""In-memory хранилище состояний сессий.

В продакшене заменили бы на Redis / Postgres. Для тестового
задания этого достаточно: каждый ``session_id`` сохраняет
``messages`` и последнюю известную ``country``.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import BaseMessage


@dataclass
class SessionState:
    messages: list[BaseMessage] = field(default_factory=list)
    country: str | None = None


class SessionMemory:
    """Потокобезопасный словарь сессий."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> SessionState:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionState()
            return self._sessions[session_id]

    def update(self, session_id: str, *, messages: list[BaseMessage], country: str | None) -> None:
        with self._lock:
            state = self._sessions.setdefault(session_id, SessionState())
            state.messages = messages
            state.country = country

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                sid: {
                    "messages": [m.type for m in s.messages],
                    "country": s.country,
                }
                for sid, s in self._sessions.items()
            }


_default_memory: SessionMemory | None = None


def get_memory_store() -> SessionMemory:
    global _default_memory
    if _default_memory is None:
        _default_memory = SessionMemory()
    return _default_memory
