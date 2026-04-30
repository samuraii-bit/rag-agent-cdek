"""Интеграционные тесты FastAPI-эндпоинта /chat."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.graph.builder import build_agent_graph
from app.graph.nodes import QueryAnalysis
from app.main import create_app
from app.memory.store import SessionMemory
from tests.conftest import FakeChatLLM, FakeKB


def _build_app_with_stubs(llm: FakeChatLLM, kb: FakeKB):
    """Собрать приложение, подменив тяжёлые зависимости стабами."""
    graph = build_agent_graph(llm=llm, kb=kb)
    memory = SessionMemory()
    return create_app(graph=graph, kb=kb, memory=memory)


def test_health(chunk_general) -> None:
    llm = FakeChatLLM()
    kb = FakeKB(chunks=[chunk_general])
    app = _build_app_with_stubs(llm, kb)
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["indexed_docs"] == 1


def test_chat_clarification_then_answer(chunk_germany, chunk_france) -> None:
    """E2E: первый ход — clarify, второй (после ответа про страну) — answer."""
    # На первом ходу LLM скажет: needs_country=True, country=none → clarify
    llm = FakeChatLLM(
        analysis=QueryAnalysis(needs_country=True, country="none"),
        answer="Уточните, пожалуйста: Германия или Франция?",
    )
    kb = FakeKB(chunks=[chunk_germany, chunk_france])
    app = _build_app_with_stubs(llm, kb)

    with TestClient(app) as client:
        r1 = client.post(
            "/chat",
            json={"session_id": "u1", "message": "Какая стипендия?"},
        )
        assert r1.status_code == 200
        b1 = r1.json()
        assert b1["type"] == "clarification"
        assert b1["sources"] == []

        # Подменяем поведение LLM на втором ходу: "Германия" указана.
        llm.analysis = QueryAnalysis(needs_country=True, country="germany")
        llm.answer = "Стипендия в Германии — 1200 евро в месяц."

        r2 = client.post(
            "/chat",
            json={"session_id": "u1", "message": "Германия"},
        )
        assert r2.status_code == 200
        b2 = r2.json()
        assert b2["type"] == "answer"
        assert "1200" in b2["answer"]
        assert any(s["source"] == "germany_rules.txt" for s in b2["sources"])


def test_chat_validation() -> None:
    """Pydantic должен ругаться на пустое сообщение."""
    llm = FakeChatLLM()
    kb = FakeKB()
    app = _build_app_with_stubs(llm, kb)
    with TestClient(app) as client:
        r = client.post("/chat", json={"session_id": "u1", "message": ""})
        assert r.status_code == 422


def test_session_isolation(chunk_germany, chunk_france) -> None:
    """Разные session_id не должны делить контекст."""
    llm = FakeChatLLM(
        analysis=QueryAnalysis(needs_country=True, country="none"),
        answer="Уточните страну.",
    )
    kb = FakeKB(chunks=[chunk_germany, chunk_france])
    app = _build_app_with_stubs(llm, kb)

    with TestClient(app) as client:
        # User A узнал про Германию
        llm.analysis = QueryAnalysis(needs_country=True, country="germany")
        llm.answer = "Германия: 1200 евро."
        client.post("/chat", json={"session_id": "A", "message": "Германия, стипендия?"})

        # User B спрашивает в общем — ему должен прийти clarify.
        llm.analysis = QueryAnalysis(needs_country=True, country="none")
        llm.answer = "Уточните страну."
        r = client.post(
            "/chat", json={"session_id": "B", "message": "А какая стипендия?"}
        )
        assert r.json()["type"] == "clarification"
