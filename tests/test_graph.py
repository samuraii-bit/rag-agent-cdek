"""Тесты графа LangGraph.

Проверяем ключевые сценарии:

1. Если вопрос требует страны, а её нет — бот задаёт уточняющий вопрос.
2. Если страна указана — выполняются retrieve + generate.
3. Контекст диалога: страна, упомянутая ранее, наследуется.
4. Если retriever пуст — бот честно отказывается, не галлюцинируя.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from app.graph.builder import build_agent_graph
from app.graph.nodes import QueryAnalysis
from tests.conftest import FakeChatLLM, FakeKB


def test_clarification_when_country_missing(chunk_germany, chunk_france) -> None:
    """Запрос про стипендию без страны → бот спрашивает страну."""
    llm = FakeChatLLM(
        analysis=QueryAnalysis(needs_country=True, country="none"),
        answer="Уточните: Германия (Берлин) или Франция (Париж)?",
    )
    kb = FakeKB(chunks=[chunk_germany, chunk_france])
    graph = build_agent_graph(llm=llm, kb=kb)

    state = graph.invoke({"messages": [HumanMessage(content="Какая стипендия?")]})

    assert state["answer_type"] == "clarification"
    assert "Германия" in state["answer"] or "Франция" in state["answer"]
    # retrieve не должен был вызываться
    assert kb.last_query is None


def test_retrieve_when_country_known(chunk_germany, chunk_france) -> None:
    """Запрос с упомянутой страной → retrieve+generate."""
    llm = FakeChatLLM(
        analysis=QueryAnalysis(needs_country=True, country="germany"),
        answer="Стипендия в Германии — 1200 евро.",
    )
    kb = FakeKB(chunks=[chunk_germany, chunk_france])
    graph = build_agent_graph(llm=llm, kb=kb)

    state = graph.invoke(
        {"messages": [HumanMessage(content="Какая стипендия в Германии?")]}
    )

    assert state["answer_type"] == "answer"
    assert "1200" in state["answer"]
    assert kb.last_country == "germany"
    # В выдаче только Германия + общие (Франция отфильтрована FakeKB).
    countries = {c.country for c in state["retrieved"]}
    assert "france" not in countries


def test_country_inherited_from_history(chunk_germany, chunk_france) -> None:
    """Страна из предыдущей реплики переиспользуется на следующем ходу."""
    # На втором ходу LLM не находит страну в новом сообщении (country='none'),
    # но граф должен подтянуть её из state.country (memory сессии).
    llm = FakeChatLLM(
        analysis=QueryAnalysis(needs_country=True, country="none"),
        answer="Налог в Германии — 15%.",
    )
    kb = FakeKB(chunks=[chunk_germany, chunk_france])
    graph = build_agent_graph(llm=llm, kb=kb)

    state = graph.invoke(
        {
            "country": "germany",  # пришло из памяти
            "messages": [
                HumanMessage(content="Расскажи про Германию."),
                AIMessage(content="Что именно интересует?"),
                HumanMessage(content="А налог?"),
            ],
        }
    )

    assert state["answer_type"] == "answer"
    assert kb.last_country == "germany", (
        "Страна должна быть унаследована из контекста сессии"
    )


def test_refusal_when_no_chunks() -> None:
    """Пустой retrieval → бот не галлюцинирует, а отказывается."""
    llm = FakeChatLLM(
        analysis=QueryAnalysis(needs_country=False, country="none"),
        answer="(не должен дойти до LLM)",
    )
    kb = FakeKB(chunks=[])
    graph = build_agent_graph(llm=llm, kb=kb)

    state = graph.invoke(
        {"messages": [HumanMessage(content="Сколько весит самолёт?")]}
    )

    assert state["answer_type"] == "refusal"
    assert "нет ответа" in state["answer"].lower() or "уточните" in state["answer"].lower()


def test_general_query_no_clarification(chunk_general) -> None:
    """Общий вопрос (про дедлайны) не требует уточнения страны."""
    llm = FakeChatLLM(
        analysis=QueryAnalysis(needs_country=False, country="none"),
        answer="Дедлайн — 25 апреля.",
    )
    kb = FakeKB(chunks=[chunk_general])
    graph = build_agent_graph(llm=llm, kb=kb)

    state = graph.invoke(
        {"messages": [HumanMessage(content="Когда дедлайн подачи документов?")]}
    )

    assert state["answer_type"] == "answer"
    assert "25" in state["answer"]


def test_keyword_fallback_when_structured_output_fails(chunk_germany, chunk_france) -> None:
    """Если LLM упала на structured output (часто бывает у мелких локальных
    моделей), keyword-fallback должен сам определить, что нужна страна
    и продолжить clarify-сценарий."""

    class BrokenStructuredLLM(FakeChatLLM):
        """LLM, у которой structured output падает с ошибкой."""

        def with_structured_output(self, schema):  # noqa: ANN001
            from langchain_core.runnables import RunnableLambda

            def _broken(_msgs):
                raise RuntimeError("invalid JSON output")

            return RunnableLambda(_broken)

    llm = BrokenStructuredLLM(answer="Уточните: Германия или Франция?")
    kb = FakeKB(chunks=[chunk_germany, chunk_france])
    graph = build_agent_graph(llm=llm, kb=kb)

    # «стипендия» — стран-зависимое слово, но страна не указана → clarify
    state = graph.invoke({"messages": [HumanMessage(content="Какая стипендия?")]})

    assert state["answer_type"] == "clarification"


def test_keyword_fallback_detects_country(chunk_germany, chunk_france) -> None:
    """Keyword-fallback должен корректно вытащить страну из текста запроса."""

    class BrokenStructuredLLM(FakeChatLLM):
        def with_structured_output(self, schema):  # noqa: ANN001
            from langchain_core.runnables import RunnableLambda

            def _broken(_msgs):
                raise RuntimeError("invalid JSON output")

            return RunnableLambda(_broken)
            return RunnableLambda(_broken)

    llm = BrokenStructuredLLM(answer="In Germany the stipend is 1200 EUR.")
    kb = FakeKB(chunks=[chunk_germany, chunk_france])
    graph = build_agent_graph(llm=llm, kb=kb)

    state = graph.invoke(
        {"messages": [HumanMessage(content="What is the stipend in Germany?")]}
    )
    assert state["answer_type"] == "answer"
    countries = {c.country for c in state["retrieved"]}
    assert "france" not in countries
