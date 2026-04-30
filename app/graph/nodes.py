"""LangGraph nodes.

Each node is a pure function from ``state`` to a dict of updates.
Dependencies (LLM, retriever) are injected explicitly so we can test
the nodes without any real backends.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from pydantic import BaseModel, Field

from app.graph.state import AgentState
from app.rag.retriever import KnowledgeBase, RetrievedChunk

if TYPE_CHECKING:  # pragma: no cover
    from langchain_core.language_models.chat_models import BaseChatModel


# =============================================================================
# Prompts
# =============================================================================

ANALYZE_SYSTEM_PROMPT = """Ты — анализатор запросов чат-бота про международную стажировку CdekStart.

База знаний бота охватывает две страны: Германию (Берлин) и Францию (Париж).
Для некоторых вопросов ответ зависит от страны (стипендия, налог, виза,
рабочий день, локация). Для других вопросов страна не нужна (общая
информация о программе, дедлайны, льготы).

Твоя задача — проанализировать ПОСЛЕДНЕЕ сообщение пользователя с учётом
истории диалога и заполнить структуру:

1) needs_country: true — если ответ объективно зависит от страны.
   false — если вопрос общий.

2) country: страна, упомянутая в текущем сообщении ИЛИ ранее в диалоге.
   - "germany" (Германия, Берлин)
   - "france" (Франция, Париж)
   - "none", если страна не упомянута

Не выдумывай страны, которых нет в этих двух вариантах.
"""


ANSWER_SYSTEM_PROMPT = """Ты — ассистент программы стажировки CdekStart.
Отвечай СТРОГО на основе предоставленного контекста из базы знаний.

Жёсткие правила:
- НИКОГДА не выдумывай факты, цифры, даты, проценты.
- Если в контексте нет ответа — честно скажи: "В моей базе знаний нет
  ответа на этот вопрос. Уточните у организаторов программы."
- Если в контексте есть данные по нескольким странам, а пользователь
  явно спросил про одну — отвечай только по нужной стране.
- Отвечай кратко, по существу, на русском языке.
- Не добавляй ссылок на источники в текст ответа.
"""


CLARIFY_SYSTEM_PROMPT = """Ты — ассистент программы стажировки CdekStart.
Пользователь задал вопрос, для которого ответ зависит от страны
стажировки, но страна не указана.

Сформулируй ОДИН короткий и вежливый уточняющий вопрос, который
просит выбрать между Германией (Берлин) и Францией (Париж).
Никаких других стран. Никакого ответа по существу — только вопрос.
Отвечай на русском языке.
"""


# =============================================================================
# Structured-output schema for the analyze node
# =============================================================================


class QueryAnalysis(BaseModel):
    """Output of the analyze node."""

    needs_country: bool = Field(
        description="True if answering this question requires a country."
    )
    country: Literal["germany", "france", "none"] = Field(
        description="Country mentioned (germany / france) or 'none'."
    )


# =============================================================================
# Keyword fallback — used when an LLM cannot return valid structured output
# (typically small local models). Keeps the clarify branch reliable.
# =============================================================================


_GERMANY_KEYWORDS = ("герман", "берлин", "немец", "germany", "berlin")
_FRANCE_KEYWORDS = ("франц", "париж", "france", "paris")
_COUNTRY_SCOPED_KEYWORDS = (
    "стипенди", "налог", "виз", "рабочий день", "часы работы",
    "ставк", "локаци", "зарплат",
    "stipend", "tax", "visa", "salary", "working hour",
)


def _keyword_country(text: str) -> str:
    text = text.lower()
    if any(k in text for k in _GERMANY_KEYWORDS):
        return "germany"
    if any(k in text for k in _FRANCE_KEYWORDS):
        return "france"
    return "none"


def _keyword_needs_country(text: str) -> bool:
    text = text.lower()
    return any(k in text for k in _COUNTRY_SCOPED_KEYWORDS)


# =============================================================================
# Helpers
# =============================================================================


def _last_user_message(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


def _format_history(messages: list[BaseMessage], limit: int = 10) -> str:
    """Compress conversation history into a text block for the prompt."""
    recent = messages[-limit:]
    lines: list[str] = []
    for msg in recent:
        if isinstance(msg, HumanMessage):
            role = "Пользователь"
        elif isinstance(msg, AIMessage):
            role = "Ассистент"
        else:
            role = "Система"
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "(пусто)"


def _format_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(контекст пуст)"
    parts: list[str] = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[Документ {i} | source={c.source} | country={c.country}]\n{c.text}")
    return "\n\n".join(parts)


# =============================================================================
# Nodes
# =============================================================================


def make_analyze_node(llm: "BaseChatModel"):
    """Analyze node.

    Uses structured output. If the LLM fails to return valid JSON
    (common on small local models), falls back to deterministic
    keyword detection.
    """
    structured = llm.with_structured_output(QueryAnalysis)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ANALYZE_SYSTEM_PROMPT),
            (
                "human",
                "История диалога:\n{history}\n\n"
                "Последнее сообщение пользователя:\n{query}",
            ),
        ]
    )

    def analyze(state: AgentState) -> dict:
        messages = state.get("messages", [])
        query = _last_user_message(messages)
        history = _format_history(messages[:-1]) if len(messages) > 1 else "(нет)"

        previous_country = state.get("country")

        try:
            analysis: QueryAnalysis = structured.invoke(
                prompt.format_messages(history=history, query=query)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "analyze: structured output failed ({}), using keyword fallback",
                exc,
            )
            country_in_text = _keyword_country(query)
            analysis = QueryAnalysis(
                needs_country=_keyword_needs_country(query),
                country=country_in_text if country_in_text != "none" else "none",
            )

        # If no country in this turn, inherit from previous turns -- this
        # is the conversational-context part of the assignment.
        country: str | None
        if analysis.country == "none":
            country = previous_country
        else:
            country = analysis.country

        logger.info(
            "analyze: needs_country={}, country={}",
            analysis.needs_country,
            country,
        )
        return {
            "query": query,
            "country": country,
            "needs_country": analysis.needs_country,
        }

    return analyze


def make_clarify_node(llm: "BaseChatModel"):
    """Clarification node — asks for the country."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CLARIFY_SYSTEM_PROMPT),
            ("human", "Вопрос пользователя: {query}"),
        ]
    )

    def clarify(state: AgentState) -> dict:
        query = state.get("query") or _last_user_message(state.get("messages", []))
        try:
            response = llm.invoke(prompt.format_messages(query=query))
            text = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("clarify: LLM failed ({}), using template", exc)
            text = (
                "Уточните, пожалуйста, для какой локации интересен ответ — "
                "Германия (Берлин) или Франция (Париж)?"
            )
        return {
            "answer": text.strip(),
            "answer_type": "clarification",
            "messages": [AIMessage(content=text.strip())],
        }

    return clarify


def make_retrieve_node(kb: KnowledgeBase):
    """Retrieval node."""

    def retrieve(state: AgentState) -> dict:
        query = state.get("query") or _last_user_message(state.get("messages", []))
        country = state.get("country")
        chunks = kb.search(query, country=country)
        logger.info(
            "retrieve: query={!r} country={} -> {} chunks",
            query,
            country,
            len(chunks),
        )
        return {"retrieved": chunks}

    return retrieve


def make_generate_node(llm: "BaseChatModel"):
    """Generation node — answers strictly from the retrieved context."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ANSWER_SYSTEM_PROMPT),
            (
                "human",
                "Контекст из базы знаний:\n{context}\n\n"
                "История диалога:\n{history}\n\n"
                "Вопрос пользователя:\n{query}",
            ),
        ]
    )

    def generate(state: AgentState) -> dict:
        chunks: list[RetrievedChunk] = state.get("retrieved", [])
        query = state.get("query") or _last_user_message(state.get("messages", []))
        history = _format_history(state.get("messages", [])[:-1])

        if not chunks:
            text = (
                "В моей базе знаний нет ответа на этот вопрос. "
                "Уточните у организаторов программы."
            )
            return {
                "answer": text,
                "answer_type": "refusal",
                "messages": [AIMessage(content=text)],
            }

        try:
            response = llm.invoke(
                prompt.format_messages(
                    context=_format_context(chunks),
                    history=history,
                    query=query,
                )
            )
            text = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("generate: LLM failed: {}", exc)
            text = (
                "Возникла техническая ошибка при генерации ответа. "
                "Попробуйте повторить запрос позже."
            )
            return {
                "answer": text,
                "answer_type": "refusal",
                "messages": [AIMessage(content=text)],
            }

        text = text.strip()
        lowered = text.lower()
        is_refusal = (
            "нет ответа" in lowered
            or "не нашёл" in lowered
            or "не нашел" in lowered
            or "нет в базе" in lowered
            or "нет в моей базе" in lowered
        )
        return {
            "answer": text,
            "answer_type": "refusal" if is_refusal else "answer",
            "messages": [AIMessage(content=text)],
        }

    return generate


# =============================================================================
# Conditional edges
# =============================================================================


def route_after_analyze(state: AgentState) -> Literal["clarify", "retrieve"]:
    """If the question requires a country and we don't know it -> clarify."""
    if state.get("needs_country") and not state.get("country"):
        return "clarify"
    return "retrieve"
