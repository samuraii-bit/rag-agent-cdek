"""Сборка LangGraph-агента."""
from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    make_analyze_node,
    make_clarify_node,
    make_generate_node,
    make_retrieve_node,
    route_after_analyze,
)
from app.graph.state import AgentState
from app.llm.factory import build_llm
from app.rag.retriever import KnowledgeBase, get_default_kb

if TYPE_CHECKING:  # pragma: no cover
    from langchain_core.language_models.chat_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph


def build_agent_graph(
    *,
    llm: "BaseChatModel | None" = None,
    kb: KnowledgeBase | None = None,
) -> "CompiledStateGraph":
    """Сборка и компиляция графа.

    Параметры можно подменить в тестах (например, фейковая LLM).
    """
    llm = llm if llm is not None else build_llm()
    kb = kb if kb is not None else get_default_kb()

    graph = StateGraph(AgentState)

    graph.add_node("analyze", make_analyze_node(llm))
    graph.add_node("clarify", make_clarify_node(llm))
    graph.add_node("retrieve", make_retrieve_node(kb))
    graph.add_node("generate", make_generate_node(llm))

    graph.add_edge(START, "analyze")
    graph.add_conditional_edges(
        "analyze",
        route_after_analyze,
        {
            "clarify": "clarify",
            "retrieve": "retrieve",
        },
    )
    graph.add_edge("retrieve", "generate")
    graph.add_edge("clarify", END)
    graph.add_edge("generate", END)

    return graph.compile()
