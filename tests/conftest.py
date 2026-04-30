"""Common test fixtures."""
from __future__ import annotations

import os

# Set safe defaults BEFORE importing app.config
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("CHROMA_PERSIST_DIR", "/tmp/cdekstart-tests-chroma")
os.environ.setdefault("KB_DATA_DIR", "data")

from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableLambda

from app.graph.nodes import QueryAnalysis
from app.rag.retriever import RetrievedChunk


# =============================================================================
# Stub LLM and KB for fast, deterministic tests.
# =============================================================================


class FakeChatLLM:
    """Minimal stand-in for BaseChatModel.

    * ``analysis`` -- what ``with_structured_output(QueryAnalysis)`` returns.
    * ``answer``   -- text returned by ``invoke`` (clarify / generate nodes).

    Both fields are read at call time, so tests can mutate them between
    invocations to simulate multi-turn conversations.
    """

    def __init__(
        self,
        *,
        analysis: QueryAnalysis | None = None,
        answer: str = "(fake answer)",
    ) -> None:
        self.analysis = analysis or QueryAnalysis(
            needs_country=False, country="none"
        )
        self.answer = answer
        self.invocations: list[Any] = []

    def with_structured_output(self, schema):  # noqa: ANN001
        # Read self.analysis at every call -- tests may mutate it between turns.
        return RunnableLambda(lambda _msgs: self.analysis)

    def invoke(self, messages: list[BaseMessage], config=None, **_kwargs):  # noqa: ANN001
        self.invocations.append(messages)
        return AIMessage(content=self.answer)


class FakeKB:
    """Stand-in for KnowledgeBase."""

    def __init__(self, chunks: list[RetrievedChunk] | None = None) -> None:
        self.chunks = chunks or []
        self.last_query: str | None = None
        self.last_country: str | None = None

    def search(
        self,
        query: str,
        *,
        country: str | None = None,
        top_k: int | None = None,  # noqa: ARG002
    ) -> list[RetrievedChunk]:
        self.last_query = query
        self.last_country = country
        if country and country != "none":
            return [c for c in self.chunks if c.country in (country, "none")]
        return list(self.chunks)

    def ensure_indexed(self) -> int:
        return len(self.chunks)

    def count(self) -> int:
        return len(self.chunks)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def chunk_germany() -> RetrievedChunk:
    return RetrievedChunk(
        text="Germany rules. Stipend 1200 EUR. Tax 15%. Visa D.",
        source="germany_rules.txt",
        country="germany",
        topic="country_rules",
        score=0.92,
    )


@pytest.fixture
def chunk_france() -> RetrievedChunk:
    return RetrievedChunk(
        text="France rules. Stipend 1300 EUR. Tax 20%. Visa VLS-TS.",
        source="france_rules.txt",
        country="france",
        topic="country_rules",
        score=0.90,
    )


@pytest.fixture
def chunk_general() -> RetrievedChunk:
    return RetrievedChunk(
        text="Deadline 25 April. Results 1 May. Start 1 June.",
        source="deadlines.txt",
        country="none",
        topic="deadlines",
        score=0.85,
    )
