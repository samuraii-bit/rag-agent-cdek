"""Провайдер-агностичная фабрика LLM.

Главный принцип: остальной код работает с интерфейсом ``BaseChatModel``
из ``langchain_core``. Чтобы переключить провайдера, достаточно поменять
переменную окружения ``LLM_PROVIDER`` — рестарт без правок кода.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from app.config import settings

if TYPE_CHECKING:  # pragma: no cover
    from langchain_core.language_models.chat_models import BaseChatModel


def build_llm(
    *,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
) -> "BaseChatModel":
    """Построить LLM по настройкам ``app.config.settings``.

    Параметры можно переопределить точечно (удобно в тестах).
    """
    provider = (provider or settings.llm_provider).lower()
    model = model or settings.llm_model
    temperature = (
        temperature if temperature is not None else settings.llm_temperature
    )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY не задан. Добавьте его в .env или переключите "
                "LLM_PROVIDER на 'anthropic' или 'ollama'."
            )
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=settings.openai_api_key,
            timeout=30,
            max_retries=2,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY не задан.")
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=settings.anthropic_api_key,
            timeout=30,
            max_retries=2,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=settings.ollama_base_url,
        )

    raise ValueError(
        f"Неизвестный LLM_PROVIDER: {provider!r}. "
        "Допустимые: openai, anthropic, ollama."
    )
