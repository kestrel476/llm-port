"""Адаптеры провайдеров. Независимы друг от друга: правка одного не трогает соседей."""

from llmport.adapters.openai_compatible import OpenAICompatible, build_payload, parse_response

__all__ = ["OpenAICompatible", "build_payload", "parse_response"]
