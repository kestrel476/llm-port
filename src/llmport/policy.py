"""Решения о повторах и переключении на запасного кандидата.

Вынесено в чистые функции нарочно. Циклы у синхронного и асинхронного провайдера
разные, а решения — одни и те же; если оставить решения внутри циклов, они разойдутся,
и проверять придётся оба. Здесь же логика тестируется без сети, без модели и без
asyncio, а циклы остаются тривиальными.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from llmport.errors import FatalError, LLMError, MalformedResponseError, RateLimitedError, TransientError


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Сколько раз и с какой паузой повторять запрос к ОДНОМУ кандидату."""

    attempts: int = 2
    """Общее число попыток, а не «повторов»: 1 означает «без повторов»."""

    base_delay_s: float = 0.5
    max_delay_s: float = 8.0
    jitter: float = 0.2
    """Доля случайного разброса: без него все клиенты возвращаются одновременно."""

    retry_malformed: bool = True
    """Повторять ли неразобранный ответ.

    По умолчанию да: самая частая причина — оборванное тело, и повтор помогает.
    Если формат несовместим в принципе, повтор просто потратит одну попытку.
    """

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError("attempts не может быть меньше 1")

    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Повторять ли после этой ошибки. `attempt` — номер уже сделанной попытки."""
        if attempt >= self.attempts:
            return False
        if isinstance(error, FatalError):
            return False
        if isinstance(error, MalformedResponseError):
            return self.retry_malformed
        return isinstance(error, TransientError)

    def delay_before(self, attempt: int, error: Exception | None = None) -> float:
        """Пауза перед попыткой номер `attempt` (нумерация с 1)."""
        if isinstance(error, RateLimitedError) and error.retry_after_s is not None:
            # Провайдер прямо сказал, сколько ждать, — гадать незачем.
            return max(0.0, error.retry_after_s)
        exponential = self.base_delay_s * (2 ** max(0, attempt - 1))
        capped = min(exponential, self.max_delay_s)
        spread = capped * self.jitter
        # random здесь не для криптографии, а для разведения одновременных повторов.
        return float(max(0.0, capped + random.uniform(-spread, spread)))


def should_failover(error: Exception) -> bool:
    """Идти ли к следующему кандидату после этой ошибки.

    На постоянных ошибках — не идти. Иначе неверный ключ, опечатка в имени модели
    или неподдерживаемое поле запроса выглядят как успех: запрос выполнит запасная
    модель, а поломанная конфигурация останется в проде незамеченной.
    """
    if isinstance(error, FatalError):
        return False
    return isinstance(error, LLMError)


def describe(error: Exception) -> str:
    """Короткое описание ошибки для события: тип и текст, без трассы."""
    return f"{type(error).__name__}: {error}"
