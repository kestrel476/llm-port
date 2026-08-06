"""Устойчивость: повторы и переключение на запасного кандидата.

Это ДЕКОРАТОР над контрактом, а не часть клиента. Провайдер остаётся простым — он
умеет один запрос, — а устойчивость надевается сверху и снимается, если сервису
нужна своя. Сейчас такой код скопирован в трёх репозиториях (`native-orchestrator`,
`k-agent`, `risk_assessor`), причём копии успели разойтись.

Отличие от существующих копий одно, но существенное: они перехватывают любое
исключение и переключаются на запасную модель всегда. Здесь решение принимает
`policy`, и на постоянных ошибках (неверный ключ, неподдерживаемое поле, нет такой
модели) фолбэка НЕ происходит — иначе поломанная конфигурация маскируется рабочим
ответом запасной модели.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

from llmport.contract import (
    AsyncLLMProvider,
    Capabilities,
    Completion,
    LLMProvider,
    Request,
)
from llmport.errors import LLMError
from llmport.events import Attempted, FailedOver, GaveUp, Listener, Retrying, ignore
from llmport.policy import RetryPolicy, describe, should_failover


class _Base:
    """Общая часть: список кандидатов, имя роли, политика, слушатель."""

    __slots__ = ("_candidates", "_listener", "_retry", "_role")

    def __init__(
        self,
        candidates: Sequence[object],
        *,
        role: str,
        retry: RetryPolicy | None = None,
        listener: Listener = ignore,
    ) -> None:
        if not candidates:
            raise ValueError("нужен хотя бы один кандидат")
        self._candidates = tuple(candidates)
        self._role = role
        self._retry = retry or RetryPolicy()
        self._listener = listener

    @property
    def role(self) -> str:
        return self._role

    @property
    def model(self) -> str:
        """Модель основного кандидата.

        Какая ответила на самом деле — смотрят в `Completion.model`: при фолбэке
        они расходятся, и это тот случай, ради которого поле там и появилось.
        """
        return str(getattr(self._candidates[0], "model", ""))

    @property
    def capabilities(self) -> Capabilities:
        """Возможности основного кандидата.

        Возможности запасного могут отличаться, и это осознанный компромисс:
        обещать пересечение — значит занижать то, что сервис получит в обычном
        режиме работы.
        """
        caps = getattr(self._candidates[0], "capabilities", None)
        return caps if isinstance(caps, Capabilities) else Capabilities()


class Resilient(_Base):
    """Синхронная обёртка над несколькими провайдерами."""

    def complete(self, request: Request) -> Completion:
        last: Exception | None = None

        for index, candidate in enumerate(self._candidates):
            provider: LLMProvider = candidate  # type: ignore[assignment]
            if index > 0 and last is not None:
                self._listener(
                    FailedOver(
                        role=self._role,
                        from_model=str(getattr(self._candidates[index - 1], "model", "")),
                        to_model=provider.model,
                        candidate_index=index,
                        error=describe(last),
                    )
                )

            for attempt in range(1, self._retry.attempts + 1):
                started = time.monotonic()
                try:
                    result = provider.complete(request)
                except Exception as exc:  # noqa: BLE001 — классифицирует policy, не тип
                    last = exc
                    self._listener(
                        Attempted(
                            role=self._role,
                            model=provider.model,
                            candidate_index=index,
                            attempt=attempt,
                            duration_s=time.monotonic() - started,
                            ok=False,
                            error=describe(exc),
                        )
                    )
                    if self._retry.should_retry(exc, attempt):
                        delay = self._retry.delay_before(attempt, exc)
                        self._listener(
                            Retrying(
                                role=self._role,
                                model=provider.model,
                                attempt=attempt + 1,
                                delay_s=delay,
                                error=describe(exc),
                            )
                        )
                        time.sleep(delay)
                        continue
                    break
                else:
                    self._listener(
                        Attempted(
                            role=self._role,
                            model=result.model,
                            candidate_index=index,
                            attempt=attempt,
                            duration_s=time.monotonic() - started,
                            ok=True,
                            usage=result.usage,
                        )
                    )
                    return result

            if last is not None and not should_failover(last):
                # Постоянная ошибка: запасной кандидат её только замаскирует.
                raise last

        failure = last if last is not None else LLMError("не выполнено ни одной попытки")
        self._listener(GaveUp(role=self._role, attempts=len(self._candidates), error=describe(failure)))
        raise failure


class AsyncResilient(_Base):
    """Асинхронная обёртка. Решения те же, отличается только ожидание."""

    async def complete(self, request: Request) -> Completion:
        last: Exception | None = None

        for index, candidate in enumerate(self._candidates):
            provider: AsyncLLMProvider = candidate  # type: ignore[assignment]
            if index > 0 and last is not None:
                self._listener(
                    FailedOver(
                        role=self._role,
                        from_model=str(getattr(self._candidates[index - 1], "model", "")),
                        to_model=provider.model,
                        candidate_index=index,
                        error=describe(last),
                    )
                )

            for attempt in range(1, self._retry.attempts + 1):
                started = time.monotonic()
                try:
                    result = await provider.complete(request)
                except Exception as exc:  # noqa: BLE001 — классифицирует policy, не тип
                    last = exc
                    self._listener(
                        Attempted(
                            role=self._role,
                            model=provider.model,
                            candidate_index=index,
                            attempt=attempt,
                            duration_s=time.monotonic() - started,
                            ok=False,
                            error=describe(exc),
                        )
                    )
                    if self._retry.should_retry(exc, attempt):
                        delay = self._retry.delay_before(attempt, exc)
                        self._listener(
                            Retrying(
                                role=self._role,
                                model=provider.model,
                                attempt=attempt + 1,
                                delay_s=delay,
                                error=describe(exc),
                            )
                        )
                        await asyncio.sleep(delay)
                        continue
                    break
                else:
                    self._listener(
                        Attempted(
                            role=self._role,
                            model=result.model,
                            candidate_index=index,
                            attempt=attempt,
                            duration_s=time.monotonic() - started,
                            ok=True,
                            usage=result.usage,
                        )
                    )
                    return result

            if last is not None and not should_failover(last):
                raise last

        failure = last if last is not None else LLMError("не выполнено ни одной попытки")
        self._listener(GaveUp(role=self._role, attempts=len(self._candidates), error=describe(failure)))
        raise failure
