"""Повторы и фолбэк.

Главное, что здесь проверяется, — исправление дефекта существующих копий. Они
перехватывают любое исключение и переключаются на запасную модель всегда, поэтому
неверный ключ и неподдерживаемое поле запроса выглядят для сервиса как успех: ответ
получен, просто не от той модели. Здесь на постоянных ошибках фолбэка не происходит.
"""

from __future__ import annotations

import asyncio

import pytest

from llmport import AuthError, BadRequestError, Capabilities, Completion, Message, Request, ServerError
from llmport.behaviours import AsyncResilient, Resilient
from llmport.events import Attempted, Event, FailedOver, Retrying
from llmport.policy import RetryPolicy

REQUEST = Request(messages=[Message(role="user", content="привет")])


class FakeProvider:
    """Провайдер со сценарием: список исходов по вызовам."""

    def __init__(self, name: str, outcomes: list[object]) -> None:
        self._name = name
        self._outcomes = list(outcomes)
        self.calls = 0

    @property
    def model(self) -> str:
        return self._name

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(vendor="fake")

    def _next(self) -> Completion:
        self.calls += 1
        outcome = self._outcomes.pop(0) if self._outcomes else Exception("сценарий исчерпан")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def complete(self, request: Request) -> Completion:  # noqa: ARG002
        return self._next()


class AsyncFakeProvider(FakeProvider):
    async def complete(self, request: Request) -> Completion:  # type: ignore[override]  # noqa: ARG002
        return self._next()


def _ok(model: str) -> Completion:
    return Completion(message=Message(role="assistant", content="ответ"), model=model)


def _instant() -> RetryPolicy:
    """Политика без пауз: тесты не должны спать."""
    return RetryPolicy(attempts=2, base_delay_s=0.0, jitter=0.0)


def test_successful_call_does_not_touch_the_spare() -> None:
    primary = FakeProvider("основная", [_ok("основная")])
    spare = FakeProvider("запасная", [_ok("запасная")])

    result = Resilient([primary, spare], role="test", retry=_instant()).complete(REQUEST)

    assert result.model == "основная"
    assert spare.calls == 0


def test_transient_error_is_retried_on_the_same_model() -> None:
    primary = FakeProvider("основная", [ServerError("503"), _ok("основная")])
    spare = FakeProvider("запасная", [_ok("запасная")])

    result = Resilient([primary, spare], role="test", retry=_instant()).complete(REQUEST)

    assert result.model == "основная"
    assert primary.calls == 2
    assert spare.calls == 0


def test_exhausted_retries_lead_to_the_spare() -> None:
    primary = FakeProvider("основная", [ServerError("503"), ServerError("503")])
    spare = FakeProvider("запасная", [_ok("запасная")])

    result = Resilient([primary, spare], role="test", retry=_instant()).complete(REQUEST)

    assert result.model == "запасная"
    assert primary.calls == 2


def test_auth_error_does_not_fall_over_to_the_spare() -> None:
    # Ключевой случай. Прежние копии ушли бы на запасную модель, сервис получил бы
    # ответ, и неверный ключ остался бы незамеченным.
    primary = FakeProvider("основная", [AuthError("ключ отвергнут")])
    spare = FakeProvider("запасная", [_ok("запасная")])

    with pytest.raises(AuthError):
        Resilient([primary, spare], role="test", retry=_instant()).complete(REQUEST)

    assert spare.calls == 0
    assert primary.calls == 1


def test_bad_request_does_not_fall_over_either() -> None:
    # «Reasoning is mandatory» — ровно этот случай: запрос несовместим с эндпоинтом.
    primary = FakeProvider("основная", [BadRequestError("400: reasoning is mandatory")])
    spare = FakeProvider("запасная", [_ok("запасная")])

    with pytest.raises(BadRequestError):
        Resilient([primary, spare], role="test", retry=_instant()).complete(REQUEST)

    assert spare.calls == 0


def test_fatal_error_is_not_retried() -> None:
    primary = FakeProvider("основная", [AuthError("ключ"), _ok("основная")])

    with pytest.raises(AuthError):
        Resilient([primary], role="test", retry=RetryPolicy(attempts=5, base_delay_s=0.0)).complete(REQUEST)

    assert primary.calls == 1


def test_failover_is_reported_as_an_event() -> None:
    # Без этого события деградация незаметна: сервис работает, но не на той модели,
    # для которой его настраивали.
    seen: list[Event] = []
    primary = FakeProvider("основная", [ServerError("503"), ServerError("503")])
    spare = FakeProvider("запасная", [_ok("запасная")])

    Resilient([primary, spare], role="skill_agent", retry=_instant(), listener=seen.append).complete(REQUEST)

    failovers = [e for e in seen if isinstance(e, FailedOver)]
    assert len(failovers) == 1
    assert failovers[0].from_model == "основная"
    assert failovers[0].to_model == "запасная"
    assert failovers[0].role == "skill_agent"


def test_every_attempt_is_reported() -> None:
    seen: list[Event] = []
    primary = FakeProvider("основная", [ServerError("503"), _ok("основная")])

    Resilient([primary], role="test", retry=_instant(), listener=seen.append).complete(REQUEST)

    attempts = [e for e in seen if isinstance(e, Attempted)]
    assert [a.ok for a in attempts] == [False, True]
    assert any(isinstance(e, Retrying) for e in seen)


def test_last_error_is_raised_when_all_candidates_fail() -> None:
    primary = FakeProvider("основная", [ServerError("503"), ServerError("503")])
    spare = FakeProvider("запасная", [ServerError("504"), ServerError("504")])

    with pytest.raises(ServerError, match="504"):
        Resilient([primary, spare], role="test", retry=_instant()).complete(REQUEST)


def test_empty_candidate_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="хотя бы один"):
        Resilient([], role="test")


def test_async_behaves_the_same_way() -> None:
    # Решения общие, различается только ожидание: поведение обязано совпадать.
    primary = AsyncFakeProvider("основная", [ServerError("503"), ServerError("503")])
    spare = AsyncFakeProvider("запасная", [_ok("запасная")])

    result = asyncio.run(AsyncResilient([primary, spare], role="test", retry=_instant()).complete(REQUEST))

    assert result.model == "запасная"


def test_async_also_refuses_to_mask_fatal_errors() -> None:
    primary = AsyncFakeProvider("основная", [AuthError("ключ")])
    spare = AsyncFakeProvider("запасная", [_ok("запасная")])

    with pytest.raises(AuthError):
        asyncio.run(AsyncResilient([primary, spare], role="test", retry=_instant()).complete(REQUEST))

    assert spare.calls == 0
