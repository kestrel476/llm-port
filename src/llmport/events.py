"""События библиотеки.

Библиотека НЕ логирует и НЕ пишет метрики: она сообщает, что произошло, а решает
сервис. Иначе пакет притащил бы structlog и prometheus во все репозитории, которые
его подключат, — и каждый со своей версией.

События типизированы, а не строкой с текстом: по строке метрику не построить,
а разбирать её регулярками — та ещё радость.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from llmport.contract import Usage


@dataclass(frozen=True, slots=True)
class Attempted:
    """Обращение к кандидату завершено — успешно или нет."""

    role: str
    model: str
    candidate_index: int
    attempt: int
    duration_s: float
    ok: bool
    error: str | None = None
    usage: Usage = field(default_factory=Usage)


@dataclass(frozen=True, slots=True)
class Retrying:
    """Повтор к тому же кандидату после временной ошибки."""

    role: str
    model: str
    attempt: int
    delay_s: float
    error: str


@dataclass(frozen=True, slots=True)
class FailedOver:
    """Переключение на следующего кандидата.

    Событие важное: именно оно означает, что сервис работает НЕ на той модели,
    которую для него настроили. Без него деградация качества незаметна.
    """

    role: str
    from_model: str
    to_model: str
    candidate_index: int
    error: str


@dataclass(frozen=True, slots=True)
class GaveUp:
    """Кандидаты закончились."""

    role: str
    attempts: int
    error: str


Event = Attempted | Retrying | FailedOver | GaveUp
Listener = Callable[[Event], None]
"""Слушатель событий. Синхронный намеренно: он вызывается и из асинхронного цикла,
а заставлять сервис писать async-логгер ради строчки в лог — перебор."""


def ignore(_event: Event) -> None:
    """Слушатель по умолчанию: события никому не нужны, пока их не запросили."""
