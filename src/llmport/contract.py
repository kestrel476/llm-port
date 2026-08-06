"""Контракт провайдера модели.

Это САМЫЙ СТАБИЛЬНЫЙ слой библиотеки: его импортируют все сервисы, и ломать его
нельзя. Всё, что склонно меняться — специфика вендоров, ретраи, фолбэк, метрики, —
живёт в адаптерах и поведениях, а не здесь.

Почему контракт такой маленький. Прошлая попытка вынести общий код (`llmconnector`
в lorium) не прижилась: её вынесли как пакет С РЕАЛИЗАЦИЕЙ, реализация не подошла
следующему сервису, его правили на месте — и копии разошлись на сотни строк.
Маленький контракт форкать незачем: с ним можно жить, даже если не подошло всё
остальное.

Протоколов два, синхронный и асинхронный. Сервисы LegalOS асинхронные, консольный
devassist синхронный, и притворяться, что одного достаточно, — значит заставить
половину потребителей крутить event loop вручную. Логика решений при этом общая
(см. `policy`), различаются только циклы.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Запрос модели на вызов инструмента."""

    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    id: str | None = None
    """Идентификатор вызова: нужен, чтобы связать результат с запросом."""


@dataclass(frozen=True, slots=True)
class Message:
    """Сообщение диалога в провайдеро-независимом виде."""

    role: Role
    content: str = ""
    name: str | None = None
    """Имя инструмента для role="tool"."""

    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    vendor_state: Mapping[str, str] = field(default_factory=dict)
    """Непрозрачное состояние вендора, которое нужно вернуть в следующем запросе.

    У GigaChat это `functions_state_id`. Хранить его в общем поле, а не в отдельном
    атрибуте под конкретного вендора, — единственный способ не тащить специфику
    в контракт.
    """


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Описание инструмента: имя, назначение и JSON-схема параметров."""

    name: str
    description: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Usage:
    """Расход токенов и стоимость, если провайдер их сообщает."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    """Отдельно от completion: у reasoning-моделей они съедают бюджет вывода.

    Именно на этом сегодня обрывался JSON — модель тратила 685 токенов из 900
    на размышления, и текста не оставалось.
    """

    total_tokens: int = 0
    cost: float | None = None

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cost=None if self.cost is None and other.cost is None else (self.cost or 0) + (other.cost or 0),
        )


@dataclass(frozen=True, slots=True)
class Completion:
    """Результат одного обращения к модели."""

    message: Message
    model: str
    """Модель, которая РЕАЛЬНО ответила.

    Не та, что запрошена: при фолбэке они расходятся, и без этого поля деградация
    качества происходит незаметно — сервис продолжает работать на запасной модели,
    а никто об этом не знает.
    """

    finish_reason: str = "stop"
    usage: Usage = field(default_factory=Usage)
    raw: Mapping[str, Any] = field(default_factory=dict)
    """Сырой ответ провайдера: для разбора нештатных ситуаций, не для логики."""

    @property
    def text(self) -> str:
        return self.message.content

    @property
    def wants_tools(self) -> bool:
        return bool(self.message.tool_calls)

    @property
    def truncated(self) -> bool:
        """Ответ оборван по лимиту вывода.

        Отдельное свойство, потому что это самая обидная ошибка: ответ формально
        получен, а JSON в нём недописан.
        """
        return self.finish_reason in {"length", "max_tokens"}


@dataclass(frozen=True, slots=True)
class Capabilities:
    """Что провайдер умеет.

    Объявляется явно, а не выясняется получением 400 из прода. Сегодня именно так
    обнаружилось, что часть эндпоинтов не даёт отключить reasoning: запрос уходил,
    сервис падал.
    """

    tools: bool = False
    structured_output: bool = False
    streaming: bool = False
    reasoning: bool = False
    reasoning_can_be_disabled: bool = False
    vendor: str = ""


@dataclass(frozen=True, slots=True)
class Request:
    """Запрос к модели.

    Собран в объект, а не рассыпан по аргументам: так его можно построить, проверить
    и залогировать, не выполняя. Половина ошибок в замерах сегодня была именно
    в сборке запроса, а не в модели.
    """

    messages: Sequence[Message]
    tools: Sequence[ToolSpec] = ()
    temperature: float | None = None
    max_tokens: int | None = None
    response_schema: Mapping[str, Any] | None = None
    """JSON-схема для структурированного вывода."""

    reasoning: bool | None = None
    """None — не трогать умолчание провайдера; False — просить отключить."""

    timeout_s: float | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)
    """Поля, специфичные для вендора (`service_tier` у RouterAI и подобное).

    Осознанная щель в абстракции: без неё сервисы начнут форкать адаптеры ради
    одного поля, и библиотека повторит судьбу llmconnector.
    """


@runtime_checkable
class LLMProvider(Protocol):
    """Синхронный провайдер."""

    @property
    def model(self) -> str: ...

    @property
    def capabilities(self) -> Capabilities: ...

    def complete(self, request: Request) -> Completion: ...


@runtime_checkable
class AsyncLLMProvider(Protocol):
    """Асинхронный провайдер."""

    @property
    def model(self) -> str: ...

    @property
    def capabilities(self) -> Capabilities: ...

    async def complete(self, request: Request) -> Completion: ...
