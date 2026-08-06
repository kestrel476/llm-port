"""Адаптер к OpenAI-совместимым эндпоинтам: RouterAI, OpenRouter, Cloud.ru.

Три провайдера в одном адаптере — не экономия, а честное отражение реальности: у них
один протокол и различаются только детали, которые вынесены в `extra` запроса
(`service_tier` у RouterAI, поля `reasoning` у OpenRouter) и в объявленные
возможности.

Сборка запроса и разбор ответа — ЧИСТЫЕ функции (`build_payload`, `parse_response`).
Это сделано намеренно: сегодня половина ошибок в замерах была именно в сборке
запроса, а не в модели, и обнаруживались они только по странным цифрам после
двадцатиминутного прогона. Чистые функции проверяются мгновенно и без сети.

Транспорт отделён от логики: адаптер принимает функцию, выполняющую HTTP. Библиотека
не навязывает ни httpx, ни aiohttp — сервисы уже живут с разными клиентами, и
притащить свой означало бы конфликт версий в четырнадцати репозиториях.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from http import HTTPStatus
from typing import Any

from llmport.contract import (
    Capabilities,
    Completion,
    Message,
    Request,
    ToolCall,
    Usage,
)
from llmport.errors import (
    AuthError,
    BadRequestError,
    MalformedResponseError,
    ModelNotFoundError,
    RateLimitedError,
    ServerError,
)

HttpResponse = tuple[int, Mapping[str, str], bytes]
"""Ответ транспорта: статус, заголовки, тело."""

SyncTransport = Callable[[str, Mapping[str, str], bytes, float | None], HttpResponse]


def build_payload(request: Request, model: str) -> dict[str, Any]:
    """Собирает тело запроса. Чистая функция — проверяется без сети."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": [_message_to_wire(m) for m in request.messages],
    }

    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.max_tokens is not None:
        payload["max_tokens"] = request.max_tokens

    if request.tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": dict(tool.parameters),
                },
            }
            for tool in request.tools
        ]

    if request.response_schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "strict": True,
                "schema": dict(request.response_schema),
            },
        }

    if request.reasoning is not None:
        # Поле верхнего уровня, а не extra_body: `extra_body` — понятие SDK OpenAI,
        # который разворачивает его в корень запроса. При прямом HTTP вложенный
        # extra_body просто игнорируется — на этом сегодня был потерян целый замер.
        payload["reasoning"] = {"enabled": request.reasoning}
        if not request.reasoning:
            payload["reasoning"]["effort"] = "none"

    # Вендорские поля идут последними и намеренно могут переопределить собранное:
    # без этой щели сервисы начнут форкать адаптер ради одного параметра.
    payload.update(dict(request.extra))
    return payload


def parse_response(status: int, body: bytes, *, provider: str, model: str) -> Completion:
    """Разбирает ответ. Ошибки классифицируются по коду и содержимому."""
    if status != HTTPStatus.OK:
        raise _error_for(status, body, provider=provider, model=model)

    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        # Самая коварная ситуация: статус 200, а тело оборвано на середине.
        raise MalformedResponseError(
            f"ответ не разобран как JSON: {exc}", provider=provider, model=model
        ) from exc

    choices = data.get("choices") or []
    if not choices:
        raise MalformedResponseError("в ответе нет ни одного варианта", provider=provider, model=model)

    choice = choices[0]
    raw_message = choice.get("message") or {}

    return Completion(
        message=Message(
            role="assistant",
            content=str(raw_message.get("content") or ""),
            tool_calls=_parse_tool_calls(raw_message.get("tool_calls")),
        ),
        model=str(data.get("model") or model),
        finish_reason=str(choice.get("finish_reason") or "stop"),
        usage=_parse_usage(data.get("usage")),
        raw=data,
    )


class OpenAICompatible:
    """Синхронный провайдер поверх переданного транспорта."""

    __slots__ = ("_capabilities", "_headers", "_model", "_provider", "_timeout_s", "_transport", "_url")

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        transport: SyncTransport,
        provider: str = "openai-compatible",
        capabilities: Capabilities | None = None,
        timeout_s: float = 60.0,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            **dict(extra_headers or {}),
        }
        self._model = model
        self._transport = transport
        self._provider = provider
        self._timeout_s = timeout_s
        self._capabilities = capabilities or Capabilities(
            tools=True,
            structured_output=True,
            streaming=True,
            reasoning=True,
            reasoning_can_be_disabled=True,
            vendor=provider,
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def capabilities(self) -> Capabilities:
        return self._capabilities

    def complete(self, request: Request) -> Completion:
        payload = build_payload(request, self._model)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        status, _headers, raw = self._transport(
            self._url, self._headers, body, request.timeout_s or self._timeout_s
        )
        return parse_response(status, raw, provider=self._provider, model=self._model)


# ── внутреннее ───────────────────────────────────────────────────────────────


def _message_to_wire(message: Message) -> dict[str, Any]:
    wire: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.name:
        wire["name"] = message.name
    if message.tool_call_id:
        wire["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        wire["tool_calls"] = [
            {
                "id": call.id or call.name,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(dict(call.arguments), ensure_ascii=False),
                },
            }
            for call in message.tool_calls
        ]
    return wire


def _parse_tool_calls(raw: Any) -> tuple[ToolCall, ...]:
    if not isinstance(raw, list):
        return ()
    calls: list[ToolCall] = []
    for item in raw:
        function = (item or {}).get("function") or {}
        name = function.get("name")
        if not name:
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                # Аргументы бывают оборваны по лимиту вывода. Терять из-за этого
                # сам факт вызова инструмента не стоит — имя уже полезно.
                arguments = {}
        calls.append(ToolCall(name=str(name), arguments=arguments or {}, id=item.get("id")))
    return tuple(calls)


def _parse_usage(raw: Any) -> Usage:
    if not isinstance(raw, dict):
        return Usage()
    details = raw.get("completion_tokens_details") or raw.get("output_tokens_details") or {}
    return Usage(
        prompt_tokens=int(raw.get("prompt_tokens") or raw.get("input_tokens") or 0),
        completion_tokens=int(raw.get("completion_tokens") or raw.get("output_tokens") or 0),
        reasoning_tokens=int((details or {}).get("reasoning_tokens") or 0),
        total_tokens=int(raw.get("total_tokens") or 0),
        cost=float(raw["cost"]) if raw.get("cost") is not None else None,
    )


def _error_for(status: int, body: bytes, *, provider: str, model: str) -> Exception:
    detail = _detail(body)
    if status in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN):
        return AuthError(detail or "ключ отвергнут", provider=provider, model=model)
    if status == HTTPStatus.NOT_FOUND:
        return ModelNotFoundError(detail or "модель не найдена", provider=provider, model=model)
    if status == HTTPStatus.TOO_MANY_REQUESTS:
        return RateLimitedError(detail or "превышен лимит запросов", provider=provider, model=model)
    if HTTPStatus.BAD_REQUEST <= status < HTTPStatus.INTERNAL_SERVER_ERROR:
        # Сюда попадает и «Reasoning is mandatory»: запрос несовместим с эндпоинтом,
        # и уводить его на запасную модель нельзя — несовместимость станет невидимой.
        return BadRequestError(f"{status}: {detail}", provider=provider, model=model)
    return ServerError(f"{status}: {detail}", provider=provider, model=model)


def _detail(body: bytes) -> str:
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return body[:200].decode("utf-8", "replace")
    error = data.get("error") if isinstance(data, dict) else None
    if isinstance(error, dict):
        return str(error.get("message") or error)
    return str(error or data)[:200]


def sequence_of(*providers: OpenAICompatible) -> Sequence[OpenAICompatible]:
    """Мелкая утилита для читаемой сборки списка кандидатов."""
    return providers
