"""Адаптер GigaChat.

Отдельно от OpenAI-совместимого не из вредности: протокол отличается по существу,
и это проверено на живом контуре, а не взято из документации.

    OpenAI-совместимые            GigaChat
    ─────────────────────────────────────────────────────────
    tools: [{type, function}]     functions: [{name, ...}]
    tool_calls: [ ... ]           function_call: { ... }   (один, не список)
    arguments — строка JSON       arguments — УЖЕ словарь
    —                             functions_state_id, который надо вернуть обратно
    Authorization: Bearer <ключ>  ключ меняется на токен, токен живёт ~30 минут

Последнее — главное: авторизация двухшаговая. Ключ отправляется на отдельный хост
(`ngw.devices.sberbank.ru:9443`) вместе с заголовком `RqUID`, в ответ приходит токен
с меткой истечения. Токен кешируется и обновляется заранее, а не по факту получения
401: получить отказ на середине пакета документов — плохой способ узнать о сроке.

Транспорт передаётся снаружи, как и в соседнем адаптере. Здесь у этого есть отдельный
смысл: сертификаты Минцифры в системном хранилище обычно отсутствуют, и хосту нужно
задать корневой сертификат явно. Прятать `verify=False` внутрь библиотеки нельзя —
отключение проверки должно быть видимым решением сервиса.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import uuid
from http import HTTPStatus
from typing import Any

from llmport.adapters.openai_compatible import SyncTransport
from llmport.contract import Capabilities, Completion, Message, Request, ToolCall, Usage
from llmport.errors import (
    AuthError,
    BadRequestError,
    MalformedResponseError,
    ModelNotFoundError,
    RateLimitedError,
    ServerError,
)

OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
API_URL = "https://gigachat.devices.sberbank.ru/api/v1"

SCOPE_PERSONAL = "GIGACHAT_API_PERS"
SCOPE_BUSINESS = "GIGACHAT_API_B2B"
SCOPE_CORPORATE = "GIGACHAT_API_CORP"
"""Области доступа. Ключ обычно открыт ровно под одну; на чужой сервер отвечает
`scope from db not fully includes consumed scope`, а не внятной ошибкой доступа."""


def build_payload(request: Request, model: str) -> dict[str, Any]:
    """Собирает тело запроса в формате GigaChat. Чистая функция."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": [_message_to_wire(m) for m in request.messages],
    }
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.max_tokens is not None:
        payload["max_tokens"] = request.max_tokens

    if request.tools:
        # Именно `functions`, а не `tools`: поле `tools` эндпоинт молча игнорирует,
        # и модель отвечает текстом вместо вызова — отладка такого стоит часа.
        payload["functions"] = [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": dict(tool.parameters),
            }
            for tool in request.tools
        ]

    if request.response_schema is not None:
        # Структурированного вывода по схеме у GigaChat нет. Ближайшее средство —
        # функция с нужной схемой: модель обязана вызвать её и заполнить поля.
        payload["functions"] = [
            {
                "name": "respond",
                "description": "Вернуть ответ в требуемой структуре",
                "parameters": dict(request.response_schema),
            }
        ]
        payload["function_call"] = {"name": "respond"}

    payload.update(dict(request.extra))
    return payload


def parse_response(status: int, body: bytes, *, model: str) -> Completion:
    """Разбирает ответ GigaChat."""
    if status != HTTPStatus.OK:
        raise _error_for(status, body, model=model)

    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MalformedResponseError(
            f"ответ не разобран как JSON: {exc}", provider="gigachat", model=model
        ) from exc

    choices = data.get("choices") or []
    if not choices:
        raise MalformedResponseError("в ответе нет ни одного варианта", provider="gigachat", model=model)

    choice = choices[0]
    raw_message = choice.get("message") or {}
    state = raw_message.get("functions_state_id")

    return Completion(
        message=Message(
            role="assistant",
            content=str(raw_message.get("content") or ""),
            tool_calls=_parse_function_call(raw_message.get("function_call")),
            # Состояние функций возвращается провайдеру в следующем запросе, иначе
            # он теряет контекст вызова. Хранится в общем поле, чтобы специфика
            # вендора не протекала в контракт.
            vendor_state={"functions_state_id": str(state)} if state else {},
        ),
        model=str(data.get("model") or model),
        finish_reason=str(choice.get("finish_reason") or "stop"),
        usage=_parse_usage(data.get("usage")),
        raw=data,
    )


class TokenCache:
    """Обмен ключа на токен с обновлением заранее.

    Обновляемся не по 401, а по метке истечения с запасом: отказ посреди пакета
    документов — дорогой способ узнать, что срок вышел.
    """

    __slots__ = ("_authorization_key", "_expires_at", "_margin_s", "_scope", "_token", "_transport", "_url")

    def __init__(
        self,
        *,
        authorization_key: str,
        transport: SyncTransport,
        scope: str = SCOPE_CORPORATE,
        oauth_url: str = OAUTH_URL,
        refresh_margin_s: float = 60.0,
    ) -> None:
        self._authorization_key = authorization_key
        self._transport = transport
        self._scope = scope
        self._url = oauth_url
        self._margin_s = refresh_margin_s
        self._token: str | None = None
        self._expires_at: float = 0.0

    @property
    def valid(self) -> bool:
        return bool(self._token) and time.time() + self._margin_s < self._expires_at

    def token(self) -> str:
        if self.valid and self._token is not None:
            return self._token
        return self.refresh()

    def refresh(self) -> str:
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            # Уникальный идентификатор запроса обязателен: без него сервер отвечает
            # отказом, не поясняя причины.
            "RqUID": str(uuid.uuid4()),
            "Authorization": f"Basic {self._authorization_key}",
        }
        body = urllib.parse.urlencode({"scope": self._scope}).encode("utf-8")
        status, _headers, raw = self._transport(self._url, headers, body, 30.0)

        if status != HTTPStatus.OK:
            raise _oauth_error(status, raw)

        try:
            data = json.loads(raw.decode("utf-8"))
            token = str(data["access_token"])
            # `expires_at` приходит в миллисекундах.
            self._expires_at = float(data["expires_at"]) / 1000.0
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise MalformedResponseError(f"токен не разобран: {exc}", provider="gigachat") from exc

        self._token = token
        return token


class GigaChat:
    """Синхронный провайдер GigaChat поверх переданного транспорта."""

    __slots__ = ("_api_url", "_model", "_timeout_s", "_tokens", "_transport")

    def __init__(
        self,
        *,
        tokens: TokenCache,
        transport: SyncTransport,
        model: str = "GigaChat-2-Max",
        api_url: str = API_URL,
        timeout_s: float = 60.0,
    ) -> None:
        self._tokens = tokens
        self._transport = transport
        self._model = model
        self._api_url = api_url.rstrip("/")
        self._timeout_s = timeout_s

    @property
    def model(self) -> str:
        return self._model

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            tools=True,
            # Схему выполняет через обязательный вызов функции, а не отдельным
            # механизмом: снаружи это неотличимо, внутри — важное различие.
            structured_output=True,
            streaming=True,
            # Рассуждение есть только у отдельной модели, а не переключателем.
            reasoning="Reasoning" in self._model,
            reasoning_can_be_disabled=False,
            vendor="gigachat",
        )

    def complete(self, request: Request) -> Completion:
        payload = build_payload(request, self._model)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        url = f"{self._api_url}/chat/completions"

        status, _headers, raw = self._transport(
            url, self._headers(), body, request.timeout_s or self._timeout_s
        )
        if status == HTTPStatus.UNAUTHORIZED:
            # Срок мог выйти раньше расчётного — обновляемся и пробуем ещё раз.
            # Ровно одна попытка: если и она даёт 401, дело в ключе, а не в сроке.
            self._tokens.refresh()
            status, _headers, raw = self._transport(
                url, self._headers(), body, request.timeout_s or self._timeout_s
            )
        return parse_response(status, raw, model=self._model)

    def models(self) -> tuple[str, ...]:
        """Список доступных моделей.

        Полезен при настройке: состав зависит от области доступа ключа, и узнать
        его заранее дешевле, чем получить 404 в проде.
        """
        status, _headers, raw = self._transport(f"{self._api_url}/models", self._headers(), b"", 30.0)
        if status != HTTPStatus.OK:
            raise _error_for(status, raw, model=self._model)
        data = json.loads(raw.decode("utf-8"))
        return tuple(str(item["id"]) for item in data.get("data", []))

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self._tokens.token()}",
        }


# ── внутреннее ───────────────────────────────────────────────────────────────


def _message_to_wire(message: Message) -> dict[str, Any]:
    wire: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.role == "tool":
        # Результат инструмента возвращается ролью `function` и его именем.
        wire["role"] = "function"
        if message.name:
            wire["name"] = message.name
    if message.tool_calls:
        call = message.tool_calls[0]
        wire["function_call"] = {"name": call.name, "arguments": dict(call.arguments)}
    state = message.vendor_state.get("functions_state_id")
    if state:
        wire["functions_state_id"] = state
    return wire


def _parse_function_call(raw: Any) -> tuple[ToolCall, ...]:
    """Разбирает `function_call`.

    Аргументы приходят СЛОВАРЁМ, а не строкой JSON, как у OpenAI. Строка тоже
    обрабатывается — на случай, если поведение разойдётся между версиями.
    """
    if not isinstance(raw, dict) or not raw.get("name"):
        return ()
    arguments = raw.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    return (ToolCall(name=str(raw["name"]), arguments=arguments or {}),)


def _parse_usage(raw: Any) -> Usage:
    if not isinstance(raw, dict):
        return Usage()
    return Usage(
        prompt_tokens=int(raw.get("prompt_tokens") or 0),
        completion_tokens=int(raw.get("completion_tokens") or 0),
        total_tokens=int(raw.get("total_tokens") or 0),
    )


def _oauth_error(status: int, body: bytes) -> Exception:
    detail = _detail(body)
    if status == HTTPStatus.BAD_REQUEST and "scope" in detail.lower():
        # Частая и сбивающая с толку ошибка: ключ выдан под другую область.
        return AuthError(
            f"область доступа не подходит ключу ({detail}); "
            f"попробуйте {SCOPE_PERSONAL}, {SCOPE_BUSINESS} или {SCOPE_CORPORATE}",
            provider="gigachat",
        )
    if status == HTTPStatus.BAD_REQUEST and "decode" in detail.lower():
        # Так выглядит испорченный при копировании ключ: разделитель между
        # client_id и секретом декодируется в недопустимый байт.
        return AuthError(f"ключ авторизации повреждён ({detail})", provider="gigachat")
    if status in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN):
        return AuthError(detail or "ключ отвергнут", provider="gigachat")
    if status >= HTTPStatus.INTERNAL_SERVER_ERROR:
        return ServerError(f"{status}: {detail}", provider="gigachat")
    return AuthError(f"{status}: {detail}", provider="gigachat")


def _error_for(status: int, body: bytes, *, model: str) -> Exception:
    detail = _detail(body)
    if status in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN):
        return AuthError(detail or "токен отвергнут", provider="gigachat", model=model)
    if status == HTTPStatus.NOT_FOUND:
        return ModelNotFoundError(detail or "модель не найдена", provider="gigachat", model=model)
    if status == HTTPStatus.TOO_MANY_REQUESTS:
        return RateLimitedError(detail or "превышен лимит", provider="gigachat", model=model)
    if HTTPStatus.BAD_REQUEST <= status < HTTPStatus.INTERNAL_SERVER_ERROR:
        return BadRequestError(f"{status}: {detail}", provider="gigachat", model=model)
    return ServerError(f"{status}: {detail}", provider="gigachat", model=model)


def _detail(body: bytes) -> str:
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return body[:200].decode("utf-8", "replace")
    if isinstance(data, dict):
        return str(data.get("message") or data.get("error") or data)[:200]
    return str(data)[:200]
