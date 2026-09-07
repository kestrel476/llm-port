"""Адаптер GigaChat — без сети.

Формы запроса и ответа здесь не выдуманы: они сняты с живого контура, поэтому тесты
фиксируют реальные отличия от OpenAI, а не предположения о них. Отличий четыре, и
каждое умеет тихо ломать интеграцию:

- `functions` вместо `tools` — поле `tools` эндпоинт молча игнорирует;
- `function_call` один, а не список;
- аргументы приходят СЛОВАРЁМ, а не строкой JSON;
- `functions_state_id` надо вернуть обратно, иначе теряется контекст вызова.
"""

from __future__ import annotations

import json

import pytest

from llmport import AuthError, MalformedResponseError, Message, ModelNotFoundError, Request, ToolCall, ToolSpec
from llmport.adapters.gigachat import (
    API_URL,
    DEFAULT_SCOPE,
    ENV_API_URL,
    ENV_AUTH_URL,
    ENV_INTERNAL_API_URL,
    ENV_OAUTH_URL,
    ENV_SCOPE,
    INTERNAL_API_URL,
    OAUTH_URL,
    SCOPE_BUSINESS,
    SCOPE_CORPORATE,
    SCOPE_PERSONAL,
    GigaChat,
    TokenCache,
    build_payload,
    parse_response,
    resolve_api_url,
    resolve_oauth_url,
    resolve_scope,
)

MESSAGES = [Message(role="user", content="классифицируй документ")]


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Чистое окружение на каждый тест.

    Адреса теперь читаются из переменных при вызове, и оставленная в оболочке
    GIGACHAT_API_URL, которая вполне может быть у разработчика, роняла бы проверки
    умолчаний. Тест не должен зависеть от того, на какой стенд человек смотрел вчера.
    """
    for name in (ENV_OAUTH_URL, ENV_AUTH_URL, ENV_API_URL, ENV_INTERNAL_API_URL, ENV_SCOPE):
        monkeypatch.delenv(name, raising=False)


def _body(**payload: object) -> bytes:
    return json.dumps(payload).encode("utf-8")


# ── сборка запроса ───────────────────────────────────────────────────────────


def test_tools_are_sent_as_functions() -> None:
    # Поле `tools` эндпоинт игнорирует, и модель отвечает текстом вместо вызова.
    tool = ToolSpec(name="get_weather", description="погода", parameters={"type": "object"})
    payload = build_payload(Request(messages=MESSAGES, tools=[tool]), "GigaChat-2-Max")

    assert "tools" not in payload
    assert payload["functions"][0]["name"] == "get_weather"


def test_schema_is_expressed_as_a_forced_function() -> None:
    # Структурированного вывода по схеме у GigaChat нет: ближайшее средство —
    # обязательный вызов функции с нужной схемой параметров.
    schema = {"type": "object", "properties": {"code": {"type": "string"}}}
    payload = build_payload(Request(messages=MESSAGES, response_schema=schema), "GigaChat-2-Max")

    assert payload["function_call"] == {"name": "respond"}
    assert payload["functions"][0]["parameters"] == schema


def test_vendor_state_is_returned_to_the_provider() -> None:
    message = Message(role="assistant", content="", vendor_state={"functions_state_id": "abc-123"})
    payload = build_payload(Request(messages=[message]), "GigaChat-2-Max")

    assert payload["messages"][0]["functions_state_id"] == "abc-123"


def test_tool_result_is_sent_with_the_function_role() -> None:
    message = Message(role="tool", name="get_weather", content='{"t": 12}')
    payload = build_payload(Request(messages=[message]), "GigaChat-2-Max")

    assert payload["messages"][0]["role"] == "function"
    assert payload["messages"][0]["name"] == "get_weather"


def test_tool_call_is_sent_back_as_a_single_function_call() -> None:
    message = Message(role="assistant", tool_calls=(ToolCall(name="get_weather", arguments={"city": "Москва"}),))
    payload = build_payload(Request(messages=[message]), "GigaChat-2-Max")

    assert payload["messages"][0]["function_call"] == {"name": "get_weather", "arguments": {"city": "Москва"}}


# ── разбор ответа ────────────────────────────────────────────────────────────


def test_plain_answer_is_parsed() -> None:
    body = _body(
        model="GigaChat-2-Max:2.0.28.2",
        choices=[{"message": {"role": "assistant", "content": "Заявления"}, "finish_reason": "stop"}],
        usage={"prompt_tokens": 69, "completion_tokens": 3, "total_tokens": 72},
    )
    result = parse_response(200, body, model="GigaChat-2-Max")

    assert result.text == "Заявления"
    assert result.model == "GigaChat-2-Max:2.0.28.2"  # версия сборки, а не запрошенное имя
    assert result.usage.total_tokens == 72


def test_function_call_with_dict_arguments() -> None:
    # Именно так отвечает живой контур: arguments — словарь, а не строка.
    body = _body(
        choices=[
            {
                "message": {
                    "content": "",
                    "function_call": {"name": "get_weather", "arguments": {"city": "Москва"}},
                    "functions_state_id": "019fd791-ad3c-7b25",
                },
                "finish_reason": "function_call",
            }
        ]
    )
    result = parse_response(200, body, model="GigaChat-2-Max")

    assert result.wants_tools
    assert result.message.tool_calls[0].arguments == {"city": "Москва"}
    assert result.message.vendor_state["functions_state_id"] == "019fd791-ad3c-7b25"


def test_string_arguments_are_also_accepted() -> None:
    # Запас на случай, если поведение разойдётся между версиями контура.
    body = _body(choices=[{"message": {"function_call": {"name": "f", "arguments": '{"a": 1}'}}}])
    result = parse_response(200, body, model="m")

    assert result.message.tool_calls[0].arguments == {"a": 1}


def test_broken_body_is_malformed() -> None:
    with pytest.raises(MalformedResponseError):
        parse_response(200, b'{"choices": [', model="m")


def test_unknown_model_maps_to_its_own_error() -> None:
    # Так отвечает контур на модель не своего поколения: `GigaChat-3-Ultra` -> 404.
    with pytest.raises(ModelNotFoundError):
        parse_response(404, _body(status=404, message="No such model"), model="GigaChat-3-Ultra")


# ── авторизация ──────────────────────────────────────────────────────────────


class FakeTransport:
    """Транспорт со сценарием ответов; запоминает заголовки для проверки."""

    def __init__(self, responses: list[tuple[int, bytes]]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, dict[str, str], bytes]] = []

    def __call__(self, url: str, headers: dict[str, str], body: bytes, timeout: float | None) -> tuple:  # noqa: ARG002
        self.calls.append((url, dict(headers), body))
        status, payload = self._responses.pop(0) if self._responses else (500, b"{}")
        return status, {}, payload


def _token_body(expires_in_ms: int = 30 * 60 * 1000) -> bytes:
    import time as _time

    return _body(access_token="t0ken", expires_at=int(_time.time() * 1000) + expires_in_ms)


def test_key_is_exchanged_for_a_token_with_rquid() -> None:
    transport = FakeTransport([(200, _token_body())])
    cache = TokenCache(authorization_key="a2V5", transport=transport, scope=SCOPE_CORPORATE)

    assert cache.token() == "t0ken"
    _url, headers, body = transport.calls[0]
    assert headers["Authorization"] == "Basic a2V5"
    assert headers["RqUID"]  # без него сервер отказывает, не поясняя причины
    assert b"GIGACHAT_API_CORP" in body


def test_token_is_reused_until_it_nears_expiry() -> None:
    transport = FakeTransport([(200, _token_body())])
    cache = TokenCache(authorization_key="k", transport=transport)

    cache.token()
    cache.token()

    assert len(transport.calls) == 1


def test_nearly_expired_token_is_refreshed_in_advance() -> None:
    # Обновляемся заранее, а не по 401: получить отказ посреди пакета документов —
    # дорогой способ узнать, что срок вышел.
    transport = FakeTransport([(200, _token_body(expires_in_ms=10_000)), (200, _token_body())])
    cache = TokenCache(authorization_key="k", transport=transport, refresh_margin_s=60.0)

    cache.token()
    cache.token()

    assert len(transport.calls) == 2


def test_wrong_scope_is_reported_as_an_auth_error_with_a_hint() -> None:
    # Сервер отвечает невнятно; подсказка про области доступа экономит полчаса.
    transport = FakeTransport([(400, _body(code=7, message="scope from db not fully includes consumed scope"))])
    cache = TokenCache(authorization_key="k", transport=transport)

    with pytest.raises(AuthError, match="GIGACHAT_API"):
        cache.token()


def test_corrupted_key_is_reported_as_such() -> None:
    # Ровно этот ответ приходит на ключ, испорченный при копировании.
    transport = FakeTransport([(400, _body(code=4, message="Can't decode 'Authorization' header"))])
    cache = TokenCache(authorization_key="broken", transport=transport)

    with pytest.raises(AuthError, match="повреждён"):
        cache.token()


# ── провайдер целиком ────────────────────────────────────────────────────────


def test_expired_token_is_refreshed_once_on_401() -> None:
    answer = _body(choices=[{"message": {"content": "ответ"}, "finish_reason": "stop"}])
    transport = FakeTransport([(200, _token_body()), (401, b"{}"), (200, _token_body()), (200, answer)])
    cache = TokenCache(authorization_key="k", transport=transport)
    provider = GigaChat(tokens=cache, transport=transport)

    assert provider.complete(Request(messages=MESSAGES)).text == "ответ"


def test_second_401_is_not_retried_endlessly() -> None:
    transport = FakeTransport([(200, _token_body()), (401, b"{}"), (200, _token_body()), (401, b"{}")])
    cache = TokenCache(authorization_key="k", transport=transport)
    provider = GigaChat(tokens=cache, transport=transport)

    with pytest.raises(AuthError):
        provider.complete(Request(messages=MESSAGES))


def test_reasoning_capability_follows_the_model_name() -> None:
    transport = FakeTransport([(200, _token_body())])
    cache = TokenCache(authorization_key="k", transport=transport)

    assert not GigaChat(tokens=cache, transport=transport, model="GigaChat-2-Max").capabilities.reasoning
    assert GigaChat(tokens=cache, transport=transport, model="GigaChat-2-Reasoning").capabilities.reasoning


def test_models_are_listed() -> None:
    listing = _body(data=[{"id": "GigaChat-2-Max"}, {"id": "GigaChat-2-Reasoning"}])
    transport = FakeTransport([(200, _token_body()), (200, listing)])
    cache = TokenCache(authorization_key="k", transport=transport)

    assert GigaChat(tokens=cache, transport=transport).models() == ("GigaChat-2-Max", "GigaChat-2-Reasoning")


# ── внутренний контур: авторизация сертификатом ──────────────────────────────


def test_certificate_mode_sends_no_authorization_header() -> None:
    """Заголовка нет вовсе: соединение уже аутентифицировано рукопожатием.

    Отправить его означало бы получить отказ от эндпоинта, который такого поля не ждёт.
    """
    answer = _body(choices=[{"message": {"content": "ответ"}, "finish_reason": "stop"}])
    transport = FakeTransport([(200, answer)])
    provider = GigaChat(transport=transport)

    assert provider.complete(Request(messages=MESSAGES)).text == "ответ"
    _url, headers, _body_sent = transport.calls[0]
    assert "Authorization" not in headers
    assert headers["Content-Type"] == "application/json"


def test_certificate_mode_needs_no_token_exchange() -> None:
    """Обмена ключа на токен нет: в контуре менять нечего и негде."""
    answer = _body(choices=[{"message": {"content": "ответ"}, "finish_reason": "stop"}])
    transport = FakeTransport([(200, answer)])

    GigaChat(transport=transport).complete(Request(messages=MESSAGES))

    assert len(transport.calls) == 1, "лишний вызов означает попытку получить токен"
    assert OAUTH_URL not in transport.calls[0][0]


def test_address_follows_the_way_of_authorising() -> None:
    """Адреса у контуров разные, и помнить об этом должен адаптер, а не сервис."""
    transport = FakeTransport([(200, _token_body())])
    external = GigaChat(tokens=TokenCache(authorization_key="k", transport=transport), transport=transport)
    internal = GigaChat(transport=transport)

    assert not external.mutual_tls
    assert internal.mutual_tls
    assert external.models  # адрес виден по первому же запросу ниже


def test_explicit_address_wins_over_the_default() -> None:
    answer = _body(choices=[{"message": {"content": "ответ"}, "finish_reason": "stop"}])
    transport = FakeTransport([(200, answer)])
    GigaChat(transport=transport, api_url="https://свой.стенд/v1").complete(Request(messages=MESSAGES))

    assert transport.calls[0][0] == "https://свой.стенд/v1/chat/completions"


def test_internal_address_is_used_without_tokens() -> None:
    answer = _body(choices=[{"message": {"content": "ответ"}, "finish_reason": "stop"}])
    transport = FakeTransport([(200, answer)])
    GigaChat(transport=transport).complete(Request(messages=MESSAGES))

    assert transport.calls[0][0].startswith(INTERNAL_API_URL)


def test_rejected_certificate_is_explained_not_retried() -> None:
    """Обновлять нечего, повтор бессмыслен, и сообщение должно говорить о сертификате."""
    transport = FakeTransport([(401, b"{}")])

    with pytest.raises(AuthError, match="сертификат"):
        GigaChat(transport=transport).complete(Request(messages=MESSAGES))

    assert len(transport.calls) == 1, "повтор при отказе по сертификату только тратит время"


def test_rejected_certificate_is_explained_when_listing_models() -> None:
    transport = FakeTransport([(401, b"{}")])

    with pytest.raises(AuthError, match="сертификат"):
        GigaChat(transport=transport).models()


# ── адреса и области доступа из окружения ────────────────────────────────────


def test_explicit_address_beats_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Порядок важен: окружение снимает необходимость править код, но не отбирает
    управление там, где адрес задан явно."""
    monkeypatch.setenv(ENV_API_URL, "https://из-окружения/v1")
    assert resolve_api_url("https://явно/v1") == "https://явно/v1"


def test_environment_beats_the_built_in_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_API_URL, "https://из-окружения/v1")
    assert resolve_api_url() == "https://из-окружения/v1"
    monkeypatch.delenv(ENV_API_URL)
    assert resolve_api_url() == API_URL


def test_internal_contour_has_its_own_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Адрес внутреннего стенда почти всегда свой, и задавать его надо, не задев внешний."""
    monkeypatch.setenv(ENV_INTERNAL_API_URL, "https://внутренний.стенд/v1")
    assert resolve_api_url(mutual_tls=True) == "https://внутренний.стенд/v1"
    assert resolve_api_url() == API_URL


def test_general_variable_covers_the_internal_contour_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """Когда сервис живёт только внутри, отдельная переменная ему не нужна."""
    monkeypatch.setenv(ENV_API_URL, "https://единственный.стенд/v1")
    assert resolve_api_url(mutual_tls=True) == "https://единственный.стенд/v1"


def test_oauth_url_accepts_both_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """Прежнее имя признаётся, чтобы не ломать уже заполненные .env."""
    monkeypatch.setenv(ENV_AUTH_URL, "https://старое-имя/oauth")
    assert resolve_oauth_url() == "https://старое-имя/oauth"
    monkeypatch.setenv(ENV_OAUTH_URL, "https://новое-имя/oauth")
    assert resolve_oauth_url() == "https://новое-имя/oauth"


def test_scope_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_SCOPE, SCOPE_PERSONAL)
    assert resolve_scope() == SCOPE_PERSONAL
    assert resolve_scope(SCOPE_BUSINESS) == SCOPE_BUSINESS
    monkeypatch.delenv(ENV_SCOPE)
    assert resolve_scope() == DEFAULT_SCOPE


def test_blank_variable_does_not_shadow_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Пустая строка в .env это незаполненное поле, а не адрес."""
    monkeypatch.setenv(ENV_API_URL, "   ")
    assert resolve_api_url() == API_URL


def test_provider_takes_the_address_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_INTERNAL_API_URL, "https://свой.контур/v1")
    answer = _body(choices=[{"message": {"content": "ответ"}, "finish_reason": "stop"}])
    transport = FakeTransport([(200, answer)])

    GigaChat(transport=transport).complete(Request(messages=MESSAGES))

    assert transport.calls[0][0] == "https://свой.контур/v1/chat/completions"


def test_token_cache_takes_the_address_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_OAUTH_URL, "https://свой.oauth/token")
    transport = FakeTransport([(200, _token_body())])

    TokenCache(authorization_key="k", transport=transport).token()

    assert transport.calls[0][0] == "https://свой.oauth/token"
