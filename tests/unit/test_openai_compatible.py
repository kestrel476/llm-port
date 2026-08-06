"""Сборка запроса и разбор ответа — без сети.

Обе функции чистые, и это главное: сегодня половина ошибок в замерах жила именно
в сборке запроса, а обнаруживалась по странным цифрам после двадцатиминутного
прогона. Здесь они ловятся за миллисекунды.
"""

from __future__ import annotations

import json

import pytest

from llmport import AuthError, BadRequestError, MalformedResponseError, Message, ModelNotFoundError, RateLimitedError, Request, ServerError, ToolSpec
from llmport.adapters import build_payload, parse_response

MESSAGES = [Message(role="system", content="ты классификатор"), Message(role="user", content="текст")]


def test_minimal_payload() -> None:
    payload = build_payload(Request(messages=MESSAGES), "z-ai/glm-5.2")

    assert payload["model"] == "z-ai/glm-5.2"
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]
    assert "temperature" not in payload  # не задано — не отправляем


def test_reasoning_goes_to_the_top_level_not_extra_body() -> None:
    # `extra_body` — понятие SDK OpenAI: он разворачивает его в корень запроса.
    # При прямом HTTP вложенный extra_body просто игнорируется, и reasoning остаётся
    # включённым. Сегодня на этом был потерян целый замер: модель тратила почти весь
    # бюджет вывода на размышления, а JSON обрывался.
    payload = build_payload(Request(messages=MESSAGES, reasoning=False), "gemini")

    assert payload["reasoning"] == {"enabled": False, "effort": "none"}
    assert "extra_body" not in payload


def test_reasoning_absent_when_not_requested() -> None:
    payload = build_payload(Request(messages=MESSAGES), "gemini")

    assert "reasoning" not in payload


def test_structured_output_schema() -> None:
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    payload = build_payload(Request(messages=MESSAGES, response_schema=schema), "model")

    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert payload["response_format"]["json_schema"]["schema"] == schema


def test_tools_are_wrapped_into_function_form() -> None:
    tool = ToolSpec(name="load_skill", description="загрузить навык", parameters={"type": "object"})
    payload = build_payload(Request(messages=MESSAGES, tools=[tool]), "model")

    assert payload["tools"][0]["type"] == "function"
    assert payload["tools"][0]["function"]["name"] == "load_skill"


def test_vendor_fields_are_passed_through() -> None:
    # Щель в абстракции оставлена намеренно: без неё сервисы начнут форкать адаптер
    # ради одного поля — ровно так и разошёлся прежний llmconnector.
    payload = build_payload(Request(messages=MESSAGES, extra={"service_tier": "priority"}), "model")

    assert payload["service_tier"] == "priority"


def _body(**payload: object) -> bytes:
    return json.dumps(payload).encode("utf-8")


def test_parses_plain_answer() -> None:
    body = _body(
        model="z-ai/glm-5.2",
        choices=[{"message": {"content": "договор"}, "finish_reason": "stop"}],
        usage={"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
    )
    result = parse_response(200, body, provider="routerai", model="запрошенная")

    assert result.text == "договор"
    assert result.model == "z-ai/glm-5.2"  # реально ответившая, а не запрошенная
    assert result.usage.total_tokens == 13
    assert not result.truncated


def test_reasoning_tokens_are_extracted() -> None:
    body = _body(
        choices=[{"message": {"content": "{}"}}],
        usage={"completion_tokens": 820, "completion_tokens_details": {"reasoning_tokens": 685}},
    )
    result = parse_response(200, body, provider="routerai", model="m")

    assert result.usage.reasoning_tokens == 685


def test_truncation_is_visible() -> None:
    body = _body(choices=[{"message": {"content": '{"suggestions": ['}, "finish_reason": "length"}])
    result = parse_response(200, body, provider="routerai", model="m")

    assert result.truncated


def test_tool_calls_are_parsed() -> None:
    body = _body(
        choices=[
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "load_skill", "arguments": '{"name": "k_plus"}'}}
                    ],
                }
            }
        ]
    )
    result = parse_response(200, body, provider="routerai", model="m")

    assert result.wants_tools
    assert result.message.tool_calls[0].name == "load_skill"
    assert result.message.tool_calls[0].arguments == {"name": "k_plus"}


def test_truncated_tool_arguments_do_not_lose_the_call() -> None:
    # Аргументы оборваны по лимиту вывода. Сам факт вызова инструмента полезнее, чем
    # исключение на месте разбора.
    body = _body(
        choices=[{"message": {"tool_calls": [{"function": {"name": "load_skill", "arguments": '{"na'}}]}}]
    )
    result = parse_response(200, body, provider="routerai", model="m")

    assert result.message.tool_calls[0].name == "load_skill"
    assert result.message.tool_calls[0].arguments == {}


def test_broken_body_is_malformed_not_success() -> None:
    # Статус 200, а тело оборвано: самая коварная ситуация, её нельзя принять за ответ.
    with pytest.raises(MalformedResponseError):
        parse_response(200, b'{"choices": [', provider="routerai", model="m")


def test_empty_choices_are_malformed() -> None:
    with pytest.raises(MalformedResponseError):
        parse_response(200, _body(choices=[]), provider="routerai", model="m")


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, AuthError),
        (403, AuthError),
        (404, ModelNotFoundError),
        (429, RateLimitedError),
        (400, BadRequestError),
        (500, ServerError),
        (503, ServerError),
    ],
)
def test_status_maps_to_error_class(status: int, expected: type[Exception]) -> None:
    body = _body(error={"message": "что-то пошло не так"})

    with pytest.raises(expected):
        parse_response(status, body, provider="routerai", model="m")


def test_error_message_is_preserved() -> None:
    body = _body(error={"message": "Reasoning is mandatory for this model"})

    with pytest.raises(BadRequestError, match="Reasoning is mandatory"):
        parse_response(400, body, provider="routerai", model="m")
