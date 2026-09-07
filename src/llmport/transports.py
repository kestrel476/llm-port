"""Тип транспорта и необязательная реализация на стандартной библиотеке.

Библиотека по-прежнему НЕ навязывает HTTP-клиент: адаптеры принимают функцию, и сервис
волен передать свой httpx или aiohttp. Здесь лежат две вещи, которые иначе каждый сервис
пишет заново.

Первое: сам тип транспорта. Раньше он жил в адаптере OpenAI-совместимых, и адаптер
GigaChat импортировал его оттуда, хотя к OpenAI отношения не имеет. Теперь у него
собственное место, а прежний импорт продолжает работать.

Второе: сборка TLS-контекста с клиентским сертификатом. Это ровно тот случай, ради
которого затевается адаптер на границе: во внутренних контурах авторизация идёт
сертификатом, деталей у неё немного, но каждая из них умеет тихо ломать соединение, и
разбираться с ними в четырнадцати сервисах по отдельности не стоит.

Реализация транспорта на urllib нужна не всем, но во внутреннем контуре она снимает
вопрос: каждое колесо туда приходится проносить отдельно и согласовывать, а здесь
достаточно стандартной библиотеки.
"""

from __future__ import annotations

import socket
import ssl
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path

from llmport.errors import RequestTimeoutError, TransportError

HttpResponse = tuple[int, Mapping[str, str], bytes]
"""Ответ транспорта: статус, заголовки, тело."""

SyncTransport = Callable[[str, Mapping[str, str], bytes, float | None], HttpResponse]

DEFAULT_TIMEOUT_S = 120.0


def build_ssl_context(
    *,
    client_cert: Path | str | None = None,
    client_key: Path | str | None = None,
    ca_bundle: Path | str | None = None,
    verify: bool = True,
) -> ssl.SSLContext:
    """Собирает TLS-контекст, при необходимости с клиентским сертификатом.

    Клиентский сертификат и есть авторизация во внутреннем контуре, поэтому его
    отсутствие или нечитаемость это ошибка настройки, а не повод продолжить без него.
    Ошибки называют недостающее поимённо: разбираться с ними будут по ту сторону контура,
    где ни отладчика, ни возможности переспросить обычно нет.

    `verify=False` отключает проверку сертификата СЕРВЕРА. Во внутренних контурах это
    обычное дело, потому что своего удостоверяющего центра в системном хранилище нет, а
    соединение всё равно остаётся взаимно аутентифицированным клиентским сертификатом.
    Умолчание при этом строгое: отключение должно быть видимым решением сервиса.
    """
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

    if verify:
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        if ca_bundle is not None:
            root = Path(ca_bundle)
            if not root.is_file():
                raise FileNotFoundError(f"корневой сертификат не найден: {root}")
            context.load_verify_locations(cafile=str(root))
        else:
            context.load_default_certs(ssl.Purpose.SERVER_AUTH)
    else:
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE

    if client_cert is None and client_key is None:
        return context
    if client_cert is None or client_key is None:
        raise ValueError("для авторизации по сертификату нужны и сертификат, и ключ")

    certificate, private_key = Path(client_cert), Path(client_key)
    if not certificate.is_file():
        raise FileNotFoundError(f"клиентский сертификат не найден: {certificate}")
    if not private_key.is_file():
        raise FileNotFoundError(f"ключ клиентского сертификата не найден: {private_key}")
    try:
        context.load_cert_chain(certfile=str(certificate), keyfile=str(private_key))
    except ssl.SSLError as exc:
        raise ValueError(
            f"клиентский сертификат не загружен ({exc}); проверьте, что это PEM "
            f"и что ключ подходит к сертификату"
        ) from exc
    return context


def urllib_transport(
    context: ssl.SSLContext | None = None,
    *,
    default_timeout_s: float = DEFAULT_TIMEOUT_S,
) -> SyncTransport:
    """Транспорт на стандартной библиотеке.

    Пустое тело означает GET: адаптеры запрашивают список моделей именно так, и метод
    выводится из тела, чтобы не расширять контракт транспорта ради одного вызова.
    """
    handler = urllib.request.HTTPSHandler(context=context) if context else urllib.request.HTTPSHandler()
    opener = urllib.request.build_opener(handler)

    def transport(
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        timeout_s: float | None,
    ) -> HttpResponse:
        timeout = timeout_s or default_timeout_s
        request = urllib.request.Request(
            url,
            data=body or None,
            headers=dict(headers),
            method="POST" if body else "GET",
        )
        try:
            with opener.open(request, timeout=timeout) as response:
                return int(response.status), dict(response.headers), response.read()
        except urllib.error.HTTPError as exc:
            # Ответ с кодом ошибки это ответ, а не сбой связи: адаптер сам разберёт тело
            # и решит, временная это ошибка или постоянная.
            return int(exc.code), dict(exc.headers or {}), exc.read()
        except TimeoutError as exc:
            raise RequestTimeoutError(f"нет ответа за {timeout} с") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, socket.timeout | TimeoutError):
                raise RequestTimeoutError(f"нет ответа за {timeout} с") from exc
            # Обрыв рукопожатия в контуре с сертификатом почти всегда означает, что
            # сервер его отверг, а не что сеть недоступна. Текст сообщения при этом
            # будет про TLS, поэтому подсказка тут не лишняя.
            raise TransportError(
                f"соединение не установлено: {exc.reason}. Если контур авторизует "
                f"клиентским сертификатом, проверьте его срок и то, что он выдан на этот стенд"
            ) from exc
        except OSError as exc:
            raise TransportError(f"обрыв соединения: {exc}") from exc

    return transport
