"""Транспорт и TLS-контекст.

Проверяется то, что ломается при развёртывании в закрытом контуре, и ломается молча:
половина пары «сертификат и ключ», путь в никуда, файл не того формата. Разбираться с
этим будут по ту сторону контура, где нет ни отладчика, ни возможности переспросить,
поэтому ошибка обязана называть недостающее поимённо.

Настоящее соединение здесь не устанавливается: для этого есть пробник, который запускают
на месте.
"""

from __future__ import annotations

import ssl
import subprocess
from pathlib import Path

import pytest

from llmport.transports import build_ssl_context, urllib_transport


@pytest.fixture(scope="module")
def certificate(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Самоподписанная пара: проверяем загрузку, а не доверие к ней."""
    directory = tmp_path_factory.mktemp("tls")
    cert, key = directory / "client.pem", directory / "client.key"
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-keyout", str(key),
         "-out", str(cert), "-days", "1", "-nodes", "-subj", "/CN=test"],
        check=True, capture_output=True,
    )
    return cert, key


def test_default_context_verifies_the_server():
    """Умолчание строгое: отключение проверки должно быть видимым решением сервиса."""
    context = build_ssl_context()
    assert context.verify_mode is ssl.CERT_REQUIRED
    assert context.check_hostname is True


def test_verification_can_be_turned_off_for_a_closed_contour():
    context = build_ssl_context(verify=False)
    assert context.verify_mode is ssl.CERT_NONE
    assert context.check_hostname is False


def test_client_certificate_is_loaded(certificate: tuple[Path, Path]):
    """Загруженный клиентский сертификат из контекста не прочитать: проверкой служит то,
    что load_cert_chain не бросил исключение, а на непригодной паре бросает."""
    cert, key = certificate
    assert isinstance(build_ssl_context(client_cert=cert, client_key=key, verify=False), ssl.SSLContext)


def test_half_a_pair_is_refused():
    """Сертификат без ключа не авторизует ничего, и молчать об этом нельзя."""
    with pytest.raises(ValueError, match="и сертификат, и ключ"):
        build_ssl_context(client_cert="client.pem")
    with pytest.raises(ValueError, match="и сертификат, и ключ"):
        build_ssl_context(client_key="client.key")


def test_missing_certificate_is_named(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="клиентский сертификат не найден"):
        build_ssl_context(client_cert=tmp_path / "нет.pem", client_key=tmp_path / "нет.key", verify=False)


def test_missing_key_is_named(certificate: tuple[Path, Path], tmp_path: Path):
    cert, _key = certificate
    with pytest.raises(FileNotFoundError, match="ключ клиентского сертификата не найден"):
        build_ssl_context(client_cert=cert, client_key=tmp_path / "нет.key", verify=False)


def test_broken_certificate_is_reported_with_a_hint(tmp_path: Path):
    """Ошибка openssl про PEM lib ничего не объясняет сама по себе."""
    cert, key = tmp_path / "client.pem", tmp_path / "client.key"
    cert.write_text("не сертификат", encoding="utf-8")
    key.write_text("не ключ", encoding="utf-8")

    with pytest.raises(ValueError, match="PEM"):
        build_ssl_context(client_cert=cert, client_key=key, verify=False)


def test_missing_root_certificate_is_named(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="корневой сертификат не найден"):
        build_ssl_context(ca_bundle=tmp_path / "нет.pem")


def test_transport_is_callable_and_carries_the_context(certificate: tuple[Path, Path]):
    cert, key = certificate
    transport = urllib_transport(build_ssl_context(client_cert=cert, client_key=key, verify=False))
    assert callable(transport)
