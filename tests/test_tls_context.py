"""The shared TLS context verifies peers whatever state the host trust store is in."""

from __future__ import annotations

import json
import re
import ssl
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import tls_context  # noqa: E402

certifi = pytest.importorskip("certifi")

PEM_BLOCK = re.compile(r"-----BEGIN CERTIFICATE-----.+?-----END CERTIFICATE-----", re.S)
GOOD_CERTIFICATES = 3
UNPARSABLE = b"\x30\x03\x02\x01\x00 definitely not a certificate"


def _bundle_der(count: int) -> list[bytes]:
    pem = Path(certifi.where()).read_text(encoding="ascii")
    return [ssl.PEM_cert_to_DER_cert(block) for block in PEM_BLOCK.findall(pem)[:count]]


def _assert_verifies(context: ssl.SSLContext) -> None:
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True


@pytest.fixture
def broken_system_store(monkeypatch):
    """The host trust store cannot be loaded, as on a Windows machine holding one unparsable certificate."""
    real = ssl.create_default_context

    def create_default_context(*args, **kwargs):
        if kwargs.get("cafile") is None and not args:
            raise ssl.SSLError("[ASN1] nested asn1 error (_ssl.c:4047)")
        return real(*args, **kwargs)

    monkeypatch.setattr(ssl, "create_default_context", create_default_context)
    monkeypatch.setitem(sys.modules, "certifi", None)


def test_certifi_bundle_is_preferred_and_verifies(monkeypatch):
    seen = {}
    real = ssl.create_default_context

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(ssl, "create_default_context", spy)
    _assert_verifies(tls_context.verified_context())
    assert seen == {"cafile": certifi.where()}


def test_platform_store_is_used_without_certifi(monkeypatch):
    calls = []
    real = ssl.create_default_context
    monkeypatch.setitem(sys.modules, "certifi", None)
    monkeypatch.setattr(ssl, "create_default_context", lambda *a, **k: calls.append((a, k)) or real(cafile=certifi.where()))
    _assert_verifies(tls_context.verified_context())
    assert calls == [((), {})]


def test_unloadable_store_falls_back_to_the_certificates_that_parse(broken_system_store, monkeypatch, caplog):
    good = _bundle_der(GOOD_CERTIFICATES)
    monkeypatch.setattr(tls_context, "_system_store_certificates", lambda: [good[0], UNPARSABLE, *good[1:]])
    with caplog.at_level("WARNING", logger="tam.tls"):
        context = tls_context.verified_context()
    _assert_verifies(context)
    assert len(context.get_ca_certs()) == GOOD_CERTIFICATES
    events = [json.loads(record.getMessage()) for record in caplog.records]
    assert [event["event"] for event in events] == ["tls_system_store_unloadable", "tls_system_store_certificates_skipped"]
    assert events[1]["skipped"] == 1 and events[1]["loaded"] == GOOD_CERTIFICATES


def test_no_usable_certificate_is_a_clear_error_not_an_unverified_context(broken_system_store, monkeypatch):
    monkeypatch.setattr(tls_context, "_system_store_certificates", lambda: [UNPARSABLE])
    with pytest.raises(tls_context.TlsTrustStoreError, match="no usable CA certificate"):
        tls_context.verified_context()


def test_other_platforms_have_no_per_certificate_fallback(broken_system_store, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(tls_context.TlsTrustStoreError):
        tls_context.verified_context()


@pytest.mark.skipif(sys.platform != "win32", reason="reads the Windows certificate stores")
def test_windows_stores_yield_der_certificates():
    certificates = list(tls_context._system_store_certificates())
    assert certificates and all(isinstance(der, bytes) for der in certificates)
