"""One TLS client context for every outbound HTTPS call.

The context always verifies the certificate chain and the hostname. Trust
anchors come from, in order:

1. the certifi bundle, when installed: python.org builds on macOS do not trust
   the system keychain, and a bundled file never depends on the host;
2. the platform trust store;
3. on Windows, when the platform store cannot be loaded as a whole (one
   certificate Python cannot parse fails the entire load with ``[ASN1] nested
   asn1 error``), the certificates that do parse, one by one.

When no trust anchor can be loaded at all, ``TlsTrustStoreError`` is raised.
Verification is never switched off.
"""

from __future__ import annotations

import json
import logging
import ssl
import sys
from collections.abc import Iterable

LOGGER = logging.getLogger("tam.tls")

WINDOWS_STORES = ("CA", "ROOT")


class TlsTrustStoreError(RuntimeError):
    """No certificate authority could be loaded, so no peer can be verified."""


def _log(event: str, **fields: object) -> None:
    LOGGER.warning(json.dumps({"event": event, **fields}))


def _system_store_certificates() -> Iterable[bytes]:
    """DER certificates the Windows system stores trust for server authentication."""
    if sys.platform != "win32":
        return
    for store in WINDOWS_STORES:
        try:
            entries = ssl.enum_certificates(store)
        except OSError as exc:
            _log("tls_system_store_unreadable", store=store, error=f"{type(exc).__name__}: {exc}")
            continue
        for der, encoding, trust in entries:
            if encoding == "x509_asn" and (trust is True or ssl.Purpose.SERVER_AUTH.oid in trust):
                yield bytes(der)


def _context_from_parsable_certificates() -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.verify_mode = ssl.CERT_REQUIRED
    context.check_hostname = True
    loaded = skipped = 0
    for der in _system_store_certificates():
        try:
            context.load_verify_locations(cadata=der)
        except ssl.SSLError:
            skipped += 1
        else:
            loaded += 1
    if skipped:
        _log("tls_system_store_certificates_skipped", loaded=loaded, skipped=skipped)
    if not loaded:
        raise TlsTrustStoreError("no usable CA certificate in the system trust store and certifi is not installed")
    return context


def verified_context() -> ssl.SSLContext:
    """A client context that verifies the chain and the hostname, or ``TlsTrustStoreError``."""
    try:
        import certifi
    except ImportError:
        pass
    else:
        return ssl.create_default_context(cafile=certifi.where())
    try:
        return ssl.create_default_context()
    except ssl.SSLError as exc:
        _log("tls_system_store_unloadable", error=f"{type(exc).__name__}: {exc}")
    return _context_from_parsable_certificates()
