"""Authenticated encryption for portable memory bundles (AES-256-GCM, scrypt key).

File layout::

    MAGIC (8 bytes) | header length (u32 BE) | header (canonical JSON)
    frame*          where frame = flag (u8) | ciphertext length (u32 BE) | ciphertext

The header names the format version, the KDF and its parameters (scrypt
n/r/p, random salt) and a random nonce prefix. Plaintext is cut into chunks of
``chunk_bytes``; chunk ``i`` is sealed with AES-256-GCM under the nonce
``prefix (7) | i (u32 BE) | flag (1)`` and the whole preamble as associated
data, so the header cannot be altered, frames cannot be reordered, and the
flag marking the last frame cannot be moved (truncation is detected). A
reader returns a chunk's plaintext only after its tag verified.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import struct
from typing import IO, Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

MAGIC = b"TAMBNDL\x01"
FORMAT_NAME = "tam-bundle"
FORMAT_VERSION = 1
CIPHER_NAME = "AES-256-GCM"
KDF_NAME = "scrypt"
KEY_BYTES = 32
SALT_BYTES = 16
NONCE_PREFIX_BYTES = 7
TAG_BYTES = 16
CHUNK_BYTES = 1024 * 1024
# scrypt cost written by new bundles: 2^17 * 8 * 128 B = 128 MiB of memory per derivation.
SCRYPT_N = 2 ** 17
SCRYPT_R = 8
SCRYPT_P = 1
# Bounds a reader accepts, so a crafted header cannot demand unbounded memory or time.
SCRYPT_MIN_N = 2 ** 14
SCRYPT_MAX_N = 2 ** 20
SCRYPT_MAX_R = 32
SCRYPT_MAX_P = 16
SCRYPT_MAX_MEMORY_BYTES = 2 * 1024 ** 3
SCRYPT_BLOCK_FACTOR = 128
MAX_HEADER_BYTES = 4096
MIN_CHUNK_BYTES = 4096
MAX_CHUNK_BYTES = 16 * 1024 * 1024
MAX_FRAMES = 2 ** 32 - 1
FLAG_MORE = 0
FLAG_FINAL = 1
_LENGTH = struct.Struct(">I")
_FRAME = struct.Struct(">BI")


class BundleCryptoError(ValueError):
    """The bundle cannot be decrypted: wrong passphrase, tampering, truncation or unsupported format."""


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(value: object, expected: int, field: str) -> bytes:
    if not isinstance(value, str):
        raise BundleCryptoError(f"header field {field} is missing")
    try:
        data = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise BundleCryptoError(f"header field {field} is not base64") from error
    if len(data) != expected:
        raise BundleCryptoError(f"header field {field} has {len(data)} bytes, expected {expected}")
    return data


def derive_key(passphrase: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    if not passphrase:
        raise BundleCryptoError("empty passphrase")
    return Scrypt(salt=salt, length=KEY_BYTES, n=n, r=r, p=p).derive(passphrase.encode("utf-8"))


def _nonce(prefix: bytes, counter: int, flag: int) -> bytes:
    return prefix + counter.to_bytes(4, "big") + bytes([flag])


class EncryptingWriter:
    """File-like sink: ``write`` plaintext, ``close`` seals the final frame. Does not close ``sink``."""

    def __init__(self, sink: IO[bytes], passphrase: str, *, scrypt_n: int | None = None, scrypt_r: int = SCRYPT_R,
                 scrypt_p: int = SCRYPT_P, chunk_bytes: int = CHUNK_BYTES) -> None:
        scrypt_n = SCRYPT_N if scrypt_n is None else scrypt_n
        if not MIN_CHUNK_BYTES <= chunk_bytes <= MAX_CHUNK_BYTES:
            raise ValueError("chunk_bytes out of range")
        salt = os.urandom(SALT_BYTES)
        self._prefix = os.urandom(NONCE_PREFIX_BYTES)
        header = {
            "format": FORMAT_NAME, "version": FORMAT_VERSION, "cipher": CIPHER_NAME, "chunk_bytes": chunk_bytes,
            "kdf": {"name": KDF_NAME, "n": scrypt_n, "r": scrypt_r, "p": scrypt_p, "salt": _b64(salt)},
            "nonce_prefix": _b64(self._prefix),
        }
        header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("ascii")
        _validate_header(header, len(header_bytes))
        self._aad = MAGIC + _LENGTH.pack(len(header_bytes)) + header_bytes
        self._aead = AESGCM(derive_key(passphrase, salt, scrypt_n, scrypt_r, scrypt_p))
        self._sink = sink
        self._chunk = chunk_bytes
        self._buffer = bytearray()
        self._counter = 0
        self._closed = False
        sink.write(self._aad)

    def writable(self) -> bool:
        return True

    def write(self, data: bytes) -> int:
        if self._closed:
            raise ValueError("write to a closed bundle writer")
        self._buffer += data
        # Keep at least one byte buffered so close() always has a final frame to seal.
        while len(self._buffer) > self._chunk:
            self._emit(bytes(self._buffer[:self._chunk]), FLAG_MORE)
            del self._buffer[:self._chunk]
        return len(data)

    def flush(self) -> None:
        self._sink.flush()

    def _emit(self, plaintext: bytes, flag: int) -> None:
        if self._counter >= MAX_FRAMES:
            raise ValueError("bundle exceeds the frame limit")
        sealed = self._aead.encrypt(_nonce(self._prefix, self._counter, flag), plaintext, self._aad)
        self._sink.write(_FRAME.pack(flag, len(sealed)) + sealed)
        self._counter += 1

    def close(self) -> None:
        if self._closed:
            return
        self._emit(bytes(self._buffer), FLAG_FINAL)
        self._buffer.clear()
        self._closed = True
        self._sink.flush()


def _int_field(container: dict[str, Any], name: str) -> int:
    value = container.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise BundleCryptoError(f"header field {name} must be an integer")
    return value


def _validate_header(header: object, size: int) -> None:
    if size > MAX_HEADER_BYTES or not isinstance(header, dict):
        raise BundleCryptoError("bundle header is malformed")
    if header.get("format") != FORMAT_NAME:
        raise BundleCryptoError("not a TAM bundle")
    version = header.get("version")
    if version != FORMAT_VERSION:
        raise BundleCryptoError(f"unsupported bundle version {version!r}; this TAM reads version {FORMAT_VERSION}")
    if header.get("cipher") != CIPHER_NAME:
        raise BundleCryptoError(f"unsupported cipher {header.get('cipher')!r}")
    chunk = _int_field(header, "chunk_bytes")
    if not MIN_CHUNK_BYTES <= chunk <= MAX_CHUNK_BYTES:
        raise BundleCryptoError("chunk size out of range")
    kdf = header.get("kdf")
    if not isinstance(kdf, dict) or kdf.get("name") != KDF_NAME:
        raise BundleCryptoError("unsupported key derivation")
    n, r, p = _int_field(kdf, "n"), _int_field(kdf, "r"), _int_field(kdf, "p")
    if (n & (n - 1) or not SCRYPT_MIN_N <= n <= SCRYPT_MAX_N or not 1 <= r <= SCRYPT_MAX_R
            or not 1 <= p <= SCRYPT_MAX_P or SCRYPT_BLOCK_FACTOR * n * r > SCRYPT_MAX_MEMORY_BYTES):
        raise BundleCryptoError("scrypt parameters out of the accepted range")


class DecryptingReader:
    """File-like source of verified plaintext. ``finish`` confirms the final frame was reached."""

    def __init__(self, source: IO[bytes], passphrase: str) -> None:
        self._source = source
        magic = self._exact(len(MAGIC), "header")
        if magic != MAGIC:
            raise BundleCryptoError("not a TAM bundle (bad magic bytes)")
        raw_length = self._exact(_LENGTH.size, "header")
        (length,) = _LENGTH.unpack(raw_length)
        if length > MAX_HEADER_BYTES:
            raise BundleCryptoError("bundle header is too large")
        header_bytes = self._exact(length, "header")
        try:
            header = json.loads(header_bytes.decode("ascii"))
        except (UnicodeDecodeError, ValueError) as error:
            raise BundleCryptoError("bundle header is not valid JSON") from error
        _validate_header(header, length)
        kdf = header["kdf"]
        salt = _unb64(kdf.get("salt"), SALT_BYTES, "kdf.salt")
        self._prefix = _unb64(header.get("nonce_prefix"), NONCE_PREFIX_BYTES, "nonce_prefix")
        self._chunk = header["chunk_bytes"]
        self._aad = magic + raw_length + header_bytes
        self._aead = AESGCM(derive_key(passphrase, salt, kdf["n"], kdf["r"], kdf["p"]))
        self._counter = 0
        self._plain = b""
        self._offset = 0
        self._final = False

    def _exact(self, size: int, what: str) -> bytes:
        data = self._source.read(size)
        if len(data) != size:
            raise BundleCryptoError(f"bundle is truncated (in the {what})")
        return data

    def readable(self) -> bool:
        return True

    def _next_frame(self) -> bool:
        if self._final:
            return False
        flag, length = _FRAME.unpack(self._exact(_FRAME.size, "frame header"))
        if flag not in (FLAG_MORE, FLAG_FINAL) or not TAG_BYTES <= length <= self._chunk + TAG_BYTES:
            raise BundleCryptoError("bundle frame is malformed (tampered or corrupted)")
        sealed = self._exact(length, "frame")
        try:
            self._plain = self._aead.decrypt(_nonce(self._prefix, self._counter, flag), sealed, self._aad)
        except InvalidTag as error:
            raise BundleCryptoError("wrong passphrase, or the bundle was modified or corrupted") from error
        self._offset = 0
        self._counter += 1
        if flag == FLAG_FINAL:
            self._final = True
            if self._source.read(1):
                raise BundleCryptoError("unexpected data after the final bundle frame")
        return True

    def read(self, size: int = -1) -> bytes:
        parts: list[bytes] = []
        wanted = size if size is not None and size >= 0 else None
        while wanted is None or wanted > 0:
            if self._offset >= len(self._plain) and not self._next_frame():
                break
            end = len(self._plain) if wanted is None else self._offset + wanted
            available = self._plain[self._offset:end]
            self._offset += len(available)
            parts.append(available)
            if wanted is not None:
                wanted -= len(available)
        return b"".join(parts)

    def finish(self) -> None:
        """Drain to the end and require the authenticated final frame (detects truncation)."""
        while self.read(self._chunk):
            continue
        if not self._final:
            raise BundleCryptoError("bundle is truncated (no final frame)")
