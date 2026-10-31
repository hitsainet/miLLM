"""Encryption at rest for operator secrets stored in the database (Feature 34).

AES-256-GCM. The key comes from ``SETTINGS_ENCRYPTION_KEY`` (in Kubernetes, the optional key of the
same name in the ``millm-secrets`` Secret), stretched to 256 bits with HKDF-SHA256. Generate one with::

    openssl rand -base64 32

THERE IS NO FALLBACK KEY, deliberately. A key generated in-process would be lost on the next restart
and every stored secret with it; a key derived from something that is not secret would make the
encryption decorative. With no key configured, `encrypt` raises `SettingsKeyMissingError` and the
caller refuses the save; nothing is ever written as plaintext.

The envelope is ``v1:<key id>:<base64(nonce || ciphertext || tag)>``. The setting's name is bound in
as associated data, so a ciphertext copied from one row to another fails authentication. The key id
(a short HKDF-derived fingerprint, not the key) lets a read tell "the key changed" apart from "the row
was tampered with" without trying to decrypt.

A failed decryption RAISES. It never hands back the stored text: miStudio once sent base64
ciphertext to a vendor in an ``Authorization`` header because a decryption failure was silent.
"""

from __future__ import annotations

import base64
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from millm.core.errors import MiLLMError

#: The environment variable (and Kubernetes Secret key) holding the key material.
KEY_ENV = "SETTINGS_ENCRYPTION_KEY"
#: Shortest key material accepted. `openssl rand -base64 32` gives 44 characters.
MIN_KEY_CHARS = 32
_VERSION = "v1"
_NONCE_BYTES = 12

#: What the operator must do, quoted by every refusal so the message is actionable on its own.
KEY_HOWTO = (
    f"Create the key {KEY_ENV} in the millm-secrets Secret, then restart the backend pod: "
    'KEY=$(openssl rand -base64 32); kubectl -n millm patch secret millm-secrets --type merge '
    f'-p "{{\\"stringData\\":{{\\"{KEY_ENV}\\":\\"$KEY\\"}}}}"; '
    "kubectl -n millm delete pod -l app=millm-backend. "
    f"With Docker Compose, set {KEY_ENV} in .env. Keep the value: a stored secret cannot be "
    "read without the key it was saved under."
)


class SettingsKeyMissingError(MiLLMError):
    """No usable encryption key, so a secret cannot be stored encrypted — the save is refused."""

    code = "SETTINGS_KEY_MISSING"
    status_code = 409


class SecretDecryptError(Exception):
    """A stored envelope could not be opened. ``reason`` is ``key_mismatch`` or ``corrupt``.

    Never carries the stored text or the key in its message."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"stored secret unreadable: {reason}")


def _key_material() -> str | None:
    raw = os.environ.get(KEY_ENV)
    if raw is None:
        from millm.core.config import settings

        secret = getattr(settings, KEY_ENV, None)
        raw = secret.get_secret_value() if secret is not None else None
    raw = (raw or "").strip()
    return raw or None


def key_problem() -> str | None:
    """Why no key is usable (``missing`` / ``too_short``), or None when one is."""
    material = _key_material()
    if material is None:
        return "missing"
    if len(material) < MIN_KEY_CHARS:
        return "too_short"
    return None


def encryption_available() -> bool:
    return key_problem() is None


def _derive(info: bytes) -> bytes:
    material = _key_material()
    problem = key_problem()
    if problem is not None or material is None:
        detail = (
            f"{KEY_ENV} is not set" if problem == "missing"
            else f"{KEY_ENV} is shorter than {MIN_KEY_CHARS} characters"
        )
        raise SettingsKeyMissingError(
            f"Cannot store the secret encrypted: {detail}. {KEY_HOWTO}",
            details={"key_env": KEY_ENV, "reason": problem},
        )
    return HKDF(
        algorithm=hashes.SHA256(), length=32, salt=b"millm-settings-v1", info=info
    ).derive(material.encode("utf-8"))


def key_id() -> str:
    """A 12-hex-character fingerprint of the current key (not reversible to the key)."""
    return _derive(b"key-id").hex()[:12]


def encrypt(plaintext: str, *, name: str) -> str:
    """The envelope for ``plaintext``, bound to the setting ``name``. Raises SettingsKeyMissingError."""
    aead = AESGCM(_derive(b"settings-encryption"))
    nonce = os.urandom(_NONCE_BYTES)
    sealed = aead.encrypt(nonce, plaintext.encode("utf-8"), name.encode("utf-8"))
    return f"{_VERSION}:{key_id()}:{base64.b64encode(nonce + sealed).decode('ascii')}"


def decrypt(envelope: str, *, name: str) -> str:
    """The plaintext, or SecretDecryptError / SettingsKeyMissingError. Never the stored text."""
    try:
        version, kid, body = envelope.split(":", 2)
    except ValueError as exc:
        raise SecretDecryptError("corrupt") from exc
    if version != _VERSION:
        raise SecretDecryptError("corrupt")
    if kid != key_id():
        raise SecretDecryptError("key_mismatch")
    try:
        raw = base64.b64decode(body, validate=True)
    except ValueError as exc:
        raise SecretDecryptError("corrupt") from exc
    if len(raw) < _NONCE_BYTES + 16:
        raise SecretDecryptError("corrupt")
    try:
        plain = AESGCM(_derive(b"settings-encryption")).decrypt(
            raw[:_NONCE_BYTES], raw[_NONCE_BYTES:], name.encode("utf-8")
        )
    except InvalidTag as exc:
        raise SecretDecryptError("corrupt") from exc
    return plain.decode("utf-8")
