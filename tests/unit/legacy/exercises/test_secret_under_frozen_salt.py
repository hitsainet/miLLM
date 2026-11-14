"""Exercise for DR-10: the HKDF salt `millm-settings-v1` (040 FR-040.3).

Every secret miLLM stores (`app_settings`: the Hugging Face token) is sealed with keys derived under
this salt. A renamed salt derives other keys, so every stored secret would read as "key mismatch"
after the rename, with the operator's key unchanged. The fixture is an envelope sealed once under
the frozen salt; it must still open, through the real settings row path.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from miserver.core import secret_box
from miserver.services import hf_token

FIXTURE = Path(__file__).resolve().parents[3] / "fixtures" / "legacy" / "secret_v1.json"


def test_stored_secret_decrypts_under_frozen_salt(monkeypatch):
    stored = json.loads(FIXTURE.read_text())
    monkeypatch.setenv(secret_box.KEY_ENV, stored["key"])

    # The key id is a derivation under the literal salt: computed here, independently of the code.
    independent = HKDF(algorithm=hashes.SHA256(), length=32, salt=b"millm-settings-v1",
                       info=b"key-id").derive(stored["key"].encode()).hex()[:12]
    assert stored["key_id"] == independent == secret_box.key_id()

    assert secret_box.decrypt(stored["envelope"], name=stored["name"]) == stored["plaintext"]

    # The app_settings row path the server takes at startup and after a save.
    row = SimpleNamespace(key=hf_token.SETTING_KEY, value=stored["envelope"], verified_at=None,
                          verified_user=None, last_check_error=None)
    hf_token._reset_memory()
    try:
        hf_token._apply_row(row)
        saved = hf_token.snapshot()
        assert saved.token == stored["plaintext"] and saved.error is None
    finally:
        hf_token._reset_memory()
