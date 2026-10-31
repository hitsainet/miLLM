"""Feature 34: the one Hugging Face credential resolver, the encrypted store, and the startup load.

Every token here is FAKE. AC-1 (precedence), AC-4 (encryption), AC-5 (refuse without a key, at the
service), AC-8 (startup load).
"""

from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from millm.core import secret_box
from millm.db.models.app_setting import AppSetting
from millm.services import hf_token
from tests.unit.batch_fixtures import batch_db  # noqa: F401

REQ = "hf_FAKErequest000000000000000000000req1"
SAVED = "hf_FAKEsaved00000000000000000000000sav2"
ENV = "hf_FAKEenv0000000000000000000000000env3"
KEY = "fake-settings-key-" + "0" * 30
OTHER_KEY = "other-settings-key-" + "1" * 30


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setenv(secret_box.KEY_ENV, KEY)
    return KEY


@pytest.fixture
def no_key(monkeypatch):
    monkeypatch.delenv(secret_box.KEY_ENV, raising=False)
    from millm.core.config import settings

    monkeypatch.setattr(settings, "SETTINGS_ENCRYPTION_KEY", None)


@pytest.fixture
def no_env(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    from millm.core.config import settings

    monkeypatch.setattr(settings, "HF_TOKEN", None)


def _put(token):
    hf_token._set_memory(token=token)


# ── AC-1: precedence ─────────────────────────────────────────────────────────

ABSENT = [None, "", "   ", "none", "None", " NONE "]


class TestPrecedence:
    """explicit > settings > env > anonymous; blank and `none` are absent at EVERY tier."""

    @pytest.mark.parametrize(
        "explicit, saved, env, expected, source",
        [
            (REQ, SAVED, ENV, REQ, "request"),
            (REQ, None, None, REQ, "request"),
            (None, SAVED, ENV, SAVED, "settings"),
            (None, SAVED, None, SAVED, "settings"),
            (None, None, ENV, ENV, "env"),
            (None, None, None, None, "none"),
            (REQ, None, ENV, REQ, "request"),
        ],
    )
    def test_table(self, monkeypatch, no_env, explicit, saved, env, expected, source):
        if env:
            monkeypatch.setenv("HF_TOKEN", env)
        _put(saved)
        assert hf_token.resolve_with_source(explicit) == (expected, source)
        assert hf_token.hub_token(explicit) == (expected if expected else False)

    @pytest.mark.parametrize("blank", ABSENT)
    def test_blank_explicit_falls_to_saved(self, no_env, blank):
        _put(SAVED)
        assert hf_token.resolve_with_source(blank) == (SAVED, "settings")

    @pytest.mark.parametrize("blank", ABSENT)
    def test_blank_env_is_anonymous_and_sends_no_credential(self, monkeypatch, no_env, blank):
        if blank is not None:
            monkeypatch.setenv("HF_TOKEN", blank)
        # False, never None: None lets huggingface_hub read HF_TOKEN itself and send "none".
        assert hf_token.hub_token(None) is False
        assert hf_token.resolve_with_source(None) == (None, "none")

    def test_false_is_anonymous_in_the_installed_hub(self, monkeypatch):
        """The resolver's `False` contract, checked against the INSTALLED huggingface_hub rather
        than assumed (re-verified for hub 2.x, 2026-10-18): `False` sends no credential even with
        HF_TOKEN set, while `None` lets the library send HF_TOKEN itself."""
        from huggingface_hub.utils import build_hf_headers

        monkeypatch.setenv("HF_TOKEN", ENV)
        assert "authorization" not in {k.lower() for k in build_hf_headers(token=False)}
        assert build_hf_headers(token=None).get("authorization") == f"Bearer {ENV}"

    @pytest.mark.parametrize("blank", ABSENT)
    def test_blank_saved_value_is_absent(self, monkeypatch, no_env, blank):
        monkeypatch.setenv("HF_TOKEN", ENV)
        _put(hf_token.clean(blank))
        assert hf_token.resolve_with_source(None) == (ENV, "env")

    def test_settings_env_file_value_is_the_env_tier(self, monkeypatch, no_env):
        from millm.core.config import settings

        monkeypatch.setattr(settings, "HF_TOKEN", ENV)
        assert hf_token.resolve_with_source(None) == (ENV, "env")

    def test_explicit_is_stripped(self, no_env):
        assert hf_token.resolve_hf_token(f"  {REQ}  ") == REQ

    def test_mask_never_reveals_the_token(self):
        masked = hf_token.mask(SAVED)
        assert masked == "hf_…sav2"
        assert SAVED not in masked
        assert hf_token.mask("short") == "…"
        assert hf_token.mask(None) is None

    def test_credential_tag_differs_per_token_and_hides_it(self):
        assert hf_token.credential_tag(SAVED) != hf_token.credential_tag(ENV)
        assert hf_token.credential_tag(False) == "anon"
        assert SAVED not in hf_token.credential_tag(SAVED)


# ── AC-4: encryption ─────────────────────────────────────────────────────────


class TestEncryption:
    def test_round_trip(self, key):
        env = secret_box.encrypt(SAVED, name="hf_token")
        assert SAVED not in env
        assert secret_box.decrypt(env, name="hf_token") == SAVED

    def test_each_encryption_uses_a_fresh_nonce(self, key):
        assert secret_box.encrypt(SAVED, name="hf_token") != secret_box.encrypt(SAVED, name="hf_token")

    def test_a_changed_key_is_named_not_returned(self, key, monkeypatch):
        env = secret_box.encrypt(SAVED, name="hf_token")
        monkeypatch.setenv(secret_box.KEY_ENV, OTHER_KEY)
        with pytest.raises(secret_box.SecretDecryptError) as exc:
            secret_box.decrypt(env, name="hf_token")
        assert exc.value.reason == "key_mismatch"
        assert SAVED not in str(exc.value)

    def test_tampering_fails_authentication(self, key):
        version, kid, body = secret_box.encrypt(SAVED, name="hf_token").split(":", 2)
        raw = bytearray(base64.b64decode(body))
        raw[-1] ^= 0x01
        tampered = f"{version}:{kid}:{base64.b64encode(bytes(raw)).decode()}"
        with pytest.raises(secret_box.SecretDecryptError) as exc:
            secret_box.decrypt(tampered, name="hf_token")
        assert exc.value.reason == "corrupt"

    def test_a_row_copied_under_another_name_fails(self, key):
        env = secret_box.encrypt(SAVED, name="hf_token")
        with pytest.raises(secret_box.SecretDecryptError):
            secret_box.decrypt(env, name="something_else")

    @pytest.mark.parametrize("garbage", ["", "plain-text-token", "v1:abc", "v2:x:y", "v1:kid:!!!"])
    def test_garbage_is_corrupt_or_mismatch_never_returned(self, key, garbage):
        with pytest.raises(secret_box.SecretDecryptError):
            secret_box.decrypt(garbage, name="hf_token")

    def test_no_key_refuses_with_the_howto(self, no_key):
        with pytest.raises(secret_box.SettingsKeyMissingError) as exc:
            secret_box.encrypt(SAVED, name="hf_token")
        assert exc.value.status_code == 409
        assert "SETTINGS_ENCRYPTION_KEY" in exc.value.message
        assert "openssl rand -base64 32" in exc.value.message
        assert "millm-secrets" in exc.value.message

    def test_a_short_key_is_refused(self, monkeypatch):
        monkeypatch.setenv(secret_box.KEY_ENV, "short")
        assert secret_box.key_problem() == "too_short"
        with pytest.raises(secret_box.SettingsKeyMissingError):
            secret_box.encrypt(SAVED, name="hf_token")

    def test_key_from_settings_secretstr(self, monkeypatch):
        from pydantic import SecretStr

        from millm.core.config import settings

        monkeypatch.delenv(secret_box.KEY_ENV, raising=False)
        monkeypatch.setattr(settings, "SETTINGS_ENCRYPTION_KEY", SecretStr(KEY))
        assert secret_box.decrypt(secret_box.encrypt(SAVED, name="n"), name="n") == SAVED


# ── Save / clear / status / startup (SQLite) ─────────────────────────────────


async def _rows(session: AsyncSession) -> list[AppSetting]:
    return list((await session.execute(select(AppSetting))).scalars().all())


class TestStore:
    async def test_save_stores_only_ciphertext_and_activates(self, key, no_env, test_session):
        await hf_token.save_token(test_session, f"  {SAVED} ")
        rows = await _rows(test_session)
        assert len(rows) == 1 and rows[0].key == "hf_token"
        assert SAVED not in rows[0].value
        assert secret_box.decrypt(rows[0].value, name="hf_token") == SAVED
        assert hf_token.resolve_with_source(None) == (SAVED, "settings")

    async def test_save_without_a_key_refuses_and_writes_nothing(self, no_key, no_env, test_session):
        with pytest.raises(secret_box.SettingsKeyMissingError):
            await hf_token.save_token(test_session, SAVED)
        assert await _rows(test_session) == []
        assert hf_token.resolve_with_source(None) == (None, "none")

    @pytest.mark.parametrize("bad", ["", "none", "hf_has space", "hf_" + "x" * 300, "hf_ünïcode000000"])
    async def test_invalid_values_are_refused_without_echo(self, key, test_session, bad):
        with pytest.raises(hf_token.HfTokenInvalidValueError) as exc:
            await hf_token.save_token(test_session, bad)
        if bad.strip():
            assert bad not in exc.value.message
        assert await _rows(test_session) == []

    async def test_resave_replaces_and_forgets_verification(self, key, no_env, test_session):
        await hf_token.save_token(test_session, SAVED)
        row = (await _rows(test_session))[0]
        row.verified_user = "someone"
        await test_session.commit()
        await hf_token.save_token(test_session, REQ)
        rows = await _rows(test_session)
        assert len(rows) == 1 and rows[0].verified_user is None
        assert hf_token.resolve_hf_token(None) == REQ

    async def test_clear_removes_row_and_memory(self, key, monkeypatch, no_env, test_session):
        monkeypatch.setenv("HF_TOKEN", ENV)
        await hf_token.save_token(test_session, SAVED)
        assert await hf_token.clear_token(test_session) is True
        assert await _rows(test_session) == []
        assert hf_token.resolve_with_source(None) == (ENV, "env")
        assert await hf_token.clear_token(test_session) is False

    async def test_status_masks_and_names_the_source(self, key, monkeypatch, no_env, test_session):
        monkeypatch.setenv("HF_TOKEN", ENV)
        await hf_token.save_token(test_session, SAVED)
        st = hf_token.status()
        assert st["configured"] and st["source"] == "settings" and st["masked"] == "hf_…sav2"
        assert st["stored"] and st["env_configured"] and st["encryption_available"]
        assert SAVED not in repr(st) and ENV not in repr(st)


class TestStartup:
    """AC-8: the saved token is loaded from the database at startup."""

    @pytest.fixture
    def factory(self, test_engine):
        return async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)

    async def test_loads_the_saved_token(self, key, no_env, factory):
        async with factory() as s:
            s.add(AppSetting(key="hf_token", value=secret_box.encrypt(SAVED, name="hf_token")))
            await s.commit()
        assert await hf_token.load_hf_token_on_startup(factory) == "settings"
        assert hf_token.resolve_hf_token(None) == SAVED

    async def test_a_key_change_falls_through_and_is_reported(self, key, monkeypatch, no_env, factory):
        async with factory() as s:
            s.add(AppSetting(key="hf_token", value=secret_box.encrypt(SAVED, name="hf_token")))
            await s.commit()
        monkeypatch.setenv(secret_box.KEY_ENV, OTHER_KEY)
        monkeypatch.setenv("HF_TOKEN", ENV)
        assert await hf_token.load_hf_token_on_startup(factory) == "env"
        st = hf_token.status()
        assert st["stored"] and st["stored_error"] == "key_mismatch" and st["source"] == "env"

    async def test_a_missing_key_falls_through(self, key, monkeypatch, no_env, factory):
        async with factory() as s:
            s.add(AppSetting(key="hf_token", value=secret_box.encrypt(SAVED, name="hf_token")))
            await s.commit()
        monkeypatch.delenv(secret_box.KEY_ENV)
        assert await hf_token.load_hf_token_on_startup(factory) == "none"
        assert hf_token.status()["stored_error"] == "key_missing"
        assert hf_token.hub_token(None) is False

    async def test_an_unmigrated_database_never_raises(self, no_env):
        def broken():
            raise RuntimeError("no such table: app_settings")

        assert await hf_token.load_hf_token_on_startup(broken) == "none"

    async def test_running_lifespan_loads_the_saved_token(self, key, no_env, monkeypatch, batch_db):
        """The wiring, RUN: a token saved in the database is active after `main.lifespan` starts.
        Removing the call from `lifespan` turns this red (control C-STARTUP)."""
        from unittest.mock import AsyncMock

        import millm.db.base as db_base
        import millm.main as main

        async with batch_db() as s:
            s.add(AppSetting(key="hf_token", value=secret_box.encrypt(SAVED, name="hf_token")))
            await s.commit()
        monkeypatch.setattr(db_base, "async_session_factory", batch_db)
        monkeypatch.setattr(main, "start_batch_api", AsyncMock())
        monkeypatch.setattr(main, "stop_batch_api", AsyncMock())
        monkeypatch.setattr(main, "disarm_probes_on_startup", AsyncMock(return_value=[]))
        monkeypatch.setattr(main, "remove_stale_temp_dirs", lambda: [])
        monkeypatch.setattr(main.settings, "AUTO_LOAD_MODEL", None, raising=False)
        assert hf_token.resolve_with_source(None) == (None, "none")
        async with main.lifespan(MagicMock()):
            assert hf_token.resolve_with_source(None) == (SAVED, "settings")

    def test_lifespan_loads_it_before_the_auto_load(self):
        import ast
        import inspect

        import millm.main as main

        assert main.load_hf_token_on_startup is hf_token.load_hf_token_on_startup
        tree = ast.parse(inspect.getsource(main.lifespan))
        lines = {
            node.func.id: node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert lines["load_hf_token_on_startup"] < lines["_auto_load_model"]


# ── Test (whoami) ────────────────────────────────────────────────────────────


class _Resp:
    def __init__(self, code):
        self.status_code = code
        self.headers = {}
        self.request = MagicMock()


class TestCheck:
    async def test_a_given_token_is_checked_and_not_recorded(self, key, no_env, test_session):
        with patch.object(hf_token, "_whoami", return_value={"name": "alice"}) as who:
            out = await hf_token.check_token(test_session, REQ)
        assert who.call_args.args == (REQ,)
        assert out["user"] == "alice" and out["source"] == "request"
        assert hf_token.status()["user"] is None

    async def test_the_saved_token_check_is_recorded(self, key, no_env, test_session):
        await hf_token.save_token(test_session, SAVED)
        with patch.object(hf_token, "_whoami", return_value={"name": "alice"}) as who:
            out = await hf_token.check_token(test_session)
        assert who.call_args.args == (SAVED,) and out["source"] == "settings"
        row = (await _rows(test_session))[0]
        assert row.verified_user == "alice" and row.verified_at is not None
        assert hf_token.status()["user"] == "alice"

    async def test_rejected_token_is_invalid_hf_token(self, key, no_env, test_session):
        from huggingface_hub.utils import HfHubHTTPError

        await hf_token.save_token(test_session, SAVED)
        err = HfHubHTTPError(f"401 Unauthorized for token {SAVED}", response=_Resp(401))
        with patch.object(hf_token, "_whoami", side_effect=err):
            with pytest.raises(Exception) as exc:
                await hf_token.check_token(test_session)
        assert exc.value.code == "INVALID_HF_TOKEN"
        assert SAVED not in exc.value.message and SAVED not in str(exc.value.details)
        assert exc.value.__cause__ is None
        assert hf_token.status()["last_check_error"] == "invalid"

    async def test_network_failure_is_named_and_not_a_verdict(self, key, no_env, test_session):
        await hf_token.save_token(test_session, SAVED)
        with patch.object(hf_token, "_whoami", side_effect=OSError("connection refused")):
            with pytest.raises(hf_token.HubUnreachableError) as exc:
                await hf_token.check_token(test_session)
        assert exc.value.code == "HF_HUB_UNREACHABLE" and exc.value.status_code == 503

    async def test_nothing_configured_is_named(self, no_env, test_session):
        with pytest.raises(hf_token.HfTokenNotConfiguredError):
            await hf_token.check_token(test_session)

    def test_whoami_passes_the_token_to_the_hub(self):
        with patch("huggingface_hub.HfApi") as api:
            api.return_value.whoami.return_value = {"name": "bob"}
            assert hf_token._whoami(REQ) == {"name": "bob"}
        assert api.return_value.whoami.call_args.kwargs == {"token": REQ}
