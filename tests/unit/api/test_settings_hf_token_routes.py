"""Feature 34 routes, through the REAL app (`create_app`) over a file-backed SQLite database.

AC-5 (save refused without a key), AC-6 (the plaintext is never in a response body or a log line),
AC-7 (registered on the live app), AC-12 (`/api` 422 bodies never echo a token). All tokens fake.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
import structlog
from fastapi import FastAPI
from sqlalchemy import select

from millm.core import secret_box
from millm.db.models.app_setting import AppSetting
from millm.services import hf_token
from tests.unit.batch_fixtures import app_with_db, batch_db, client_for  # noqa: F401

SAVED = "hf_FAKEsaved00000000000000000000000sav2"
OTHER = "hf_FAKEother00000000000000000000000oth4"
ENV = "hf_FAKEenv0000000000000000000000000env3"
KEY = "fake-settings-key-" + "0" * 30
BASE = "/api/settings/hf-token"


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setenv(secret_box.KEY_ENV, KEY)


@pytest.fixture
def no_key(monkeypatch):
    from millm.core.config import settings

    monkeypatch.delenv(secret_box.KEY_ENV, raising=False)
    monkeypatch.setattr(settings, "SETTINGS_ENCRYPTION_KEY", None)


@pytest.fixture(autouse=True)
def env_token(monkeypatch):
    from millm.core.config import settings

    monkeypatch.setattr(settings, "HF_TOKEN", None)
    monkeypatch.setenv("HF_TOKEN", ENV)


@pytest.fixture
def logs():
    with structlog.testing.capture_logs() as captured:
        yield captured


def assert_no_plaintext(*texts: str, secrets=(SAVED, OTHER, ENV)) -> None:
    """The redaction check every test below runs over bodies and logs."""
    for text in texts:
        for secret in secrets:
            assert secret not in text, "a route or a log line carried a token in plaintext"
            assert secret[3:] not in text, "a route or a log line carried a token without its prefix"


async def _rows(factory):
    async with factory() as s:
        return list((await s.execute(select(AppSetting))).scalars().all())


class TestRegistered:
    def test_the_live_app_serves_all_four(self):
        """Read from the live app's own route table (its OpenAPI document), not the module."""
        from millm.main import create_app

        paths = create_app().openapi()["paths"]
        assert set(paths.get(BASE, {})) == {"get", "put", "delete"}, "settings routes not registered"
        assert set(paths.get(f"{BASE}/test", {})) == {"post"}


class TestRoutes:
    async def test_status_save_test_clear(self, key, batch_db, logs):
        app = app_with_db(batch_db)
        bodies = []
        async with client_for(app) as c:
            r = await c.get(BASE)
            bodies.append(r.text)
            assert r.status_code == 200
            assert r.json()["data"]["source"] == "env" and r.json()["data"]["masked"] == "hf_…env3"

            r = await c.put(BASE, json={"token": SAVED})
            bodies.append(r.text)
            assert r.status_code == 200, r.text
            d = r.json()["data"]
            assert (d["source"], d["masked"], d["stored"], d["configured"]) == ("settings", "hf_…sav2", True, True)

            rows = await _rows(batch_db)
            assert len(rows) == 1 and SAVED not in rows[0].value

            with patch.object(hf_token, "_whoami", return_value={"name": "alice"}) as who:
                r = await c.post(f"{BASE}/test")
            bodies.append(r.text)
            assert r.status_code == 200 and r.json()["data"]["user"] == "alice"
            assert who.call_count == 1 and who.call_args.args == (SAVED,)

            r = await c.get(BASE)
            bodies.append(r.text)
            assert r.json()["data"]["user"] == "alice" and r.json()["data"]["last_verified_at"]

            r = await c.delete(BASE)
            bodies.append(r.text)
            assert r.status_code == 200 and r.json()["data"]["source"] == "env"
            assert await _rows(batch_db) == []
        assert_no_plaintext(*bodies, json.dumps(logs, default=str))

    async def test_save_without_a_key_is_refused_with_the_howto(self, no_key, batch_db, logs):
        async with client_for(app_with_db(batch_db)) as c:
            r = await c.put(BASE, json={"token": SAVED})
            status = (await c.get(BASE)).json()["data"]
        assert r.status_code == 409
        err = r.json()["error"]
        assert err["code"] == "SETTINGS_KEY_MISSING"
        assert "SETTINGS_ENCRYPTION_KEY" in err["message"] and "openssl rand -base64 32" in err["message"]
        assert await _rows(batch_db) == []
        assert status["encryption_available"] is False and status["source"] == "env"
        assert_no_plaintext(r.text, json.dumps(logs, default=str))

    async def test_test_a_typed_token_without_saving(self, key, batch_db):
        async with client_for(app_with_db(batch_db)) as c:
            with patch.object(hf_token, "_whoami", return_value={"name": "bob"}) as who:
                r = await c.post(f"{BASE}/test", json={"token": OTHER})
        assert r.status_code == 200 and r.json()["data"] == {
            "ok": True, "user": "bob", "source": "request", "verified_at": r.json()["data"]["verified_at"],
        }
        assert who.call_args.args == (OTHER,)
        assert await _rows(batch_db) == []
        assert_no_plaintext(r.text)

    async def test_invalid_and_network_are_named(self, key, batch_db, logs):
        from huggingface_hub.utils import HfHubHTTPError

        from unittest.mock import MagicMock

        Resp = lambda: MagicMock(status_code=401, headers={})  # noqa: E731

        async with client_for(app_with_db(batch_db)) as c:
            await c.put(BASE, json={"token": SAVED})
            with patch.object(hf_token, "_whoami",
                              side_effect=HfHubHTTPError(f"401 bad {SAVED}", response=Resp())):
                bad = await c.post(f"{BASE}/test")
            with patch.object(hf_token, "_whoami", side_effect=ConnectionError(f"down {SAVED}")):
                down = await c.post(f"{BASE}/test")
            status = (await c.get(BASE)).text
        assert (bad.status_code, bad.json()["error"]["code"]) == (401, "INVALID_HF_TOKEN")
        assert (down.status_code, down.json()["error"]["code"]) == (503, "HF_HUB_UNREACHABLE")
        assert json.loads(status)["data"]["last_check_error"] == "network"
        assert_no_plaintext(bad.text, down.text, status, json.dumps(logs, default=str))

    async def test_nothing_to_test(self, key, batch_db, monkeypatch):
        monkeypatch.delenv("HF_TOKEN")
        async with client_for(app_with_db(batch_db)) as c:
            r = await c.post(f"{BASE}/test")
        assert (r.status_code, r.json()["error"]["code"]) == (404, "HF_TOKEN_NOT_CONFIGURED")

    @pytest.mark.parametrize("body", [{"token": ""}, {"token": "none"}, {"token": "hf_a b"},
                                      {"token": 12345}, {"token": None}, {}])
    async def test_bad_bodies_never_echo_or_store(self, key, batch_db, body):
        async with client_for(app_with_db(batch_db)) as c:
            r = await c.put(BASE, json=body)
        assert r.status_code in (400, 422), r.text
        assert await _rows(batch_db) == []
        assert "hf_a b" not in r.text and "12345" not in r.text


class TestCheckIsRecorded:
    """A Test's outcome is STORED with the saved token, and status returns it (2026-10-18 defect).

    On the deployed pod the operator typed a token, pressed Test (passed), then Save: the Save
    cleared the verification the Test had just made, and GET read `user: null,
    last_verified_at: null`. A failed Test lived only in process memory. Every case below runs
    through the real routes and reads the stored row back, then reloads it as a restart would.
    """

    @staticmethod
    def _invalid():
        from unittest.mock import MagicMock

        from huggingface_hub.utils import HfHubHTTPError

        return HfHubHTTPError("401 Unauthorized", response=MagicMock(status_code=401, headers={}))

    @staticmethod
    async def _restart(batch_db) -> dict:
        """What a fresh process reports: memory dropped, then the startup load from the row."""
        hf_token._reset_memory()
        await hf_token.load_hf_token_on_startup(batch_db)
        return hf_token.status()

    async def test_test_the_typed_token_then_save_records_the_check(self, key, batch_db, logs):
        bodies = []
        async with client_for(app_with_db(batch_db)) as c:
            with patch.object(hf_token, "_whoami", return_value={"name": "mistudio"}) as who:
                t = await c.post(f"{BASE}/test", json={"token": SAVED})
            bodies.append(t.text)
            assert t.status_code == 200 and t.json()["data"]["source"] == "request"
            assert who.call_count == 1 and who.call_args.args == (SAVED,)
            r = await c.put(BASE, json={"token": SAVED})
            bodies.append(r.text)
            got = (await c.get(BASE)).json()["data"]
        assert r.json()["data"]["user"] == "mistudio"
        assert got["user"] == "mistudio" and got["last_verified_at"] == t.json()["data"]["verified_at"]
        assert got["last_check_error"] is None
        (row,) = await _rows(batch_db)
        assert row.verified_user == "mistudio" and row.verified_at is not None
        after = await self._restart(batch_db)
        assert after["user"] == "mistudio" and after["last_verified_at"] is not None
        assert_no_plaintext(*bodies, json.dumps(logs, default=str))

    async def test_save_then_test_records_the_check(self, key, batch_db):
        async with client_for(app_with_db(batch_db)) as c:
            await c.put(BASE, json={"token": SAVED})
            with patch.object(hf_token, "_whoami", return_value={"name": "mistudio"}):
                await c.post(f"{BASE}/test")
            got = (await c.get(BASE)).json()["data"]
        assert got["user"] == "mistudio" and got["last_verified_at"]
        (row,) = await _rows(batch_db)
        assert row.verified_user == "mistudio"

    async def test_typing_the_saved_token_and_testing_records_it(self, key, batch_db):
        async with client_for(app_with_db(batch_db)) as c:
            await c.put(BASE, json={"token": SAVED})
            with patch.object(hf_token, "_whoami", return_value={"name": "mistudio"}):
                await c.post(f"{BASE}/test", json={"token": SAVED})
            got = (await c.get(BASE)).json()["data"]
        assert got["user"] == "mistudio"
        (row,) = await _rows(batch_db)
        assert row.verified_user == "mistudio"

    async def test_a_resave_clears_stale_verification(self, key, batch_db):
        async with client_for(app_with_db(batch_db)) as c:
            with patch.object(hf_token, "_whoami", return_value={"name": "mistudio"}):
                await c.post(f"{BASE}/test", json={"token": SAVED})
            await c.put(BASE, json={"token": SAVED})
            assert (await c.get(BASE)).json()["data"]["user"] == "mistudio"
            # The same token again, no new Test: the earlier check is not re-used twice.
            same = (await c.put(BASE, json={"token": SAVED})).json()["data"]
            # A different token: the old check describes a token that is no longer saved.
            other = (await c.put(BASE, json={"token": OTHER})).json()["data"]
        for d in (same, other):
            assert (d["user"], d["last_verified_at"], d["last_check_error"]) == (None, None, None)
        (row,) = await _rows(batch_db)
        assert (row.verified_user, row.verified_at, row.last_check_error) == (None, None, None)

    async def test_a_check_of_another_token_is_not_carried(self, key, batch_db):
        async with client_for(app_with_db(batch_db)) as c:
            with patch.object(hf_token, "_whoami", return_value={"name": "someone"}):
                await c.post(f"{BASE}/test", json={"token": OTHER})
            d = (await c.put(BASE, json={"token": SAVED})).json()["data"]
        assert (d["user"], d["last_verified_at"]) == (None, None)

    async def test_a_stale_check_is_not_carried(self, key, batch_db, monkeypatch):
        from datetime import timedelta

        monkeypatch.setattr(hf_token, "PENDING_CHECK_TTL", timedelta(seconds=-1))
        async with client_for(app_with_db(batch_db)) as c:
            with patch.object(hf_token, "_whoami", return_value={"name": "mistudio"}):
                await c.post(f"{BASE}/test", json={"token": SAVED})
            d = (await c.put(BASE, json={"token": SAVED})).json()["data"]
        assert (d["user"], d["last_verified_at"]) == (None, None)

    async def test_a_clear_discards_a_pending_check(self, key, batch_db):
        async with client_for(app_with_db(batch_db)) as c:
            with patch.object(hf_token, "_whoami", return_value={"name": "mistudio"}):
                await c.post(f"{BASE}/test", json={"token": SAVED})
            await c.delete(BASE)
            d = (await c.put(BASE, json={"token": SAVED})).json()["data"]
        assert d["user"] is None

    async def test_a_failed_test_of_the_saved_token_is_stored(self, key, batch_db, logs):
        bodies = []
        async with client_for(app_with_db(batch_db)) as c:
            await c.put(BASE, json={"token": SAVED})
            with patch.object(hf_token, "_whoami", return_value={"name": "mistudio"}):
                await c.post(f"{BASE}/test")
            with patch.object(hf_token, "_whoami", side_effect=self._invalid()):
                bad = await c.post(f"{BASE}/test")
            bodies.append(bad.text)
            got = (await c.get(BASE)).json()["data"]
        assert bad.status_code == 401
        assert got["last_check_error"] == "invalid"
        # The last SUCCESS is kept: it is the last verified time, not a claim the token works now.
        assert got["user"] == "mistudio" and got["last_verified_at"]
        (row,) = await _rows(batch_db)
        assert row.last_check_error == "invalid"
        after = await self._restart(batch_db)
        assert after["last_check_error"] == "invalid", "a failed check was not persisted"
        async with client_for(app_with_db(batch_db)) as c:
            with patch.object(hf_token, "_whoami", return_value={"name": "mistudio"}):
                await c.post(f"{BASE}/test")
            cleared = (await c.get(BASE)).json()["data"]
        assert cleared["last_check_error"] is None
        (row,) = await _rows(batch_db)
        assert row.last_check_error is None
        assert_no_plaintext(*bodies, json.dumps(logs, default=str))

    async def test_a_failed_typed_test_then_save_records_the_failure(self, key, batch_db):
        async with client_for(app_with_db(batch_db)) as c:
            with patch.object(hf_token, "_whoami", side_effect=self._invalid()):
                await c.post(f"{BASE}/test", json={"token": SAVED})
            d = (await c.put(BASE, json={"token": SAVED})).json()["data"]
        assert (d["last_check_error"], d["user"], d["last_verified_at"]) == ("invalid", None, None)
        (row,) = await _rows(batch_db)
        assert row.last_check_error == "invalid"


class TestNoPlaintextNegativeControl:
    """The scanner above must catch a route that DOES return the token, or it proves nothing."""

    async def test_a_leaking_route_is_caught(self):
        import httpx

        app = FastAPI()

        @app.get("/leak")
        async def leak():
            return {"token": SAVED}

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            text = (await c.get("/leak")).text
        with pytest.raises(AssertionError):
            assert_no_plaintext(text)


class TestValidationRedaction:
    """AC-12, a pre-existing leak: a model-level 422 on `/api/models` echoed the whole body."""

    async def test_model_download_422_does_not_echo_hf_token(self, batch_db, tmp_path, monkeypatch):
        from millm.core.config import settings

        monkeypatch.setattr(settings, "MODEL_CACHE_DIR", str(tmp_path / "models"))
        async with client_for(app_with_db(batch_db)) as c:
            r = await c.post("/api/models", json={"source": "huggingface", "hf_token": SAVED})
        assert r.status_code == 422, r.text
        assert_no_plaintext(r.text)
        assert "[redacted]" in r.text

    async def test_a_field_level_error_on_the_token_is_redacted(self, batch_db):
        async with client_for(app_with_db(batch_db)) as c:
            r = await c.put(BASE, json={"token": ["hf_FAKEsaved00000000000000000000000sav2"]})
        assert r.status_code == 422
        assert_no_plaintext(r.text)

    def test_redaction_walks_nested_values(self):
        from millm.api.exception_handlers import redact_secrets

        out = redact_secrets([{"loc": ["body"], "input": {"repo_id": "x", "nested": {"hf_token": SAVED}}},
                              {"loc": ["body", "token"], "input": SAVED, "msg": "m"}])
        assert_no_plaintext(json.dumps(out))
        assert out[0]["input"]["repo_id"] == "x"
