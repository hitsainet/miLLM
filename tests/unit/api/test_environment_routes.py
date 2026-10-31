"""Feature 35 routes, through the REAL app (`create_app`) over a file-backed SQLite database.

Registered on the live app; every writable key reaches the singleton through PUT and back through
DELETE; every never-writable key is refused by name; a secret set in the environment appears in no
response; LOG_LEVEL takes effect now; and switching to DEBUG while serving still sends no
exception text.
"""

from __future__ import annotations

import json
import logging

import pytest
from fastapi import APIRouter
from sqlalchemy import select

from millm.core.config import settings
from millm.db.models.env_override import EnvOverride, EnvOverrideChange
from millm.services import runtime_settings as rs
from tests.unit.batch_fixtures import app_with_db, batch_db, client_for  # noqa: F401
from tests.unit.services.test_runtime_settings import VALID_VALUES

BASE = "/api/environment"
SENTINELS = {
    "DATABASE_URL": "postgresql+asyncpg://u:SENTINEL-DB-PASS-91@db/x",
    "HF_TOKEN": "hf_SENTINELenvtoken000000000000000000x9",
    "SETTINGS_ENCRYPTION_KEY": "SENTINEL-ENCRYPTION-KEY-000000000000000000",
    "WEIGHTS_TOKENS": '[{"label":"t","role":"trainer","sha256":"SENTINELDIGEST"}]',
    # 038 S2 (FTDD §5.8): the receiver's credentials and the client key's path.
    "TELEMETRY_HEADERS": "authorization=Bearer SENTINEL-TELEMETRY-HEADER",
    "TELEMETRY_TLS_CLIENT_KEY_FILE": "/run/secrets/SENTINEL-CLIENT-KEY.pem",
}


class TestRegistered:
    def test_the_live_app_serves_the_routes(self):
        """Read from the live app's own route table (its OpenAPI document), not the module."""
        from millm.main import create_app

        paths = create_app().openapi()["paths"]
        # ENH-006 D75: `PUT /api/environment` saves several settings at once.
        assert set(paths.get(BASE, {})) == {"get", "put"}, "environment routes not registered"
        assert set(paths.get(f"{BASE}/{{key}}", {})) == {"put", "delete"}
        assert set(paths.get(f"{BASE}/history", {})) == {"get"}


class TestListing:
    async def test_lists_every_field_grouped_with_effect(self, batch_db):
        async with client_for(app_with_db(batch_db)) as c:
            r = await c.get(BASE)
        assert r.status_code == 200, r.text
        data = r.json()["data"]
        # 30 (Feature 35) + 6 (Feature 36 probe sessions) + 5 (Feature 37 probe budget, capacity).
        # 039 S1: +8 runtime keys; 039 S2: +5 (the rule switch, budget and three limits).
        # 039 S3: +5 runtime (operator-code call bounds, calls per request, code bytes per
        # import) and +1 restart (the runner socket). ENH-008: +1 runtime (APPROVAL_TTL_HOURS).
        # ENH-006 D75: +21 runtime (every telemetry setting but the five environment-only ones).
        # ENH-006 D83: −4 runtime (the behaviour count and three recommendation ceilings, retired)
        # +1 runtime (PROBE_SLOWDOWN_BUDGET_PCT).
        assert data["counts"] == {"runtime": 78, "model_load": 10, "restart": 14,
                                  "environment": len(data["settings"]) - 102}
        rows = {row["key"]: row for row in data["settings"]}
        assert set(rows) == set(type(settings).model_fields)
        row = rows["MAX_PENDING_REQUESTS"]
        assert (row["group"], row["effect"], row["runtime_writable"], row["source"]) == (
            "runtime", "next queue admission", True, "env")
        assert row["label"] == "Queue length" and row["description"]
        assert (row["value"], row["env_value"], row["minimum"], row["maximum"]) == (10, 10, 1, 1000)
        assert rows["TORCH_COMPILE"]["group"] == "model_load"
        assert rows["TORCH_COMPILE"]["runtime_writable"] is False
        assert "restart millm-backend" in rows["KV_CACHE_MODE"]["effect"]
        assert rows["DEBUG"]["group"] == "environment"
        assert data["runtime_keys"][0] == "LOG_LEVEL"

    async def test_no_secret_appears_anywhere(self, batch_db, monkeypatch):
        """Whole-body search for a sentinel in each secret, set in the environment and on the
        singleton; each secret's row says only that it is set."""
        from pydantic import SecretStr

        for key, value in SENTINELS.items():
            monkeypatch.setenv(key, value)
            monkeypatch.setattr(settings, key, SecretStr(value) if key == "SETTINGS_ENCRYPTION_KEY"
                                else value)
        async with client_for(app_with_db(batch_db)) as c:
            listing = await c.get(BASE)
            await c.put(f"{BASE}/PROBE_MAX_ARMED", json={"value": 3})
            history = await c.get(f"{BASE}/history")
            refused = [await c.put(f"{BASE}/{key}", json={"value": "x"}) for key in SENTINELS]
        bodies = [listing.text, history.text] + [r.text for r in refused]
        for body in bodies:
            assert "SENTINEL" not in body, "a secret value reached a response"
        # 038 S2: a credential inside the receiver URL is not shown either (only its host/path).
        assert "SENTINEL" not in repr(settings)
        rows = {row["key"]: row for row in listing.json()["data"]["settings"]}
        for key in SENTINELS:
            assert rows[key]["secret"] is True and rows[key]["set"] is True
            assert rows[key]["value"] is None and rows[key]["env_value"] is None
        async with batch_db() as session:
            stored = json.dumps([(c.key, c.previous_value, c.new_value)
                                 for c in (await session.execute(select(EnvOverrideChange))).scalars()])
        assert "SENTINEL" not in stored and all(k not in stored for k in SENTINELS)

    async def test_the_receiver_url_is_shown_without_its_credentials(self, batch_db, monkeypatch):
        url = "https://user:SENTINEL-URL-PASSWORD@otlp.example:4318/base?api_key=SENTINEL-Q#f"
        monkeypatch.setenv("TELEMETRY_ENDPOINT", url)
        monkeypatch.setattr(settings, "TELEMETRY_ENDPOINT", url)
        async with client_for(app_with_db(batch_db)) as c:
            listing = await c.get(BASE)
        assert listing.status_code == 200
        assert "SENTINEL" not in listing.text
        rows = {row["key"]: row for row in listing.json()["data"]["settings"]}
        assert rows["TELEMETRY_ENDPOINT"]["value"] == "https://otlp.example:4318/base"
        assert rows["TELEMETRY_ENDPOINT"]["env_value"] == "https://otlp.example:4318/base"
        # Review R1 of S2: nor in the repr of the settings or of the resolved telemetry view (the
        # earlier sentinel test never put a credential in the URL, so its repr check could not
        # see this). Controls: `repr=True` on either field.
        from millm.telemetry.settings import load

        assert "SENTINEL" not in repr(settings)
        assert "SENTINEL" not in repr(load(settings.model_copy(update={
            "TELEMETRY_LOGS_ENABLED": True, "TELEMETRY_ENDPOINT": url})))

    async def test_an_unset_secret_reads_not_set(self, batch_db, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setattr(settings, "HF_TOKEN", None)
        async with client_for(app_with_db(batch_db)) as c:
            rows = {r["key"]: r for r in (await c.get(BASE)).json()["data"]["settings"]}
        assert rows["HF_TOKEN"]["set"] is False


class TestWrites:
    async def test_each_runtime_key_round_trips_through_the_singleton(self, batch_db):
        """One database for all 41 keys (a database per key cost ~3 s each)."""
        async with client_for(app_with_db(batch_db)) as c:
            for key in sorted(VALID_VALUES):
                before = getattr(settings, key)
                value = VALID_VALUES[key]
                r = await c.put(f"{BASE}/{key}", json={"value": value, "reason": f"try {key}"})
                assert r.status_code == 200, (key, r.text)
                row = r.json()["data"]
                assert (row["value"], row["env_value"], row["source"]) == (
                    value, before, "override"), key
                assert getattr(settings, key) == value, f"{key}: never reached the singleton"
                assert (row["updated_by"], row["reason"]) == ("operator", f"try {key}"), key
                r = await c.delete(f"{BASE}/{key}", params={"reason": f"back {key}"})
                assert r.status_code == 200, (key, r.text)
                assert r.json()["data"]["source"] == "env", key
                assert getattr(settings, key) == before, f"{key}: reset did not restore"
                changes = (await c.get(f"{BASE}/history", params={"limit": 2})).json()["data"]
                assert [(ch["key"], ch["action"], ch["reason"]) for ch in changes["changes"]] == [
                    (key, "reset", f"back {key}"), (key, "set", f"try {key}")], key
        async with batch_db() as session:
            assert (await session.execute(select(EnvOverride))).scalars().all() == []
            assert len((await session.execute(select(EnvOverrideChange))).scalars().all()) == (
                2 * len(VALID_VALUES))

    async def test_never_writable_by_name(self, batch_db):
        async with client_for(app_with_db(batch_db)) as c:
            for key in sorted(rs.NEVER_WRITABLE):
                before = getattr(settings, key)
                r = await c.put(f"{BASE}/{key}", json={"value": 1})
                d = await c.delete(f"{BASE}/{key}")
                assert r.status_code == 422, (key, r.status_code)
                assert r.json()["error"]["code"] == "SETTING_NOT_RUNTIME", key
                assert d.status_code == 422, key
                assert getattr(settings, key) == before, key

    async def test_refusals(self, batch_db):
        async with client_for(app_with_db(batch_db)) as c:
            unknown = await c.put(f"{BASE}/NOT_A_SETTING", json={"value": 1})
            bad = await c.put(f"{BASE}/MAX_PENDING_REQUESTS", json={"value": 0})
            pair = await c.put(f"{BASE}/CLUSTER_INTENSITY_MIN", json={"value": 9})
            gpu = await c.put(f"{BASE}/SCORING_FLOAT32_DEVICE", json={"value": "cuda:1"})
            not_overridden = await c.delete(f"{BASE}/MAX_PENDING_REQUESTS")
            long_reason = await c.put(f"{BASE}/PROBE_MAX_ARMED",
                                      json={"value": 3, "reason": "x" * 501})
        assert unknown.status_code == 404 and unknown.json()["error"]["code"] == "SETTING_NOT_FOUND"
        assert bad.status_code == 422 and "at least 1" in bad.json()["error"]["message"]
        assert pair.status_code == 422 and "must be below" in pair.json()["error"]["message"]
        assert gpu.status_code == 422
        assert not_overridden.status_code == 404
        assert not_overridden.json()["error"]["code"] == "SETTING_NOT_OVERRIDDEN"
        assert long_reason.status_code == 422
        assert settings.MAX_PENDING_REQUESTS == 10 and settings.PROBE_MAX_ARMED == 64  # 039 default

    async def test_history_limit_is_bounded(self, batch_db):
        async with client_for(app_with_db(batch_db)) as c:
            assert (await c.get(f"{BASE}/history", params={"limit": 0})).status_code == 422
            assert (await c.get(f"{BASE}/history", params={"limit": 201})).status_code == 422


class TestLogLevelNow:
    async def test_a_route_change_reaches_every_millm_logger(self, batch_db):
        probe = logging.getLogger("millm.services.inference_service")
        async with client_for(app_with_db(batch_db)) as c:
            await c.put(f"{BASE}/LOG_LEVEL", json={"value": "error"})
            assert not probe.isEnabledFor(logging.WARNING)
            await c.put(f"{BASE}/LOG_LEVEL", json={"value": "DEBUG"})
            assert probe.isEnabledFor(logging.DEBUG)
            await c.delete(f"{BASE}/LOG_LEVEL")
        assert probe.getEffectiveLevel() == getattr(logging, settings.LOG_LEVEL.upper())

    async def test_debug_while_serving_sends_no_exception_text(self, batch_db):
        """The spec's DEBUG defect: switched at run time, a 500 still carries only an error id."""
        app = app_with_db(batch_db)
        boom = APIRouter()

        @boom.get("/api/_boom")
        async def _boom():
            raise RuntimeError("SECRET-EXCEPTION-TEXT /etc/passwd")

        app.include_router(boom)
        async with client_for(app) as c:
            assert (await c.put(f"{BASE}/LOG_LEVEL", json={"value": "DEBUG"})).status_code == 200
            r = await c.get("/api/_boom")
        assert r.status_code == 500
        assert "SECRET-EXCEPTION-TEXT" not in r.text and "RuntimeError" not in r.text
