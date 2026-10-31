"""D-35: a vendored checkpoint-manifest schema miLLM cannot use is NOT fatal at startup.

Serving starts; the failure is logged at error level naming the offending fields; every 032 save
and load refuses `INVALID_MANIFEST` with `details.reason == "schema_ownership"` until it is fixed.

The bad schema is REAL: a copy of the vendored file with one property whose owner cannot be
derived (neither the trainer sentence nor the writer sentence), so the real
`assert_schema_ownership` raises the real `SchemaOwnershipError`. No stand-in raises for it.

Mutation controls, each red here:
  lifespan calls `assert_schema_ownership()` raw (fatal again) -> TestLifespan.*
  the save refusal removed                                     -> TestService.test_save_refuses
  the load refusal removed                                     -> TestService.test_load_refuses
  the startup log dropped                                      -> TestStartupCheck.test_logs_*
"""

from __future__ import annotations

import hashlib
import json
from unittest.mock import AsyncMock

import pytest

import millm.services.weights_checkpoint_service as svc_module
from millm.api.schemas.weights_checkpoints import LoadRequest
from millm.core.config import settings
from millm.core.errors import InvalidManifestError
from millm.services import checkpoint_manifest
from millm.services.weights_checkpoint_service import (
    WeightsCheckpointService,
    check_schema_ownership_on_startup,
    schema_ownership_error,
)
from tests.unit.batch_fixtures import batch_db  # noqa: F401 - a fixture
from tests.unit.checkpoint_service_fixtures import (  # noqa: F401 - harness is a fixture
    ATTACHMENT_ID,
    OPERATOR,
    TRAINER,
    attach_holder,
    harness,
    manifest_fields,
    resident_llama,
)

ORPHAN = "w3a_orphan_field"
TOKEN = "trainer-token-for-d35"


def _clear_caches() -> None:
    for name in ("_load_schema_cached", "ownership", "_projection", "_full_validator",
                 "_projection_validator"):
        getattr(checkpoint_manifest, name).cache_clear()


@pytest.fixture
def bad_schema(tmp_path, monkeypatch):
    """The vendored schema plus one top-level property nobody owns."""
    schema = json.loads(checkpoint_manifest.SCHEMA_PATH.read_text(encoding="utf-8"))
    definition = checkpoint_manifest.manifest_definition(schema)
    definition["properties"][ORPHAN] = {"type": "string", "description": "Nobody says who."}
    path = tmp_path / "checkpoint-manifest-v1.json"
    path.write_text(json.dumps(schema), encoding="utf-8")
    monkeypatch.setattr(checkpoint_manifest, "SCHEMA_PATH", path)
    _clear_caches()
    yield path
    monkeypatch.undo()
    _clear_caches()


class Recorder:
    """structlog stand-in (see test_probe_scoring TestPrivacy for why not capture_logs)."""

    def __init__(self) -> None:
        self.records: list[tuple[str, str, dict]] = []

    def __getattr__(self, level):
        return lambda event, **kw: self.records.append((level, event, kw))


class TestStartupCheck:
    def test_the_vendored_schema_is_usable(self):
        assert schema_ownership_error() is None
        assert check_schema_ownership_on_startup() is None

    def test_logs_at_error_level_naming_the_field_and_never_raises(self, bad_schema,
                                                                   monkeypatch):
        recorder = Recorder()
        monkeypatch.setattr(svc_module, "logger", recorder)
        error = check_schema_ownership_on_startup()
        assert error is not None and "SchemaOwnershipError" in error and ORPHAN in error
        errors = [(e, kw) for lvl, e, kw in recorder.records if lvl == "error"]
        assert len(errors) == 1
        event, kw = errors[0]
        assert event == "weights_schema_ownership_failed" and ORPHAN in kw["error"]

    def test_an_unreadable_schema_is_the_same_refusal(self, tmp_path, monkeypatch):
        monkeypatch.setattr(checkpoint_manifest, "SCHEMA_PATH", tmp_path / "missing.json")
        _clear_caches()
        try:
            assert schema_ownership_error() is not None
            assert check_schema_ownership_on_startup() is not None  # and does not raise
        finally:
            monkeypatch.undo()
            _clear_caches()


class TestService:
    async def test_save_refuses(self, harness, bad_schema):
        h = harness
        resident_llama()
        attach_holder(h.weights, h.clock)
        with pytest.raises(InvalidManifestError) as info:
            await WeightsCheckpointService(h.models).save(TRAINER, ATTACHMENT_ID,
                                                          manifest_fields())
        assert info.value.details["reason"] == "schema_ownership"
        assert info.value.details["operation"] == "save"
        assert ORPHAN in info.value.details["errors"][0]["message"]
        assert not any(h.root.iterdir())  # nothing written

    async def test_load_refuses(self, harness, bad_schema):
        h = harness
        loaded = resident_llama()
        with pytest.raises(InvalidManifestError) as info:
            await WeightsCheckpointService(h.models).load(OPERATOR, LoadRequest(model_id=1))
        assert info.value.details["reason"] == "schema_ownership"
        assert info.value.details["operation"] == "load"
        assert h.names() == [] and h.barrier.snapshot().weights_state == "clean"
        from millm.ml.model_loader import LoadedModelState

        assert LoadedModelState().current is loaded


class TestLifespan:
    async def test_serving_starts_and_the_routes_refuse(self, bad_schema, monkeypatch,
                                                        batch_db, tmp_path):
        """The app starts through the REAL lifespan with the real (bad) schema check; serving
        answers; save and load answer 400 INVALID_MANIFEST reason schema_ownership."""
        import httpx

        import millm.db.base as db_base
        import millm.main as main

        monkeypatch.setattr(db_base, "async_session_factory", batch_db)
        monkeypatch.setattr(main, "start_batch_api", AsyncMock())
        monkeypatch.setattr(main, "stop_batch_api", AsyncMock())
        monkeypatch.setattr(main, "disarm_probes_on_startup", AsyncMock(return_value=[]))
        monkeypatch.setattr(main.settings, "AUTO_LOAD_MODEL", None, raising=False)
        monkeypatch.setattr(main, "remove_stale_temp_dirs", lambda: [])
        monkeypatch.setattr(settings, "MODEL_CACHE_DIR", str(tmp_path / "cache"))
        monkeypatch.setattr(settings, "WEIGHTS_TOKENS", json.dumps([{
            "label": "forge", "role": "trainer",
            "sha256": hashlib.sha256(TOKEN.encode()).hexdigest()}]))
        recorder = Recorder()
        monkeypatch.setattr(svc_module, "logger", recorder)
        app = main.create_app()
        auth = {"Authorization": f"Bearer {TOKEN}"}
        async with main.lifespan(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                         base_url="http://test") as client:
                status = await client.get("/api/weights/status")
                save = await client.post("/api/weights/save", headers=auth,
                                         json={"manifest": {"run_id": "r", "step": 1}})
                load = await client.post("/api/weights/load", headers=auth,
                                         json={"model_id": 1})
        assert status.status_code == 200  # serving started
        assert [e for lvl, e, _ in recorder.records if lvl == "error"] == [
            "weights_schema_ownership_failed"]
        for response, operation in ((save, "save"), (load, "load")):
            assert response.status_code == 400, response.text
            error = response.json()["error"]
            assert error["code"] == "INVALID_MANIFEST"
            assert error["details"]["reason"] == "schema_ownership"
            assert error["details"]["operation"] == operation
