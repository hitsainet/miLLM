"""The `/v1` model resolver and the served name on the wire (032 FR-32.6.2-4, FR-32.6.7-9;
032 review 1.9, 5.3, 5.9; 033 review findings 7 and 17.5).

Real FastAPI app, real InferenceService over a tiny real Llama; only the database-backed
ModelService is a stand-in, and it answers by NAME like the repository does (an exact name
matches, anything else is unknown), so a request reaching the resident without a row by that
name is proven to have taken the resident path, not a lucky catch-all mock.

MUTATION CONTROLS (recorded in the W0b report):
  C4  the weights-origin carrier returns `Resolution(row, reload=True)` instead of the refusal
      -> test_a_weights_origin_carrier_whose_identity_moved_is_refused
  C7  `X-miLLM-Model` dropped from `post_generation`  -> the header tests
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from millm.api.routes.openai.model_resolution import (
    LOAD_THROUGH_WEIGHTS_SENTENCE,
    batch_model_resolver,
    resolve_requested_model,
)
from millm.core.errors import ModelNotFoundError
from millm.ml.model_loader import LoadedModelState
from millm.services import resident_identity
from millm.services.resident_identity import ResidentIdentity
from tests.unit.f25_fixtures import clear_loaded, make_service, word_model, word_tokenizer


def _row(name: str, model_id: int = 1, weights_origin=None) -> SimpleNamespace:
    from datetime import datetime

    return SimpleNamespace(
        id=model_id, name=name, architecture="text-generation", gguf_files=None,
        revision="abc", quantization=SimpleNamespace(value="FP16"), gguf_label="",
        weights_origin=weights_origin, repo_id=None, loaded_at=datetime(2026, 10, 6),
        created_at=datetime(2026, 10, 6), status="loaded",
    )


class _Service:
    """The ModelService surface the resolver and routes use, answering by exact name."""

    def __init__(self, rows):
        self.rows = {r.name: r for r in rows}
        self.by_id = {r.id: r for r in rows}
        self.find_model_by_name = AsyncMock(side_effect=lambda name: self.rows.get(name))
        self.load_model_and_wait = AsyncMock()
        self.get_locked_model = AsyncMock(return_value=None)
        self.get_available_models = AsyncMock(return_value=list(rows))

    async def get_model(self, model_id):
        if model_id not in self.by_id:
            raise ModelNotFoundError(f"Model with ID {model_id} not found")
        return self.by_id[model_id]


@pytest.fixture
def inference():
    svc = make_service(word_model(), word_tokenizer(), name="tiny")
    yield svc
    clear_loaded()


@pytest.fixture
def overlay():
    state = {"name": None}

    def fn():
        return state["name"]

    resident_identity.register_served_name_overlay(fn)
    try:
        yield state
    finally:
        with resident_identity._lock:
            resident_identity._overlays.remove(fn)


def _move_identity(name: str, origin: str = "checkpoint") -> None:
    """What 032's in-place load does: the resident identity moves off the carrier row."""
    LoadedModelState().current.weights_identity = ResidentIdentity(
        name=name, base_model="tiny", origin=origin, checkpoint_dir="/c/run/step-500",
        manifest_sha256="0" * 64, expected_parent=None, commits_at_identity=0,
    )


# ---------------------------------------------------------------- resolver cases


async def test_the_resident_by_its_served_name_needs_no_row_by_that_name(inference, overlay):
    overlay["name"] = "tiny:adapter-x"
    service = _Service([_row("tiny")])
    resolution = await resolve_requested_model("tiny:adapter-x", service, inference)
    assert resolution.resident and resolution.row.name == "tiny"
    assert resolution.served_name == "tiny:adapter-x"
    assert resolution.refusal is None


async def test_the_bare_base_name_still_means_the_resident(inference, overlay):
    overlay["name"] = "tiny:adapter-x"
    resolution = await resolve_requested_model("tiny", _Service([_row("tiny")]), inference)
    assert resolution.resident and resolution.row.name == "tiny"


async def test_a_checkpoint_identity_name_resolves_to_the_resident(inference):
    _move_identity("tiny:run:step-500")
    service = _Service([_row("tiny:run:step-400", weights_origin="checkpoint")])
    LoadedModelState().current.model_name = "tiny:run:step-400"
    resolution = await resolve_requested_model("tiny:run:step-500", service, inference)
    assert resolution.resident and resolution.row.name == "tiny:run:step-400"


async def test_an_unknown_name_is_the_existing_404(inference):
    resolution = await resolve_requested_model("nope", _Service([_row("tiny")]), inference)
    assert resolution.refusal is not None and resolution.refusal.status_code == 404
    body = json.loads(resolution.refusal.body)
    assert body["error"]["code"] == "model_not_found"
    assert body["error"]["message"] == (
        "The model 'nope' does not exist or has not been downloaded. "
        "Download it first using the Management API.")


async def test_a_weights_origin_carrier_whose_identity_moved_is_refused(inference):
    """Review 5.3: naming the carrier's OLD name must never reload its directory unverified."""
    LoadedModelState().current.model_name = "tiny:run:step-400"
    _move_identity("tiny:run:step-500")
    service = _Service([_row("tiny:run:step-400", weights_origin="checkpoint")])
    resolution = await resolve_requested_model("tiny:run:step-400", service, inference)
    assert resolution.refusal is not None and not resolution.reload
    assert resolution.refusal.status_code == 404
    body = json.loads(resolution.refusal.body)
    assert LOAD_THROUGH_WEIGHTS_SENTENCE in body["error"]["message"]
    assert "/api/weights/load" in body["error"]["message"]


async def test_a_plain_carrier_whose_identity_moved_is_refused_not_reloaded(inference):
    """D-38: naming the row after a save or load moved the identity off it is the unknown-model
    404, naming what is resident and the deliberate load; never an implicit reload."""
    _move_identity("tiny:run:step-500")
    # The carrier is the resident's own row: same id (2026-10-18 D56, the carrier is known by id).
    LoadedModelState().current.model_id = 7
    service = _Service([_row("tiny", model_id=7)])
    resolution = await resolve_requested_model("tiny", service, inference)
    assert not resolution.reload and not resolution.resident and resolution.row is None
    assert resolution.refusal_reason == "identity_moved"
    assert resolution.refusal.status_code == 404
    error = json.loads(resolution.refusal.body)["error"]
    assert error["code"] == "model_not_found" and error["param"] == "model"
    assert error["type"] == "invalid_request_error"
    assert "'tiny:run:step-500'" in error["message"]
    assert 'POST /api/weights/load {"model_id": 7}' in error["message"]
    assert error["message"] == resolution.refusal_message
    service.load_model_and_wait.assert_not_called()


async def test_the_identity_moved_refusal_has_the_unknown_model_shape(inference):
    """Same status, code, type and param as the unknown-model 404, so a client handling one
    handles the other."""
    unknown = await resolve_requested_model("nope", _Service([_row("tiny")]), inference)
    _move_identity("tiny:run:step-500")
    moved = await resolve_requested_model("tiny", _Service([_row("tiny")]), inference)
    a = json.loads(unknown.refusal.body)["error"]
    b = json.loads(moved.refusal.body)["error"]
    assert unknown.refusal.status_code == moved.refusal.status_code == 404
    keys = ("code", "type", "param")
    assert {k: a[k] for k in keys} == {k: b[k] for k in keys}


async def test_a_weights_origin_row_that_is_not_resident_is_refused(inference):
    service = _Service([_row("tiny"), _row("other:run:step-1", 2, weights_origin="checkpoint")])
    resolution = await resolve_requested_model("other:run:step-1", service, inference)
    assert resolution.refusal is not None and resolution.refusal_reason == "weights_origin"


async def test_a_plain_row_that_is_not_resident_loads_as_before(inference):
    service = _Service([_row("tiny"), _row("other", 2)])
    resolution = await resolve_requested_model("other", service, inference)
    assert resolution.row.name == "other" and not resolution.resident and not resolution.reload


async def test_nothing_resident_resolves_by_row(inference):
    clear_loaded()
    resolution = await resolve_requested_model("tiny", _Service([_row("tiny")]), inference)
    assert resolution.row.name == "tiny" and not resolution.resident


class TestBatchAdapter:
    """`validate_file(resolve_model=...)` keeps its signature (032 review 1.9)."""

    async def test_unknown_is_none_so_the_validator_writes_its_own_404(self, inference):
        resolve = batch_model_resolver(_Service([_row("tiny")]), inference)
        assert await resolve("nope") is None

    async def test_resident_served_name_returns_the_carrier(self, inference, overlay):
        overlay["name"] = "tiny:adapter-x"
        resolve = batch_model_resolver(_Service([_row("tiny")]), inference)
        assert (await resolve("tiny:adapter-x")).name == "tiny"

    async def test_a_refused_row_raises_a_millm_error(self, inference):
        service = _Service([_row("tiny"), _row("ck", 2, weights_origin="checkpoint")])
        with pytest.raises(ModelNotFoundError) as exc:
            await batch_model_resolver(service, inference)("ck")
        assert "/api/weights/load" in exc.value.message

    async def test_an_identity_moved_row_raises_the_same_refusal(self, inference):
        """D-38 through the batch adapter: every line naming the row is refused with the same
        message, never resolved to the row (which the runner would then load)."""
        _move_identity("tiny:run:step-500")
        LoadedModelState().current.model_id = 3  # the carrier is the resident's own row
        with pytest.raises(ModelNotFoundError) as exc:
            await batch_model_resolver(_Service([_row("tiny", model_id=3)]), inference)("tiny")
        assert 'POST /api/weights/load {"model_id": 3}' in exc.value.message
        assert "'tiny:run:step-500'" in exc.value.message
        assert exc.value.details["reason"] == "identity_moved"

    def test_the_runner_passes_the_adapter(self):
        """Wiring: the batch runner hands the validator THIS resolver, not the bare lookup."""
        import ast
        import inspect

        from millm.services.batch import runner

        tree = ast.parse(inspect.getsource(runner))
        calls = [
            kw.value for node in ast.walk(tree) if isinstance(node, ast.Call)
            for kw in node.keywords if kw.arg == "resolve_model"
        ]
        assert len(calls) == 1
        assert isinstance(calls[0], ast.Call) and calls[0].func.id == "batch_model_resolver"


# ---------------------------------------------------------------- routes


def _client(inference, service):
    from fastapi.testclient import TestClient

    from millm.api.dependencies import get_inference_service, get_model_service
    from millm.main import create_app

    app = create_app()
    app.dependency_overrides[get_model_service] = lambda: service
    app.dependency_overrides[get_inference_service] = lambda: inference
    return TestClient(app)


CHAT = {"messages": [{"role": "user", "content": "w1 w2"}], "max_tokens": 2,
        "temperature": 0.0}


class TestRoutes:
    def test_chat_naming_the_served_name_is_served_without_a_load(self, inference, overlay):
        overlay["name"] = "tiny:adapter-x"
        service = _Service([_row("tiny")])
        r = _client(inference, service).post(
            "/v1/chat/completions", json={"model": "tiny:adapter-x", **CHAT})
        assert r.status_code == 200, r.text
        assert r.json()["model"] == "tiny:adapter-x"
        assert r.headers["X-miLLM-Model"].startswith('"tiny:adapter-x";origin=model')
        service.load_model_and_wait.assert_not_called()

    @pytest.mark.parametrize("path, body", [
        ("/v1/chat/completions", CHAT),
        ("/v1/completions", {"prompt": "w1 w2", "max_tokens": 2, "temperature": 0.0}),
        ("/v1/embeddings", {"input": "w1 w2"}),
    ])
    def test_every_non_streaming_route_carries_the_header(self, inference, overlay, path, body):
        overlay["name"] = 'odd"name\\x'
        r = _client(inference, _Service([_row("tiny")])).post(
            path, json={"model": "tiny", **body})
        assert r.status_code == 200, r.text
        assert r.json()["model"] == 'odd"name\\x'
        assert r.headers["X-miLLM-Model"].startswith('"odd\\"name\\\\x";origin=model')

    def test_a_stream_carries_the_name_in_chunks_and_millm_model_and_no_header(
        self, inference, overlay,
    ):
        overlay["name"] = "tiny:adapter-x"
        client = _client(inference, _Service([_row("tiny")]))
        with client.stream("POST", "/v1/chat/completions",
                           json={"model": "tiny", "stream": True, **CHAT}) as r:
            assert r.status_code == 200
            assert "X-miLLM-Model" not in r.headers  # FR-32.6.8: it could not be truthful
            lines = [ln for ln in r.iter_lines() if ln.startswith("data: {")]
        payloads = [json.loads(ln[len("data: "):]) for ln in lines]
        assert payloads and all(p["model"] == "tiny:adapter-x" for p in payloads)
        terminal = [p for p in payloads if "millm_steering" in p]
        assert terminal[0]["millm_model"].startswith('"tiny:adapter-x";origin=model')

    def test_unknown_model_is_the_existing_404(self, inference):
        r = _client(inference, _Service([_row("tiny")])).post(
            "/v1/chat/completions", json={"model": "nope", **CHAT})
        assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_found"

    def test_a_refused_weights_row_is_never_loaded(self, inference):
        service = _Service([_row("tiny"), _row("ck", 2, weights_origin="checkpoint")])
        r = _client(inference, service).post("/v1/chat/completions", json={"model": "ck", **CHAT})
        assert r.status_code == 404
        assert "/api/weights/load" in r.json()["error"]["message"]
        service.load_model_and_wait.assert_not_called()

    def test_v1_models_lists_the_served_name_and_hides_idle_weights_rows(
        self, inference, overlay,
    ):
        overlay["name"] = "tiny:adapter-x"
        service = _Service([
            _row("tiny"), _row("other", 2), _row("ck", 3, weights_origin="checkpoint"),
        ])
        ids = [m["id"] for m in _client(inference, service).get("/v1/models").json()["data"]]
        assert ids == ["tiny:adapter-x", "other"]

    def test_v1_models_without_an_overlay_lists_the_row_name(self, inference):
        ids = [m["id"] for m in
               _client(inference, _Service([_row("tiny")])).get("/v1/models").json()["data"]]
        assert ids == ["tiny"]

    def test_v1_models_locked_lists_the_served_name(self, inference, overlay):
        overlay["name"] = "tiny:adapter-x"
        service = _Service([_row("tiny")])
        service.get_locked_model = AsyncMock(return_value=service.rows["tiny"])
        ids = [m["id"] for m in _client(inference, service).get("/v1/models").json()["data"]]
        assert ids == ["tiny:adapter-x"]

    def test_get_model_by_served_name(self, inference, overlay):
        overlay["name"] = "tiny:adapter-x"
        client = _client(inference, _Service([_row("tiny")]))
        r = client.get("/v1/models/tiny:adapter-x")
        assert r.status_code == 200 and r.json()["id"] == "tiny:adapter-x"
        assert client.get("/v1/models/nope").status_code == 404

    def test_inline_steering_naming_the_served_name_is_not_called_not_resident(
        self, inference, overlay,
    ):
        """`refuse_inline_steering_before_load` reads the same resident names."""
        from millm.api.routes.openai.load_policy import refuse_inline_steering_before_load

        overlay["name"] = "tiny:adapter-x"
        request = MagicMock(model="tiny:adapter-x")
        request.steering.features = [1]
        refuse_inline_steering_before_load(request, inference)  # does not raise


# ------------------------------------------- the carrier is the resident's row BY ID (D56)


async def test_the_carrier_is_the_resident_id_not_a_namesake(inference):
    """2026-10-18 (D56): rows 7 (Q8) and 13 (FP16) of one repo shared a name and the carrier was
    whichever the name lookup returned. Here the lookup by name returns a namesake (id 9); the
    resident is row 1, and its row is the one the resolution carries."""
    service = _Service([_row("tiny", model_id=1), _row("tiny", model_id=9)])
    assert (await service.find_model_by_name("tiny")).id == 9  # the namesake answers by name

    resolution = await resolve_requested_model("tiny", service, inference)

    assert resolution.resident and resolution.row.id == 1


async def test_a_namesake_of_a_moved_resident_is_not_its_carrier(inference):
    """The identity moved off row 7 (a checkpoint was loaded over it); a DIFFERENT row 9 that
    shares the name is not the carrier, so naming it is not D-38's refusal: it is that row,
    loaded on demand like any other."""
    _move_identity("tiny:run:step-500")
    LoadedModelState().current.model_id = 7
    service = _Service([_row("tiny", model_id=9)])

    resolution = await resolve_requested_model("tiny", service, inference)

    assert resolution.refusal is None and not resolution.resident
    assert resolution.row.id == 9
