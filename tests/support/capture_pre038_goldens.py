"""Pre-038 goldens: one golden per probe surface, captured from the REAL routes (038 FTASKS 0.4).

Captured from `af328581` (after the handoff commit `129a3e29`) BEFORE any S1 code change, so the
files under `tests/fixtures/goldens/pre038/` record what every surface said before the result
contract existed. Feature 038 is additive (FR-38.3): every key present here must still be present,
with the same value, after S1 — except the three free-text reason forms FR-38.5 turns into codes.

The harness is shared. `capture_all()` drives the same scenarios the goldens test drives on the
current code, so a golden is always compared with the payload of the identical request:

* a tiny real Llama on the CPU (`tests.unit.f25_fixtures`), real probe rows imported through
  `ProbeService`, armed through `armed_probe_from_row` on a real hook;
* the real app (`create_app`) over a real in-memory SQLite database behind
  `millm.db.base.async_session_factory`;
* the socket captured at `progress_emitter.emit_probe_event`, the only patched seam (039: at
  `emit_probe_results`, one message per request, flattened to the per-result list it carries).

Volatile values (completion ids, timestamps, row ids, random probe and session ids) are replaced
by placeholders in `normalise`, so a golden is byte-stable across runs.

Run (writes the goldens; review the diff before committing):

    venv/bin/python -m tests.support.capture_pre038_goldens
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Optional
from unittest.mock import AsyncMock, patch

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "goldens" / "pre038"

#: Every surface FR-38.3 moves onto the builder.
SURFACES = (
    "header", "stream_chunk", "body", "event_rows", "socket", "events_list", "event_detail",
    "session_cells", "score", "batch_line",
)

#: The behaviour keys the fixtures carry. Placeholders, never a real behaviour (FR-38.1).
ALPHA_KEY = "b-alpha"
BETA_KEY = "b-beta"


@dataclass
class ProbeSpec:
    name: str
    layer: int
    threshold: Optional[float]
    weight: float
    behaviour: str
    windows: tuple[str, ...] = ("all",)
    #: `raise` (encoder_failed), `wrong_width` (head_mismatch) or None.
    encoder: Optional[str] = None
    #: Remove the layer's hook after arming, so the probe never observes (`no_scored_tokens`).
    unhook: bool = False


ALPHA = ProbeSpec("probe-alpha", layer=0, threshold=0.0, weight=0.25, behaviour=ALPHA_KEY)
BETA = ProbeSpec("probe-beta", layer=1, threshold=None, weight=-0.5, behaviour=BETA_KEY,
                 windows=("all", "response"))


@dataclass
class Scenario:
    name: str
    probes: list[ProbeSpec]
    surfaces: tuple[str, ...]
    request: dict[str, Any] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    #: Applied around the request: a context manager factory taking the inference service.
    patches: Optional[Callable[[Any], Any]] = None


def _prompt_length(value_fn: Callable[[Any, int], None]) -> Callable[[Any], Any]:
    @contextlib.contextmanager
    def cm(svc: Any) -> Iterator[None]:
        real = svc._probe_note_prompt_length

        def note(context: Any, n: int) -> None:
            if context is None:
                return real(context, n)
            value_fn(context, int(n))

        svc._probe_note_prompt_length = note
        try:
            yield
        finally:
            svc._probe_note_prompt_length = real
    return cm


def _no_last_user(svc: Any) -> Any:
    @contextlib.contextmanager
    def cm() -> Iterator[None]:
        real = svc._probe_note_last_user_span
        svc._probe_note_last_user_span = lambda *a, **k: None
        try:
            yield
        finally:
            svc._probe_note_last_user_span = real
    return cm()


def _cbm(svc: Any) -> Any:
    @contextlib.contextmanager
    def cm() -> Iterator[None]:
        from millm.core.config import settings
        from tests.support.generation_entry_points import EventLog, FakeCBM

        old = settings.PROBE_FORCE_SERIAL
        settings.PROBE_FORCE_SERIAL = False
        svc._cbm_backend = FakeCBM(EventLog())
        try:
            yield
        finally:
            svc._cbm_backend = None
            settings.PROBE_FORCE_SERIAL = old
    return cm()


_ALL = SURFACES
_ROWS = ("header", "body", "event_rows", "socket", "events_list")

SCENARIOS: list[Scenario] = [
    Scenario("armed", [ALPHA, BETA], _ALL, headers={"X-miLLM-Session": "golden"}),
    Scenario("unarmed", [], ("header", "stream_chunk", "body", "batch_line")),
    Scenario("judge_traffic", [ALPHA], _ROWS + ("stream_chunk",),
             headers={"X-miLLM-Origin": "judge"}),
    Scenario("batched_request", [ALPHA], _ROWS, request={"n": 2}),
    Scenario("continuous_batching", [ALPHA], _ROWS, patches=_cbm),
    Scenario("encoder_failed", [ProbeSpec(**{**ALPHA.__dict__, "encoder": "raise"})], _ROWS),
    Scenario("head_mismatch", [ProbeSpec(**{**ALPHA.__dict__, "encoder": "wrong_width"})], _ROWS),
    Scenario("prompt_boundary_invalid", [ALPHA], _ROWS,
             patches=_prompt_length(lambda ctx, n: ctx.set_prompt_length(-5))),
    Scenario("prompt_boundary_conflict", [ALPHA], _ROWS,
             patches=_prompt_length(
                 lambda ctx, n: (ctx.set_prompt_length(n), ctx.set_prompt_length(n + 3)))),
    Scenario("prompt_boundary_unknown", [ProbeSpec(**{**ALPHA.__dict__, "windows": ("prompt",)})],
             _ROWS, patches=_prompt_length(lambda ctx, n: None)),
    Scenario("last_user_span_unknown",
             [ProbeSpec(**{**ALPHA.__dict__, "windows": ("last_user",)})], _ROWS,
             patches=_no_last_user),
    Scenario("no_scored_tokens", [ProbeSpec(**{**ALPHA.__dict__, "unhook": True})], _ROWS),
]


# ── normalisation ────────────────────────────────────────────────────────────────

_PATTERNS = [
    (re.compile(r"chatcmpl-[A-Za-z0-9]+"), "<completion_id>"),
    (re.compile(r"\bcmpl-[A-Za-z0-9]+"), "<completion_id>"),
    (re.compile(r"batch_req_[0-9a-f]+"), "<batch_req>"),
    (re.compile(r"\breq_[0-9a-f]{24}\b"), "<req>"),
    (re.compile(r"\bps_[0-9a-f]+\b"), "<session>"),
    (re.compile(r"\bbatch_[0-9a-f]{12,}\b"), "<batch>"),
]
_VOLATILE_KEYS = {"created", "created_at", "at", "checked_at", "event_id", "updated_at"}


def normalise(value: Any, probe_ids: dict[str, str]) -> Any:
    """Placeholders for every value that differs between two runs of the same request."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key in _VOLATILE_KEYS and item is not None:
                out[key] = f"<{key}>"
            elif key == "id" and isinstance(item, int):
                out[key] = "<row_id>"
            else:
                out[key] = normalise(item, probe_ids)
        return out
    if isinstance(value, list):
        return [normalise(item, probe_ids) for item in value]
    if isinstance(value, str):
        for real, placeholder in probe_ids.items():
            value = value.replace(real, placeholder)
        for pattern, placeholder in _PATTERNS:
            value = pattern.sub(placeholder, value)
        return value
    return value


# ── the harness ──────────────────────────────────────────────────────────────────


def _definition(spec: ProbeSpec) -> dict:
    from tests.unit.score_fixtures import tiny_definition

    doc = tiny_definition(spec.name, layer=spec.layer, threshold=spec.threshold,
                          weights=[spec.weight] * 16)
    doc["concept"] = None
    doc["behaviour"] = {"key": spec.behaviour, "name": spec.behaviour.replace("-", " "),
                        "version": None}
    return copy.deepcopy(doc)


def _encoder(kind: Optional[str]) -> Any:
    import torch

    if kind == "raise":
        def broken(_x: Any) -> Any:
            raise RuntimeError("encoder exploded: " + "detail " * 40)
        return broken
    if kind == "wrong_width":
        return lambda x: torch.zeros(x.shape[0], 3)
    return None


class Harness:
    """One fresh app, database and model per scenario (state never leaks between them)."""

    def __init__(self) -> None:
        self.socket: list[dict] = []
        self.probe_ids: dict[str, str] = {}

    async def __aenter__(self) -> "Harness":
        from sqlalchemy import event
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

        import millm.api.dependencies as deps
        import millm.db.base as db_base
        from millm.db.base import Base
        from millm.services import probe_sessions
        from millm.services.probe_runtime import ProbeRuntimeState
        from millm.services.sae_service import AttachedSAEState
        from tests.unit.f25_fixtures import make_client, make_service, model_row
        from tests.unit.f25_fixtures import word_model, word_tokenizer
        from tests.unit.score_fixtures import tiny_identity

        self._stack = contextlib.ExitStack()
        from millm.rules import registry as rule_registry

        rule_registry.forget_for_tests()  # 039 S2: no attachment leaks between scenarios
        AttachedSAEState().reset_for_tests()
        ProbeRuntimeState.reset_for_tests()
        probe_sessions._forget_for_tests()
        engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)

        @event.listens_for(engine.sync_engine, "connect")
        def _fk(dbapi_connection: Any, _record: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        self.engine = engine
        self.factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        self._stack.enter_context(patch.object(db_base, "async_session_factory", self.factory))
        self._stack.enter_context(patch.object(deps, "_probe_event_service", None, create=True))
        self.model, self.tokenizer = word_model(), word_tokenizer()
        self._stack.enter_context(
            patch("millm.services.inference_service.torch.cuda.is_available", return_value=False)
        )
        self.svc = make_service(self.model, self.tokenizer)
        self._stack.enter_context(patch(
            "millm.services.probe_arm_bridge.loaded_identity",
            AsyncMock(return_value=(tiny_identity(), self.model, self.tokenizer)),
        ))
        from millm.sockets.progress import progress_emitter

        # 039 (TD39-28): one `probe:results` message per request. Flattened here into the
        # per-result list the pre-038 goldens recorded from `probe:event`, so the same golden
        # compares; `self.socket_messages` keeps the messages as emitted.
        self.socket_messages: list[dict] = []

        def _results(payload: dict) -> None:
            self.socket_messages.append(copy.deepcopy(payload))
            self.socket.extend(copy.deepcopy(payload.get("results") or []))

        self._stack.enter_context(patch.object(progress_emitter, "emit_probe_results", _results))
        self.client, _ = make_client(self.svc, model_row())
        from millm.api.dependencies import get_db

        async def _db() -> Any:
            async with self.factory() as session:
                yield session

        self.client.app.dependency_overrides[get_db] = _db
        return self

    async def __aexit__(self, *exc: Any) -> None:
        from millm.services import probe_sessions
        from millm.services.probe_runtime import ProbeRuntimeState
        from millm.services.sae_service import AttachedSAEState
        from tests.unit.f25_fixtures import clear_loaded

        self._stack.close()
        from millm.rules import registry as rule_registry

        rule_registry.forget_for_tests()
        ProbeRuntimeState.reset_for_tests()
        AttachedSAEState().reset_for_tests()
        probe_sessions._forget_for_tests()
        clear_loaded()
        await self.engine.dispose()

    async def arm(self, specs: list[ProbeSpec]) -> list[Any]:
        from millm.db.repositories.probe_repository import ProbeRepository
        from millm.services.probe_arming import armed_probe_from_row
        from millm.services.probe_runtime import ProbeRuntimeState
        from millm.services.probe_service import ProbeService

        rows = []
        state = ProbeRuntimeState()
        for i, spec in enumerate(specs):
            async with self.factory() as session:
                row = await ProbeService(ProbeRepository(session)).import_definition(
                    _definition(spec), raw_bytes=1, on_conflict="rename", origin="file"
                )
            self.probe_ids[row.id] = f"<probe:{spec.name}>"
            armed = armed_probe_from_row(row, encoder=_encoder(spec.encoder),
                                         windows=list(spec.windows))
            state.arm(armed, self.model)
            if spec.unhook:
                handle = state._handles.pop(spec.layer)
                state._hooker.remove(handle)
            rows.append(row)
        return rows

    def _fresh_socket_throttle(self) -> None:
        import millm.api.dependencies as deps

        deps._probe_event_service = None

    def post_chat(self, body: dict, headers: dict[str, str]) -> Any:
        self._fresh_socket_throttle()
        return self.client.post("/v1/chat/completions", json=body, headers=headers)

    def stream_chat(self, body: dict, headers: dict[str, str]) -> str:
        self._fresh_socket_throttle()
        with self.client.stream("POST", "/v1/chat/completions", json={**body, "stream": True},
                                headers=headers) as response:
            assert response.status_code == 200, response.read()
            return "".join(response.iter_text())

    async def event_rows(self, request_id: str) -> list[dict]:
        from sqlalchemy import select

        from millm.db.models.probe import ProbeEvent

        async with self.factory() as session:
            rows = (await session.execute(
                select(ProbeEvent).where(ProbeEvent.request_id == request_id)
                .order_by(ProbeEvent.id)
            )).scalars().all()
        return [
            {c.name: _jsonable(getattr(row, c.name)) for c in ProbeEvent.__table__.columns}
            for row in rows
        ]

    async def batch_line(self, body: dict) -> dict:
        from millm.services.batch.executors import RowExecutor

        row = SimpleNamespace(line_no=1, custom_id="golden-1", kind="chat")
        line = json.dumps({"custom_id": "golden-1", "method": "POST",
                           "url": "/v1/chat/completions", "body": body}).encode()

        async def read_line(_row: Any) -> bytes:
            return line

        from tests.unit.f25_fixtures import model_row

        executor = RowExecutor(
            batch_id="batch_0123456789abcdef", endpoint="/v1/chat/completions",
            inference=self.svc, model_row=model_row(), read_line=read_line,
            session_factory=self.factory,
        )
        self._fresh_socket_throttle()
        result = await executor.run_one(row)
        return result.result


def _jsonable(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _stream_chunk(text: str) -> Optional[dict]:
    """The terminal probe chunk of an SSE body, or None when the stream carried none."""
    for line in text.splitlines():
        if line.startswith("data: ") and "millm_probe_verdicts" in line:
            return json.loads(line[len("data: "):])
    return None


def _stream_shape(text: str) -> list[Any]:
    """The whole stream, chunk by chunk (the unarmed case must be byte-identical)."""
    out: list[Any] = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        data = line[len("data: "):]
        out.append(data if data == "[DONE]" else json.loads(data))
    return out


BODY = {"model": "tiny", "messages": [{"role": "user", "content": "w1 w2 w3"}],
        "max_tokens": 3, "temperature": 0.0}


async def capture(scenario: Scenario) -> dict[str, Any]:
    """Every requested surface for one scenario, normalised."""
    out: dict[str, Any] = {}
    async with Harness() as h:
        rows = await h.arm(scenario.probes)
        body = {**BODY, **scenario.request}
        cm = scenario.patches(h.svc) if scenario.patches else contextlib.nullcontext()
        with cm:
            response = h.post_chat(body, scenario.headers)
            assert response.status_code == 200, response.text
            request_id = response.json()["id"]
            if "header" in scenario.surfaces:
                out["header"] = response.headers.get("X-miLLM-Probe-Verdicts")
            if "body" in scenario.surfaces:
                out["body"] = response.json()
            if "event_rows" in scenario.surfaces:
                out["event_rows"] = await h.event_rows(request_id)
            if "socket" in scenario.surfaces:
                out["socket"] = list(h.socket)
            if "events_list" in scenario.surfaces:
                listed = h.client.get("/api/probes/events", params={"request_id": request_id})
                out["events_list"] = listed.json()["data"]
            if "event_detail" in scenario.surfaces:
                listed = h.client.get("/api/probes/events", params={"request_id": request_id})
                out["event_detail"] = [
                    h.client.get(f"/api/probes/events/{e['id']}").json()["data"]
                    for e in sorted(listed.json()["data"], key=lambda e: e["id"])
                ]
            if "session_cells" in scenario.surfaces:
                session_id = response.headers.get("X-miLLM-Session-Id")
                detail = h.client.get(f"/api/probes/sessions/{session_id}").json()["data"]
                out["session_cells"] = [cell for turn in detail["turns"]
                                        for cell in turn["verdicts"]]
            if "stream_chunk" in scenario.surfaces:
                text = h.stream_chat(body, scenario.headers)
                out["stream_chunk"] = (
                    _stream_chunk(text) if scenario.probes else _stream_shape(text)
                )
        if "score" in scenario.surfaces:
            scored = h.client.post("/api/probes/score", json={
                "probe_ids": [r.id for r in rows],
                "inputs": [{"token_ids": [0, 2, 6, 7, 8], "prompt_tokens": 3}],
                "windows": ["all", "prompt"],
            })
            assert scored.status_code == 200, scored.text
            out["score"] = scored.json()["data"]
        if "batch_line" in scenario.surfaces:
            out["batch_line"] = await h.batch_line(body)
        return normalise(out, h.probe_ids)


async def capture_all() -> dict[str, dict[str, Any]]:
    return {scenario.name: await capture(scenario) for scenario in SCENARIOS}


def golden_path(scenario: str) -> Path:
    return GOLDEN_DIR / f"{scenario}.json"


def main() -> None:
    import logging

    logging.disable(logging.CRITICAL)
    captured = asyncio.run(capture_all())
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for name, surfaces in captured.items():
        golden_path(name).write_text(json.dumps(surfaces, indent=2, sort_keys=True) + "\n")
        print(f"{name}: {', '.join(sorted(surfaces))}")


if __name__ == "__main__":
    main()
