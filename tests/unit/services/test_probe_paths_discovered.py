"""Every generation path, DISCOVERED, opens a probe context or records why it cannot (FR-27.9).

⚠ **THE LIST IS NOT THE AUTHORITY.** `test_probe_wiring.py` named four methods and missed three:
batched chat and both continuous-batching non-streaming paths reached `generate` with a probe
armed and recorded nothing at all — no header, no event, no reason (BRD-04 §5.13). A hand-kept
list can only ever hold the paths its author remembered, so this guard derives the paths from
`InferenceService` itself:

1. **Discovery (AST).** A *generation site* is a method whose body — nested closures included —
   calls, or passes as a callable, a generation primitive. Passing counts because
   `asyncio.to_thread(self._generate_sync, …)` and `Thread(target=self._generate_in_thread)` are
   how the real paths reach them, and because `_llamacpp_text_completion` reaches
   `self._model.create_completion` only through a local `_complete`.
2. **Entry points.** The `self.<method>` call graph; the public coroutines from which a site is
   reachable. Asserted non-empty and to contain the three known ones, so a parser that finds
   nothing cannot pass by checking nothing.
3. **Behaviour.** Each site is DRIVEN with a probe armed. Primitives are wrapped (the real ones
   run on a tiny real model where possible) and every event — site entered, probe context opened,
   generation reached, event recorded — goes to one lock-guarded log. A log rather than a context
   lookup at generation time: `_generate_in_thread` runs on a plain `Thread`, which sees neither
   the caller's ContextVars nor a reliable stack.
4. **The scenario table is checked for EQUALITY against discovery.** A new generation site fails
   red until somebody writes the request that reaches it.

Mutations this catches (FTID §8): deleting any path's `_probe_begin`/`_probe_begin_detached`
(M1–M10), and a new method that calls `self._generate_sync` with no scenario (M11).
"""

from __future__ import annotations

import ast
import functools
import inspect
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

import millm.services.inference_service as inference_module
from millm.api.schemas.openai import ChatCompletionRequest, ChatMessage, TextCompletionRequest
from millm.ml.model_loader import ENGINE_LLAMACPP, LoadedModel, LoadedModelState
from millm.ml.probe_head import ProbeHead
from millm.services.inference_service import InferenceService
from millm.services.probe_runtime import ArmedProbe, ProbeRuntimeState
from tests.unit.f25_fixtures import clear_loaded, make_service, word_model, word_tokenizer

from tests.support.generation_entry_points import (  # noqa: E402,F401 - shared (FR-27.9)
    EXEMPT,
    KNOWN_ENTRY_POINTS,
    PRIMITIVE_ATTRS,
    PRIMITIVE_METHODS,
    SCENARIOS,
    EventLog,
    FakeCBM,
    FakeLlama,
    _chat,
    _class_methods,
    _is_primitive,
    _is_self_attr,
    _referenced,
    _text,
    discover_entry_points,
    discover_sites,
)


# ── discovery ────────────────────────────────────────────────────────────────────


class TestDiscovery:
    def test_sites_are_found_and_include_the_ones_the_old_list_missed(self):
        sites = discover_sites()
        assert sites, "discovery found nothing — a broken parser must not pass by checking nothing"
        for missed in ("_generate_batch_chunk", "_cbm_chat_completion", "_cbm_text_completion"):
            assert missed in sites, f"{missed} reaches generation and was not discovered"

    def test_nested_closures_are_walked(self):
        """`_llamacpp_text_completion` reaches `create_completion` only inside `_complete`."""
        assert "_llamacpp_text_completion" in discover_sites()
        assert "_llamacpp_stream_chat_completion" in discover_sites()

    def test_entry_points_are_found(self):
        entries = discover_entry_points()
        assert entries, "no entry points discovered"
        assert KNOWN_ENTRY_POINTS <= entries, f"missing: {KNOWN_ENTRY_POINTS - entries}"

    def test_no_exempt_method_is_a_generation_site(self):
        sites = discover_sites()
        assert not (set(EXEMPT) & sites), f"exempt methods that generate: {set(EXEMPT) & sites}"
        for name, reason in EXEMPT.items():
            assert reason.strip(), f"exemption {name} carries no reason"

    def test_a_primitive_passed_as_a_callable_counts(self):
        node = ast.parse(
            "async def f(self):\n    await asyncio.to_thread(self._generate_sync, kw)\n"
        ).body[0]
        assert _referenced(node, _is_primitive)

    def test_a_comment_naming_a_primitive_does_not_count(self):
        node = ast.parse('def f(self):\n    "self._generate_sync(x)"\n    return 1\n').body[0]
        assert not _referenced(node, _is_primitive)


# ── the behavioural harness ──────────────────────────────────────────────────────


def test_the_scenario_table_equals_discovery():
    """⚠ EQUALITY, not a subset. A new site fails here until someone writes a way to reach it."""
    sites = discover_sites()
    assert set(SCENARIOS) == sites, (
        f"sites with no scenario: {sorted(sites - set(SCENARIOS))}; "
        f"scenarios for no site: {sorted(set(SCENARIOS) - sites)}"
    )


def _probe(d: int = 16) -> ArmedProbe:
    return ArmedProbe(
        probe_id="pr_guard", name="guard", head=ProbeHead(weight=torch.ones(d), bias=0.0, layer=0),
        rule="mean", scope="all", layer=0, rung=2, rung_language="detects on unseen tasks",
        threshold=1.0, windows=("all",),
    )


@pytest.fixture
def harness(monkeypatch):
    ProbeRuntimeState.reset_for_tests()
    log = EventLog()

    @asynccontextmanager
    async def no_db():  # the record path must not reach a database here
        raise RuntimeError("no database in the path guard")
        yield  # pragma: no cover

    monkeypatch.setattr("millm.db.base.async_session_factory", no_db)
    yield log
    ProbeRuntimeState.reset_for_tests()
    clear_loaded()


def _build(log: EventLog, scenario: dict, monkeypatch) -> InferenceService:
    model, tokenizer = word_model(), word_tokenizer()
    svc = make_service(model, tokenizer)
    # The probe is armed on a real model so the serial paths score through a real hook.
    ProbeRuntimeState().arm(_probe(), model)
    if scenario.get("cbm"):
        from millm.core.config import settings

        monkeypatch.setattr(settings, "PROBE_FORCE_SERIAL", False)
        svc._cbm_backend = FakeCBM(log)
    if scenario["engine"] == "llamacpp":
        LoadedModelState().set(LoadedModel(
            model_id=1, model_name="tiny", model=FakeLlama(log), tokenizer=None,
            loaded_at=datetime(2026, 10, 6), engine=ENGINE_LLAMACPP,
        ))
    _instrument(svc, log)
    return svc


def _instrument(svc: InferenceService, log: EventLog) -> None:
    for name in ("_generate_sync", "_generate_in_thread"):
        real = getattr(svc, name)

        def primitive(*args, _real=real, _name=name, **kwargs):
            log.add("gen", _name)
            return _real(*args, **kwargs)

        setattr(svc, name, primitive)

    for name in ("_probe_begin", "_probe_begin_detached"):
        if not hasattr(svc, name):
            continue
        real = getattr(svc, name)

        def begin(*args, _real=real, _name=name, **kwargs):
            context = _real(*args, **kwargs)
            if context is not None:
                log.add("begin", _name)
            return context

        setattr(svc, name, begin)

    # 039 S2 (FR-39.23, FTASKS 13.6): `_rules_finish` is awaited before every `_probe_finish`.
    real_rules = svc._rules_finish

    async def rules(context, *args, **kwargs):
        if context is not None:
            log.add("rules", None)
        return await real_rules(context, *args, **kwargs)

    svc._rules_finish = rules

    real_record = svc._probe_record

    async def record(context, verdicts=None, full_ids=None, **kwargs):
        if context is not None:
            computed = verdicts if verdicts is not None else context.finish()
            log.add("record", list(computed))
            # 038 FR-38.6: what the path SAID about how it served the request, and what the
            # finalized results carry (they are built from the same context).
            execution = getattr(context, "execution", None)
            log.add("mode", (getattr(execution, "engine", None), getattr(execution, "mode", None)))
            for result in getattr(context, "results", None) or ():
                log.add("result_mode", (result.engine, result.execution_mode))
            verdicts = computed
        return await real_record(context, verdicts, full_ids=full_ids, **kwargs)

    svc._probe_record = record

    for site in discover_sites():
        real = getattr(svc, site)
        if inspect.isasyncgenfunction(real):
            def wrap(real=real, site=site):
                @functools.wraps(real)
                async def gen(*args, **kwargs):
                    log.add("enter", site)
                    async for item in real(*args, **kwargs):
                        yield item
                return gen
        elif inspect.iscoroutinefunction(real):
            def wrap(real=real, site=site):
                @functools.wraps(real)
                async def coro(*args, **kwargs):
                    log.add("enter", site)
                    return await real(*args, **kwargs)
                return coro
        else:
            def wrap(real=real, site=site):
                @functools.wraps(real)
                def plain(*args, **kwargs):
                    log.add("enter", site)
                    return real(*args, **kwargs)
                return plain
        setattr(svc, site, wrap())


async def _drive(svc: InferenceService, scenario: dict) -> None:
    extra = scenario.get("request", {})
    if scenario["call"] == "chat":
        await svc.create_chat_completion(_chat(**extra))
    elif scenario["call"] == "stream":
        async for _ in svc.stream_chat_completion(_chat(stream=True, **extra)):
            pass
    else:
        await svc.create_text_completion(_text(**extra))


#: ⚠ THE EXPECTED RED, RECORDED MECHANICALLY (task 1.6). Six sites (batched chat, CBM chat and
#: text, three llama.cpp paths) reached generation with a probe armed and opened no context on the
#: code this guard was written against (`08c1c53`). `strict=True`: the
#: moment one is fixed its xfail turns into a failure, so this set must shrink to empty as FR-27.8
#: lands — a guard that is green before the fix is not guarding the fix.
#: All six were wired by FR-27.8 (task 2); the set is kept, empty, so a regression has somewhere
#: obvious to be recorded rather than silently re-listed.
PENDING_FR_27_8: set[str] = set()


@pytest.mark.parametrize("site", [
    pytest.param(s, marks=pytest.mark.xfail(strict=True, reason="FR-27.8 not yet wired"))
    if s in PENDING_FR_27_8 else s
    for s in sorted(SCENARIOS)
])
async def test_every_generation_site_opens_a_context_or_records_why_not(site, harness, monkeypatch):
    scenario = SCENARIOS[site]
    svc = _build(harness, scenario, monkeypatch)
    with patch("millm.services.inference_service.torch.cuda.is_available", return_value=False):
        await _drive(svc, scenario)
    events = harness.events

    assert ("enter", site) in events, f"the scenario for {site} did not reach it: {events}"
    gens = [i for i, (kind, _) in enumerate(events) if kind == "gen"]
    assert gens, f"{site}: no generation primitive was reached"
    for i in gens:
        assert any(kind == "begin" for kind, _ in events[:i]), (
            f"{site}: generation reached with no probe context open — the request is silent"
        )
    records = [payload for kind, payload in events if kind == "record"]
    assert len(records) == 1, f"{site}: expected exactly one recorded event, got {len(records)}"
    # 039 S2: the rule hook ran on this path too, once, before the record (nothing silent).
    kinds = [kind for kind, _ in events]
    assert kinds.count("rules") == 1, f"{site}: `_rules_finish` entered {kinds.count('rules')}x"
    assert kinds.index("rules") < kinds.index("record"), f"{site}: rules after the record"
    verdicts = records[0]
    assert verdicts, f"{site}: the record carried no verdict"
    for v in verdicts:
        assert (v.scored and v.score is not None) or v.not_scored_reason, (
            f"{site}: a verdict with neither a score nor a reason: {v}"
        )
    assert ProbeRuntimeState().current_request() is None, (
        f"{site}: a probe context was left open; the next request's begin would refuse"
    )


# ── Feature 36: every record passes the reply ───────────────────────────────────


def test_every_probe_record_call_passes_the_reply_text():
    """⚠ BY AST, OVER EVERY CALL (036 FTDD §10 "Path discovery").

    The turn writer hashes the reply into the key the conversation's NEXT request is matched by,
    and records it when transcripts are on. A path that records without it leaves every following
    turn of that conversation unjoinable, silently — so a `_probe_record` call added later without
    `reply_text=` fails here. The count is asserted too, so a parser that finds nothing cannot pass.
    """
    tree = ast.parse(Path(inference_module.__file__).read_text())
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == "_record_probe_shielded"
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "self"
    ]
    assert len(calls) >= 10, f"found only {len(calls)} record calls — parser broken?"
    missing = [c.lineno for c in calls if not any(k.arg == "reply_text" for k in c.keywords)]
    assert not missing, f"record calls without reply_text= at lines {missing}"


def test_no_record_is_awaited_bare():
    """⚠ 038 D38.33 (FTASKS 3.10). Every record is the TRACKED, SHIELDED task: an
    `await self._probe_record(` in a streaming `finally` is cancelled with the request and the
    event is lost. `_probe_record` may be called only by `_spawn_probe_record`."""
    tree = ast.parse(Path(inference_module.__file__).read_text())
    callers = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "_probe_record"):
                callers.append(fn.name)
    assert callers == ["_spawn_probe_record"], callers


# ── 038 FR-38.6: every path states its engine and execution mode ──────────────────

#: site -> (engine, execution mode) the probe context must carry. Control M6: hard-code `serial`.
EXPECTED_EXECUTION = {
    "create_chat_completion": ("transformers", "serial"),
    "stream_chat_completion": ("transformers", "serial"),
    "create_text_completion": ("transformers", "serial"),
    "_generate_batch_chunk": ("transformers", "batched"),
    "_cbm_chat_completion": ("cbm", "continuous_batching"),
    "_cbm_stream_chat_completion": ("cbm", "continuous_batching"),
    "_cbm_text_completion": ("cbm", "continuous_batching"),
    "_llamacpp_chat_completion": ("llamacpp", "serial"),
    "_llamacpp_stream_chat_completion": ("llamacpp", "serial"),
    "_llamacpp_text_completion": ("llamacpp", "serial"),
}


def test_every_site_has_an_expected_execution():
    assert set(EXPECTED_EXECUTION) == set(SCENARIOS), "a new site needs its execution facts"


@pytest.mark.parametrize("site", sorted(EXPECTED_EXECUTION))
async def test_execution_mode_per_path(site, harness, monkeypatch):
    scenario = SCENARIOS[site]
    svc = _build(harness, scenario, monkeypatch)
    with patch("millm.services.inference_service.torch.cuda.is_available", return_value=False):
        await _drive(svc, scenario)
    await InferenceService.drain_probe_records(5.0)
    modes = [payload for kind, payload in harness.events if kind == "mode"]
    assert modes == [EXPECTED_EXECUTION[site]], f"{site}: {modes}"
    result_modes = {payload for kind, payload in harness.events if kind == "result_mode"}
    assert result_modes <= {EXPECTED_EXECUTION[site]}, f"{site}: results say {result_modes}"


# ── 038 FR-38.9 (review R1): every STREAMING path names a disconnect ──────────────────
#: Removing the CBM stream's `_probe_mark_interrupted` survived the suite: the record then
#: defaults to `generation_error`, blaming the server for a client that went away.
STREAM_SITES = sorted(s for s, sc in SCENARIOS.items() if sc["call"] == "stream")


def test_the_streaming_sites_are_the_three_engines():
    assert set(STREAM_SITES) == {"stream_chat_completion", "_cbm_stream_chat_completion",
                                 "_llamacpp_stream_chat_completion"}, STREAM_SITES


@pytest.mark.parametrize("site", STREAM_SITES)
async def test_a_disconnect_on_every_streaming_path_says_client_disconnected(site, harness,
                                                                           monkeypatch):
    scenario = SCENARIOS[site]
    svc = _build(harness, scenario, monkeypatch)
    seen: list = []
    inner = svc._probe_record

    async def record(context, verdicts=None, *a, **kw):
        if context is not None:
            seen.append((verdicts is None, getattr(context, "interrupted", None)))
        return await inner(context, verdicts, *a, **kw)

    svc._probe_record = record
    agen = svc.stream_chat_completion(_chat(stream=True, **scenario.get("request", {})))
    with patch("millm.services.inference_service.torch.cuda.is_available", return_value=False):
        await agen.__anext__()
        await agen.aclose()
        # The engine generators are delegated to with `async for`; closing the outer one leaves
        # the inner to the event loop's async-generator finalizer, as in production.
        import asyncio
        import gc

        for _ in range(20):
            if seen:
                break
            gc.collect()
            await asyncio.sleep(0.05)
    await InferenceService.drain_probe_records(5.0)
    assert seen == [(True, "client_disconnected")], f"{site}: {seen}"


# ── 038 FR-38.15: every entry point opens THE inference span (FTASKS 8.4) ─────────────


def _span_decorator(node: ast.AST) -> str | None:
    """`traced_entry` / `traced_stream_entry` when the method carries one, else None."""
    for deco in getattr(node, "decorator_list", ()):
        call = deco.func if isinstance(deco, ast.Call) else deco
        name = call.attr if isinstance(call, ast.Attribute) else getattr(call, "id", None)
        if name in ("traced_entry", "traced_stream_entry"):
            return name
    return None


def test_every_generation_site_is_reached_only_through_a_span_entry_point():
    """Discovered, never listed: each public method from which a generation site is reachable
    carries the inference-span decorator, and no other method does (one span per request, not
    one per layer of the call graph). A new public path that generates turns this red until it
    opens the span too."""
    methods = _class_methods()
    entries = discover_entry_points()
    assert entries == KNOWN_ENTRY_POINTS, entries
    decorated = {name for name, node in methods.items() if _span_decorator(node)}
    assert decorated == entries, (decorated, entries)
    assert _span_decorator(methods["stream_chat_completion"]) == "traced_stream_entry"
    for name in entries - {"stream_chat_completion"}:
        assert _span_decorator(methods[name]) == "traced_entry", name
    # And the live class object carries the wrapper (the AST is not the only witness).
    for name in entries:
        assert getattr(getattr(InferenceService, name), "__millm_inference_span__", None), name


# ── 038 FR-38.15 (12.5a F2): every path's inference span states its engine and mode ─────
#: `millm.engine` / `millm.execution_mode` were allowlisted and documented but set on no span.
#: Read back from the receiver fed by the real export path, armed AND unarmed (the attributes
#: describe the request, not the probes). Control F2a: delete `note_execution` in `_probe_begin`;
#: F2b: in `_probe_begin_detached`.


@pytest.fixture
def otel_stub():
    from millm import telemetry
    from tests.support.otlp_stub import OtlpStub
    from tests.unit.telemetry.conftest import reset_all, telemetry_settings

    stub = OtlpStub().start()
    reset_all()
    telemetry.configure(telemetry_settings(stub.endpoint))
    yield stub
    reset_all()
    stub.stop()


@pytest.mark.parametrize("armed", [True, False], ids=["armed", "unarmed"])
@pytest.mark.parametrize("site", sorted(EXPECTED_EXECUTION))
async def test_the_inference_span_states_the_paths_engine_and_mode(site, armed, harness,
                                                                   otel_stub, monkeypatch):
    from millm import telemetry
    from millm.telemetry import allowlist

    scenario = SCENARIOS[site]
    svc = _build(harness, scenario, monkeypatch)
    if not armed:
        ProbeRuntimeState.reset_for_tests()
    with patch("millm.services.inference_service.torch.cuda.is_available", return_value=False):
        await _drive(svc, scenario)
    await InferenceService.drain_probe_records(5.0)
    telemetry.shutdown(5.0)
    roots = [s for s in otel_stub.spans if s["parent_span_id"] == ""]
    assert len(roots) == 1, [s["name"] for s in otel_stub.spans]
    attrs = roots[0]["attributes"]
    assert (attrs.get("millm.engine"), attrs.get("millm.execution_mode")) == (
        EXPECTED_EXECUTION[site]), f"{site}: {attrs}"
    assert allowlist.attributes_dropped() == 0
