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

    real_record = svc._probe_record

    async def record(context, verdicts=None, full_ids=None, **kwargs):
        if context is not None:
            computed = verdicts if verdicts is not None else context.finish()
            log.add("record", list(computed))
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
    verdicts = records[0]
    assert verdicts, f"{site}: the record carried no verdict"
    for v in verdicts:
        assert (v.scored and v.score is not None) or v.not_scored_reason, (
            f"{site}: a verdict with neither a score nor a reason: {v}"
        )
    assert ProbeRuntimeState().current_request() is None, (
        f"{site}: a probe context was left open; the next request's begin would refuse"
    )
