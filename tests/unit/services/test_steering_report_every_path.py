"""Every generation path, DISCOVERED, publishes a steering report (Feature 28, FR-28.3.1;
FTASKS 5.5, 5.8, 6.3).

The sites come from `tests/support/generation_entry_points.py` — the AST discovery Feature 27
built for probes (FR-27.9) — and the scenario table is checked for EQUALITY against it, so a new
generation site fails here until somebody drives it. A hand-kept list can only hold the paths its
author remembered; the probe guard's own history is three paths a list missed.

Each site is driven twice:

* nothing attached → `none` (and, for a stream, exactly ONE `millm_steering` chunk, last before
  `[DONE]`, even for `none`);
* one SAE attached with live steering → `manual` with the hash of the live values on the
  transformers and batching-manager paths, and `unknown;reason=llamacpp_entries` on llama.cpp,
  where the hooks cannot have run. A path that published nothing, or published the default, fails.

Scoring (exempt from generation) publishes the constant `none` (X-09).
"""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import patch

import pytest

from millm.ml.model_loader import ENGINE_LLAMACPP, LoadedModel, LoadedModelState
from millm.services.inference_service import get_steering_report, reset_steering_memo
from tests.support.generation_entry_points import (
    EXEMPT,
    SCENARIOS,
    EventLog,
    FakeCBM,
    FakeLlama,
    _chat,
    _text,
    discover_sites,
)
from tests.unit.f25_fixtures import make_service, word_model, word_tokenizer
from tests.unit.f28_fixtures import (
    attach,
    clean_state,  # noqa: F401 - fixture
    db,  # noqa: F401 - fixture
    independent_hash,
)


def test_the_scenario_table_equals_discovery():
    sites = discover_sites()
    assert sites, "discovery found nothing — a broken parser must not pass by checking nothing"
    assert set(SCENARIOS) == sites, (
        f"sites with no scenario: {sorted(sites - set(SCENARIOS))}; "
        f"scenarios for no site: {sorted(set(SCENARIOS) - sites)}"
    )


def _build(scenario: dict, monkeypatch, steered: bool):
    model, tokenizer = word_model(), word_tokenizer()
    svc = make_service(model, tokenizer)
    log = EventLog()
    handles = []
    if steered:
        served = attach(model, [("sae_path", 0, 5)])
        handles = served.handles
        sae = served.sae("sae_path", 0)
        sae.set_steering_batch({3: 2.5})
        sae.enable_steering(True)
    if scenario.get("cbm"):
        svc._cbm_backend = FakeCBM(log)
    if scenario["engine"] == "llamacpp":
        LoadedModelState().set(LoadedModel(
            model_id=1, model_name="tiny", model=FakeLlama(log), tokenizer=None,
            loaded_at=datetime(2026, 10, 6), engine=ENGINE_LLAMACPP,
        ))
    # Instrument every site so the scenario is proven to reach the one it names.
    for site in discover_sites():
        real = getattr(svc, site)
        import inspect

        if inspect.isasyncgenfunction(real):
            def wrap(real=real, site=site):
                async def gen(*a, **k):
                    log.add("enter", site)
                    async for item in real(*a, **k):
                        yield item
                return gen
        elif inspect.iscoroutinefunction(real):
            def wrap(real=real, site=site):
                async def coro(*a, **k):
                    log.add("enter", site)
                    return await real(*a, **k)
                return coro
        else:
            def wrap(real=real, site=site):
                def plain(*a, **k):
                    log.add("enter", site)
                    return real(*a, **k)
                return plain
        monkeypatch.setattr(svc, site, wrap())
    return svc, log, handles


async def _drive(svc, scenario) -> list[str]:
    extra = scenario.get("request", {})
    chunks: list[str] = []
    with patch("millm.services.inference_service.torch.cuda.is_available", return_value=False):
        if scenario["call"] == "chat":
            await svc.create_chat_completion(_chat(**extra))
        elif scenario["call"] == "stream":
            async for chunk in svc.stream_chat_completion(_chat(stream=True, **extra)):
                chunks.append(chunk)
        else:
            await svc.create_text_completion(_text(**extra))
    return chunks


def _expected(scenario: dict, steered: bool) -> str:
    if not steered:
        return "none"
    if scenario["engine"] == "llamacpp":
        return "unknown;reason=llamacpp_entries"
    return (f'manual;sae="sae_path";layer=0;features=1;'
            f'hash="{independent_hash("sae_path", {3: 2.5})}"')


@pytest.mark.parametrize("steered", [False, True], ids=["nothing-attached", "live-steering"])
@pytest.mark.parametrize("site", sorted(SCENARIOS))
async def test_every_generation_site_publishes_a_report(site, steered, clean_state, db,  # noqa: F811
                                                        monkeypatch):
    scenario = SCENARIOS[site]
    reset_steering_memo()
    svc, log, handles = _build(scenario, monkeypatch, steered)
    try:
        chunks = await _drive(svc, scenario)
    finally:
        for h in handles:
            h.remove()
    assert ("enter", site) in log.events, f"the scenario for {site} did not reach it"
    report = get_steering_report()
    assert report is not None, f"{site}: no steering report was published"
    assert report.header == _expected(scenario, steered), site
    if scenario["call"] == "stream":
        steering_chunks = [c for c in chunks if '"millm_steering"' in c]
        assert len(steering_chunks) == 1, f"{site}: expected exactly one steering chunk"
        assert chunks[-1] == "data: [DONE]\n\n"
        assert chunks[-2] == steering_chunks[0], f"{site}: the steering chunk must be last"
        payload = json.loads(chunks[-2][len("data: "):])
        assert payload["choices"] == [] and payload["object"] == "chat.completion.chunk"
        assert payload["millm_steering"] == report.header, "the chunk carries the exact header"


@pytest.mark.parametrize("call", ["text", "chat"])
async def test_scoring_publishes_none_even_with_live_steering(call, clean_state):  # noqa: F811
    """X-09 / FR-28.3.1: scoring runs under `_unsteered`; its report is the constant `none`."""
    assert "_score_prompts" in EXEMPT
    model, tokenizer = word_model(), word_tokenizer()
    svc = make_service(model, tokenizer)
    served = attach(model, [("sae_path", 0, 5)])
    try:
        sae = served.sae("sae_path", 0)
        sae.set_steering_batch({3: 2.5})
        sae.enable_steering(True)
        reset_steering_memo()
        with patch("millm.services.inference_service.torch.cuda.is_available",
                   return_value=False):
            if call == "text":
                await svc.create_text_completion(_text(max_tokens=1, logprobs=2))
            else:
                await svc.create_chat_completion(_chat(max_tokens=1, logprobs=True))
        assert get_steering_report().header == "none"
    finally:
        for h in served.handles:
            h.remove()
