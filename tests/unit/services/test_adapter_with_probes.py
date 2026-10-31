"""Probes, steering and model work while an adapter is applied (033 FTASKS 6.1-6.3; addendum D-14).

A probe is armed on decoder layer L of a tiny real Llama (its hook is PREPENDED on the layer, as in
production); a steering-shaped hook is APPENDED on the same layer, adding a vector to its output as
SAE steering does (`sae_hooker.py`); the adapter hooks layer L's projections. So:

- the probe reads the ADAPTED residual — its score moves when the adapter is applied and returns to
  the base value exactly when it is removed;
- the probe still ignores steering (prepended before the steering hook);
- every verdict produced while the adapter is applied is `provisional`, and carries `served_model`
  naming the adapter — the reason a reader can see;
- model work that suppresses SAEs (`run_model_work`, used by scoring and parity) still runs with
  the adapter: the adapter is the model, not steering;
- probe parity is refused with `ADAPTER_APPLIED` while applied (it compares against base scores).

This file's steering is a steering-shaped hook; the same geometry with a REAL `LoadedSAE` installed
by `SAEHooker` (hook order, provisional marking, exact restore) is
`test_adapter_with_real_sae.py`, which closes FTASKS 6.2.
"""

from __future__ import annotations

import pytest
import torch

from millm.core.errors import AdapterAppliedError
from millm.ml.probe_head import ProbeHead
from millm.services.probe_parity import ProbeParityEngine
from millm.services.probe_runtime import ArmedProbe, ProbeRequestContext, ProbeRuntimeState
from tests.unit.adapter_service_fixtures import ADAPTER_NAME, PROMPT_IDS, build_env, identity

LAYER = 1


@pytest.fixture
def env(tmp_path):
    ProbeRuntimeState.reset_for_tests()
    e = build_env(tmp_path)
    yield e
    ProbeRuntimeState.reset_for_tests()
    e.close()


def _probe() -> ArmedProbe:
    torch.manual_seed(3)
    return ArmedProbe(
        probe_id="pr_1", name="concept", head=ProbeHead(weight=torch.randn(48), bias=0.0,
                                                        layer=LAYER),
        rule="mean", scope="all", layer=LAYER, rung=2, rung_language="detects on unseen tasks",
        threshold=0.0,
    )


def _verdict(env, request_id: str):
    state = ProbeRuntimeState()
    state.begin_request(request_id)
    try:
        with torch.no_grad():
            env.model(PROMPT_IDS)
    finally:
        context = state.end_request()
    return context.finish()[0]


def _steering_hook(module, args, output):
    if isinstance(output, tuple):
        return (output[0] + 5.0,) + tuple(output[1:])
    return output + 5.0


async def test_probe_reads_the_adapted_residual_provisionally_and_ignores_steering(env):
    rel = env.write()
    ProbeRuntimeState().arm(_probe(), env.model)
    base = _verdict(env, "base")
    assert base.scored and base.provisional is False and base.served_model == "tiny"

    await env.service.load(identity(), rel)
    adapted = _verdict(env, "adapted")
    assert adapted.scored and adapted.score != base.score
    assert adapted.provisional is True
    assert adapted.served_model == ADAPTER_NAME

    layer = env.model.model.layers[LAYER]
    handle = layer.register_forward_hook(_steering_hook)  # appended, as SAE steering is
    try:
        steered = _verdict(env, "steered")
    finally:
        handle.remove()
    assert steered.score == adapted.score, "the probe reads before steering"
    assert steered.provisional is True

    await env.service.unload(identity())
    after = _verdict(env, "after")
    assert after.score == base.score
    assert after.provisional is False and after.served_model == "tiny"


async def test_a_not_scored_verdict_is_provisional_too(env):
    rel = env.write()
    await env.service.load(identity(), rel)
    context = ProbeRequestContext("x", [_probe()])
    verdict = context.finish()[0]
    assert verdict.scored is False and verdict.provisional is True
    assert verdict.served_model == ADAPTER_NAME


async def test_model_work_that_suppresses_saes_still_runs_adapted(env):
    """6.3: `run_model_work` (scoring, embeddings' unsteered path, parity) suppresses SAEs only."""
    base = env.logits()
    rel = env.write()
    await env.service.load(identity(), rel)
    adapted = await env.inference.run_model_work(env.logits)
    assert not torch.equal(adapted, base)
    assert torch.equal(adapted, env.logits())


async def test_parity_is_refused_while_an_adapter_is_applied(env):
    rel = env.write()
    probe = _probe()
    definition = {"test_vectors": {"tolerance": 0.001, "vectors": []}}
    ProbeParityEngine(lambda ids, ctx: None).run(probe, definition, tolerance=0.001)  # base: runs
    await env.service.load(identity(), rel)
    with pytest.raises(AdapterAppliedError) as exc:
        ProbeParityEngine(lambda ids, ctx: None).run(probe, definition, tolerance=0.001)
    assert exc.value.details["adapter"] == ADAPTER_NAME
    assert exc.value.status_code == 409
    await env.service.unload(identity())
    report = ProbeParityEngine(lambda ids, ctx: None).run(probe, definition, tolerance=0.001)
    assert report.error  # no vectors: an ordinary report again, not a refusal
