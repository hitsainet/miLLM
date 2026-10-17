"""033 FTASKS 6.2 with a REAL loaded SAE: a probe on layer L, an SAE steering layer L, an adapter
on layer L's projections (FR-33.5.5; addendum D-14).

`test_adapter_with_probes.py` proves the same geometry with a steering-SHAPED hook. This file
replaces that stand-in with the production objects: a `LoadedSAE` installed by the real
`SAEHooker` (`millm/ml/sae_hooker.py`), steering one feature with its decoder direction, on the
same decoder layer the probe reads and whose projections the adapter hooks. It asserts:

- **Hook order.** On decoder layer L the probe hook runs FIRST (it is prepended,
  `probe_hooker.py`) and the SAE steering hook after it; the adapter's hooks live on layer L's
  projections, inside the layer's own forward, so they run before both.
- **What the probe reads.** The adapted residual: the score moves when the adapter is applied,
  and the SAE's steering (which demonstrably runs and changes the logits) does not move it.
- **Provisional marking.** Every verdict while the adapter is applied is `provisional` and names
  the adapter in `served_model`.
- **Exact restore.** After unload: the probe score equals the base score bit for bit, the steered
  logits equal the pre-adapter steered logits bit for bit, no projection carries a hook, and the
  SAE hook is still installed and still steering.

Negative control (recorded in `attachable_weights_acceptance_2026-10-14.md`): register the probe
hook without `prepend=True` (`probe_hooker.py`) — the order assertion and the "steering does not
move the probe" assertion both go red.
"""

from __future__ import annotations

import pytest
import torch

from millm.ml.probe_head import ProbeHead
from millm.ml.sae_config import SAEConfig
from millm.ml.sae_hooker import SAEHooker
from millm.ml.sae_wrapper import LoadedSAE
from millm.services.probe_runtime import ArmedProbe, ProbeRuntimeState
from tests.unit.adapter_service_fixtures import ADAPTER_NAME, PROMPT_IDS, build_env, identity

LAYER = 1
HIDDEN = 48
D_SAE = 16
FEATURE = 3
STEER = 8.0


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
        probe_id="pr_sae", name="concept",
        head=ProbeHead(weight=torch.randn(HIDDEN), bias=0.0, layer=LAYER),
        rule="mean", scope="all", layer=LAYER, rung=2, rung_language="detects on unseen tasks",
        threshold=0.0,
    )


def _sae() -> LoadedSAE:
    torch.manual_seed(17)
    config = SAEConfig(d_in=HIDDEN, d_sae=D_SAE, model_name="tiny",
                       hook_name=f"blocks.{LAYER}.hook_resid_post", hook_layer=LAYER,
                       dtype="float32")
    return LoadedSAE(W_enc=torch.randn(HIDDEN, D_SAE), b_enc=torch.zeros(D_SAE),
                     W_dec=torch.randn(D_SAE, HIDDEN), b_dec=torch.zeros(HIDDEN),
                     config=config)


def _verdict(env, request_id: str):
    state = ProbeRuntimeState()
    state.begin_request(request_id)
    try:
        with torch.no_grad():
            env.model(PROMPT_IDS)
    finally:
        context = state.end_request()
    return context.finish()[0]


def _projection_hooks(layer: torch.nn.Module) -> int:
    return sum(len(m._forward_hooks) for name, m in layer.named_modules() if name)


async def test_probe_sae_and_adapter_on_one_layer(env):
    layer = env.model.model.layers[LAYER]
    assert _projection_hooks(layer) == 0

    # The SAE is attached FIRST and the probe armed after it: the order production can see
    # (an operator steering, then arming a monitor). Only the probe's `prepend=True` puts it ahead.
    sae = _sae()
    hooker = SAEHooker()
    sae_handle = hooker.install(env.model, LAYER, sae)
    try:
        ProbeRuntimeState().arm(_probe(), env.model)
        assert hooker.last_resolved_module_path == f"model.layers.{LAYER}"

        # Hook order on decoder layer L: the probe's hook runs before the SAE's.
        order = list(layer._forward_hooks.keys())
        assert len(order) == 2, order
        assert order[-1] == sae_handle.id, "the SAE steering hook must run after the probe's"

        unsteered_logits = env.logits()
        base = _verdict(env, "base")
        assert base.scored and base.provisional is False and base.served_model == "tiny"

        sae.set_steering(FEATURE, STEER)
        sae.enable_steering(True)
        applied_before = sae.steering_apply_count
        steered_base_logits = env.logits()
        assert sae.steering_apply_count > applied_before, "the real SAE did not steer"
        assert not torch.equal(steered_base_logits, unsteered_logits)
        steered_base = _verdict(env, "steered-base")
        assert steered_base.score == base.score, "the probe reads before SAE steering"

        # The adapter hooks layer L's projections (inside the layer's forward).
        rel = env.write()
        await env.service.load(identity(), rel)
        assert _projection_hooks(layer) > 0
        assert list(layer._forward_hooks.keys()) == order, "the adapter must not touch layer L's own hooks"

        adapted = _verdict(env, "adapted")
        assert adapted.scored and adapted.score != base.score, "the probe reads the ADAPTED residual"
        assert adapted.provisional is True
        assert adapted.served_model == ADAPTER_NAME
        assert sae.steering_apply_count > applied_before
        steered_adapted_logits = env.logits()
        assert not torch.equal(steered_adapted_logits, steered_base_logits)

        sae.enable_steering(False)
        unsteered_adapted = _verdict(env, "adapted-unsteered")
        sae.enable_steering(True)
        assert unsteered_adapted.score == adapted.score, "SAE steering must not move the probe"
        assert unsteered_adapted.provisional is True

        await env.service.unload(identity())

        # Exact restore: the adapter's hooks are gone, the probe and the SAE are not.
        assert _projection_hooks(layer) == 0
        assert list(layer._forward_hooks.keys()) == order
        after = _verdict(env, "after")
        assert after.score == base.score
        assert after.provisional is False and after.served_model == "tiny"
        assert torch.equal(env.logits(), steered_base_logits), "steered logits not restored exactly"
        sae.enable_steering(False)
        assert torch.equal(env.logits(), unsteered_logits)
    finally:
        hooker.remove(sae_handle)
