"""`AdapterService` (Feature 33, FTASKS 4.3-4.12, 5.4, 11.1): the refusal order, the slot, the
pause, the epoch, the drop, and the status block.

Real objects throughout (`tests/unit/adapter_service_fixtures.py`): a tiny real Llama, the real
`WeightBarrier.serving_pause` holding a real InferenceService's admission slot, the vendored
schema, the real hooks. Every refusal is asserted with spies proving that NOTHING after it ran:
no row read, no artifact read, no pause entered, no hook installed, and the slot released.

Every load/unload test runs twice, with nothing attached and with an attachment reported, and the
attachment service records ZERO reads (FR-33.5.4: no adapter precondition reads it).
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import threading
from unittest.mock import MagicMock, patch

import pytest
import torch

from millm.core.errors import (
    AdapterAlreadyLoadedError,
    AdapterBaseChangedError,
    AdapterHashMismatchError,
    AdapterInstallFailedError,
    AdapterNotServableError,
    AdapterShapeMismatchError,
    ModelBusyError,
    ModelLeasedError,
    ModelNotLoadedError,
    NoAdapterLoadedError,
    WeightsBarrierUnavailableError,
    WeightsNotCleanError,
)
from millm.ml.adapter_hooks import LoraHookSet
from millm.ml.model_loader import ENGINE_LLAMACPP, LoadedModelState, ModelLoader
from millm.services import adapter_service, resident_identity
from millm.services.adapter_service import get_adapter_state
from millm.services.request_queue import QueueFullError
from millm.services.weight_attachment import WeightService, set_weight_service
from tests.unit.adapter_fixtures import param_fingerprint, target_paths
from tests.unit.adapter_service_fixtures import ADAPTER_NAME, build_env, identity

@pytest.fixture(params=[False, True], ids=["nothing_attached", "attached"])
def attached(request):
    """With `True`, the attachment service reports an attachment; it must never be read."""
    if not request.param:
        set_weight_service(None)
        yield None
        set_weight_service(None)
        return
    spy = MagicMock(spec=WeightService)
    spy.is_attached.return_value = True
    set_weight_service(spy)
    yield spy
    set_weight_service(None)
    # FR-33.5.4: no adapter path read whether anything is attached.
    assert spy.method_calls == [], spy.method_calls


@pytest.fixture
def env(tmp_path, attached):
    e = build_env(tmp_path)
    yield e
    e.close()


@pytest.fixture
def audit():
    with patch.object(adapter_service, "weights_audit") as spy:
        yield spy


def _hook_count(model) -> int:
    return sum(len(m._forward_hooks) for m in model.modules())


class _Spies:
    """Every step after the refusals, so a refused load can be proven to have run none of them."""

    def __init__(self, env):
        self.prepare = patch.object(
            adapter_service, "prepare_adapter", wraps=adapter_service.prepare_adapter
        )
        self.pause = patch.object(env.barrier, "serving_pause", wraps=env.barrier.serving_pause)
        self.install = patch.object(LoraHookSet, "install", wraps=LoraHookSet.install)

    def __enter__(self):
        self.p, self.s, self.i = self.prepare.__enter__(), self.pause.__enter__(), self.install.__enter__()
        return self

    def __exit__(self, *exc):
        for cm in (self.install, self.pause, self.prepare):
            cm.__exit__(*exc)

    def assert_nothing_ran(self, env):
        assert self.p.call_count == 0
        assert self.s.call_count == 0
        assert self.i.call_count == 0
        env.row_lookup.assert_not_awaited()
        assert get_adapter_state().current is None
        assert not get_adapter_state().loading, "the slot must be released"
        assert _hook_count(env.model) == 0


# ---------------------------------------------------------------------------------- refusals


class TestRefusalOrder:
    async def test_no_model(self, env, audit):
        rel = env.write()
        LoadedModelState()._loaded = None
        with _Spies(env) as spies, pytest.raises(ModelNotLoadedError):
            await env.service.load(identity(), rel)
        spies.assert_nothing_ran(env)
        audit.assert_called_once()
        assert audit.call_args.args[:2] == ("adapter/load", "refused")
        assert audit.call_args.kwargs["code"] == "MODEL_NOT_LOADED"

    async def test_gguf_engine_is_not_servable(self, env, audit):
        rel = env.write()
        LoadedModelState().current.engine = ENGINE_LLAMACPP
        with _Spies(env) as spies, pytest.raises(AdapterNotServableError) as exc:
            await env.service.load(identity(), rel)
        assert exc.value.details["reason"] == "engine"
        spies.assert_nothing_ran(env)
        assert audit.call_args.kwargs["reason"] == "engine"

    @pytest.mark.parametrize("method", ["bitsandbytes", "gguf:Q4_K_M", "unknown"])
    async def test_quantized_weights_are_not_servable(self, env, audit, method):
        rel = env.write()
        LoadedModelState().current.quantization_method = method
        with _Spies(env) as spies, pytest.raises(AdapterNotServableError) as exc:
            await env.service.load(identity(), rel)
        assert exc.value.details["reason"] == "quantization"
        spies.assert_nothing_ran(env)

    async def test_continuous_batching_refuses_with_reason_cbm(self, env, audit):
        rel = env.write()
        env.inference._cbm_backend = MagicMock(is_running=True)
        try:
            with _Spies(env) as spies, pytest.raises(AdapterNotServableError) as exc:
                await env.service.load(identity(), rel)
        finally:
            env.inference._cbm_backend = None
        assert exc.value.details["reason"] == "cbm"
        spies.assert_nothing_ran(env)

    async def test_more_than_one_slot_refuses_before_the_artifact_is_read(self, env):
        rel = env.write()
        env.inference._request_queue._max_concurrent = 2
        with _Spies(env) as spies, pytest.raises(WeightsBarrierUnavailableError) as exc:
            await env.service.load(identity(), rel)
        assert exc.value.details["reason"] == "concurrency"
        spies.assert_nothing_ran(env)

    async def test_a_foreign_lease_refuses_naming_the_holder(self, env, audit):
        rel = env.write()
        env.leases.grant(1, "tiny", "midataworks", "label run 7", 120)
        with _Spies(env) as spies, pytest.raises(ModelLeasedError) as exc:
            await env.service.load(identity(), rel, lease_id="not-the-lease")
        assert exc.value.details["holder"] == "midataworks"
        spies.assert_nothing_ran(env)
        assert audit.call_args.kwargs["code"] == "MODEL_LEASED"

    async def test_the_lease_holder_passes(self, env):
        rel = env.write()
        grant = env.leases.grant(1, "tiny", "midataworks", "label run 7", 120)
        result = await env.service.load(identity(), rel, lease_id=grant.lease_id)
        assert result["adapter"]["name"] == ADAPTER_NAME

    async def test_weights_not_clean(self, env):
        rel = env.write()
        env.barrier._weights_state = "torn"
        with _Spies(env) as spies, pytest.raises(WeightsNotCleanError):
            await env.service.load(identity(), rel)
        spies.assert_nothing_ran(env)

    async def test_an_applied_adapter_refuses_a_second_load(self, env):
        rel = env.write()
        await env.service.load(identity(), rel)
        with _Spies(env) as spies, pytest.raises(AdapterAlreadyLoadedError) as exc:
            await env.service.load(identity(), rel)
        assert exc.value.details["adapter"] == ADAPTER_NAME
        assert spies.p.call_count == 0 and spies.s.call_count == 0

    async def test_a_corrupt_weights_file_refuses_before_any_tensor_or_pause(self, env, audit):
        rel = env.write(corrupt_byte=True)
        with _Spies(env) as spies, patch.object(LoraHookSet, "build") as build, \
                pytest.raises(AdapterHashMismatchError):
            await env.service.load(identity(), rel)
        assert build.call_count == 0
        assert spies.s.call_count == 0 and spies.i.call_count == 0
        assert get_adapter_state().current is None and not get_adapter_state().loading
        assert audit.call_args.kwargs["code"] == "ADAPTER_HASH_MISMATCH"

    async def test_one_dimension_off_refuses_naming_the_module(self, env):
        off = target_paths(env.model, ["down_proj"])[0]
        rel = env.write(shape_off_by_one=off)
        with _Spies(env) as spies, pytest.raises(AdapterShapeMismatchError) as exc:
            await env.service.load(identity(), rel)
        assert exc.value.details["first"]["module"] == off
        assert spies.s.call_count == 0 and _hook_count(env.model) == 0


# ---------------------------------------------------------------------------------- load/unload


class TestLoadAndUnload:
    async def test_load_applies_without_touching_the_base(self, env, audit):
        rel = env.write()
        before_logits, before = env.logits(), param_fingerprint(env.model)
        names = [n for n, _ in env.model.named_parameters()]
        buffers = [n for n, _ in env.model.named_buffers()]
        result = await env.service.load(identity(), rel)
        assert result["adapter"]["name"] == ADAPTER_NAME
        assert result["adapter"]["target_count"] == 6  # q, v, down on two layers
        assert result["resolved_model_name"] == ADAPTER_NAME
        assert resident_identity.served_name() == ADAPTER_NAME
        assert resident_identity.base_served_name() == "tiny"
        assert not torch.equal(env.logits(), before_logits)
        assert param_fingerprint(env.model) == before
        assert [n for n, _ in env.model.named_parameters()] == names
        assert [n for n, _ in env.model.named_buffers()] == buffers
        assert _hook_count(env.model) == 6
        assert not get_adapter_state().loading
        audit.assert_called_once()
        call = audit.call_args
        assert call.args[:2] == ("adapter/load", "accepted")
        assert call.kwargs["adapter"] == ADAPTER_NAME
        assert call.kwargs["weights_sha256"] == result["adapter"]["weights_sha256"]
        assert call.kwargs["path"] == rel

    async def test_unload_returns_the_base_bit_for_bit(self, env, audit):
        rel = env.write()
        base_logits, before = env.logits(), param_fingerprint(env.model)
        await env.service.load(identity(), rel)
        applied = get_adapter_state().current
        result = await env.service.unload(identity())
        assert result["unloaded"]["name"] == ADAPTER_NAME
        assert result["resolved_model_name"] == "tiny"
        assert torch.equal(env.logits(), base_logits)
        assert param_fingerprint(env.model) == before
        assert _hook_count(env.model) == 0
        assert applied.prepared.weights == {}, "tensors are released"
        assert audit.call_args.args[:2] == ("adapter/unload", "accepted")

    async def test_unload_with_nothing_applied(self, env, audit):
        with patch.object(env.barrier, "serving_pause", wraps=env.barrier.serving_pause) as pause, \
                pytest.raises(NoAdapterLoadedError):
            await env.service.unload(identity())
        assert pause.call_count == 0
        assert audit.call_args.kwargs["code"] == "NO_ADAPTER_LOADED"

    async def test_unload_honours_a_foreign_lease(self, env):
        rel = env.write()
        await env.service.load(identity(), rel)
        env.leases.grant(1, "tiny", "midataworks", "run", 120)
        with pytest.raises(ModelLeasedError):
            await env.service.unload(identity())
        assert get_adapter_state().current is not None

    async def test_two_concurrent_loads_give_one_adapter_and_one_409(self, env):
        rel = env.write()
        results = await asyncio.gather(
            env.service.load(identity(), rel), env.service.load(identity(), rel),
            return_exceptions=True,
        )
        errors = [r for r in results if isinstance(r, BaseException)]
        assert len(errors) == 1 and isinstance(errors[0], AdapterAlreadyLoadedError)
        assert errors[0].details["reason"] == "load_in_progress"
        assert _hook_count(env.model) == 6

    async def test_the_base_changed_between_verification_and_the_pause(self, env, audit):
        rel = env.write()
        real = adapter_service.prepare_adapter

        def prepare_then_replace(*args, **kwargs):
            artifact = real(*args, **kwargs)
            resident_identity.advance_base_epoch("test: replaced while verifying")
            return artifact

        with patch.object(adapter_service, "prepare_adapter", side_effect=prepare_then_replace), \
                patch.object(env.barrier, "serving_pause", wraps=env.barrier.serving_pause) as pause, \
                pytest.raises(AdapterBaseChangedError):
            await env.service.load(identity(), rel)
        assert pause.call_count == 1, "refused INSIDE the pause"
        assert _hook_count(env.model) == 0
        assert resident_identity.served_name() == "tiny"
        assert not get_adapter_state().loading
        assert audit.call_args.kwargs["code"] == "ADAPTER_BASE_CHANGED"
        # The slot is free again: the same adapter loads against the (same) base now.
        assert (await env.service.load(identity(), rel))["adapter"]["name"] == ADAPTER_NAME

    async def test_an_install_failure_rolls_back(self, env):
        rel = env.write()
        calls = {"n": 0}
        real_register = torch.nn.Module.register_forward_hook

        def flaky(self, *a, **k):
            calls["n"] += 1
            if calls["n"] == 3:
                raise RuntimeError("boom")
            return real_register(self, *a, **k)

        with patch.object(torch.nn.Module, "register_forward_hook", flaky), \
                pytest.raises(AdapterInstallFailedError):
            await env.service.load(identity(), rel)
        assert _hook_count(env.model) == 0
        assert get_adapter_state().current is None and not get_adapter_state().loading
        assert resident_identity.served_name() == "tiny"

    @pytest.mark.parametrize("error", [
        QueueFullError("queue full", details={}),
        ModelBusyError("unloading", details={}),
    ], ids=["queue_full", "model_busy"])
    async def test_a_refusal_from_the_pause_installs_nothing(self, env, audit, error):
        """033 review 13: `_admit` can refuse the barrier's own hold."""
        rel = env.write()

        def refusing():
            raise error

        with patch.object(env.inference, "hold_admission", side_effect=refusing), \
                pytest.raises(type(error)):
            await env.service.load(identity(), rel)
        assert _hook_count(env.model) == 0
        assert get_adapter_state().current is None and not get_adapter_state().loading
        assert resident_identity.served_name() == "tiny"
        assert audit.call_args.kwargs["code"] == error.code

    async def test_no_commit_bracket_and_no_state_change(self, env):
        rel = env.write()
        before = env.barrier.snapshot()
        with patch.object(env.barrier, "commit_begin") as begin, \
                patch.object(env.barrier, "commit_end") as end:
            await env.service.load(identity(), rel)
            await env.service.unload(identity())
        assert begin.call_count == 0 and end.call_count == 0
        after = env.barrier.snapshot()
        assert (after.weights_state, after.steps_since_load) == (
            before.weights_state, before.steps_since_load)

    async def test_names_and_hooks_flip_inside_one_pause(self, env):
        """Observed from inside the pause: before the commit neither has flipped; the request
        admitted after the pause sees both."""
        rel = env.write()
        seen = {}
        real_commit = get_adapter_state().commit

        def observing_commit(applied, model, epoch):
            seen["held"] = env.barrier.holds_slot()
            seen["before"] = (resident_identity.served_name(), _hook_count(env.model))
            real_commit(applied, model, epoch)
            seen["after"] = (resident_identity.served_name(), _hook_count(env.model))

        with patch.object(get_adapter_state(), "commit", side_effect=observing_commit):
            await env.service.load(identity(), rel)
        assert seen["held"] is True
        assert seen["before"] == ("tiny", 0)
        assert seen["after"] == (ADAPTER_NAME, 6)


# ---------------------------------------------------------------------------------- status


class TestStatus:
    async def test_nothing_applied(self, env):
        status = env.service.status()
        assert status["applied"] is False and status["base_steps_since_load"] is None

    async def test_applied_and_a_commit_while_applied(self, env):
        rel = env.write()
        await env.service.load(identity(), rel)
        status = env.service.status()
        assert status["applied"] is True
        assert status["name"] == ADAPTER_NAME and status["base_name"] == "tiny"
        assert status["rank"] == 4 and status["target_count"] == 6
        assert status["base_steps_since_load"] == 0
        with env.barrier._lock:  # what a commit does to the counter (D-16)
            env.barrier._steps_since_load += 1
        assert env.service.status()["base_steps_since_load"] == 1

    async def test_not_recorded_is_null_never_zero(self, env):
        rel = env.write()
        await env.service.load(identity(), rel)
        get_adapter_state().current.steps_at_load = None
        assert env.service.status()["base_steps_since_load"] is None


# ---------------------------------------------------------------------------------- the drop


class TestDropOnBaseReplacement:
    def test_the_drop_and_overlay_are_registered_in_the_live_registry(self):
        from millm.main import create_app

        create_app()
        assert adapter_service.drop_adapter_for_base_change in (
            resident_identity.base_replaced_callbacks())
        assert adapter_service.applied_adapter_name in resident_identity.served_name_overlays()

    async def test_a_real_unload_drops_once_and_releases(self, env, audit):
        rel = env.write()
        await env.service.load(identity(), rel)
        applied = get_adapter_state().current
        state = get_adapter_state()
        with patch.object(state, "take", wraps=state.take) as take:
            assert ModelLoader().unload() is True  # the real path every unload ends in
        assert take.call_count == 1
        assert state.current is None and applied.prepared.weights == {}
        assert _hook_count(env.model) == 0
        drop = [c for c in audit.call_args_list if c.args[0] == "adapter/drop"]
        assert len(drop) == 1 and drop[0].kwargs["reason"] == "model_unload"
        assert drop[0].kwargs["adapter"] == ADAPTER_NAME

    async def test_loaded_model_state_clear_drops_before_the_weights_move(self, env):
        rel = env.write()
        await env.service.load(identity(), rel)
        seen = {}
        real_take = get_adapter_state().take

        def take():
            seen["resident_at_drop"] = LoadedModelState().current is not None
            return real_take()

        with patch.object(get_adapter_state(), "take", side_effect=take):
            LoadedModelState().clear()
        assert seen["resident_at_drop"] is True
        assert get_adapter_state().current is None

    async def test_one_drop_call_per_base_replacement(self, env):
        from millm.ml.model_loader import LoadedModel
        from datetime import datetime

        state = get_adapter_state()
        with patch.object(state, "take", wraps=state.take) as take:
            LoadedModelState().set(LoadedModel(
                model_id=2, model_name="other", model=env.model, tokenizer=None,
                loaded_at=datetime(2026, 10, 14)))
            assert take.call_count == 1
            LoadedModelState().clear()
            assert take.call_count == 2

    async def test_the_drop_never_raises(self, env):
        rel = env.write()
        await env.service.load(identity(), rel)
        with patch.object(LoraHookSet, "detach", side_effect=RuntimeError("boom")):
            assert adapter_service.drop_adapter_for_base_change("test") is False
        # The replacement itself proceeds.
        LoadedModelState().clear()
        assert LoadedModelState().current is None

    async def test_a_drop_with_nothing_applied_is_a_no_op(self, env):
        assert adapter_service.drop_adapter_for_base_change("test") is False

    async def test_a_replacement_during_the_commit_is_serialised_by_the_lock(self, env):
        """033 review 11, mutation M34. A base replacement in another thread (the unload
        worker) fires while the load is committing. With the lock the drop waits, then removes
        what the load installed; without it, the drop finds nothing and the stale adapter stays
        hooked onto a replaced base."""
        rel = env.write()
        real_install = LoraHookSet.install
        threads = []

        def install_with_concurrent_replacement(model, prepared):
            t = threading.Thread(
                target=resident_identity.begin_base_replacement, args=("forced_unload",))
            t.start()
            t.join(timeout=0.3)  # with the lock the drop blocks here; without it, it finishes
            threads.append(t)
            return real_install(model, prepared)

        with patch.object(LoraHookSet, "install", side_effect=install_with_concurrent_replacement):
            await env.service.load(identity(), rel)
        threads[0].join(timeout=5)
        assert not threads[0].is_alive()
        assert get_adapter_state().current is None
        assert _hook_count(env.model) == 0


# ---------------------------------------------------------------------------------- wiring by AST


def _names(module) -> set[str]:
    tree = ast.parse(inspect.getsource(module))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            out.update(a.name for a in node.names)
            if isinstance(node, ast.ImportFrom) and node.module:
                out.add(node.module)
    return out


class TestNoAttachmentNoCommit:
    @pytest.mark.parametrize("module_name", [
        "millm.services.adapter_service", "millm.ml.adapter_hooks",
    ])
    def test_never_references_the_attachment_module(self, module_name):
        import importlib

        names = _names(importlib.import_module(module_name))
        forbidden = {"weight_attachment", "millm.services.weight_attachment",
                     "get_weight_service", "is_attached", "attachment_view"}
        assert not names & forbidden

    def test_never_calls_the_commit_bracket(self):
        names = _names(adapter_service)
        assert not names & {"commit_begin", "commit_end", "internal_write"}

    def test_the_walk_sees_a_known_name(self):
        """Guard the guard: an AST walk that saw nothing would pass every row."""
        assert {"serving_pause", "register_on_base_replaced"} <= _names(adapter_service)
