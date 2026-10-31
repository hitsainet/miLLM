"""R1-A review regressions for `AdapterService` (033, review round 1, 2026-10-14).

Each test drives the REAL object a fixture used to stand in for. The earlier epoch test advanced
the counter by hand; a real unload also DELETES `LoadedModel.model`, and the load read that
attribute before its epoch check — an AttributeError 500 with no audit line, where the contract
says `409 ADAPTER_BASE_CHANGED`.

MUTATION CONTROLS (R1-A): C1 read `loaded.model` before the epoch check -> TestRealReplacement;
C2 delete the in-pause lease re-check -> TestLeaseGrantedMidLoad; C3 delete the `_failed` audit
branch -> TestUnexpectedExceptionsAreAudited. Survivors of the first R1-A pass, now pinned:
m24 (unload's in-pause None check), m27/m28 (unload's pre-pause clean/CBM/concurrency refusals),
m29 (status after a counter reset), m42 (`committed = True`) -> TestUnloadAndStatusGaps;
m44 (`resident_base` back on the loop) -> TestNoFileIoOnTheLoop.
"""

from __future__ import annotations

import contextlib
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from millm.core.errors import (
    AdapterBaseChangedError,
    AdapterNotServableError,
    ModelLeasedError,
    NoAdapterLoadedError,
    WeightsBarrierUnavailableError,
    WeightsNotCleanError,
)
from millm.ml.model_loader import LoadedModel, LoadedModelState
from millm.services import adapter_service, resident_identity
from millm.services.adapter_service import get_adapter_state
from tests.unit.adapter_fixtures import tiny_llama
from tests.unit.adapter_service_fixtures import ADAPTER_NAME, build_env, identity


@pytest.fixture
def env(tmp_path):
    e = build_env(tmp_path)
    yield e
    e.close()


@pytest.fixture
def audit():
    with patch.object(adapter_service, "weights_audit") as spy:
        yield spy


def _hooks(model) -> int:
    return sum(len(m._forward_hooks) for m in model.modules())


def _prepare_then(action):
    """Wrap the real `prepare_adapter` so `action` runs after verification, before the pause;
    the prepared adapter is captured so the test can check its tensors were released."""
    real = adapter_service.prepare_adapter
    seen = {}

    def run(*args, **kwargs):
        artifact = real(*args, **kwargs)
        seen["prepared"] = artifact.prepared
        action()
        return artifact

    return run, seen


class TestRealReplacement:
    async def test_a_real_unload_while_verifying_is_base_changed_not_a_500(self, env, audit):
        rel = env.write()
        side, seen = _prepare_then(lambda: LoadedModelState().clear())
        with patch.object(adapter_service, "prepare_adapter", side_effect=side), \
                pytest.raises(AdapterBaseChangedError):
            await env.service.load(identity(), rel)
        assert audit.call_args.kwargs["code"] == "ADAPTER_BASE_CHANGED"
        assert audit.call_args.args[1] == "refused"
        state = get_adapter_state()
        assert state.current is None and not state.loading
        assert seen["prepared"].weights == {} and seen["prepared"].modules == {}
        assert _hooks(env.model) == 0
        assert resident_identity.served_name() is None

    async def test_a_real_replacement_while_verifying_hooks_neither_model(self, env, audit):
        rel = env.write()
        other = tiny_llama()

        def replace():
            LoadedModelState().clear()
            LoadedModelState().set(LoadedModel(
                model_id=2, model_name="other", model=other, tokenizer=None,
                loaded_at=datetime(2026, 10, 14)))

        side, seen = _prepare_then(replace)
        with patch.object(adapter_service, "prepare_adapter", side_effect=side), \
                pytest.raises(AdapterBaseChangedError):
            await env.service.load(identity(), rel)
        assert audit.call_args.kwargs["code"] == "ADAPTER_BASE_CHANGED"
        assert _hooks(other) == 0 and _hooks(env.model) == 0
        assert get_adapter_state().current is None and not get_adapter_state().loading
        assert seen["prepared"].weights == {}
        assert resident_identity.served_name() == "other"


class TestReplacedBeforeThePrepareStarts:
    """R2-C: R1-A finding 1 on the EARLIER path. `_load` read `loaded.model` after three awaits
    (row lookup, registered names, resident identity); a forced unload in any of them deletes
    `LoadedModel.model`, so the read raised AttributeError -> a 500. Negative control: pass
    `loaded.model` to the thread again -> AttributeError, not ADAPTER_BASE_CHANGED."""

    async def test_an_unload_during_the_name_lookup_is_base_changed(self, env, audit):
        rel = env.write()
        real = env.service._names_lookup

        async def unloading():
            names = await real()
            LoadedModelState().clear()
            return names

        env.service._names_lookup = unloading
        with patch.object(adapter_service, "prepare_adapter") as prepare, \
                pytest.raises(AdapterBaseChangedError):
            await env.service.load(identity(), rel)
        assert prepare.call_count == 0, "nothing is read once the base is gone"
        assert audit.call_args.args[1] == "refused"
        assert audit.call_args.kwargs["code"] == "ADAPTER_BASE_CHANGED"
        state = get_adapter_state()
        assert state.current is None and not state.loading

    async def test_a_replacement_during_the_name_lookup_never_prepares_on_either(self, env, audit):
        rel = env.write()
        other = tiny_llama()
        real = env.service._names_lookup

        async def replacing():
            names = await real()
            LoadedModelState().clear()
            LoadedModelState().set(LoadedModel(
                model_id=2, model_name="other", model=other, tokenizer=None,
                loaded_at=datetime(2026, 10, 14)))
            return names

        env.service._names_lookup = replacing
        with patch.object(adapter_service, "prepare_adapter") as prepare, \
                pytest.raises(AdapterBaseChangedError):
            await env.service.load(identity(), rel)
        assert prepare.call_count == 0
        assert _hooks(other) == 0 and not get_adapter_state().loading


class TestLeaseGrantedMidLoad:
    async def test_a_lease_granted_while_verifying_refuses_inside_the_pause(self, env, audit):
        rel = env.write()
        side, seen = _prepare_then(
            lambda: env.leases.grant(1, "tiny", "midataworks", "label run 7", 120))
        with patch.object(adapter_service, "prepare_adapter", side_effect=side), \
                patch.object(env.barrier, "serving_pause",
                             wraps=env.barrier.serving_pause) as pause, \
                pytest.raises(ModelLeasedError) as exc:
            await env.service.load(identity(), rel)
        assert pause.call_count == 1, "refused inside the pause"
        assert exc.value.details["holder"] == "midataworks"
        assert audit.call_args.kwargs["code"] == "MODEL_LEASED"
        assert _hooks(env.model) == 0
        assert get_adapter_state().current is None and not get_adapter_state().loading
        assert seen["prepared"].weights == {}
        assert resident_identity.served_name() == "tiny"

    async def test_a_lease_granted_while_unload_waits_for_the_slot_refuses_inside_the_pause(
        self, env, audit
    ):
        """R2-C: unload checked the lease only before the pause, which waits for the request in
        flight (a whole stream). Negative control: drop the in-pause re-check -> the adapter is
        removed under the new leaseholder."""
        rel = env.write()
        await env.service.load(identity(), rel)
        real_pause = env.barrier.serving_pause

        @contextlib.asynccontextmanager
        async def lease_granted_while_waiting(*args, **kwargs):
            env.leases.grant(1, "tiny", "midataworks", "label run 8", 120)
            async with real_pause(*args, **kwargs) as guard:
                yield guard

        with patch.object(env.barrier, "serving_pause", lease_granted_while_waiting), \
                pytest.raises(ModelLeasedError) as exc:
            await env.service.unload(identity())
        assert exc.value.details["holder"] == "midataworks"
        assert audit.call_args.kwargs["code"] == "MODEL_LEASED"
        assert get_adapter_state().current is not None, "the adapter is still applied"
        assert _hooks(env.model) > 0

    async def test_the_holder_passes_both_checks(self, env):
        """The in-pause re-check must still let the holder's `X-miLLM-Lease` through."""
        rel = env.write()
        grant = env.leases.grant(1, "tiny", "forge", "run 9", 120)
        with patch.object(env.service, "_refuse_if_leased",
                          wraps=env.service._refuse_if_leased) as check:
            result = await env.service.load(identity(), rel, lease_id=grant.lease_id)
        assert result["adapter"]["name"] == ADAPTER_NAME
        assert check.call_count == 2, "checked before the read AND inside the pause"
        assert all(c.args[2] == grant.lease_id for c in check.call_args_list)


class TestUnexpectedExceptionsAreAudited:
    async def test_an_unexpected_load_failure_writes_one_refused_audit_line(self, env, audit):
        rel = env.write()
        env.row_lookup.side_effect = RuntimeError("database is down: secret-ish detail")
        with pytest.raises(RuntimeError):
            await env.service.load(identity(), rel)
        assert audit.call_count == 1
        call = audit.call_args
        assert call.args[:2] == ("adapter/load", "refused")
        assert call.kwargs["code"] == "INTERNAL_ERROR"
        assert call.kwargs["reason"] == "RuntimeError"
        assert "secret-ish" not in repr(call)
        assert not get_adapter_state().loading

    async def test_an_unexpected_unload_failure_writes_one_refused_audit_line(self, env, audit):
        rel = env.write()
        await env.service.load(identity(), rel)
        audit.reset_mock()
        with patch.object(get_adapter_state(), "take", side_effect=RuntimeError("boom")), \
                pytest.raises(RuntimeError):
            await env.service.unload(identity())
        assert audit.call_count == 1
        assert audit.call_args.args[:2] == ("adapter/unload", "refused")
        assert audit.call_args.kwargs["code"] == "INTERNAL_ERROR"


class TestUnloadAndStatusGaps:
    async def test_a_drop_while_the_unload_pause_is_taken_is_no_adapter_loaded(self, env, audit):
        """The base was replaced between the unload's check and its pause: the drop already
        removed everything, so the unload is a clean 409, never an AttributeError 500."""
        rel = env.write()
        await env.service.load(identity(), rel)
        real = env.barrier.serving_pause

        @contextlib.asynccontextmanager
        async def pause_then_drop(operation, who):
            async with real(operation, who) as guard:
                assert adapter_service.drop_adapter_for_base_change("test: forced unload")
                yield guard

        audit.reset_mock()
        with patch.object(env.barrier, "serving_pause", side_effect=pause_then_drop), \
                pytest.raises(NoAdapterLoadedError):
            await env.service.unload(identity())
        refused = [c for c in audit.call_args_list if c.args[:2] == ("adapter/unload", "refused")]
        assert len(refused) == 1 and refused[0].kwargs["code"] == "NO_ADAPTER_LOADED"
        assert _hooks(env.model) == 0

    async def test_unload_while_not_clean_refuses_before_the_pause(self, env):
        rel = env.write()
        await env.service.load(identity(), rel)
        env.barrier._weights_state = "torn"
        try:
            with patch.object(env.barrier, "serving_pause",
                              wraps=env.barrier.serving_pause) as pause, \
                    pytest.raises(WeightsNotCleanError):
                await env.service.unload(identity())
        finally:
            env.barrier._weights_state = "clean"
        assert pause.call_count == 0
        assert get_adapter_state().current is not None

    async def test_unload_with_cbm_is_adapter_not_servable_reason_cbm(self, env):
        """D-13: the adapter routes refuse CBM as ADAPTER_NOT_SERVABLE / cbm, not as the
        barrier's BARRIER_UNAVAILABLE."""
        rel = env.write()
        await env.service.load(identity(), rel)
        env.inference._cbm_backend = MagicMock(is_running=True)
        try:
            with patch.object(env.barrier, "serving_pause",
                              wraps=env.barrier.serving_pause) as pause, \
                    pytest.raises(AdapterNotServableError) as exc:
                await env.service.unload(identity())
        finally:
            env.inference._cbm_backend = None
        assert exc.value.details["reason"] == "cbm"
        assert pause.call_count == 0

    async def test_unload_with_two_slots_refuses_concurrency_before_the_pause(self, env):
        rel = env.write()
        await env.service.load(identity(), rel)
        env.inference._request_queue._max_concurrent = 2
        try:
            with patch.object(env.barrier, "serving_pause",
                              wraps=env.barrier.serving_pause) as pause, \
                    pytest.raises(WeightsBarrierUnavailableError) as exc:
                await env.service.unload(identity())
        finally:
            env.inference._request_queue._max_concurrent = 1
        assert exc.value.details["reason"] == "concurrency"
        assert pause.call_count == 0

    async def test_a_counter_reset_reads_not_recorded_never_negative(self, env):
        rel = env.write()
        await env.service.load(identity(), rel)
        get_adapter_state().current.steps_at_load = 5
        assert env.barrier.snapshot().steps_since_load == 0
        assert env.service.status()["base_steps_since_load"] is None

    async def test_the_applied_adapter_keeps_its_tensors_until_unload(self, env):
        """A successful load must not run the failure path's release: the applied record holds
        the tensors (and the unload releases them)."""
        rel = env.write()
        await env.service.load(identity(), rel)
        applied = get_adapter_state().current
        assert len(applied.prepared.weights) == 6 and len(applied.prepared.modules) == 6
        await env.service.unload(identity())
        assert applied.prepared.weights == {} and applied.prepared.modules == {}


class TestNoFileIoOnTheLoop:
    async def test_the_resident_identity_is_read_off_the_event_loop(self, env):
        """D-09: a LOCAL row's identity hashes a file. Control m44: call `resident_base`
        directly on the loop -> red."""
        import threading

        rel = env.write()
        seen = []
        real = adapter_service.resident_base

        def spy(*args, **kwargs):
            seen.append(threading.current_thread() is threading.main_thread())
            return real(*args, **kwargs)

        with patch.object(adapter_service, "resident_base", side_effect=spy):
            await env.service.load(identity(), rel)
        assert seen == [False]
