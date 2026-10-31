"""W3 integration regressions for `AdapterService` (033, 2026-10-14).

Each class pins one line the W3 pass added or changed, and each has a mutation control (the
mutation was applied, this file went red, the line was restored):

- TestRegisteredNames: the database's row names reach `prepare_adapter` (D-32 in full). Control:
  pass `registered_names=()` instead of the looked-up names.
- TestUnexpectedPrepareErrorsAreRefusals: a torch / safetensors `RuntimeError` or an `OSError`
  escaping `prepare_adapter` becomes `ADAPTER_INSTALL_FAILED` / `ADAPTER_INVALID` with a refused
  audit line (R1-B). Controls: call `prepare_adapter` instead of `prepare_adapter_mapped`; swap
  the two mapped classes.
- TestCancellation: a cancellation during the worker thread keeps the claim until the thread is
  done (R1-A finding 8); a cancellation after the commit still audits `accepted`. Controls:
  release the claim unconditionally; drop the `_cancelled` branch.
- TestDynamoResetOutsideTheLock: `AdapterState.take` resets Dynamo after releasing its lock
  (R1-A finding 7). Control: call `LoraHookSet.remove` under the lock again.
- TestTheSharedLeaseRule: the adapter refuses through `model_lease.refuse_if_leased`, the rule
  `ModelService` uses. Control: re-inline a lookup that bypasses it.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import torch

from millm.core.errors import (
    AdapterAlreadyLoadedError,
    AdapterHashMismatchError,
    AdapterInstallFailedError,
    AdapterInvalidError,
    AdapterManifestInvalidError,
    InsufficientMemoryError,
)
from millm.ml import adapter_hooks
from millm.services import adapter_service, model_lease
from millm.services.adapter_service import (
    OP_LOAD,
    OP_UNLOAD,
    addressable_names,
    get_adapter_state,
)
from tests.unit.adapter_fixtures import BASE_NAME
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


def _audits(spy, operation):
    return [c for c in spy.call_args_list if c.args[0] == operation]


# --- D-32: registered names ---------------------------------------------------------------------


class TestRegisteredNames:
    async def test_the_looked_up_names_reach_prepare(self, env):
        env.names_lookup.return_value = ["tiny", BASE_NAME, "other-row"]
        rel = env.write()
        with patch.object(adapter_service, "prepare_adapter",
                          wraps=adapter_service.prepare_adapter) as prepare:
            await env.service.load(identity(), rel)
        env.names_lookup.assert_awaited_once_with()
        assert prepare.call_count == 1
        assert prepare.call_args.kwargs["registered_names"] == ("tiny", BASE_NAME, "other-row")

    async def test_an_adapter_named_like_another_row_is_refused(self, env, audit):
        env.names_lookup.return_value = ["tiny", BASE_NAME, "other-row"]
        rel = env.write(manifest_overrides={"name": "other-row"})
        with pytest.raises(AdapterManifestInvalidError) as exc:
            await env.service.load(identity(), rel)
        assert exc.value.details["reason"] == "name"
        assert exc.value.details["problem"] == "registered_name"
        assert audit.call_args.args[:2] == (OP_LOAD, "refused")
        assert audit.call_args.kwargs["code"] == "ADAPTER_MANIFEST_INVALID"
        assert _hooks(env.model) == 0
        assert not get_adapter_state().loading

    def test_addressable_names_cover_bare_names_and_repo_ids(self):
        rows = [
            SimpleNamespace(name="org/m-GGUF:Q4_K_M", repo_id="org/m-GGUF", weights_origin=None),
            SimpleNamespace(name="tiny", repo_id=None, weights_origin=None),
            # A checkpoint carrier is reachable only by its full name (find_by_name).
            SimpleNamespace(name="base:r9:step-5", repo_id=None, weights_origin="saved"),
        ]
        assert addressable_names(rows) == {
            "org/m-GGUF:Q4_K_M", "org/m-GGUF", "tiny", "base:r9:step-5",
        }

    async def test_the_default_lookup_reads_every_row(self):
        rows = [SimpleNamespace(name="a:Q8", repo_id="org/a", weights_origin=None)]

        class _Session:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

        with patch("millm.db.base.async_session_factory", lambda: _Session()), \
                patch("millm.db.repositories.model_repository.ModelRepository.get_all",
                      AsyncMock(return_value=rows)):
            names = await adapter_service._default_registered_names()
        assert names == ["a", "a:Q8", "org/a"]

    def test_the_live_service_uses_the_database_lookup(self):
        assert adapter_service.AdapterService()._names_lookup is (
            adapter_service._default_registered_names
        )


# --- R1-B: unexpected errors from prepare --------------------------------------------------------


class TestUnexpectedPrepareErrorsAreRefusals:
    @pytest.mark.parametrize("raised, expected, reason", [
        (RuntimeError("CUDA error: /secret/path"), AdapterInstallFailedError, "prepare_failed"),
        (ValueError("odd /secret/path"), AdapterInstallFailedError, "prepare_failed"),
        (OSError(5, "I/O error", "/secret/path"), AdapterInvalidError, "unreadable"),
        (PermissionError("/secret/path"), AdapterInvalidError, "unreadable"),
    ])
    async def test_mapped_and_audited(self, env, audit, raised, expected, reason):
        rel = env.write()
        with patch.object(adapter_service, "prepare_adapter", side_effect=raised), \
                pytest.raises(expected) as exc:
            await env.service.load(identity(), rel)
        assert exc.value.details == {"reason": reason, "error": type(raised).__name__}
        assert "/secret/path" not in exc.value.message
        refused = _audits(audit, OP_LOAD)
        assert len(refused) == 1
        assert refused[0].args[1] == "refused"
        assert refused[0].kwargs["code"] == expected.code
        assert refused[0].kwargs["reason"] == reason
        assert "/secret/path" not in repr(refused[0])
        assert not get_adapter_state().loading and get_adapter_state().current is None

    @pytest.mark.parametrize("raised", [MemoryError(), torch.OutOfMemoryError("CUDA out of memory")])
    async def test_out_of_memory_is_507_not_an_install_failure(self, env, audit, raised):
        """Out of memory anywhere in the preparation is INSUFFICIENT_MEMORY (507), as the
        placement already is — never ADAPTER_INSTALL_FAILED (R2-C). Control: drop the
        MemoryError branch -> AdapterInstallFailedError."""
        rel = env.write()
        with patch.object(adapter_service, "prepare_adapter", side_effect=raised), \
                pytest.raises(InsufficientMemoryError) as exc:
            await env.service.load(identity(), rel)
        assert exc.value.details == {"reason": "prepare_out_of_memory",
                                     "error": type(raised).__name__}
        refused = _audits(audit, OP_LOAD)
        assert len(refused) == 1 and refused[0].kwargs["code"] == InsufficientMemoryError.code
        assert not get_adapter_state().loading

    def test_out_of_memory_while_parsing_the_weights_is_507(self, env, monkeypatch):
        """`load_safetensors` running out of memory is not an unreadable file. Control: drop the
        MemoryError branch in `prepare_adapter` -> AdapterInvalidError (unreadable)."""
        import safetensors.torch

        from millm.services import adapter_artifact

        rel = env.write()
        monkeypatch.setattr(safetensors.torch, "load",
                            lambda data: (_ for _ in ()).throw(MemoryError()))
        resident = adapter_artifact.ResidentBase(
            manifest_sha256=None, names=frozenset({BASE_NAME}), served_names=frozenset({BASE_NAME})
        )
        with pytest.raises(InsufficientMemoryError):
            adapter_artifact.prepare_adapter(
                rel, env.model, resident, roots=[("primary", str(env.root))]
            )

    async def test_a_public_refusal_passes_through_unchanged(self, env):
        rel = env.write()
        original = AdapterHashMismatchError("x", details={"expected": "a", "actual": "b"})
        with patch.object(adapter_service, "prepare_adapter", side_effect=original), \
                pytest.raises(AdapterHashMismatchError) as exc:
            await env.service.load(identity(), rel)
        assert exc.value is original

    async def test_cancellation_is_not_mapped(self):
        with patch.object(adapter_service, "prepare_adapter",
                          side_effect=KeyboardInterrupt), pytest.raises(KeyboardInterrupt):
            adapter_service.prepare_adapter_mapped("x")


# --- R1-A finding 8: cancellation ----------------------------------------------------------------


class TestCancellation:
    async def test_a_cancel_during_prepare_keeps_the_claim_until_the_thread_ends(self, env, audit):
        rel = env.write()
        started, go = threading.Event(), threading.Event()
        real = adapter_service.prepare_adapter
        built = {}

        def slow(*args, **kwargs):
            started.set()
            assert go.wait(10)
            artifact = real(*args, **kwargs)
            built["prepared"] = artifact.prepared
            return artifact

        with patch.object(adapter_service, "prepare_adapter", side_effect=slow):
            task = asyncio.create_task(env.service.load(identity(), rel))
            assert await asyncio.to_thread(started.wait, 10)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            # The thread still runs: the slot must stay claimed, so a second load cannot
            # prepare alongside it.
            assert get_adapter_state().loading
            with pytest.raises(AdapterAlreadyLoadedError) as second:
                await env.service.load(identity(), rel)
            assert second.value.details["reason"] == "load_in_progress"
            go.set()
            deadline = time.monotonic() + 10
            while get_adapter_state().loading and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
        assert not get_adapter_state().loading, "released once the thread finished"
        assert get_adapter_state().current is None
        assert built["prepared"].weights == {} and built["prepared"].modules == {}
        assert _hooks(env.model) == 0
        first = _audits(audit, OP_LOAD)[0]
        assert first.args[1] == "refused" and first.kwargs["code"] == "CANCELLED"

    async def test_a_cancel_that_lands_after_the_thread_finished_drops_its_tensors(
        self, env, audit
    ):
        """The cancellation can win the race to the frame after the worker thread has finished:
        `prepare.done()` is then true, no done-callback is registered, and the tensors it built
        must be dropped here, with the claim released once (R2-C). Negative control: remove the
        `else` branch -> the built tensors survive the cancel."""
        rel = env.write()
        real = adapter_service.prepare_adapter
        built = {}

        def capture(*args, **kwargs):
            artifact = real(*args, **kwargs)
            built["prepared"] = artifact.prepared
            return artifact

        async def finished_then_cancelled(fut):
            await fut
            raise asyncio.CancelledError()

        with patch.object(adapter_service, "prepare_adapter", side_effect=capture), \
                patch.object(adapter_service.asyncio, "shield", finished_then_cancelled), \
                pytest.raises(asyncio.CancelledError):
            await env.service.load(identity(), rel)
        assert built["prepared"].weights == {} and built["prepared"].modules == {}
        assert not get_adapter_state().loading, "the claim is released (once, in finally)"
        assert get_adapter_state().current is None and _hooks(env.model) == 0
        lines = _audits(audit, OP_LOAD)
        assert len(lines) == 1 and lines[0].kwargs["code"] == "CANCELLED"
        # And the slot is usable again: a second load succeeds.
        await env.service.load(identity(), rel)
        assert get_adapter_state().current is not None

    async def test_a_cancel_after_the_commit_still_audits_accepted(self, env, audit):
        rel = env.write()
        real_pause = env.barrier.serving_pause

        @contextlib.asynccontextmanager
        async def cancelled_on_exit(*args, **kwargs):
            async with real_pause(*args, **kwargs) as guard:
                yield guard
            raise asyncio.CancelledError()

        with patch.object(env.barrier, "serving_pause", cancelled_on_exit), \
                pytest.raises(asyncio.CancelledError):
            await env.service.load(identity(), rel)
        assert get_adapter_state().current is not None, "the commit took effect"
        assert _hooks(env.model) > 0
        lines = _audits(audit, OP_LOAD)
        assert len(lines) == 1
        assert lines[0].args[1] == "accepted"
        assert lines[0].kwargs["adapter"] == ADAPTER_NAME
        assert not get_adapter_state().loading

    async def test_a_cancel_after_the_unload_took_effect_still_audits_accepted(self, env, audit):
        rel = env.write()
        await env.service.load(identity(), rel)
        prepared = get_adapter_state().current.prepared
        real_pause = env.barrier.serving_pause

        @contextlib.asynccontextmanager
        async def cancelled_on_exit(*args, **kwargs):
            async with real_pause(*args, **kwargs) as guard:
                yield guard
            raise asyncio.CancelledError()

        with patch.object(env.barrier, "serving_pause", cancelled_on_exit), \
                pytest.raises(asyncio.CancelledError):
            await env.service.unload(identity())
        assert get_adapter_state().current is None and _hooks(env.model) == 0
        lines = _audits(audit, OP_UNLOAD)
        assert len(lines) == 1 and lines[0].args[1] == "accepted"
        assert prepared.weights == {}, "tensors released even though the caller was cancelled"


# --- R1-A finding 7: Dynamo reset outside the lock ------------------------------------------------


class TestDynamoResetOutsideTheLock:
    async def test_an_overlay_read_is_not_blocked_by_a_slow_reset(self, env):
        rel = env.write()
        await env.service.load(identity(), rel)
        in_reset, release = threading.Event(), threading.Event()
        calls = []

        def slow_reset():
            calls.append(threading.current_thread().name)
            in_reset.set()
            release.wait(10)

        state = get_adapter_state()
        with patch.object(adapter_hooks, "reset_dynamo_for_hook_change", slow_reset):
            dropper = threading.Thread(target=state.take, name="unload-worker")
            dropper.start()
            try:
                assert in_reset.wait(10), "take resets Dynamo"
                reader = {}

                def read():
                    reader["name"] = state.applied_name()

                t = threading.Thread(target=read)
                t.start()
                t.join(timeout=1.0)
                assert not t.is_alive(), "the overlay read waited on the Dynamo reset"
                assert reader["name"] is None
            finally:
                release.set()
                dropper.join(10)
        assert calls == ["unload-worker"], "exactly one reset, by the dropping thread"
        assert _hooks(env.model) == 0

    def test_remove_still_resets(self, monkeypatch):
        calls = []
        monkeypatch.setattr(adapter_hooks, "reset_dynamo_for_hook_change",
                            lambda: calls.append(1))
        adapter_hooks.LoraHookSet.remove([])
        assert calls == [1]
        adapter_hooks.LoraHookSet.detach([])
        assert calls == [1], "detach alone does not reset"


# --- W2-H / R1-A: one lease rule -----------------------------------------------------------------


class TestTheSharedLeaseRule:
    async def test_load_refuses_through_the_shared_rule(self, env):
        rel = env.write()
        with patch.object(model_lease, "refuse_if_leased",
                          wraps=model_lease.refuse_if_leased) as rule:
            await env.service.load(identity(), rel)
        assert rule.call_count == 2, "before the read and inside the pause"
        for call in rule.call_args_list:
            assert call.args == (env.leases,)
            assert call.kwargs == {
                "operation": OP_LOAD, "resident_model_id": 1, "target_model_id": 1,
                "lease_id": None,
            }

    async def test_unload_refuses_through_the_shared_rule(self, env):
        rel = env.write()
        await env.service.load(identity(), rel)
        with patch.object(model_lease, "refuse_if_leased",
                          wraps=model_lease.refuse_if_leased) as rule:
            await env.service.unload(identity(), lease_id="abc")
        assert rule.call_count == 2, "before the pause and inside it (R2-C)"
        for call in rule.call_args_list:
            assert call.args == (env.leases,)
            assert call.kwargs == {
                "operation": OP_UNLOAD, "resident_model_id": 1, "target_model_id": 1,
                "lease_id": "abc",
            }
