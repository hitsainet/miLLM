"""The weight-attachment service (Feature 31, FTASKS 5.2, 5.3, 5.7).

The attach, holder, transition and staleness rules run against a recording FAKE barrier (they
read only its snapshot). Everything that changes the barrier — detach, clear-stale, the holder
calls and the address checks (`TestDetach`, `TestClearStale`, `TestHolderCallsAndAddresses`) —
runs against the REAL `WeightBarrier` holding a real one-slot admission (converted from the fake
after the W1-A merge, W2-F), with `wraps=` spies asserting each barrier call's payload and count.

MUTATION CONTROLS (each turns this file red): store the attachment ID instead of its digest;
stale computed with `<` or without the timeout; NOT_ATTACHMENT_HOLDER before NOT_ATTACHED;
detach in `writing` leaving `writing` (M19); clear-stale accepting an active attachment (M20);
stale no longer blocking attach (M21); drop any transition check.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import patch

import pytest
import torch

import millm.services.weight_barrier as barrier_module
from millm.core.config import settings
from millm.core.errors import (
    AlreadyAttachedError,
    AttachmentNotStaleError,
    ModelNotLoadedError,
    NotAttachableError,
    NotAttachedError,
    NotAttachmentHolderError,
    WeightsForbiddenError,
    WeightsTornError,
)
from millm.core.weights_auth import WeightsIdentity
from millm.ml.model_loader import LoadedModel
from millm.services.weight_attachment import (
    WeightService,
    get_weight_service,
    set_weight_service,
)
from millm.services.weight_barrier import BarrierSnapshot, WeightBarrier
from millm.services.weight_handles import ExportedHandles

TRAINER = WeightsIdentity("miforge", "trainer")
OTHER = WeightsIdentity("intruder", "trainer")
OPERATOR = WeightsIdentity("ops", "operator")
TIMEOUT = float(settings.WEIGHTS_HEARTBEAT_TIMEOUT_S)
HANDLE = "SECRET-HANDLE-BYTES-b64"


# --- fakes -------------------------------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeBarrier:
    """The barrier's public surface, recording calls (W1-A's real one replaces it)."""

    def __init__(self) -> None:
        self.state = "clean"
        self.mode = "serving"
        self.owner: Optional[str] = None
        self.steps = 0
        self.calls: list[tuple[Any, ...]] = []
        self.on_quiesce: Any = None
        self.resume_raises: Optional[Exception] = None

    def snapshot(self) -> BarrierSnapshot:
        return BarrierSnapshot(
            weights_state=self.state,  # type: ignore[arg-type]
            barrier=self.mode,  # type: ignore[arg-type]
            hold_owner=self.owner,
            episode=1,
            steps_since_load=self.steps,
            commit_overran=False,
            last_save=None,
            last_hold_timeout_at=None,
            last_overrun=None,
            held_model_id=None,
        )

    async def quiesce(self, owner_digest: str, identity: WeightsIdentity) -> dict:
        self.calls.append(("quiesce", owner_digest, identity))
        if self.on_quiesce is not None:
            self.on_quiesce()
        self.mode, self.owner = "quiesced", "holder"
        return {"held_requests": 0}

    def resume(self, owner_digest: str) -> None:
        self.calls.append(("resume", owner_digest))
        if self.resume_raises is not None:
            raise self.resume_raises
        self.mode, self.owner = "serving", None

    def commit_begin(self, owner_digest: str) -> None:
        self.calls.append(("commit_begin", owner_digest))
        self.state = "writing"

    def commit_end(self, owner_digest: str) -> dict:
        self.calls.append(("commit_end", owner_digest))
        self.state = "clean"
        self.steps += 1
        return {"steps_since_load": self.steps}

    def mark_torn(self, reason: str) -> None:
        self.calls.append(("mark_torn", reason))
        self.state = "torn"

    def on_holder_detached(self, owner_digest: str, *, tear_writing: bool = True) -> str:
        self.calls.append(("on_holder_detached", owner_digest, tear_writing))
        return self.state

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]


class FakeState:
    def __init__(self, loaded: Any) -> None:
        self.current = loaded
        self.is_unloading = False


@dataclass
class Fakes:
    barrier: FakeBarrier
    state: FakeState
    inference: Any
    clock: Clock
    load_claimed: list[bool]
    exports: list[Any] = field(default_factory=list)
    moved: list[str] = field(default_factory=list)
    ipc_collects: list[int] = field(default_factory=list)
    on_export: Any = None
    export_raises: Optional[BaseException] = None


def _loaded() -> LoadedModel:
    model = SimpleNamespace(
        named_parameters=lambda remove_duplicate=True: iter(
            [("w", SimpleNamespace(device=torch.device("cuda", 0), dtype=torch.bfloat16))]
        )
    )
    return LoadedModel(
        model_id=7,
        model_name="tiny-model",
        model=model,
        tokenizer=None,
        loaded_at=datetime(2026, 10, 14),
        quantization_method="none",
        attachable_allocation=True,
        gpu_indices=[0],
    )


def _build(**fake_overrides: Any) -> tuple[WeightService, Fakes]:
    fakes = Fakes(
        barrier=FakeBarrier(),
        state=FakeState(_loaded()),
        inference=SimpleNamespace(
            cbm_enabled=lambda: False, request_queue=SimpleNamespace(max_concurrent=1)
        ),
        clock=Clock(),
        load_claimed=[False],
    )
    for key, value in fake_overrides.items():
        setattr(fakes, key, value)

    def export(model: Any) -> ExportedHandles:
        fakes.exports.append(model)
        if fakes.on_export is not None:
            fakes.on_export()
        if fakes.export_raises is not None:
            raise fakes.export_raises
        return ExportedHandles(
            storages=[{"index": 0, "handle": HANDLE, "storage_size_bytes": 4096},
                      {"index": 1, "handle": HANDLE, "storage_size_bytes": 1024}],
            parameters=[{"name": "w", "storage": 0}],
            addresses={"w": 4096},
            keep_alive=["storage-object"],
        )

    service = WeightService(
        barrier=lambda: fakes.barrier,
        inference=lambda: fakes.inference,
        loaded_state=lambda: fakes.state,
        load_claimed=lambda: fakes.load_claimed[0],
        export=export,
        producer=lambda: {"torch": "2.10.0", "cuda": "12.8", "driver": "580.82.07"},
        moved=lambda model, fp: list(fakes.moved),
        ipc_collect=lambda: fakes.ipc_collects.append(1),
        clock=fakes.clock,
        wall=lambda: datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc),
    )
    return service, fakes


@pytest.fixture
def built():
    return _build()


async def _attached(built) -> tuple[WeightService, Fakes, str]:
    service, fakes = built
    result = await service.attach(TRAINER)
    return service, fakes, result.attachment_id


# --- attach ----------------------------------------------------------------------------------


class TestAttach:
    async def test_returns_the_id_once_and_the_contract_fields(self, built):
        service, fakes = built
        result = await service.attach(TRAINER)
        body = result.to_dict()
        assert set(body) == {
            "attachment_id", "model", "weights_state", "steps_since_load", "heartbeat",
            "producer", "storages", "parameters", "release", "config",
        }
        # The fake model has no Hugging Face config: stated as null, never omitted (D39).
        assert body["config"] is None
        assert body["model"] == {"model_id": 7, "name": "tiny-model", "manifest_hash": None}
        assert body["heartbeat"] == {
            "timeout_s": TIMEOUT,
            "interval_hint_s": float(settings.WEIGHTS_HEARTBEAT_INTERVAL_HINT_S),
        }
        assert body["producer"]["driver"] == "580.82.07"
        assert body["weights_state"] == "clean" and body["steps_since_load"] == 0
        assert len(result.attachment_id) >= 32
        assert len(fakes.exports) == 1 and fakes.exports[0] is fakes.state.current.model
        assert service.is_attached() is True

    async def test_the_served_config_is_the_loaded_one_exactly(self):
        """D39: the trainer rebuilds the graph from THIS process's config, so the response
        carries the loaded object's own serialisation - not a Hub copy, not a subset."""
        from transformers import LlamaConfig

        from millm.api.schemas.weights import AttachResponse

        config = LlamaConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2, tie_word_embeddings=True,
        )
        loaded = _loaded()
        loaded.model.config = config
        service, fakes = _build(state=FakeState(loaded))
        body = (await service.attach(TRAINER)).to_dict()
        assert body["config"] == json.loads(config.to_json_string(use_diff=False))
        assert body["config"]["model_type"] == "llama"
        assert body["config"]["tie_word_embeddings"] is True
        # Through the route's response model too: the field survives `extra="forbid"`.
        assert AttachResponse.model_validate(body).config == body["config"]

    async def test_only_the_digest_is_stored(self, built):
        service, _, attachment_id = await _attached(built)
        attachment = service._attachment
        assert attachment.digest == hashlib.sha256(attachment_id.encode()).hexdigest()
        assert attachment.ref == attachment.digest[:8]
        stored = json.dumps({k: repr(v) for k, v in vars(attachment).items()})
        assert attachment_id not in stored
        assert attachment_id not in repr(service.__dict__)

    async def test_the_result_repr_hides_the_id_and_handles(self, built):
        service, _ = built
        result = await service.attach(TRAINER)
        assert result.attachment_id not in repr(result) and HANDLE not in repr(result)

    async def test_each_attach_gets_a_fresh_random_id(self):
        first_service, _ = _build()
        second_service, _ = _build()
        a = (await first_service.attach(TRAINER)).attachment_id
        b = (await second_service.attach(TRAINER)).attachment_id
        assert a != b

    async def test_a_second_attach_names_the_holder_and_not_the_id(self, built):
        service, _, attachment_id = await _attached(built)
        with pytest.raises(AlreadyAttachedError) as info:
            await service.attach(OTHER)
        details = info.value.details
        assert details["role"] == "trainer" and details["label"] == "miforge"
        assert details["state"] == "active" and details["attached_at"].startswith("2026-10-14")
        assert attachment_id not in str(details) and attachment_id not in str(info.value)

    async def test_a_stale_attachment_still_blocks_attach(self, built):
        """M21 / FR-31.2.1."""
        service, fakes, _ = await _attached(built)
        fakes.clock.advance(TIMEOUT + 1)
        with pytest.raises(AlreadyAttachedError) as info:
            await service.attach(OTHER)
        assert info.value.details["state"] == "stale"

    async def test_two_concurrent_attaches_give_one_attachment(self, built):
        service, fakes = built
        results = await asyncio.gather(
            service.attach(TRAINER), service.attach(OTHER), return_exceptions=True
        )
        ok = [r for r in results if not isinstance(r, BaseException)]
        refused = [r for r in results if isinstance(r, AlreadyAttachedError)]
        assert len(ok) == 1 and len(refused) == 1
        assert len(fakes.exports) == 1

    async def test_the_reservation_blocks_loads_during_the_export(self, built):
        service, fakes = built
        seen: list[Any] = []

        def during() -> None:
            seen.append(service.is_attached())
            with pytest.raises(AlreadyAttachedError):
                service.refuse_if_attached("model_load")
            # Nobody holds a reservation yet: holder calls answer NOT_ATTACHED.
            with pytest.raises(NotAttachedError):
                service.require_holder(TRAINER, "anything")
            seen.append("checked")

        fakes.on_export = during
        await service.attach(TRAINER)
        assert seen == [True, "checked"]

    async def test_a_failed_export_clears_the_reservation(self, built):
        service, fakes = built
        fakes.export_raises = RuntimeError("share failed")
        with pytest.raises(RuntimeError):
            await service.attach(TRAINER)
        assert service.is_attached() is False
        fakes.export_raises = None
        await service.attach(TRAINER)

    async def test_not_attachable_is_refused_with_nothing_reserved(self, built):
        service, fakes = built
        fakes.state.current = replace(fakes.state.current, quantization_method="bitsandbytes")
        with pytest.raises(NotAttachableError) as info:
            await service.attach(TRAINER)
        assert info.value.details["reason"] == "quantized"
        assert service.is_attached() is False and fakes.exports == []

    async def test_nothing_resident_is_model_not_loaded(self, built):
        service, fakes = built
        fakes.state.current = None
        with pytest.raises(ModelNotLoadedError):
            await service.attach(TRAINER)

    @pytest.mark.parametrize(
        "facts,reason",
        [
            ({"cbm": True}, "continuous_batching_running"),
            ({"slots": 2}, "concurrency"),
        ],
    )
    async def test_deployment_facts_reach_the_check(self, facts, reason):
        service, fakes = _build()
        fakes.inference = SimpleNamespace(
            cbm_enabled=lambda: facts.get("cbm", False),
            request_queue=SimpleNamespace(max_concurrent=facts.get("slots", 1)),
        )
        with pytest.raises(NotAttachableError) as info:
            await service.attach(TRAINER)
        assert info.value.details["reason"] == reason


class TestTransitions:
    """031 review D3: no handles while the model is about to move."""

    async def _refused(self, service) -> dict:
        with pytest.raises(NotAttachableError) as info:
            await service.attach(TRAINER)
        assert info.value.details["reason"] == "model_transition"
        assert service.is_attached() is False
        return info.value.details

    async def test_unloading(self, built):
        service, fakes = built
        fakes.state.is_unloading = True
        assert (await self._refused(service))["transition"] == "unloading"

    async def test_load_slot_claimed(self, built):
        service, fakes = built
        fakes.load_claimed[0] = True
        assert (await self._refused(service))["transition"] == "loading"

    @pytest.mark.parametrize("owner", ["internal:weights_save", "internal:adapter_load"])
    async def test_internal_write_or_pause(self, built, owner):
        service, fakes = built
        fakes.barrier.owner = owner
        assert (await self._refused(service))["transition"] == "internal_write"

    async def test_a_holder_hold_left_torn_is_not_a_transition(self, built):
        """Re-attaching to a torn set is how a trainer finds out (acceptance 7)."""
        service, fakes = built
        fakes.barrier.owner, fakes.barrier.state, fakes.barrier.mode = "holder", "torn", "quiesced"
        result = await service.attach(TRAINER)
        assert result.weights_state == "torn"

    async def test_a_commit_left_open_is_not_attachable(self, built):
        """Addendum D-40: a commit clear-stale left `writing` for the watchdog is refused with
        reason `weights_writing`, not joined mid-commit; no handles are exported."""
        service, fakes = built
        fakes.barrier.owner, fakes.barrier.state, fakes.barrier.mode = (
            "holder", "writing", "quiesced")
        with pytest.raises(NotAttachableError) as info:
            await service.attach(TRAINER)
        assert info.value.details == {"reason": "weights_writing", "weights_state": "writing"}
        assert service.is_attached() is False and fakes.exports == []

    async def test_a_transition_that_starts_during_the_export_refuses(self, built):
        service, fakes = built
        fakes.on_export = lambda: setattr(fakes.state, "is_unloading", True)
        details = await self._refused(service)
        assert details["transition"] == "unloading"

    async def test_a_model_replaced_during_the_export_refuses(self, built):
        service, fakes = built
        fakes.on_export = lambda: setattr(fakes.state, "current", _loaded())
        assert (await self._refused(service))["transition"] == "model_replaced"

    def test_the_default_load_probe_reads_the_load_slot(self, monkeypatch):
        """The production probe reads the slot `ModelService` claims (until W2-F's accessor)."""
        from millm.services import model_service
        from millm.services.weight_attachment import _default_load_claimed

        monkeypatch.setitem(model_service._LOAD_SLOT, "model_id", None)
        assert _default_load_claimed() is False
        monkeypatch.setitem(model_service._LOAD_SLOT, "model_id", 7)
        assert _default_load_claimed() is True


# --- holder checks -------------------------------------------------------------------------


class TestHolder:
    async def test_the_holder_is_recognised(self, built):
        service, _, attachment_id = await _attached(built)
        assert service.holder_of(TRAINER, attachment_id) is service._attachment
        assert service.require_holder(TRAINER, attachment_id) is service._attachment

    async def test_a_wrong_id_is_not_the_holder(self, built):
        service, _, _ = await _attached(built)
        assert service.holder_of(TRAINER, "wrong") is None
        with pytest.raises(NotAttachmentHolderError) as info:
            service.require_holder(TRAINER, "wrong")
        assert info.value.status_code == 403

    async def test_the_right_id_from_another_label_is_not_the_holder(self, built):
        service, _, attachment_id = await _attached(built)
        assert service.holder_of(OTHER, attachment_id) is None
        with pytest.raises(NotAttachmentHolderError):
            service.require_holder(OTHER, attachment_id)

    async def test_a_missing_header_is_not_the_holder(self, built):
        service, _, _ = await _attached(built)
        assert service.holder_of(TRAINER, None) is None
        with pytest.raises(NotAttachmentHolderError) as info:
            service.require_holder(TRAINER, None)
        assert info.value.details["header_present"] is False

    @pytest.mark.parametrize("presented", [None, "", "wrong"])
    def test_not_attached_comes_before_not_holder(self, built, presented):
        """I11: with nothing attached, a missing or wrong header is NOT_ATTACHED (409)."""
        service, _ = built
        assert service.holder_of(TRAINER, presented) is None
        with pytest.raises(NotAttachedError) as info:
            service.require_holder(TRAINER, presented)
        assert info.value.status_code == 409

    async def test_a_stale_holder_is_still_the_holder(self, built):
        service, fakes, attachment_id = await _attached(built)
        fakes.clock.advance(TIMEOUT * 3)
        assert service.require_holder(TRAINER, attachment_id) is service._attachment

    async def test_refuse_if_attached_names_the_state(self, built):
        service, fakes, _ = await _attached(built)
        with pytest.raises(AlreadyAttachedError) as info:
            service.refuse_if_attached("model_unload")
        assert info.value.details["state"] == "active"
        assert info.value.details["operation"] == "model_unload"
        fakes.clock.advance(TIMEOUT + 0.5)
        with pytest.raises(AlreadyAttachedError) as info:
            service.refuse_if_attached("model_load")
        assert info.value.details["state"] == "stale"


# --- staleness and heartbeat ---------------------------------------------------------------


class TestStale:
    async def test_stale_is_computed_on_read_with_no_tick(self, built):
        service, fakes, _ = await _attached(built)
        assert service.attachment_view().stale is False
        fakes.clock.advance(TIMEOUT)
        assert service.attachment_view().stale is False, "exactly at the window is not stale"
        fakes.clock.advance(0.001)
        view = service.attachment_view()
        assert view.stale is True and view.holder_label == "miforge"
        status = service.status().attachment
        assert status.state == "stale" and status.stale_since is not None

    async def test_stale_changes_nothing_about_serving(self, built):
        service, fakes, _ = await _attached(built)
        fakes.clock.advance(TIMEOUT * 10)
        service.tick()
        assert fakes.barrier.calls == [] and fakes.barrier.state == "clean"

    async def test_a_heartbeat_keeps_it_active(self, built):
        service, fakes, attachment_id = await _attached(built)
        fakes.clock.advance(TIMEOUT - 1)
        reply = service.heartbeat(TRAINER, attachment_id)
        assert reply == {
            "attachment_state": "active",
            "timeout_s": TIMEOUT,
            "interval_hint_s": float(settings.WEIGHTS_HEARTBEAT_INTERVAL_HINT_S),
            "weights_state": "clean",
        }
        fakes.clock.advance(TIMEOUT - 1)
        assert service.attachment_view().stale is False

    async def test_a_heartbeat_revives_a_stale_uncleared_attachment(self, built):
        from structlog.testing import capture_logs

        service, fakes, attachment_id = await _attached(built)
        fakes.clock.advance(TIMEOUT + 5)
        with capture_logs() as logs:
            service.heartbeat(TRAINER, attachment_id)
        assert service.attachment_view().stale is False
        revived = [e for e in logs if e["event"] == "weights_attachment_revived"]
        assert len(revived) == 1 and revived[0]["stale_s"] == pytest.approx(TIMEOUT + 5)

    async def test_a_heartbeat_from_another_label_is_refused(self, built):
        service, _, attachment_id = await _attached(built)
        with pytest.raises(NotAttachmentHolderError):
            service.heartbeat(OTHER, attachment_id)

    async def test_tick_logs_stale_once_per_episode(self, built):
        from structlog.testing import capture_logs

        service, fakes, attachment_id = await _attached(built)
        with capture_logs() as logs:
            service.tick()
            fakes.clock.advance(TIMEOUT + 1)
            service.tick()
            service.tick()
            service.heartbeat(TRAINER, attachment_id)
            fakes.clock.advance(TIMEOUT + 1)
            service.tick()
        assert [e["event"] for e in logs].count("weights_attachment_stale") == 2


# --- the REAL barrier (W2-F: converted from RERUN_WITH_REAL_BARRIER) -------------------------


@dataclass
class Real:
    service: WeightService
    fakes: Fakes
    barrier: WeightBarrier
    inference: Any
    attachment_id: str = ""

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.attachment_id.encode()).hexdigest()

    def state(self) -> tuple[str, str, Optional[str]]:
        snap = self.barrier.snapshot()
        return snap.weights_state, snap.barrier, snap.hold_owner


class _RealInference:
    """What the real barrier reads, with a real one-slot admission it can hold."""

    def __init__(self) -> None:
        self.request_queue = SimpleNamespace(
            max_concurrent=1, pending_count=0, holding_count=0, background_waiting_count=0
        )
        self.slot = asyncio.Semaphore(1)

    def cbm_enabled(self) -> bool:
        return False

    def invalidate_request_caches(self) -> int:
        return 0

    @contextlib.asynccontextmanager
    async def hold_admission(self):
        async with self.slot:
            yield


@pytest.fixture
def real(monkeypatch):
    """A WeightService over the REAL WeightBarrier (fake clock for staleness only)."""
    monkeypatch.setattr(barrier_module, "_synchronize_resident_devices", lambda: None)
    monkeypatch.setattr(settings, "WEIGHTS_QUIESCE_DRAIN_TIMEOUT_S", 2.0)
    inference = _RealInference()
    barrier = WeightBarrier(inference=lambda: inference)
    service, fakes = _build(inference=inference)
    service._barrier = lambda: barrier
    fakes.barrier = barrier  # type: ignore[assignment]
    yield Real(service=service, fakes=fakes, barrier=barrier, inference=inference)
    barrier.reset("test teardown")


async def _real_attached(real: Real) -> Real:
    real.attachment_id = (await real.service.attach(TRAINER)).attachment_id
    return real


def _spy(r: Real, name: str) -> Any:
    """A `wraps=` spy on one real barrier method: the real call runs, its payload is kept."""
    return patch.object(r.barrier, name, wraps=getattr(r.barrier, name))


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


def _fire_watchdog(barrier: WeightBarrier) -> None:
    """Expire the open commit now: the watchdog's own callback, past its deadline."""
    barrier._watchdog_deadline_mono = 0.0
    barrier._on_watchdog(barrier._watchdog_generation)


# --- detach (D16) ------------------------------------------------------------------------------


class TestDetach:
    async def test_detach_while_serving_touches_no_hold(self, real):
        r = await _real_attached(real)
        with _spy(r, "on_holder_detached") as spy:
            reply = await r.service.detach(TRAINER, r.attachment_id)
        assert reply == {"weights_state": "clean", "barrier": "serving"}
        assert spy.call_count == 1 and spy.call_args.args == (r.digest,)
        assert spy.call_args.kwargs == {}, "detach tears an open commit (the default)"
        assert r.state() == ("clean", "serving", None)
        assert r.service.is_attached() is False and r.fakes.ipc_collects == [1]

    async def test_detach_from_a_clean_quiesce_releases_the_slot(self, real):
        r = await _real_attached(real)
        await r.service.quiesce(TRAINER, r.attachment_id)
        assert r.inference.slot.locked()
        reply = await r.service.detach(TRAINER, r.attachment_id)
        await _settle()
        assert reply["barrier"] == "serving" and r.state() == ("clean", "serving", None)
        assert not r.inference.slot.locked() and not r.barrier.holds_slot()

    async def test_detach_while_writing_tears_at_once(self, real):
        """M19 / FR-31.3.2: torn, the slot kept, every request refused."""
        r = await _real_attached(real)
        await r.service.quiesce(TRAINER, r.attachment_id)
        r.service.commit_begin(TRAINER, r.attachment_id)
        reply = await r.service.detach(TRAINER, r.attachment_id)
        assert reply["weights_state"] == "torn" and r.service.is_attached() is False
        assert r.barrier.holds_slot() and r.barrier.admission_refusal() is not None

    async def test_detach_while_torn_stays_torn_and_reattach_sees_it(self, real):
        """Acceptance 7, unit half."""
        r = await _real_attached(real)
        await r.service.quiesce(TRAINER, r.attachment_id)
        r.service.commit_begin(TRAINER, r.attachment_id)
        _fire_watchdog(r.barrier)
        assert r.state()[0] == "torn"
        reply = await r.service.detach(TRAINER, r.attachment_id)
        assert reply["weights_state"] == "torn"
        again = await r.service.attach(TRAINER)
        assert again.weights_state == "torn"
        # The re-attached holder did not begin that commit: it cannot end it (FR-31.12.2).
        with pytest.raises(NotAttachmentHolderError):
            r.service.commit_end(TRAINER, again.attachment_id)

    async def test_a_pause_that_adopted_the_quiesce_releases_it_after_a_detach(self, real):
        """An internal op adopted the holder's quiesce; the holder leaves meanwhile: on the
        op's exit the hold is released, not handed back to nobody."""
        r = await _real_attached(real)
        await r.service.quiesce(TRAINER, r.attachment_id)
        async with r.barrier.serving_pause("adapter_load", OPERATOR):
            assert r.state() == ("clean", "quiesced", "internal:adapter_load")
            await r.service.detach(TRAINER, r.attachment_id)
            assert r.barrier.holds_slot(), "the pause still holds while it runs"
        await _settle()
        assert r.state() == ("clean", "serving", None)
        assert not r.inference.slot.locked()

    async def test_detach_never_touches_someone_elses_hold(self, real):
        r = await _real_attached(real)
        async with r.barrier.serving_pause("weights_save", OPERATOR):
            await r.service.detach(TRAINER, r.attachment_id)
            assert r.state() == ("clean", "quiesced", "internal:weights_save")
            assert r.barrier.holds_slot()
        assert r.state() == ("clean", "serving", None)

    async def test_detach_by_a_non_holder_is_refused_and_changes_nothing(self, real):
        r = await _real_attached(real)
        with _spy(r, "on_holder_detached") as spy:
            with pytest.raises(NotAttachmentHolderError):
                await r.service.detach(OTHER, "wrong")
        assert spy.call_count == 0
        assert r.service.is_attached() is True and r.fakes.ipc_collects == []

    async def test_detach_with_nothing_attached(self, real):
        with pytest.raises(NotAttachedError):
            await real.service.detach(TRAINER, None)


# --- clear-stale -------------------------------------------------------------------------------


class TestClearStale:
    async def test_a_trainer_token_is_forbidden(self, real):
        r = await _real_attached(real)
        r.fakes.clock.advance(TIMEOUT + 1)
        with pytest.raises(WeightsForbiddenError):
            await r.service.clear_stale(TRAINER)
        assert r.service.is_attached() is True

    async def test_an_active_attachment_is_not_cleared(self, real):
        """M20 / BRD-02 R-02.34."""
        r = await _real_attached(real)
        with pytest.raises(AttachmentNotStaleError) as info:
            await r.service.clear_stale(OPERATOR)
        assert info.value.details == {"role": "trainer", "label": "miforge"}
        assert r.service.is_attached() is True

    async def test_nothing_attached(self, real):
        with pytest.raises(NotAttachedError):
            await real.service.clear_stale(OPERATOR)

    async def test_a_stale_clean_quiesce_is_cleared_and_resumed(self, real):
        r = await _real_attached(real)
        await r.service.quiesce(TRAINER, r.attachment_id)
        r.fakes.clock.advance(TIMEOUT + 1)
        with _spy(r, "on_holder_detached") as spy:
            reply = await r.service.clear_stale(OPERATOR)
        await _settle()
        assert spy.call_count == 1
        assert spy.call_args.args == (r.digest,) and spy.call_args.kwargs == {"tear_writing": False}
        assert reply["cleared"]["label"] == "miforge" and reply["cleared"]["role"] == "trainer"
        assert reply["cleared"]["stale_since"] is not None
        assert reply["barrier"] == "serving" and r.service.is_attached() is False
        assert not r.inference.slot.locked() and r.fakes.ipc_collects == [1]

    async def test_writing_is_left_for_the_watchdog(self, real):
        """FR-31.4.5: clearing never tears by itself; the watchdog still does."""
        r = await _real_attached(real)
        await r.service.quiesce(TRAINER, r.attachment_id)
        r.service.commit_begin(TRAINER, r.attachment_id)
        r.fakes.clock.advance(TIMEOUT + 1)
        reply = await r.service.clear_stale(OPERATOR)
        assert reply["weights_state"] == "writing" and r.barrier.holds_slot()
        _fire_watchdog(r.barrier)
        assert r.state()[0] == "torn"

    async def test_the_watchdog_after_clearing_logs_who_was_writing(self, real):
        """R-01.22 / FR-31.11.2 (R1 RX F3): clear-stale forgets the holder, but the torn line the
        watchdog writes later still names who began the commit."""
        from structlog.testing import capture_logs

        r = await _real_attached(real)
        await r.service.quiesce(TRAINER, r.attachment_id)
        r.service.commit_begin(TRAINER, r.attachment_id)
        r.fakes.clock.advance(TIMEOUT + 1)
        await r.service.clear_stale(OPERATOR)
        with capture_logs() as logs:
            _fire_watchdog(r.barrier)
        torn = [e for e in logs if e["event"] == "weights_torn"]
        assert len(torn) == 1 and torn[0]["reason"] == "watchdog"
        assert (torn[0]["role"], torn[0]["label"]) == ("trainer", "miforge")

    async def test_torn_is_left_torn(self, real):
        r = await _real_attached(real)
        await r.service.quiesce(TRAINER, r.attachment_id)
        r.service.commit_begin(TRAINER, r.attachment_id)
        _fire_watchdog(r.barrier)
        r.fakes.clock.advance(TIMEOUT + 1)
        reply = await r.service.clear_stale(OPERATOR)
        assert reply["weights_state"] == "torn" and r.barrier.holds_slot()

    async def test_attach_after_clearing_a_writer_waits_for_the_watchdog(self, real):
        """Addendum D-40 over the REAL barrier: after clear-stale left the commit `writing`, a
        new attach is `NOT_ATTACHABLE` (`weights_writing`); once the watchdog tears the set,
        re-attaching to `torn` is allowed (acceptance 7)."""
        r = await _real_attached(real)
        await r.service.quiesce(TRAINER, r.attachment_id)
        r.service.commit_begin(TRAINER, r.attachment_id)
        r.fakes.clock.advance(TIMEOUT + 1)
        await r.service.clear_stale(OPERATOR)
        with pytest.raises(NotAttachableError) as info:
            await r.service.attach(OTHER)
        assert info.value.details["reason"] == "weights_writing"
        assert r.service.is_attached() is False
        _fire_watchdog(r.barrier)
        result = await r.service.attach(OTHER)
        assert result.weights_state == "torn"

    async def test_after_clearing_a_new_attach_succeeds(self, real):
        r = await _real_attached(real)
        r.fakes.clock.advance(TIMEOUT + 1)
        await r.service.clear_stale(OPERATOR)
        await r.service.attach(OTHER)
        assert r.service.attachment_view().holder_label == "intruder"


# --- holder wrappers and fixed addresses ---------------------------------------------------


class TestHolderCallsAndAddresses:
    async def test_quiesce_commit_resume_pass_the_digest(self, real):
        r = await _real_attached(real)
        spies = {
            name: _spy(r, name)
            for name in ("quiesce", "commit_begin", "commit_end", "resume")
        }
        started = {name: p.start() for name, p in spies.items()}
        try:
            await r.service.quiesce(TRAINER, r.attachment_id)
            began = r.service.commit_begin(TRAINER, r.attachment_id)
            ended = r.service.commit_end(TRAINER, r.attachment_id)
            resumed = r.service.resume(TRAINER, r.attachment_id)
        finally:
            for p in spies.values():
                p.stop()
        assert began.weights_state == "writing" and began.state_before == "clean"
        assert (ended.weights_state, ended.steps_since_load, ended.overran) == ("clean", 1, False)
        assert resumed.released is True and resumed.barrier == "serving"
        assert started["quiesce"].call_args.args == (r.digest, TRAINER)
        for name in ("commit_begin", "commit_end", "resume"):
            assert started[name].call_count == 1 and started[name].call_args.args == (r.digest,)

    @pytest.mark.parametrize("call", ["resume", "commit_begin", "commit_end"])
    async def test_a_non_holder_reaches_no_barrier_method(self, real, call):
        r = await _real_attached(real)
        with _spy(r, call) as spy:
            with pytest.raises(NotAttachmentHolderError):
                getattr(r.service, call)(OTHER, "wrong")
        assert spy.call_count == 0

    async def test_a_moved_address_at_quiesce_tears(self, real):
        r = await _real_attached(real)
        r.fakes.moved = ["model.layers.0.weight"]
        with pytest.raises(WeightsTornError) as info:
            await r.service.quiesce(TRAINER, r.attachment_id)
        assert info.value.details["reason"] == "address_moved"
        assert r.state()[0] == "torn" and r.barrier.holds_slot()

    async def test_commit_end_checks_addresses_after_the_commit(self, real):
        """Checked after: a tear before would be undone by the late commit-end."""
        r = await _real_attached(real)
        await r.service.quiesce(TRAINER, r.attachment_id)
        r.service.commit_begin(TRAINER, r.attachment_id)
        r.fakes.moved = ["w"]
        with pytest.raises(WeightsTornError):
            r.service.commit_end(TRAINER, r.attachment_id)
        assert r.state()[0] == "torn"

    async def test_a_moved_address_at_detach_tears(self, real):
        r = await _real_attached(real)
        r.fakes.moved = ["w"]
        reply = await r.service.detach(TRAINER, r.attachment_id)
        assert reply["weights_state"] == "torn"

    async def test_unmoved_addresses_change_nothing(self, real):
        r = await _real_attached(real)
        assert r.service.verify_addresses("quiesce") == []
        assert r.state() == ("clean", "serving", None)

    async def test_a_detach_during_the_drain_resumes_and_refuses(self, real):
        """031 review D14: no hold may outlive its holder. The detach lands while quiesce
        waits for an in-flight request; once the barrier holds the slot it releases it."""
        r = await _real_attached(real)
        await r.inference.slot.acquire()  # an in-flight request
        quiesce = asyncio.create_task(r.service.quiesce(TRAINER, r.attachment_id))
        await _settle()
        assert r.barrier.snapshot().hold_owner == "holder" and not r.barrier.holds_slot()
        await r.service.detach(TRAINER, r.attachment_id)
        r.inference.slot.release()  # the request finishes; the barrier takes the slot
        with pytest.raises(NotAttachedError):
            await quiesce
        await _settle()
        assert r.state() == ("clean", "serving", None)
        assert not r.inference.slot.locked() and not r.barrier.holds_slot()


# --- status and logs ---------------------------------------------------------------------------


class TestStatusAndLogs:
    def test_nothing_attached(self):
        service = WeightService()
        assert service.is_attached() is False
        assert service.attachment_view() is None
        assert service.holder_of(TRAINER, "an-id") is None
        assert service.holder_of(TRAINER, None) is None

    def test_require_holder_refuses_not_attached_without_echoing(self):
        with pytest.raises(NotAttachedError) as info:
            WeightService().require_holder(TRAINER, "an-id")
        assert info.value.status_code == 409
        assert "an-id" not in str(info.value.details)

    async def test_the_resident_name_is_the_base_while_an_adapter_is_applied(
        self, built, monkeypatch
    ):
        """Addendum D-41: the attached process sees BASE tensors (R-01.27) and `manifest_hash`
        is the base's, so attach's `model.name` and status's `resident_model.name` are the base
        served name even while an overlay (an applied adapter) renames what `/v1` serves."""
        from millm.services import resident_identity

        monkeypatch.setattr(resident_identity, "_overlays", [lambda: "org/tiny:r:adapter-1"])
        service, fakes = built
        # The fixture must not agree by construction: /v1 really does serve the adapter name.
        assert resident_identity.served_name(fakes.state.current) == "org/tiny:r:adapter-1"
        result = await service.attach(TRAINER)
        assert result.to_dict()["model"]["name"] == "tiny-model"
        assert service.status().resident_model.name == "tiny-model"

    def test_refuse_if_attached_is_a_no_op_while_unattached(self):
        assert WeightService().refuse_if_attached("model_load") is None

    def test_status_joins_the_real_barrier(self):
        status = WeightService().status()
        assert status.attached is False and status.attachment is None
        assert status.weights_state == "clean" and status.barrier == "serving"

    async def test_status_never_carries_the_id(self, built):
        """M14 / FR-31.16.2."""
        service, _, attachment_id = await _attached(built)
        status = service.status().model_dump(mode="json")
        assert attachment_id not in json.dumps(status)
        assert service._attachment.digest not in json.dumps(status)
        assert set(status["attachment"]) == {
            "state", "role", "label", "attached_at", "last_heartbeat_at", "stale_since",
        }
        assert status["resident_model"] == {
            "model_id": 7, "name": "tiny-model", "manifest_hash": None,
        }

    async def test_no_log_line_carries_the_id_or_a_handle(self, built):
        from structlog.testing import capture_logs

        service, fakes = built
        with capture_logs() as logs:
            result = await service.attach(TRAINER)
            attachment_id = result.attachment_id
            service.heartbeat(TRAINER, attachment_id)
            await service.quiesce(TRAINER, attachment_id)
            service.commit_begin(TRAINER, attachment_id)
            service.commit_end(TRAINER, attachment_id)
            service.resume(TRAINER, attachment_id)
            fakes.clock.advance(TIMEOUT + 1)
            service.tick()
            fakes.moved = ["w"]
            service.verify_addresses("check")
            fakes.moved = []
            await service.detach(TRAINER, attachment_id)
        text = str(logs)
        assert logs, "the flow logs something"
        assert attachment_id not in text and HANDLE not in text
        assert hashlib.sha256(attachment_id.encode()).hexdigest() not in text

    def test_the_singleton_and_its_reset(self):
        set_weight_service(None)
        first = get_weight_service()
        assert get_weight_service() is first
        set_weight_service(None)
        assert get_weight_service() is not first


async def test_a_reattached_holders_commit_begin_on_a_torn_set_is_not_clean(real):
    """Addendum D-43 (FR-31.10.1): after a watchdog tear the slot stays held under the departed
    holder's quiesce. A holder that re-attaches (allowed to a torn set, acceptance 7) and calls
    commit-begin is told 409 WEIGHTS_NOT_CLEAN, the set's real state, not 403
    NOT_ATTACHMENT_HOLDER because its digest is not the departed quiesce's.

    MUTATION CONTROL (R2-A): the owner check moved back above the state check in
    `WeightBarrier.commit_begin` -> red (NOT_ATTACHMENT_HOLDER)."""
    from millm.core.errors import WeightsNotCleanError

    r = await _real_attached(real)
    await r.service.quiesce(TRAINER, r.attachment_id)
    r.service.commit_begin(TRAINER, r.attachment_id)
    _fire_watchdog(r.barrier)
    assert r.barrier.snapshot().weights_state == "torn"
    await r.service.detach(TRAINER, r.attachment_id)

    again = (await r.service.attach(TRAINER)).attachment_id
    with pytest.raises(WeightsNotCleanError) as info:
        r.service.commit_begin(TRAINER, again)
    assert info.value.status_code == 409
    assert info.value.details == {"operation": "commit_begin", "weights_state": "torn"}
    assert r.barrier.snapshot().weights_state == "torn"


async def test_a_reattached_holders_resume_on_a_torn_set_is_not_clean(real):
    """Addendum D-44 (as D-43 for commit-begin): a holder re-attached after a watchdog tear that
    calls resume is told 409 WEIGHTS_NOT_CLEAN, not 403 NOT_ATTACHMENT_HOLDER.

    MUTATION CONTROL: the torn check removed from `WeightBarrier.resume` -> red."""
    from millm.core.errors import WeightsNotCleanError

    r = await _real_attached(real)
    await r.service.quiesce(TRAINER, r.attachment_id)
    r.service.commit_begin(TRAINER, r.attachment_id)
    _fire_watchdog(r.barrier)
    await r.service.detach(TRAINER, r.attachment_id)

    again = (await r.service.attach(TRAINER)).attachment_id
    with pytest.raises(WeightsNotCleanError) as info:
        r.service.resume(TRAINER, again)
    assert info.value.status_code == 409
    assert info.value.details == {"operation": "resume", "weights_state": "torn"}
    assert r.barrier.snapshot().weights_state == "torn"


@pytest.mark.parametrize(
    "cbm, slots, expected",
    [(False, 1, None), (True, 1, "cbm"), (False, 2, "concurrency"), (True, 2, "cbm")],
)
async def test_status_reports_why_the_barrier_cannot_pause(real, cbm, slots, expected):
    """`pause_unavailable` (miForge F11 asked to learn BARRIER_UNAVAILABLE before submitting): the
    status states exactly what a quiesce would be refused for, from the one shared check.

    MUTATION CONTROL: status passes `pause_unavailable=None` unconditionally -> red."""
    from millm.core.errors import WeightsBarrierUnavailableError

    r = await _real_attached(real)  # attach first: an attach is itself refused under CBM
    r.inference.cbm_enabled = lambda: cbm  # type: ignore[method-assign]
    r.inference.request_queue.max_concurrent = slots
    status = r.service.status()
    assert status.pause_unavailable == expected
    if expected is None:
        return
    with pytest.raises(WeightsBarrierUnavailableError) as info:
        await r.service.quiesce(TRAINER, r.attachment_id)
    assert info.value.details["reason"] == expected, "status and the refusal must agree"


# --- the release contract and what status can observe (addendum D-47, hardware finding 2) -----


class TestReleaseContract:
    async def test_attach_states_what_stays_pinned_and_how_to_release_it(self, built):
        service, _ = built
        body = (await service.attach(TRAINER)).to_dict()
        assert body["release"]["pinned_bytes"] == 4096 + 1024
        contract = body["release"]["contract"]
        assert "exactly once" in contract and "until miLLM restarts" in contract
        assert "ipc_collect" in contract

    async def test_status_counts_every_export(self, built):
        service, fakes = built
        assert service.status().exports_since_start == 0
        result = await service.attach(TRAINER)
        assert service.status().exports_since_start == 1
        await service.detach(TRAINER, result.attachment_id)
        await service.attach(TRAINER)
        assert service.status().exports_since_start == 2
        assert len(fakes.exports) == 2

    def test_status_reports_the_allocator_per_card(self, built, monkeypatch):
        service, _ = built
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda i: (i + 1) * 3 * 1024 * 1024)
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda i: (i + 1) * 5 * 1024 * 1024)
        mem = [m.model_dump() for m in service.status().cuda_memory]
        assert mem == [
            {"device_index": 0, "allocated_mb": 3, "reserved_mb": 5},
            {"device_index": 1, "allocated_mb": 6, "reserved_mb": 10},
        ]

    def test_status_cuda_memory_is_null_without_cuda(self, built, monkeypatch):
        service, _ = built
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert service.status().cuda_memory is None

    def test_a_failing_allocator_read_never_fails_the_status(self, built, monkeypatch):
        service, _ = built
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        def boom(i):
            raise RuntimeError("driver gone")

        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(torch.cuda, "memory_allocated", boom)
        assert service.status().cuda_memory is None
