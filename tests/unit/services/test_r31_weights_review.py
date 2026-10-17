"""Review round 1 of Feature 031 (R31): the lines no earlier test caught, and the fixes made.

Each test names the mutation control that survived the existing suite (or the defect fixed) and
was re-run as a negative control against this file. Record:
`0xcc/reviews/attachable_weights_031_review_R1_2026-10-14.md` (the coordinator's), R31 report.

- R31-M6   `WeightService.quiesce` re-checks the attachment after the barrier holds the slot.
- R31-M7   `heartbeat` never revives an attachment removed between the holder check and the write.
- R31-M11  a parameter-set replacement never cancels a quiesce still taking the slot.
- R31-M14  a late commit-end re-arms the hold timer only while the holder task holds the slot.
- R31-M32  an attachment still being exported (pending) is never stale.
- R31-M36  commit-end checks addresses AFTER closing the commit, so the tear cannot be undone.
- R31-M38  clear-stale refuses a pending reservation (it would orphan the attach in flight).
- R31-F4   attach refuses when its reservation was removed during the export.
- R31-F5   the watchdog and a tearing detach tear in their own locked section.
- R31-F6   an internal write's first write voids the inherited commit in status too.
- R31-F2/F3 `WEIGHTS_TOKENS` parsing never raises on deep nesting, and rejects a digest with a
  trailing newline instead of storing a dead entry.
"""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import patch

import pytest

from millm.core.errors import NoOpenCommitError, NotAttachedError, WeightsTornError
from millm.core.weights_auth import parse_weights_tokens
from tests.unit.services.test_weight_attachment import (
    OPERATOR,
    TIMEOUT,
    TRAINER,
    _build,
    _real_attached,
    _settle,
    real,  # noqa: F401 - fixture
)
from tests.unit.services.test_weight_barrier_states import (
    D,
    OPS,
    _fire_watchdog,
    _torn,
    _writing,
    env,  # noqa: F401 - fixture
    synced,  # noqa: F401 - fixture
)


# --- WEIGHTS_TOKENS parsing (F2, F3) ------------------------------------------------------------


def test_deeply_nested_tokens_are_malformed_never_a_raise():
    tokens = parse_weights_tokens("[" * 100_000)
    assert tokens.entries == () and tokens.refusal_reason == "tokens_malformed"


def test_a_digest_with_a_trailing_newline_is_malformed_not_a_dead_entry():
    raw = '[{"label": "a", "role": "trainer", "sha256": "' + "a" * 64 + '\\n"}]'
    tokens = parse_weights_tokens(raw)
    assert tokens.refusal_reason == "tokens_malformed"
    assert "sha256" in (tokens.problem or "")


# --- attachment service (fake barrier) ------------------------------------------------------------


async def test_quiesce_rechecks_the_attachment_after_the_barrier_holds():
    """R31-M6: an attachment gone by the time the barrier returns is refused and the hold
    handed back through `on_holder_detached(tear_writing=False)`."""
    service, fakes = _build()
    attachment_id = (await service.attach(TRAINER)).attachment_id
    digest = service._attachment.digest
    fakes.barrier.on_quiesce = lambda: setattr(service, "_attachment", None)
    with pytest.raises(NotAttachedError):
        await service.quiesce(TRAINER, attachment_id)
    assert ("on_holder_detached", digest, False) in fakes.barrier.calls
    assert fakes.barrier.names().count("on_holder_detached") == 1


async def test_heartbeat_never_revives_an_attachment_removed_after_the_holder_check():
    """R31-M7: `require_holder` and the write are two locked sections."""
    service, fakes = _build()
    attachment_id = (await service.attach(TRAINER)).attachment_id
    old = service._attachment
    before = old.last_heartbeat_mono
    fakes.clock.advance(5)
    original = service._match

    def match_then_clear(identity, presented):
        result = original(identity, presented)
        service._attachment = None  # a clear-stale landing between the two sections
        return result

    with patch.object(service, "_match", side_effect=match_then_clear):
        with pytest.raises(NotAttachedError):
            service.heartbeat(TRAINER, attachment_id)
    assert old.last_heartbeat_mono == before
    assert service.is_attached() is False


async def _attach_blocked_in_export(service, fakes):
    started, release = threading.Event(), threading.Event()

    def block() -> None:
        started.set()
        assert release.wait(5)

    fakes.on_export = block
    task = asyncio.create_task(service.attach(TRAINER))
    assert await asyncio.to_thread(started.wait, 5)
    return task, release


async def test_a_pending_attachment_is_never_stale_and_clear_stale_refuses_it():
    """R31-M32, R31-M38: clearing a reservation mid-export would let the attach finish with an
    ID nothing recognises and handles open on unprotected weights."""
    service, fakes = _build()
    task, release = await _attach_blocked_in_export(service, fakes)
    try:
        fakes.clock.advance(TIMEOUT * 10)
        view = service.attachment_view()
        assert view is not None and view.stale is False
        with pytest.raises(NotAttachedError):
            await service.clear_stale(OPERATOR)
        assert service.is_attached() is True
    finally:
        release.set()
    result = await task
    assert service.holder_of(TRAINER, result.attachment_id) is not None


async def test_attach_refuses_when_its_reservation_was_removed_during_the_export():
    """R31-F4: the post-export section re-checks the reservation is still registered."""
    service, fakes = _build()
    fakes.on_export = lambda: setattr(service, "_attachment", None)
    with pytest.raises(NotAttachedError):
        await service.attach(TRAINER)
    assert service.is_attached() is False


# --- the real barrier ------------------------------------------------------------------------------


async def test_commit_end_tear_from_moved_addresses_cannot_be_undone_by_a_second_commit_end(real):
    """R31-M36: checked BEFORE the commit closed, the tear is a torn set with the commit still
    open, which the trainer's next commit-end would 'recover' over moved addresses."""
    r = await _real_attached(real)
    await r.service.quiesce(TRAINER, r.attachment_id)
    r.service.commit_begin(TRAINER, r.attachment_id)
    r.fakes.moved = ["w"]
    with pytest.raises(WeightsTornError):
        r.service.commit_end(TRAINER, r.attachment_id)
    r.fakes.moved = []
    with pytest.raises(NoOpenCommitError):
        r.service.commit_end(TRAINER, r.attachment_id)
    assert r.state()[0] == "torn"


async def test_parameter_set_replaced_never_cancels_a_quiesce_taking_the_slot(env):
    """R31-M11: the holder-owned hold is still being TAKEN; releasing it would cancel the
    holder task under the quiesce."""
    await env.inference.slot.acquire()  # in-flight work holds the one slot
    quiesce = asyncio.create_task(env.barrier.quiesce(D, OPS))
    await _settle()
    assert env.barrier._quiescing is True
    env.barrier.on_parameter_set_replaced("model_load")
    env.inference.slot.release()
    result = await quiesce
    assert result.weights_state == "clean"
    assert env.barrier.holds_slot() and env.barrier.snapshot().hold_owner == "holder"


async def test_a_late_commit_end_arms_no_hold_timer_without_a_held_slot(env):
    """R31-M14: a hold deadline on status with no slot held would be a fiction, and its timer
    would 'release' a hold nobody has."""
    await _torn(env)
    task = env.barrier._holder_task
    task.cancel()
    await _settle()
    assert env.barrier.holds_slot() is False
    env.barrier.commit_end(D)
    assert env.barrier.snapshot().hold_deadline is None
    assert env.barrier._hold_deadline_mono is None


async def test_the_watchdog_tears_inside_its_own_locked_section(env):
    """R31-F5: deciding under the lock and tearing after it let a commit-end (or a load
    completing in a worker thread) land in between and be overwritten with `torn`."""
    await _writing(env)
    with patch.object(env.barrier, "mark_torn", side_effect=AssertionError("unlocked tear")):
        _fire_watchdog(env)
    snap = env.barrier.snapshot()
    assert snap.weights_state == "torn" and snap.commit_overran is True
    assert env.barrier.admission_refusal() is not None


async def test_a_detach_while_writing_tears_inside_its_own_locked_section(env):
    """R31-F5, detach form: the tear is decided and made in one locked section."""
    await _writing(env)
    with patch.object(env.barrier, "mark_torn", side_effect=AssertionError("unlocked tear")):
        assert env.barrier.on_holder_detached(D) == "torn"
    assert env.barrier.snapshot().weights_state == "torn"
    assert env.barrier.admission_refusal() is not None


async def test_clear_stale_form_still_leaves_writing_for_the_watchdog(env):
    await _writing(env)
    assert env.barrier.on_holder_detached(D, tear_writing=False) == "writing"


async def test_an_internal_writes_first_write_voids_the_inherited_commit_in_status(env):
    """R31-F6: after the first write the trainer's commit is void; status must not keep showing
    it as open with a growing `open_ms`."""
    await _torn(env)
    assert env.barrier.snapshot().commit_began_at is not None
    with pytest.raises(RuntimeError):
        async with env.barrier.internal_write("weights_load", OPS) as handle:
            handle.mark_first_write()
            raise RuntimeError("half-loaded")
    snap = env.barrier.snapshot()
    assert snap.weights_state == "torn"
    assert snap.commit_began_at is None and snap.commit_open_ms is None

