"""Feature 29 task 2.7: the lease registry, by an injected clock — never `sleep`.

Fixtures disagree with the defects they guard: two holders, a resident model (1) different
from another model (2), and a clock that moves past expiry.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from millm.core.errors import LeaseExpiredError, LeaseNotFoundError, ModelLeasedError
from millm.services.model_lease import (
    UNKNOWN_LEASE_MESSAGE,
    LeaseRegistry,
    clear_leases_on_startup,
    get_lease_registry,
    set_lease_registry,
)


class Clock:
    def __init__(self) -> None:
        self.mono = 1000.0
        self.wall = datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc)

    def advance(self, seconds: float) -> None:
        self.mono += seconds
        self.wall += timedelta(seconds=seconds)


class Resident:
    def __init__(self, model_id):
        self.model_id = model_id

    def __call__(self):
        return self.model_id


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def resident():
    return Resident(1)


@pytest.fixture
def reg(clock, resident):
    return LeaseRegistry(
        resident_model_id=resident, monotonic=lambda: clock.mono, now=lambda: clock.wall,
        ended_memory=4,
    )


def _grant(reg, holder="midataworks", ttl=120, model_id=1):
    return reg.grant(model_id, f"m{model_id}", holder, "label run 7", ttl)


class TestGrant:
    def test_grant_returns_an_id_and_stores_only_its_digest(self, reg):
        grant = _grant(reg)
        assert len(grant.lease_id) >= 32
        record = reg.current(1)
        assert record is not None
        assert grant.lease_id not in repr(record)
        assert record.digest != grant.lease_id and len(record.digest) == 64
        assert record.lease_ref == record.digest[:8]
        assert record.expires_at == datetime(2026, 10, 6, 12, 2, tzinfo=timezone.utc)

    def test_two_grants_give_two_ids(self, reg, resident):
        a = _grant(reg)
        reg.release(1, a.lease_id)
        b = _grant(reg)
        assert a.lease_id != b.lease_id

    def test_second_grant_on_a_live_lease_is_refused_naming_the_holder(self, reg):
        _grant(reg, holder="midataworks")
        with pytest.raises(ModelLeasedError) as exc:
            _grant(reg, holder="mistudio-agent")
        assert exc.value.details["holder"] == "midataworks"
        assert exc.value.details["expires_at"] == "2026-10-06T12:02:00+00:00"

    def test_same_holder_string_is_still_a_second_caller(self, reg):
        """Holder is a label, not an identity: the same string does not get the lease again."""
        _grant(reg, holder="midataworks")
        with pytest.raises(ModelLeasedError):
            _grant(reg, holder="midataworks")


class TestExpiry:
    def test_honoured_until_the_deadline_and_gone_at_it(self, reg, clock):
        _grant(reg, ttl=120)
        clock.advance(119.999)
        assert reg.current(1) is not None
        clock.advance(0.001)  # exactly at the deadline: `<=` means gone (M5)
        assert reg.current(1) is None
        assert reg.last_ended(1).end_reason == "expired"

    def test_expired_lease_can_be_replaced(self, reg, clock):
        _grant(reg, holder="midataworks", ttl=10)
        clock.advance(11)
        assert _grant(reg, holder="mistudio-agent").record.holder == "mistudio-agent"

    def test_wall_clock_step_does_not_extend_a_lease(self, reg, clock):
        _grant(reg, ttl=10)
        clock.wall -= timedelta(hours=5)  # wall clock steps back; monotonic does not
        clock.mono += 10
        assert reg.current(1) is None

    def test_expiry_is_logged_once(self, reg, clock):
        from structlog.testing import capture_logs

        _grant(reg, ttl=10)
        clock.advance(10)
        with capture_logs() as logs:
            reg.current(1)
            reg.current(1)
        assert [e["event"] for e in logs].count("lease_expired") == 1

    def test_seconds_remaining(self, reg, clock):
        record = _grant(reg, ttl=120).record
        clock.advance(30.2)
        assert reg.seconds_remaining(record) == 90


class TestRenew:
    def test_new_expiry_is_now_plus_ttl_not_old_plus_ttl(self, reg, clock):
        grant = _grant(reg, ttl=120)
        clock.advance(100)
        renewed = reg.renew(1, grant.lease_id, 60)
        # now (12:01:40) + 60 = 12:02:40, not 12:02:00 + 60 = 12:03:00
        assert renewed.expires_at == datetime(2026, 10, 6, 12, 2, 40, tzinfo=timezone.utc)
        assert renewed.ttl_seconds == 60
        clock.advance(59)
        assert reg.current(1) is not None
        clock.advance(1)
        assert reg.current(1) is None

    def test_unknown_id_is_404_naming_the_restart(self, reg):
        _grant(reg)
        with pytest.raises(LeaseNotFoundError) as exc:
            reg.renew(1, "not-a-lease", 60)
        assert UNKNOWN_LEASE_MESSAGE in exc.value.message

    def test_expired_id_is_409_with_end_reason(self, reg, clock):
        grant = _grant(reg, ttl=10)
        clock.advance(10)
        with pytest.raises(LeaseExpiredError) as exc:
            reg.renew(1, grant.lease_id, 60)
        assert exc.value.details["end_reason"] == "expired"

    def test_a_lease_for_another_model_is_404(self, reg, resident):
        grant = _grant(reg, model_id=1)
        with pytest.raises(LeaseNotFoundError):
            reg.renew(2, grant.lease_id, 60)
        # ... and nothing changed on model 1
        assert reg.current(1).expires_at == grant.record.expires_at

    def test_an_ended_lease_for_another_model_is_still_404(self, reg):
        grant = _grant(reg, model_id=1)
        reg.release(1, grant.lease_id)
        with pytest.raises(LeaseNotFoundError):
            reg.release(2, grant.lease_id)


class TestRelease:
    def test_release_ends_at_once(self, reg):
        grant = _grant(reg)
        ended = reg.release(1, grant.lease_id)
        assert ended.end_reason == "released"
        assert reg.current(1) is None

    def test_release_twice_is_409(self, reg):
        grant = _grant(reg)
        reg.release(1, grant.lease_id)
        with pytest.raises(LeaseExpiredError) as exc:
            reg.release(1, grant.lease_id)
        assert exc.value.details["end_reason"] == "released"

    def test_wrong_id_changes_nothing(self, reg):
        _grant(reg)
        with pytest.raises(LeaseNotFoundError):
            reg.release(1, "wrong")
        assert reg.current(1) is not None

    def test_ended_memory_is_bounded(self, reg):
        ids = []
        for _ in range(6):
            grant = _grant(reg)
            ids.append(grant.lease_id)
            reg.release(1, grant.lease_id)
        # memory 4: the first two are forgotten → 404; the last is remembered → 409
        with pytest.raises(LeaseNotFoundError):
            reg.release(1, ids[0])
        with pytest.raises(LeaseExpiredError):
            reg.release(1, ids[-1])


class TestResolveAndMatches:
    def test_resolve_returns_live_only(self, reg, clock):
        grant = _grant(reg, ttl=10)
        assert reg.resolve(grant.lease_id).model_id == 1
        assert reg.resolve("wrong") is None
        assert reg.resolve(None) is None
        clock.advance(10)
        assert reg.resolve(grant.lease_id) is None

    def test_matches(self, reg):
        grant = _grant(reg)
        record = reg.current(1)
        assert reg.matches(record, grant.lease_id) is True
        assert reg.matches(record, grant.lease_id + "x") is False
        assert reg.matches(record, None) is False
        assert reg.matches(record, "") is False


class TestResidencySelfHeal:
    def test_a_lease_whose_model_left_reads_as_none(self, reg, resident):
        _grant(reg, model_id=1)
        resident.model_id = None  # the loader was emptied without unload_model's success branch
        assert reg.current(1) is None
        assert reg.last_ended(1).end_reason == "model_unloaded"

    def test_a_swap_ends_the_lease(self, reg, resident):
        _grant(reg, model_id=1)
        resident.model_id = 2
        assert reg.current(1) is None


class TestEndAndClear:
    def test_end_for_model(self, reg):
        _grant(reg)
        assert reg.end_for_model(1, "model_unloaded").end_reason == "model_unloaded"
        assert reg.current(1) is None
        assert reg.end_for_model(1, "model_unloaded") is None

    def test_clear_ends_every_lease_with_restart(self, reg):
        _grant(reg, model_id=1)
        assert reg.clear() == 1
        assert reg.current(1) is None
        assert reg.last_ended(1).end_reason == "restart"


class TestProcessRegistry:
    def test_get_returns_one_registry(self):
        assert get_lease_registry() is get_lease_registry()

    def test_clear_on_startup_never_raises(self, monkeypatch):
        class Broken:
            def clear(self, reason):
                raise RuntimeError("boom")

        set_lease_registry(Broken())  # type: ignore[arg-type]
        assert clear_leases_on_startup() == 0


class TestNoIdInLogs:
    def test_no_event_carries_the_lease_id(self, reg, clock, resident):
        from structlog.testing import capture_logs

        with capture_logs() as logs:
            grant = _grant(reg, ttl=100)
            reg.renew(1, grant.lease_id, 50)
            reg.release(1, grant.lease_id)
            second = _grant(reg, ttl=5)
            clock.advance(5)
            reg.current(1)
            third = _grant(reg, ttl=50)
            resident.model_id = 2
            reg.current(1)
        events = {e["event"] for e in logs}
        assert {"lease_granted", "lease_renewed", "lease_released", "lease_expired",
                "lease_ended"} <= events
        secrets_ = [grant.lease_id, second.lease_id, third.lease_id]
        for event in logs:
            for value in event.values():
                assert all(s not in str(value) for s in secrets_), event
        # Every lease event carries the ref, holder and model.
        for event in logs:
            if event["event"].startswith("lease_"):
                assert {"lease_ref", "holder", "model_id"} <= set(event)
