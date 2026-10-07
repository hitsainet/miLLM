"""The steering report is computed from what the hooks applied, never echoed (Feature 28,
FR-28.3.2, FR-28.3.6, FR-28.3.7; FTASKS 5.1 – 5.3, 5.6, 5.7, 5.9).

⚠ THE HONESTY TESTS INJECT A MISMATCH ON PURPOSE (FTID §8): a request record claiming one set
while the snapshot holds another must report the snapshot. Hashes are computed by
`independent_hash` (struct + hashlib in the fixture module), never through the code under test.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import millm.services.steering_report as report_module
from millm.services.inference_service import (
    get_steering_report,
    reset_steering_memo,
    set_steering_report,
)
from millm.services.sae_service import AttachedSAEState
from millm.services.steering_report import (
    EntrySnapshot,
    RequestSteeringRecord,
    SteeringReport,
    SteeringSnapshot,
    SteeringStateReader,
    steering_report_for_row,
)
from tests.unit.f28_fixtures import (
    SAE_A,
    SAE_B,
    add_profile,
    build,
    chat,
    clean_state,  # noqa: F401 - fixture
    db,  # noqa: F401 - fixture
    independent_hash,
    no_cuda,
    text,
)


def snap(*entries, epoch=0) -> SteeringSnapshot:
    return SteeringSnapshot(epoch=epoch, entries=tuple(
        EntrySnapshot(sae_id=s, layer=layer, enabled=en, applied=dict(vals))
        for s, layer, en, vals in entries))


class Recorder:
    def __init__(self):
        self.events: list[tuple[str, str, dict]] = []

    def __getattr__(self, level):
        return lambda event, **kw: self.events.append((level, event, kw))


@pytest.fixture
def log(monkeypatch):
    recorder = Recorder()
    monkeypatch.setattr(report_module, "logger", recorder)
    return recorder


# ── 5.1 capture ─────────────────────────────────────────────────────────────────────────────


class TestCapture:
    def test_copies_values_without_zeros_enabled_and_epoch(self, clean_state):  # noqa: F811
        served = build([(SAE_A, 0, 11), (SAE_B, 1, 22)])
        try:
            a = served.sae(SAE_A, 0)
            a.set_steering_batch({1: 2.0, 2: 0.0, 3: -0.0})
            a.enable_steering(True)
            served.sae(SAE_B, 1).set_steering_batch({4: 1.0})
            AttachedSAEState().bump_steering_epoch("t")
            s = SteeringSnapshot.capture()
            assert s.failed is False and s.epoch == AttachedSAEState().steering_epoch
            by = {(e.sae_id, e.layer): e for e in s.entries}
            assert by[(SAE_A, 0)].applied == {1: 2.0} and by[(SAE_A, 0)].enabled is True
            assert by[(SAE_B, 1)].applied == {4: 1.0} and by[(SAE_B, 1)].enabled is False
            # A copy: a later write does not reach the snapshot.
            a.set_steering_batch({1: 9.0})
            assert by[(SAE_A, 0)].applied == {1: 2.0}
            assert [e.sae_id for e in s.steered()] == [SAE_A]
        finally:
            for h in served.handles:
                h.remove()

    def test_never_raises(self, log):
        class Broken:
            def entries(self):
                raise RuntimeError("registry gone")

        s = SteeringSnapshot.capture(Broken())
        assert s.failed is True and s.entries == ()
        assert any(e == "steering_snapshot_failed" for _l, e, _k in log.events)


# ── 5.2 request-scoped, reset per request ───────────────────────────────────────────────────


def test_a_stale_report_is_never_returned_after_the_reset():
    set_steering_report(SteeringReport.from_items([]))
    assert get_steering_report() is not None
    reset_steering_memo()
    assert get_steering_report() is None


# ── 5.3 labels ──────────────────────────────────────────────────────────────────────────────


class TestLabels:
    async def test_nothing_steered_is_none(self):
        r = await SteeringStateReader().describe(snap((SAE_A, 0, False, {1: 2.0})), None)
        assert r.header == "none"

    async def test_inline_by_equal_values(self):
        record = RequestSteeringRecord(kind="inline", epoch_at_admission=0, sae_id=SAE_A,
                                       layer=0, applied={3: 200.0}, clamped=1)
        r = await SteeringStateReader().describe(snap((SAE_A, 0, True, {3: 200.0})), record)
        assert r.header == (f'inline;sae="{SAE_A}";layer=0;features=1;'
                            f'hash="{independent_hash(SAE_A, {3: 200.0})}";clamped=1')

    async def test_profile_request_by_equal_values(self):
        record = RequestSteeringRecord(kind="profile", epoch_at_admission=0, sae_id=SAE_A,
                                       layer=0, applied={1: 1.5}, profile_name="humor",
                                       profile_source="request", intensity=0.4375)
        r = await SteeringStateReader().describe(snap((SAE_A, 0, True, {1: 1.5})), record)
        assert r.header.startswith('profile;name="humor";source=request;intensity="0.4375";')

    async def test_no_record_no_circuit_no_profile_is_manual(self, db):  # noqa: F811
        r = await SteeringStateReader().describe(snap((SAE_A, 0, True, {2: -4.0})), None)
        assert r.header == (f'manual;sae="{SAE_A}";layer=0;features=1;'
                            f'hash="{independent_hash(SAE_A, {2: -4.0})}"')

    async def test_active_profile_by_equal_values(self, db, log):  # noqa: F811
        await add_profile(db, "calm", {2: 4.0, 3: 400.0}, active=True, intensity=0.75,
                          sae_id="sae_other")
        live = {2: 3.0, 3: 200.0}  # 4.0 * 0.75, and clamp(400 * 0.75)
        r = await SteeringStateReader().describe(snap((SAE_A, 0, True, live)), None)
        assert r.header == (f'profile;name="calm";source=active;intensity="0.75";sae="{SAE_A}";'
                            f'layer=0;features=2;hash="{independent_hash(SAE_A, live)}";'
                            'clamped=1')
        mismatch = [kw for _l, e, kw in log.events if e == "profile_sae_mismatch"]
        assert mismatch == [{"profile": "calm", "profile_sae_id": "sae_other",
                             "applied_sae_id": SAE_A, "layer": 0}]

    async def test_an_active_profile_with_other_values_does_not_lend_its_name(self, db):  # noqa: F811
        await add_profile(db, "calm", {2: 4.0}, active=True, intensity=0.75)
        r = await SteeringStateReader().describe(snap((SAE_A, 0, True, {2: 3.0000001})), None)
        assert r.header.startswith("manual;")

    async def test_unknown_when_the_snapshot_failed(self):
        failed = SteeringSnapshot(epoch=None, failed=True, error="x")
        assert (await SteeringStateReader().describe(failed, None)).header == (
            "unknown;reason=read_failed")

    async def test_llamacpp_with_steered_entries_is_unknown(self):
        r = await SteeringStateReader().describe(
            snap((SAE_A, 0, True, {1: 1.0})), None, engine="llamacpp")
        assert r.header == "unknown;reason=llamacpp_entries"
        r = await SteeringStateReader().describe(snap(), None, engine="llamacpp")
        assert r.header == "none"


class TestCircuitLabel:
    @pytest.fixture
    def circuit(self, monkeypatch, db):  # noqa: F811
        """One FULL-serving circuit whose plan claims (SAE_A, 0) and (SAE_B, 1)."""
        row = SimpleNamespace(id="crc_x", serving_mode="full", intensity=0.9, circuit_meta={})
        claims = {"composed": False, "raise": False}

        async def list_active(self):
            return [row]

        async def live_claims(self):
            if claims["raise"]:
                raise RuntimeError("claims table gone")
            return [SimpleNamespace(composed=claims["composed"])]

        monkeypatch.setattr(
            "millm.db.repositories.circuit_repository.CircuitRepository.list_active", list_active)
        monkeypatch.setattr(
            "millm.services.circuit_claim_registry.CircuitClaimRegistry.live_claims", live_claims)
        monkeypatch.setattr("millm.api.schemas.circuit.CircuitDefinitionV1.model_validate",
                            classmethod(lambda cls, meta: SimpleNamespace()))
        monkeypatch.setattr(
            "millm.ml.circuit_steering.CircuitSteeringEngine.plan_for",
            lambda self, d, c=None, intensity=None: SimpleNamespace(
                claimed_entries=(SimpleNamespace(sae_id=SAE_A, layer=0),
                                 SimpleNamespace(sae_id=SAE_B, layer=1)),
                intensity=0.5))
        return claims

    async def test_entries_a_circuit_claims_collapse_into_one_item(self, circuit):
        s = snap((SAE_A, 0, True, {1: 1.0}), (SAE_B, 1, True, {2: 2.0}), ("sae_c", 3, True,
                                                                           {5: 1.0}))
        r = await SteeringStateReader().describe(s, None)
        members = r.header.split(", ")
        assert members[0] == 'circuit;id="crc_x";intensity="0.5"'
        assert members[1].startswith('manual;sae="sae_c";layer=3;')
        assert len(members) == 2

    async def test_the_dial_lambda_wins_when_the_request_dialled_the_circuit(self, circuit):
        record = RequestSteeringRecord(kind="circuit", epoch_at_admission=0, intensity=1.25)
        r = await SteeringStateReader().describe(snap((SAE_A, 0, True, {1: 1.0})), record)
        assert r.header == 'circuit;id="crc_x";intensity="1.25"'

    async def test_composed(self, circuit):
        circuit["composed"] = True
        r = await SteeringStateReader().describe(snap((SAE_A, 0, True, {1: 1.0})), None)
        assert r.header == 'circuit;id="crc_x";intensity="0.5";composed'

    async def test_unreadable_claims_fail_closed(self, circuit, log):
        """FTDD TD11: the rung echo fails open; this header is an honesty statement."""
        circuit["raise"] = True
        r = await SteeringStateReader().describe(snap((SAE_A, 0, True, {1: 1.0})), None)
        assert r.header == "unknown;reason=claims_unreadable"
        assert any(e == "steering_report_unknown" and kw["reason"] == "claims_unreadable"
                   for _l, e, kw in log.events)

    async def test_an_unreadable_circuit_table_is_unknown_not_manual(self, monkeypatch, db, log):  # noqa: F811
        async def broken(self):
            raise RuntimeError("postgres blip")

        monkeypatch.setattr(
            "millm.db.repositories.circuit_repository.CircuitRepository.list_active", broken)
        r = await SteeringStateReader().describe(snap((SAE_A, 0, True, {1: 1.0})), None)
        assert r.header == "unknown;reason=read_failed", (
            "a database blip must not relabel circuit steering as `manual`"
        )


# ── 5.6 honesty ─────────────────────────────────────────────────────────────────────────────


class TestHonesty:
    async def test_the_record_lies_and_the_snapshot_wins(self, db):  # noqa: F811
        """The request CLAIMS inline {5: 8.0}; the hooks ran {5: 4.0}."""
        record = RequestSteeringRecord(kind="inline", epoch_at_admission=0, sae_id=SAE_A,
                                       layer=0, applied={5: 8.0})
        r = await SteeringStateReader().describe(snap((SAE_A, 0, True, {5: 4.0})), record)
        assert r.header == (f'manual;sae="{SAE_A}";layer=0;features=1;'
                            f'hash="{independent_hash(SAE_A, {5: 4.0})}"')
        assert independent_hash(SAE_A, {5: 8.0}) not in r.header

    async def test_a_record_on_another_entry_does_not_label_this_one(self, db):  # noqa: F811
        record = RequestSteeringRecord(kind="inline", epoch_at_admission=0, sae_id=SAE_B,
                                       layer=1, applied={5: 4.0})
        r = await SteeringStateReader().describe(snap((SAE_A, 0, True, {5: 4.0})), record)
        assert r.header.startswith("manual;")

    async def test_an_epoch_move_marks_every_member_changed(self, db):  # noqa: F811
        record = RequestSteeringRecord(kind="inline", epoch_at_admission=3, sae_id=SAE_A,
                                       layer=0, applied={1: 1.0})
        s = snap((SAE_A, 0, True, {1: 1.0}), (SAE_B, 1, True, {2: 2.0}), epoch=4)
        r = await SteeringStateReader().describe(s, record)
        members = r.header.split(", ")
        assert len(members) == 2 and all(m.endswith(";changed") for m in members)
        r = await SteeringStateReader().describe(snap(epoch=4), record)
        assert r.header == "none;changed"

    async def test_an_operator_write_during_generation_is_reported(self, clean_state, db,  # noqa: F811
                                                                    monkeypatch):
        """T-81, end to end: an authoritative write lands INSIDE generation. The header describes
        the values at the end and carries `changed`; the superseded restore is skipped."""
        from millm.services.sae_service import SAEService

        served = build([(SAE_A, 0, 11)])
        try:
            svc = served.svc
            real = svc._generate_sync

            def operator_writes_mid_generation(*args, **kwargs):
                # The operator steering route's service method: an authoritative, epoch-bumping
                # write (sae_service.set_steering_batch).
                SAEService.for_registry().set_steering_batch({6: 2.5})
                return real(*args, **kwargs)

            monkeypatch.setattr(svc, "_generate_sync", operator_writes_mid_generation)
            with no_cuda():
                await svc.create_chat_completion(
                    chat(steering={"features": [{"index": 1, "strength": 4.0}]}))
            header = get_steering_report().header
            assert header.endswith(";changed"), header
            assert independent_hash(SAE_A, {1: 4.0, 6: 2.5}) in header
            assert header.startswith("manual;")
        finally:
            for h in served.handles:
                h.remove()

    async def test_a_live_global_profile_is_reported_with_its_intensity(self, clean_state,  # noqa: F811
                                                                         db):  # noqa: F811
        """US-4: no steering field; a profile is active at λ = 0.75."""
        served = build([(SAE_A, 0, 11)])
        try:
            await add_profile(db, "humor", {2: 4.0, 5: -2.0}, active=True, intensity=0.75)
            a = served.sae(SAE_A, 0)
            a.set_steering_batch({2: 3.0, 5: -1.5})   # what activation wrote: stored × 0.75
            a.enable_steering(True)
            with no_cuda():
                await served.svc.create_chat_completion(chat())
            assert get_steering_report().header == (
                f'profile;name="humor";source=active;intensity="0.75";sae="{SAE_A}";layer=0;'
                f'features=2;hash="{independent_hash(SAE_A, {2: 3.0, 5: -1.5})}"'
            )
        finally:
            for h in served.handles:
                h.remove()


# ── 5.7 failure is unknown, never a failed request ──────────────────────────────────────────


async def test_describe_raising_is_unknown_and_logged_and_the_request_succeeds(
    clean_state, monkeypatch, log  # noqa: F811
):
    served = build([(SAE_A, 0, 11)])
    try:
        async def boom(self, *args, **kwargs):
            raise RuntimeError("reader exploded")

        monkeypatch.setattr(SteeringStateReader, "_describe", boom)
        with no_cuda():
            result = await served.svc.create_text_completion(text())
        assert result.choices, "the request must still succeed (FR-28.3.7)"
        assert get_steering_report().header == "unknown;reason=read_failed"
        assert any(lvl == "warning" and e == "steering_report_unknown"
                   and kw["reason"] == "read_failed" for lvl, e, kw in log.events)
    finally:
        for h in served.handles:
            h.remove()


# ── 5.9 the batch seam ──────────────────────────────────────────────────────────────────────


async def test_steering_report_for_row_is_the_header_string(db):  # noqa: F811
    s = snap((SAE_A, 0, True, {3: 200.0}))
    record = RequestSteeringRecord(kind="inline", epoch_at_admission=0, sae_id=SAE_A, layer=0,
                                   applied={3: 200.0}, clamped=1)
    header = (await SteeringStateReader().describe(s, record)).header
    assert await steering_report_for_row(s, record) == header
    assert header.startswith("inline;")
