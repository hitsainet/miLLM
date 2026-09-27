"""Probe and probe-event persistence, and the retention that keeps it bounded.

The tests worth reading twice are in `TestRetention`. A monitor that quietly fills a disk stops
monitoring, so pruning is not housekeeping here — and `prune_to_cap` has a specific trap: every
armed probe writes an event for the same request within the same millisecond, so any retention
rule that orders by timestamp alone is ambiguous exactly when it is exercised.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from millm.db.models.probe import Probe, ProbeEvent
from millm.db.repositories.probe_repository import ProbeEventRepository, ProbeRepository
from tests.unit.probe_fixtures import probe_definition

pytestmark = pytest.mark.asyncio


def probe_row(probe_id: str = "pr_0001", name: str = "high-stakes", **over) -> dict:
    doc = probe_definition()
    row = dict(
        id=probe_id,
        name=name,
        definition=doc,
        hf_id=doc["model"]["hf_id"],
        revision=doc["model"]["revision"],
        d_model=doc["model"]["d_model"],
        n_layers=doc["model"]["n_layers"],
        template_sha256=doc["model"]["chat_template_sha256"],
        layer=doc["read"]["layer"],
        rule=doc["aggregation"]["rule"],
        streamable=doc["aggregation"]["streamable"],
        scope=doc["scope"],
        basis=doc["basis"],
        threshold=doc["decision"]["threshold"],
        target_fpr=doc["decision"]["target_fpr"],
        rung=doc["evidence"]["rung"],
    )
    row.update(over)
    return row


@pytest.fixture
async def repo(test_session):
    return ProbeRepository(test_session)


@pytest.fixture
async def events(test_session):
    return ProbeEventRepository(test_session)


class TestProbeCrud:
    async def test_create_and_get(self, repo):
        created = await repo.create(**probe_row())
        assert created.id == "pr_0001"
        fetched = await repo.get("pr_0001")
        assert fetched is not None and fetched.name == "high-stakes"

    async def test_the_whole_definition_round_trips(self, repo):
        """The typed columns are a projection; the definition is the source of truth."""
        await repo.create(**probe_row())
        got = await repo.get("pr_0001")
        assert got.definition["test_vectors"]["vectors"][0]["token_ids"] == [1, 2, 3, 4]
        assert got.definition["provenance"]["mistudio_probe_id"] == "pm_fixture01"

    async def test_get_by_name(self, repo):
        await repo.create(**probe_row())
        assert (await repo.get_by_name("high-stakes")).id == "pr_0001"
        assert await repo.get_by_name("nope") is None

    async def test_a_missing_probe_is_none_not_an_error(self, repo):
        assert await repo.get("pr_nope") is None

    async def test_update(self, repo):
        probe = await repo.create(**probe_row())
        await repo.update(probe, armed=True, paused_reason=None)
        assert (await repo.get("pr_0001")).armed is True

    async def test_delete_takes_its_events_with_it(self, repo, events, test_session):
        probe = await repo.create(**probe_row())
        await events.create_many([{"probe_id": probe.id, "scored": True, "score": 1.0}])
        assert await events.count(probe.id) == 1
        await repo.delete(probe)
        assert await events.count(probe.id) == 0

    async def test_defaults_are_not_armed(self, repo):
        probe = await repo.create(**probe_row())
        assert probe.armed is False
        assert probe.basis == "residual"


class TestListing:
    async def test_armed_filter(self, repo):
        await repo.create(**probe_row("pr_a", "a"))
        await repo.create(**probe_row("pr_b", "b", armed=True))
        assert {p.id for p in await repo.list()} == {"pr_a", "pr_b"}
        assert {p.id for p in await repo.list_armed()} == {"pr_b"}
        assert await repo.count_armed() == 1

    async def test_names_taken_supports_rename_on_conflict(self, repo):
        await repo.create(**probe_row("pr_a", "high-stakes"))
        await repo.create(**probe_row("pr_b", "high-stakes-2"))
        await repo.create(**probe_row("pr_c", "other"))
        assert await repo.names_taken("high-stakes") == {"high-stakes", "high-stakes-2"}

    async def test_disarm_all_records_why(self, repo):
        """⚠ The reason is written, not just the flag cleared.

        "Disarmed because the model changed" and "disarmed by an operator" are different facts,
        and a page that cannot tell them apart reports silence either way.
        """
        await repo.create(**probe_row("pr_a", "a", armed=True))
        await repo.create(**probe_row("pr_b", "b", armed=True))
        await repo.create(**probe_row("pr_c", "c"))

        assert await repo.disarm_all("model_changed") == 2
        assert await repo.count_armed() == 0
        assert (await repo.get("pr_a")).paused_reason == "model_changed"
        # The one that was never armed is untouched, not given a spurious reason.
        assert (await repo.get("pr_c")).paused_reason is None


class TestEvents:
    async def test_create_many_and_list_newest_first(self, repo, events):
        await repo.create(**probe_row())
        await events.create_many(
            [{"probe_id": "pr_0001", "scored": True, "score": float(i)} for i in range(3)]
        )
        listed = await events.list_events(probe_id="pr_0001")
        assert [e.score for e in listed] == [2.0, 1.0, 0.0]

    async def test_an_unscored_event_carries_its_reason(self, repo, events):
        await repo.create(**probe_row())
        await events.create_many(
            [{"probe_id": "pr_0001", "scored": False, "not_scored_reason": "batched_request"}]
        )
        event = (await events.list_events(probe_id="pr_0001"))[0]
        assert event.scored is False
        assert event.not_scored_reason == "batched_request"
        assert event.score is None

    async def test_events_are_findable_by_request_id(self, repo, events):
        """`request_id` is the ONLY link between a `/v1` response and its verdict."""
        await repo.create(**probe_row())
        await events.create_many(
            [
                {"probe_id": "pr_0001", "request_id": "chatcmpl-aaa", "scored": True, "score": 1.0},
                {"probe_id": "pr_0001", "request_id": "chatcmpl-bbb", "scored": True, "score": 2.0},
            ]
        )
        found = await events.list_events(request_id="chatcmpl-bbb")
        assert len(found) == 1 and found[0].score == 2.0

    async def test_the_rung_is_stored_per_event(self, repo, events):
        """Denormalised on purpose: an event must keep describing the evidence that was true when
        it was observed, not the probe's current rung."""
        await repo.create(**probe_row())
        await events.create_many([{"probe_id": "pr_0001", "scored": True, "score": 1.0, "rung": 2}])
        probe = await repo.get("pr_0001")
        await repo.update(probe, rung=3)
        assert (await events.list_events(probe_id="pr_0001"))[0].rung == 2

    async def test_clear_is_scoped_to_one_probe(self, repo, events):
        await repo.create(**probe_row("pr_a", "a"))
        await repo.create(**probe_row("pr_b", "b"))
        await events.create_many(
            [{"probe_id": "pr_a", "scored": True}, {"probe_id": "pr_b", "scored": True}]
        )
        assert await events.clear("pr_a") == 1
        assert await events.count("pr_a") == 0
        assert await events.count("pr_b") == 1


class TestRetention:
    async def test_prune_to_cap_keeps_the_newest(self, repo, events):
        await repo.create(**probe_row())
        await events.create_many(
            [{"probe_id": "pr_0001", "scored": True, "score": float(i)} for i in range(10)]
        )
        removed = await events.prune_to_cap("pr_0001", cap=3)
        assert removed == 7
        assert {e.score for e in await events.list_events(probe_id="pr_0001")} == {7.0, 8.0, 9.0}

    async def test_prune_to_cap_is_unambiguous_when_timestamps_COLLIDE(self, repo, events):
        """⚠ This is the case that actually happens.

        Every armed probe writes an event for the same request inside the same millisecond, so
        ordering by `created_at` alone is ambiguous exactly when retention runs. An offset-based
        delete ("everything after row N") can then remove the wrong rows or none. The id is the
        tiebreak, and this fixture gives every row an identical timestamp to force the issue.
        """
        await repo.create(**probe_row())
        same = datetime.now(timezone.utc)
        await events.create_many(
            [
                {"probe_id": "pr_0001", "scored": True, "score": float(i), "created_at": same}
                for i in range(6)
            ]
        )
        assert await events.prune_to_cap("pr_0001", cap=2) == 4
        assert await events.count("pr_0001") == 2

    async def test_prune_to_cap_is_scoped_to_one_probe(self, repo, events):
        await repo.create(**probe_row("pr_a", "a"))
        await repo.create(**probe_row("pr_b", "b"))
        await events.create_many([{"probe_id": "pr_a", "scored": True} for _ in range(5)])
        await events.create_many([{"probe_id": "pr_b", "scored": True} for _ in range(5)])
        await events.prune_to_cap("pr_a", cap=1)
        assert await events.count("pr_a") == 1
        assert await events.count("pr_b") == 5, "pruning one probe removed another's events"

    async def test_prune_aged_removes_only_the_old(self, repo, events):
        await repo.create(**probe_row())
        old = datetime.now(timezone.utc) - timedelta(days=45)
        recent = datetime.now(timezone.utc) - timedelta(days=1)
        await events.create_many(
            [
                {"probe_id": "pr_0001", "scored": True, "score": 1.0, "created_at": old},
                {"probe_id": "pr_0001", "scored": True, "score": 2.0, "created_at": recent},
            ]
        )
        assert await events.prune_aged(30) == 1
        remaining = await events.list_events(probe_id="pr_0001")
        assert [e.score for e in remaining] == [2.0]

    async def test_a_zero_age_window_disables_age_pruning_rather_than_deleting_everything(
        self, repo, events
    ):
        """0 must mean "no age limit", not "older than now" — which would be everything."""
        await repo.create(**probe_row())
        await events.create_many([{"probe_id": "pr_0001", "scored": True}])
        assert await events.prune_aged(0) == 0
        assert await events.count("pr_0001") == 1

    async def test_a_zero_cap_is_a_no_op_not_a_purge(self, repo, events):
        await repo.create(**probe_row())
        await events.create_many([{"probe_id": "pr_0001", "scored": True}])
        assert await events.prune_to_cap("pr_0001", cap=0) == 0
        assert await events.count("pr_0001") == 1

    async def test_prune_applies_both_rules(self, repo, events):
        await repo.create(**probe_row())
        old = datetime.now(timezone.utc) - timedelta(days=60)
        await events.create_many(
            [{"probe_id": "pr_0001", "scored": True, "created_at": old} for _ in range(3)]
        )
        await events.create_many([{"probe_id": "pr_0001", "scored": True} for _ in range(5)])
        removed = await events.prune("pr_0001", cap=2, max_age_days=30)
        assert removed == 6
        assert await events.count("pr_0001") == 2
