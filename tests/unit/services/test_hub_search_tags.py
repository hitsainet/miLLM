"""T30: the hub search finds both tags (040 FR-040.42, TPD-11; FTDD §5.4; TID §3.8).

Before miWorkbench writes the `miworkbench-*-definition` tag (its P6), the probe and cluster hub
searches return repos carrying the configured tag (`mistudio-*-definition`, frozen, DR-06) OR its
twin, each repo once. The old tag is never dropped. The Hub is a recording fake: payload and call
count are asserted, and each tag's answer is its own.

Control NC-88 (0xcc/reviews/040_P4_controls_2026-11-11.md): make `hub_tags` return the configured
tag alone -> `test_old_only_new_only_and_both_each_returned_once` red.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from miserver.core.config import settings
from miserver.services import cluster_hub_service
from miserver.services.cluster_hub_service import ClusterHubService, hub_tag_twin
from miserver.services.probe_hub_service import ProbeHubService

OLD, NEW = "mistudio-probe-definition", "miworkbench-probe-definition"


def _repo(repo_id: str, *tags: str) -> SimpleNamespace:
    return SimpleNamespace(id=repo_id, likes=0, downloads=0, last_modified=None, tags=list(tags))


class _Hub:
    """`HfApi` stand-in: answers per tag (the first filter), records every call's keywords."""

    calls: list[dict] = []
    by_tag: dict[str, list] = {}

    def list_models(self, **kwargs):
        type(self).calls.append(kwargs)
        return list(type(self).by_tag.get(kwargs["filter"][0], []))[:kwargs["limit"]]


@pytest.fixture
def hub(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "SAE_CACHE_DIR", str(tmp_path))
    _Hub.calls, _Hub.by_tag = [], {}
    with patch.object(cluster_hub_service, "HfApi", _Hub):
        yield _Hub


def test_twin_rule():
    assert hub_tag_twin("mistudio-probe-definition") == "miworkbench-probe-definition"
    assert hub_tag_twin("mistudio-cluster-definition") == "miworkbench-cluster-definition"
    assert hub_tag_twin("mistudio-circuit-definition") == "miworkbench-circuit-definition"
    assert hub_tag_twin("miworkbench-probe-definition") == "mistudio-probe-definition"
    assert hub_tag_twin(hub_tag_twin("mistudio-cluster-definition")) == "mistudio-cluster-definition"
    assert hub_tag_twin("my-own-tag") is None, "an operator's own tag has no twin"
    assert hub_tag_twin("mistudio-probe-definition-x") is None
    assert hub_tag_twin("miworkbench-other-definition") is None, "only the three registered kinds"
    assert ProbeHubService().hub_tags == (OLD, NEW), "configured tag first, then its twin"
    assert ClusterHubService().hub_tags == ("mistudio-cluster-definition",
                                            "miworkbench-cluster-definition")


async def test_one_call_per_tag_payload_and_count(hub):
    await ProbeHubService().search(query="pii", base_model="org/base", limit=7)
    assert len(hub.calls) == 2, hub.calls
    assert [c["filter"] for c in hub.calls] == [[OLD, "base_model:org/base"],
                                                [NEW, "base_model:org/base"]]
    assert [(c["search"], c["limit"]) for c in hub.calls] == [("pii", 7), ("pii", 7)]


async def test_a_tag_without_a_twin_makes_one_call(hub, monkeypatch):
    monkeypatch.setattr(settings, "PROBE_HUB_TAG", "my-own-tag")
    await ProbeHubService().search(query=None, base_model=None, limit=5)
    assert [c["filter"] for c in hub.calls] == [["my-own-tag"]]


async def test_old_only_new_only_and_both_each_returned_once(hub):
    hub.by_tag = {OLD: [_repo("a/old-only", OLD), _repo("a/both", OLD, NEW)],
                  NEW: [_repo("a/both", OLD, NEW), _repo("a/new-only", NEW)]}
    found = await ProbeHubService().search(query=None, base_model=None, limit=30)
    assert [r.repo_id for r in found] == ["a/old-only", "a/both", "a/new-only"]
    assert len(hub.calls) == 2


async def test_merge_dedupes_in_first_seen_order_and_cuts_at_limit(hub):
    hub.by_tag = {OLD: [_repo("a/1"), _repo("a/2"), _repo("a/3")],
                  NEW: [_repo("a/3"), _repo("a/4"), _repo("a/1"), _repo("a/5")]}
    found = await ProbeHubService().search(query=None, base_model=None, limit=4)
    assert [r.repo_id for r in found] == ["a/1", "a/2", "a/3", "a/4"], "configured tag first, cut at limit"


async def test_cache_key_includes_both_tags(hub, monkeypatch):
    service = ProbeHubService(cache_ttl_s=300)
    hub.by_tag = {OLD: [_repo("a/old")], NEW: [_repo("a/new")]}
    first = await service.search(query="q", base_model=None, limit=5)
    again = await service.search(query="q", base_model=None, limit=5)
    assert again == first and len(hub.calls) == 2, "the second identical search is served from the cache"
    assert any(str(service.hub_tags) in key for key in service._cache), list(service._cache)

    # The tag is a runtime setting: after it changes, the cached answer for the old tags is not served.
    monkeypatch.setattr(settings, "PROBE_HUB_TAG", "my-own-tag")
    hub.by_tag["my-own-tag"] = [_repo("a/mine")]
    changed = await service.search(query="q", base_model=None, limit=5)
    assert [r.repo_id for r in changed] == ["a/mine"]
    assert [c["filter"] for c in hub.calls][2:] == [["my-own-tag"]]


async def test_cluster_search_uses_its_own_pair(hub):
    hub.by_tag = {"miworkbench-cluster-definition": [_repo("c/new-only")]}
    found = await ClusterHubService().search(query=None, base_model=None, limit=5)
    assert [r.repo_id for r in found] == ["c/new-only"]
    assert [c["filter"] for c in hub.calls] == [["mistudio-cluster-definition"],
                                                ["miworkbench-cluster-definition"]]
