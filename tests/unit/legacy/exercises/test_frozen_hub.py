"""Exercises for DR-06 (Hub tags `mistudio-*-definition`), DR-07 (SAE repo ids under `mistudio/`) and
DR-30 (the Hugging Face account `mistudio`) — 040 FR-040.7.

Published packs carry these tags and ids; renaming the default tag would make every search find
nothing, and a stored `sae.hf_repo` must resolve as written. The Hub is a recording fake: payload and
call count are asserted, not merely "was called".
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.unit.api.test_probe_event_filters import client  # noqa: F401 - fixture
from tests.unit.f28_fixtures import db  # noqa: F401 - fixture

REPO = Path(__file__).resolve().parents[4]


class _RecordingHub:
    calls: list[dict] = []
    answer: list = []

    def list_models(self, **kwargs):
        type(self).calls.append(kwargs)
        return list(type(self).answer)


@pytest.fixture
def hub(monkeypatch, tmp_path):
    from miserver.core.config import settings

    monkeypatch.setattr(settings, "SAE_CACHE_DIR", str(tmp_path))
    _RecordingHub.calls, _RecordingHub.answer = [], []
    with patch("miserver.services.cluster_hub_service.HfApi", _RecordingHub):
        yield _RecordingHub


async def test_hub_searches_send_the_frozen_tags(hub):
    from miserver.services.cluster_hub_service import ClusterHubService
    from miserver.services.probe_hub_service import ProbeHubService

    await ProbeHubService().search(query="q", base_model=None, limit=5)
    await ClusterHubService().search(query="q", base_model=None, limit=5)
    # 040 FR-040.42 (P4): each search also asks for the tag's `miworkbench-*` twin, AFTER the frozen
    # tag, which is never dropped. Two calls per search: the frozen tag first.
    assert [c["filter"] for c in hub.calls] == [["mistudio-probe-definition"],
                                                ["miworkbench-probe-definition"],
                                                ["mistudio-cluster-definition"],
                                                ["miworkbench-cluster-definition"]]
    assert [c["search"] for c in hub.calls] == ["q"] * 4 and len(hub.calls) == 4


async def test_mistudio_account_repo_ids_pass_through(hub):
    from miserver.services.probe_hub_service import ProbeHubService

    hub.answer = [SimpleNamespace(id="mistudio/probe-monitors-test", likes=1, downloads=2,
                                  last_modified=None, tags=["mistudio-probe-definition"])]
    found = await ProbeHubService().search(query=None, base_model=None, limit=5)
    assert [r.repo_id for r in found] == ["mistudio/probe-monitors-test"], "once, though both tags answer it"
    assert len(hub.calls) == 2  # the frozen tag and its twin (040 FR-040.42)


def test_sae_repo_ids_under_mistudio_resolve(client, monkeypatch, tmp_path):  # noqa: F811
    from miserver.core.config import settings

    monkeypatch.setattr(settings, "SAE_CACHE_DIR", str(tmp_path))
    doc = json.loads((REPO / "tests" / "fixtures" / "lfm2_probe_definition_sae.json").read_text())
    repo = doc["sae"]["hf_repo"]
    assert repo.startswith("mistudio/sae-")
    response = client.post("/api/probes/import", json=doc)
    assert response.status_code == 200, response.text
    probe_id = response.json()["data"]["id"]
    stored = client.get(f"/api/probes/{probe_id}").json()["data"]
    assert repo in json.dumps(stored), "the stored SAE reference keeps the mistudio/ repo id verbatim"
