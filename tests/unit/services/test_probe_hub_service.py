"""Probe Hub browsing, with the Hub mocked.

The point of these is NOT to re-test the breaker, the cache or the manifest parsing — those belong
to `ClusterHubService` and are covered by `test_cluster_hub_service.py`. What is tested here is
that the four knobs are actually wired: the probe tag reaches the Hub query, the probe suffix gates
what can be imported, downloads land in the probes cache directory, and a fetched document is
validated against the PROBE contract rather than the cluster one.

⚠ That last one is the reason this file exists. If `DEFINITION_MODEL` were left inherited, a probe
fetched from the Hub would be validated as a cluster definition — and a cluster mirror with
`extra="allow"` accepts almost any object, so it would import cleanly and fail much later, in the
runtime, with nothing pointing back here.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from millm.api.schemas.probe import ProbeDefinitionV1
from millm.core.errors import ValidationError
from millm.services.probe_hub_service import ProbeHubService
from tests.unit.probe_fixtures import probe_definition

pytestmark = pytest.mark.asyncio


@pytest.fixture
def service(tmp_path):
    with patch("millm.services.cluster_hub_service.settings") as st:
        st.PROBE_HUB_CACHE_TTL_S = 300
        st.PROBE_HUB_TAG = "mistudio-probe-definition"
        st.SAE_CACHE_DIR = str(tmp_path)
        with patch("millm.services.probe_hub_service.settings", st, create=True):
            yield ProbeHubService()


class TestTheKnobsAreWired:
    async def test_search_uses_the_PROBE_tag(self, service):
        with patch("millm.services.cluster_hub_service._list_models_sync") as list_models:
            list_models.return_value = []
            await service.search(query="stakes", base_model="LiquidAI/LFM2.5-1.2B-Instruct")
        list_models.assert_called_once_with(
            "mistudio-probe-definition", "stakes", "LiquidAI/LFM2.5-1.2B-Instruct", 30
        )

    async def test_the_cache_directory_is_probes_not_clusters(self, service, tmp_path):
        assert service._cache_dir == str(tmp_path / "probes")

    def test_the_suffix_and_model_are_the_probe_ones(self, service):
        assert service.DEFINITION_SUFFIX == ".probe.json"
        assert service.DEFINITION_MODEL is ProbeDefinitionV1
        assert service.ARTIFACT_NAME == "probe"


class TestFetch:
    async def test_it_validates_against_the_PROBE_contract(self, service, tmp_path):
        """⚠ The knob that matters most.

        A cluster mirror with `extra="allow"` accepts nearly any object, so validating a probe
        against it would import cleanly and only fail later in the runtime.
        """
        path = tmp_path / "high-stakes.probe.json"
        path.write_text(json.dumps(probe_definition()))
        with patch(
            "millm.services.cluster_hub_service._download_file_sync", return_value=str(path)
        ):
            definition, raw, hub_ref = await service.fetch_definition(
                "mistudio/probe-monitors-test", "high-stakes.probe.json"
            )
        assert isinstance(definition, ProbeDefinitionV1)
        assert definition.read.layer == 11
        assert raw["test_vectors"]["vectors"][0]["token_ids"] == [1, 2, 3, 4]
        assert hub_ref["repo_id"] == "mistudio/probe-monitors-test"

    async def test_a_cluster_file_cannot_be_imported_as_a_probe(self, service):
        with pytest.raises(ValidationError, match=r"Only \.probe\.json files"):
            await service.fetch_definition("org/pack", "fear.cluster.json")

    async def test_path_traversal_is_refused(self, service):
        for bad in ("../../etc/passwd.probe.json", "/etc/passwd.probe.json"):
            with pytest.raises(ValidationError):
                await service.fetch_definition("org/pack", bad)

    async def test_the_raw_payload_survives_for_lossless_re_export(self, service, tmp_path):
        doc = probe_definition()
        doc["future_field"] = {"kept": True}
        path = tmp_path / "x.probe.json"
        path.write_text(json.dumps(doc))
        with patch(
            "millm.services.cluster_hub_service._download_file_sync", return_value=str(path)
        ):
            _definition, raw, _ref = await service.fetch_definition("org/p", "x.probe.json")
        assert raw["future_field"] == {"kept": True}


class TestListing:
    async def test_loose_files_are_filtered_by_the_probe_suffix(self, service):
        with patch("millm.services.cluster_hub_service._list_repo_files_sync") as files:
            files.return_value = [
                "README.md",
                "high-stakes.probe.json",
                "fear.cluster.json",
                "deception.probe.json",
            ]
            listed = await service.list_definitions("mistudio/probe-monitors-test")
        names = {item["file"] if isinstance(item, dict) else item.file for item in listed}
        assert names == {"high-stakes.probe.json", "deception.probe.json"}
