"""
Unit tests for ClusterHubService (Feature 8, Task 3.3/3.4) with the Hub
mocked: search filter composition, manifest vs loose-file listing, caps,
suffix/size validation, provenance, and TTL caching.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from millm.core.errors import ValidationError
from millm.services.cluster_hub_service import (
    MAX_LISTED_DEFINITIONS,
    ClusterHubService,
)


def make_definition_payload(name="fear cluster") -> dict:
    return {
        "kind": "mistudio.cluster-definition",
        "schema_version": "1",
        "name": name,
        "members": [{"feature_idx": 1, "strength": 1.0}],
    }


@pytest.fixture
def service(tmp_path):
    with patch("millm.services.cluster_hub_service.settings") as st:
        st.CLUSTER_HUB_CACHE_TTL_S = 300
        st.CLUSTER_HUB_TAG = "mistudio-cluster-definition"
        st.SAE_CACHE_DIR = str(tmp_path)
        yield ClusterHubService()


class TestSearch:
    async def test_search_composes_tag_and_base_model_filters(self, service):
        with patch(
            "millm.services.cluster_hub_service._list_models_sync"
        ) as list_models:
            list_models.return_value = [
                MagicMock(id="org/pack", likes=3, downloads=10,
                          last_modified=None, tags=["mistudio"]),
            ]
            result = await service.search(query="fear", base_model="google/gemma-2-2b")
        list_models.assert_called_once_with(
            "mistudio-cluster-definition", "fear", "google/gemma-2-2b", 30
        )
        assert result[0].repo_id == "org/pack"
        assert result[0].likes == 3

    async def test_search_limit_capped_at_50(self, service):
        with patch(
            "millm.services.cluster_hub_service._list_models_sync", return_value=[]
        ) as list_models:
            await service.search(limit=500)
        assert list_models.call_args[0][3] == 50

    async def test_search_results_cached(self, service):
        with patch(
            "millm.services.cluster_hub_service._list_models_sync", return_value=[]
        ) as list_models:
            await service.search(query="x")
            await service.search(query="x")
        list_models.assert_called_once()


class TestListDefinitions:
    async def test_loose_files_fallback_filters_suffix_and_caps(self, service):
        files = [f"c{i}.cluster.json" for i in range(MAX_LISTED_DEFINITIONS + 20)]
        files += ["README.md", "evil.json", "manifest.txt"]
        with patch(
            "millm.services.cluster_hub_service._list_repo_files_sync",
            return_value=files,
        ):
            refs = await service.list_definitions("org/pack")
        assert len(refs) == MAX_LISTED_DEFINITIONS
        assert all(r.file.endswith(".cluster.json") for r in refs)

    async def test_manifest_preferred_and_malformed_lines_skipped(self, service, tmp_path):
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text(
            json.dumps({"file": "a.cluster.json", "name": "A", "member_count": 3})
            + "\nnot json at all\n"
            + json.dumps({"file": "evil.json", "name": "skipped-wrong-suffix"})
            + "\n"
            + json.dumps({"file": "b.cluster.json", "name": "B"})
            + "\n"
        )
        with patch(
            "millm.services.cluster_hub_service._list_repo_files_sync",
            return_value=["manifest.jsonl", "a.cluster.json", "b.cluster.json"],
        ), patch(
            "millm.services.cluster_hub_service._download_file_sync",
            return_value=str(manifest),
        ):
            refs = await service.list_definitions("org/pack")
        assert [r.file for r in refs] == ["a.cluster.json", "b.cluster.json"]
        assert refs[0].member_count == 3


class TestFetchDefinition:
    async def test_fetch_validates_and_returns_provenance(self, service, tmp_path):
        f = tmp_path / "fear.cluster.json"
        f.write_text(json.dumps(make_definition_payload()))
        with patch(
            "millm.services.cluster_hub_service._download_file_sync",
            return_value=str(f),
        ):
            definition, raw, hub_ref = await service.fetch_definition(
                "org/pack", "fear.cluster.json", revision="abc123"
            )
        assert definition.name == "fear cluster"
        assert raw["name"] == "fear cluster"   # raw payload travels to storage
        assert hub_ref == {"repo_id": "org/pack", "revision": "abc123",
                           "path": "fear.cluster.json"}

    async def test_fetch_rejects_wrong_suffix(self, service):
        with pytest.raises(ValidationError, match="cluster.json"):
            await service.fetch_definition("org/pack", "model.safetensors")

    @pytest.mark.parametrize("bad", ["../etc/passwd.cluster.json", "/abs.cluster.json"])
    async def test_fetch_rejects_traversal(self, service, bad):
        with pytest.raises(ValidationError, match="Invalid filename"):
            await service.fetch_definition("org/pack", bad)

    async def test_fetch_rejects_oversize(self, service, tmp_path):
        f = tmp_path / "big.cluster.json"
        f.write_text("x" * 1_100_000)
        with patch(
            "millm.services.cluster_hub_service._download_file_sync",
            return_value=str(f),
        ):
            with pytest.raises(ValidationError, match="exceeds"):
                await service.fetch_definition("org/pack", "big.cluster.json")
