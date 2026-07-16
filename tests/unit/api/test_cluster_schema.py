"""
Validation tests for the cluster interchange mirror (Feature 8, Task 2.4):
valid documents, hostile payloads, and every cap the contract enforces.
"""

import pytest
from pydantic import ValidationError

from millm.api.schemas.cluster import (
    MAX_BUNDLE,
    MAX_MEMBERS,
    ClusterBundleV1,
    ClusterDefinitionV1,
    DefinitionSAERef,
    ProfileMember,
)
from millm.core.steering_range import STEERING_RANGE, clamp_steering, would_clamp


def make_definition(**overrides) -> dict:
    base = {
        "kind": "mistudio.cluster-definition",
        "schema_version": "1",
        "name": "fear cluster",
        "narrative": "Steers toward fear-adjacent tokens.",
        "display_token": "fear",
        "model": {"hf_id": "google/gemma-2-2b"},
        "sae": {"mistudio_sae_id": "sae_a", "layer": 12, "n_features": 16384},
        "members": [
            {"feature_idx": 100, "strength": 1.2, "sign": 1, "max_activation": 3.1},
            {"feature_idx": 200, "strength": -0.8, "sign": -1},
        ],
        "budget": {"B": 2.4, "intensity": 1.0, "intensity_range": [0.5, 1.5]},
        "provenance": {"mistudio_version": "0.5.0"},
    }
    base.update(overrides)
    return base


class TestValidDocuments:
    def test_definition_parses(self):
        d = ClusterDefinitionV1.model_validate(make_definition())
        assert d.name == "fear cluster"
        assert d.members[1].sign == -1
        assert d.budget.intensity_range == [0.5, 1.5]

    def test_bundle_parses(self):
        b = ClusterBundleV1.model_validate({
            "kind": "mistudio.cluster-bundle",
            "schema_version": "1",
            "definitions": [make_definition(), make_definition(name="joy")],
        })
        assert [d.name for d in b.definitions] == ["fear cluster", "joy"]

    def test_minimal_definition(self):
        d = ClusterDefinitionV1.model_validate({
            "kind": "mistudio.cluster-definition",
            "schema_version": "1",
            "name": "x",
            "members": [{"feature_idx": 0, "strength": 1.0}],
        })
        assert d.sae.n_features is None
        assert d.budget is None


class TestHostilePayloads:
    def test_wrong_kind_rejected(self):
        with pytest.raises(ValidationError):
            ClusterDefinitionV1.model_validate(make_definition(kind="mistudio.evil"))

    def test_wrong_schema_major_rejected(self):
        with pytest.raises(ValidationError):
            ClusterDefinitionV1.model_validate(make_definition(schema_version="2"))

    @pytest.mark.parametrize("bad", ["/data/saes/x", "~/saes/x", "../x", "C:\\saes\\x"])
    def test_source_hint_paths_rejected(self, bad):
        with pytest.raises(ValidationError):
            DefinitionSAERef(source_hint=bad)

    def test_source_hint_hf_style_allowed(self):
        assert DefinitionSAERef(source_hint="hf:repo/path").source_hint == "hf:repo/path"

    def test_strength_out_of_contract_range_rejected(self):
        with pytest.raises(ValidationError):
            ProfileMember(feature_idx=0, strength=301.0)

    def test_negative_feature_idx_rejected(self):
        with pytest.raises(ValidationError):
            ProfileMember(feature_idx=-1, strength=1.0)

    def test_empty_members_rejected(self):
        with pytest.raises(ValidationError):
            ClusterDefinitionV1.model_validate(make_definition(members=[]))

    def test_member_cap_enforced(self):
        members = [{"feature_idx": i, "strength": 1.0} for i in range(MAX_MEMBERS + 1)]
        with pytest.raises(ValidationError):
            ClusterDefinitionV1.model_validate(make_definition(members=members))

    def test_bundle_cap_enforced(self):
        defs = [make_definition(name=f"c{i}") for i in range(MAX_BUNDLE + 1)]
        with pytest.raises(ValidationError):
            ClusterBundleV1.model_validate({
                "kind": "mistudio.cluster-bundle", "schema_version": "1", "definitions": defs,
            })

    def test_narrative_cap_enforced(self):
        with pytest.raises(ValidationError):
            ClusterDefinitionV1.model_validate(make_definition(narrative="x" * 10_001))

    def test_intensity_bounds(self):
        with pytest.raises(ValidationError):
            ClusterDefinitionV1.model_validate(
                make_definition(budget={"intensity": 2.5}))


class TestSteeringRangeClamp:
    def test_within_range_untouched(self):
        assert clamp_steering(150.0) == 150.0
        assert clamp_steering(-199.9) == -199.9

    def test_clamps_both_sides(self):
        assert clamp_steering(600.0) == STEERING_RANGE
        assert clamp_steering(-250.0) == -STEERING_RANGE

    def test_would_clamp(self):
        assert would_clamp(200.1) is True
        assert would_clamp(-201.0) is True
        assert would_clamp(200.0) is False
