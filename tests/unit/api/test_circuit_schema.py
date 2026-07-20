"""Hostile-payload + validation tests for the circuit contract (F13, task 2.4).

Imported definitions are strictly DATA: miLLM never executes them, enforces
size/count caps, and refuses filesystem paths or credential-like content. These
tests pin the refusals that keep a malicious or malformed pack from doing harm.
"""

import json

import pytest
from pydantic import ValidationError

from millm.api.schemas.circuit import (
    MAX_CIRCUIT_IMPORT_BYTES,
    MAX_EDGES,
    MAX_MEMBERS_PER_LAYER,
    MAX_SAES,
    CircuitDefinitionV1,
)


def make_doc(**overrides) -> dict:
    doc = {
        "kind": "mistudio.circuit-definition",
        "schema_version": "1",
        "name": "fear→threat",
        "saes": [{"layer": 10, "n_features": 8192}, {"layer": 13, "n_features": 8192}],
        "members": [
            {"layer": 10, "feature": {"feature_idx": 1, "strength": 40.0}},
            {"layer": 13, "feature": {"feature_idx": 2, "strength": 30.0}},
        ],
        "edges": [
            {
                "up": {"layer": 10, "feature_idx": 1},
                "down": {"layer": 13, "feature_idx": 2},
                "rung": 2,
                "effect_size": 0.4,
            }
        ],
    }
    doc.update(overrides)
    return doc


class TestHappyPath:
    def test_valid_document_parses(self):
        c = CircuitDefinitionV1.model_validate(make_doc())
        assert c.name == "fear→threat"
        assert c.layers() == [10, 13]
        assert c.sae_for_layer(13).n_features == 8192
        assert c.edges[0].rung == 2

    def test_edges_optional(self):
        c = CircuitDefinitionV1.model_validate(make_doc(edges=[]))
        assert c.edges == []


class TestKindAndVersion:
    def test_unknown_kind_rejected(self):
        with pytest.raises(ValidationError):
            CircuitDefinitionV1.model_validate(
                make_doc(kind="mistudio.cluster-definition")
            )

    def test_totally_unknown_kind_rejected(self):
        with pytest.raises(ValidationError):
            CircuitDefinitionV1.model_validate(make_doc(kind="evil.payload"))

    def test_major_version_mismatch_rejected(self):
        with pytest.raises(ValidationError):
            CircuitDefinitionV1.model_validate(make_doc(schema_version="2"))


class TestCaps:
    def test_too_many_saes_rejected(self):
        saes = [{"layer": i, "n_features": 8192} for i in range(MAX_SAES + 1)]
        with pytest.raises(ValidationError):
            CircuitDefinitionV1.model_validate(make_doc(saes=saes))

    def test_max_saes_allowed(self):
        saes = [{"layer": i, "n_features": 8192} for i in range(MAX_SAES)]
        CircuitDefinitionV1.model_validate(make_doc(saes=saes))

    def test_too_many_edges_rejected(self):
        edges = [
            {"up": {"layer": 10, "feature_idx": i}, "down": {"layer": 13, "feature_idx": i}}
            for i in range(MAX_EDGES + 1)
        ]
        with pytest.raises(ValidationError):
            CircuitDefinitionV1.model_validate(make_doc(edges=edges))

    def test_too_many_members_on_one_layer_rejected(self):
        members = [
            {"layer": 10, "feature": {"feature_idx": i, "strength": 1.0}}
            for i in range(MAX_MEMBERS_PER_LAYER + 1)
        ]
        with pytest.raises(ValidationError, match="max"):
            CircuitDefinitionV1.model_validate(make_doc(members=members))

    def test_members_spread_across_layers_allowed(self):
        """The per-LAYER cap must not become a global cap — a circuit legitimately
        spans layers with up to 20 members each."""
        members = (
            [{"layer": 10, "feature": {"feature_idx": i, "strength": 1.0}} for i in range(20)]
            + [{"layer": 13, "feature": {"feature_idx": i, "strength": 1.0}} for i in range(20)]
        )
        c = CircuitDefinitionV1.model_validate(make_doc(members=members))
        assert len(c.members) == 40

    def test_cluster_ref_expanded_members_count_toward_layer_cap(self):
        members = [
            {
                "layer": 10,
                "member_kind": "cluster_ref",
                "expanded_members": [
                    {"feature_idx": i, "strength": 1.0} for i in range(MAX_MEMBERS_PER_LAYER + 1)
                ],
            }
        ]
        with pytest.raises(ValidationError, match="max"):
            CircuitDefinitionV1.model_validate(make_doc(members=members))

    def test_empty_members_rejected(self):
        with pytest.raises(ValidationError):
            CircuitDefinitionV1.model_validate(make_doc(members=[]))

    def test_empty_saes_rejected(self):
        with pytest.raises(ValidationError):
            CircuitDefinitionV1.model_validate(make_doc(saes=[]))

    def test_import_byte_cap_is_one_mb(self):
        assert MAX_CIRCUIT_IMPORT_BYTES == 1_048_576

    def test_oversize_payload_detectable_by_byte_cap(self):
        """A definition near the cap is hostile — the service refuses on bytes
        before parsing (this pins the measurement the service uses)."""
        doc = make_doc(narrative="x" * 20_000)
        raw = json.dumps(doc).encode()
        assert len(raw) < MAX_CIRCUIT_IMPORT_BYTES  # sane doc fits
        huge = json.dumps(make_doc(narrative="x" * 2_000_000)).encode()
        assert len(huge) > MAX_CIRCUIT_IMPORT_BYTES


class TestNoLocalPathsOrCredentials:
    @pytest.mark.parametrize(
        "hint",
        ["/etc/passwd", "~/secrets/sae", "../../../root", "C:\\Users\\me\\sae"],
    )
    def test_filesystem_paths_in_source_hint_rejected(self, hint):
        """Reused from the cluster contract — the format must stay portable and
        must never smuggle a local path."""
        with pytest.raises(ValidationError):
            CircuitDefinitionV1.model_validate(
                make_doc(saes=[{"layer": 10, "source_hint": hint}])
            )

    def test_hf_style_source_hint_allowed(self):
        c = CircuitDefinitionV1.model_validate(
            make_doc(saes=[{"layer": 10, "source_hint": "hf:google/gemma-scope"}])
        )
        assert c.saes[0].source_hint == "hf:google/gemma-scope"


class TestFieldConstraints:
    def test_negative_layer_rejected(self):
        with pytest.raises(ValidationError):
            CircuitDefinitionV1.model_validate(
                make_doc(members=[{"layer": -1, "feature": {"feature_idx": 1, "strength": 1.0}}])
            )

    def test_rung_out_of_range_rejected(self):
        bad = {"up": {"layer": 1, "feature_idx": 0}, "down": {"layer": 2, "feature_idx": 0}, "rung": 9}
        with pytest.raises(ValidationError):
            CircuitDefinitionV1.model_validate(make_doc(edges=[bad]))

    def test_member_strength_out_of_contract_range_rejected(self):
        with pytest.raises(ValidationError):
            CircuitDefinitionV1.model_validate(
                make_doc(members=[{"layer": 10, "feature": {"feature_idx": 1, "strength": 5000.0}}])
            )

    def test_intensity_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            CircuitDefinitionV1.model_validate(
                make_doc(budget={"layers": {}, "intensity": 9.0})
            )

    def test_name_length_capped(self):
        with pytest.raises(ValidationError):
            CircuitDefinitionV1.model_validate(make_doc(name="x" * 500))

    def test_empty_name_rejected(self):
        with pytest.raises(ValidationError):
            CircuitDefinitionV1.model_validate(make_doc(name=""))
