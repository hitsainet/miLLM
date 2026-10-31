"""What the probe contract refuses, and why each refusal exists.

Every validator here guards a failure that is **silent** rather than loud — a definition that
imports cleanly and then detects something other than what it claims. miStudio validates the same
things on the way out, and that is not a reason to skip them on the way in: a document can reach
miLLM by a route that never passed through miStudio's endpoint (a hand-edited file, an older
export, a third party's repo). A contract validated only at the producer is not validated.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from millm.api.schemas.probe import ProbeDefinitionV1
from tests.unit.probe_fixtures import acknowledged, probe_definition, sae_probe_definition


def parse(doc: dict) -> ProbeDefinitionV1:
    return ProbeDefinitionV1.model_validate(doc)


class TestTheFixtureIsActuallyValid:
    def test_a_dense_probe_parses(self):
        p = parse(probe_definition())
        assert p.name == "high-stakes"
        assert p.read.layer == 11
        assert p.aggregation.rule == "mean"

    def test_an_sae_probe_parses(self):
        p = parse(sae_probe_definition())
        assert p.basis == "sae_features"
        assert p.sae is not None
        assert len(p.head.weights) == len(p.sae.feature_indices)


class TestTheBasisMustBeUnambiguous:
    def test_an_sae_basis_without_an_sae_block_is_refused(self):
        doc = probe_definition(basis="sae_features")
        with pytest.raises(ValidationError, match="no `sae` block"):
            parse(doc)

    def test_a_residual_basis_carrying_an_sae_block_is_refused(self):
        """Carrying both does not say which basis the weights are in."""
        doc = sae_probe_definition()
        doc["basis"] = "residual"
        with pytest.raises(ValidationError, match="does not say which basis"):
            parse(doc)

    def test_a_dense_head_must_be_d_model_wide(self):
        doc = probe_definition()
        doc["head"]["weights"] = [1.0, 2.0]
        doc["head"]["norm_mean"] = [0.0, 0.0]
        doc["head"]["norm_std"] = [1.0, 1.0]
        with pytest.raises(ValidationError, match="d_model=8"):
            parse(doc)

    def test_an_sae_head_must_match_its_feature_count(self):
        doc = sae_probe_definition(k=4)
        doc["sae"]["feature_indices"] = [3, 11]
        with pytest.raises(ValidationError, match="they pair by position"):
            parse(doc)

    def test_feature_indices_must_be_sorted_and_unique(self):
        """Each weight pairs with one feature BY POSITION, so a reorder re-points every weight."""
        doc = sae_probe_definition(k=4)
        doc["sae"]["feature_indices"] = [47, 3, 11, 29]
        with pytest.raises(ValidationError, match="sorted and unique"):
            parse(doc)

        doc["sae"]["feature_indices"] = [3, 3, 11, 29]
        with pytest.raises(ValidationError, match="sorted and unique"):
            parse(doc)


class TestTheReadPoint:
    def test_a_hook_point_other_than_resid_post_is_refused(self):
        """miStudio has already shipped an estate-wide defect from this exact confusion: every
        SAE trained before `4e334f28` learned a normalised pre-MLP signal because "residual"
        resolved to a norm module."""
        doc = probe_definition()
        doc["read"]["hook_point"] = "ffn_norm"
        with pytest.raises(ValidationError, match="hook_point must be 'resid_post'"):
            parse(doc)

    def test_a_layer_outside_the_model_is_refused(self):
        doc = probe_definition()
        doc["read"]["layer"] = 99
        with pytest.raises(ValidationError, match="outside a 16-layer model"):
            parse(doc)


class TestStandardisation:
    def test_a_zero_in_norm_std_is_refused(self):
        """Clamping a 0 std to eps turns a 0.001 drift into 1000."""
        doc = probe_definition()
        doc["head"]["norm_std"] = [1.0] * 7 + [0.0]
        with pytest.raises(ValidationError, match="norm_std contains a zero"):
            parse(doc)

    def test_mismatched_normalisation_width_is_refused(self):
        doc = probe_definition()
        doc["head"]["norm_mean"] = [0.0] * 3
        with pytest.raises(ValidationError, match="normalisation must match its weights"):
            parse(doc)


class TestAggregation:
    def test_an_unknown_rule_is_refused(self):
        doc = probe_definition()
        doc["aggregation"]["rule"] = "meen"
        with pytest.raises(ValidationError, match="unknown rule"):
            parse(doc)

    def test_streamable_must_be_derived_not_asserted(self):
        """A document claiming `last` streams would make the runtime emit a verdict that is a
        guess the row is about to end."""
        doc = probe_definition()
        doc["aggregation"]["rule"] = "last"
        doc["aggregation"]["streamable"] = True
        with pytest.raises(ValidationError, match="not defined until the sequence ends"):
            parse(doc)

    def test_last_is_accepted_when_it_declares_itself_non_streamable(self):
        doc = probe_definition()
        doc["aggregation"] = {"rule": "last", "params": {}, "streamable": False}
        assert parse(doc).aggregation.rule == "last"

    def test_attention_without_a_query_is_refused(self):
        """Without the query the rule silently becomes softmax at tau=1."""
        doc = probe_definition()
        doc["aggregation"] = {"rule": "attention", "params": {}, "streamable": True}
        with pytest.raises(ValidationError, match="needs `head.attention_query`"):
            parse(doc)

    def test_attention_with_a_query_is_accepted(self):
        doc = probe_definition()
        doc["aggregation"] = {"rule": "attention", "params": {}, "streamable": True}
        doc["head"]["attention_query"] = [0.1] * 8
        assert parse(doc).aggregation.rule == "attention"


class TestEvidence:
    def test_below_rung_two_needs_an_acknowledgement_in_the_DOCUMENT(self):
        """⚠ The gate is in the contract, not only at the endpoint.

        An endpoint gate is bypassed by a hand-edited file or a future caller, and the refusal
        should travel with the format.
        """
        doc = probe_definition()
        doc["evidence"] = {"rung": 1, "rung_language": "detects on held-out data",
                           "acknowledgement": None, "evaluations": []}
        with pytest.raises(ValidationError, match="must record an acknowledgement"):
            parse(doc)

    def test_an_acknowledged_low_rung_probe_is_accepted(self):
        doc = probe_definition()
        doc["evidence"] = {"rung": 0, "rung_language": "trained",
                           "acknowledgement": acknowledged(), "evaluations": []}
        assert parse(doc).evidence.rung == 0

    def test_rung_two_and_above_needs_no_acknowledgement(self):
        assert parse(probe_definition()).evidence.acknowledgement is None

    def test_a_rung_outside_the_ladder_is_refused(self):
        doc = probe_definition()
        doc["evidence"]["rung"] = 4
        with pytest.raises(ValidationError):
            parse(doc)

    def test_the_language_travels_with_the_document(self):
        """miStudio owns the vocabulary; a local number→phrase map would be a second vocabulary
        free to drift, and language rising above evidence is the thing most likely to drift."""
        doc = probe_definition()
        doc["evidence"]["rung_language"] = "whatever miStudio said"
        assert parse(doc).evidence.rung_language == "whatever miStudio said"


class TestKindAndScope:
    def test_another_kind_is_refused(self):
        doc = probe_definition(kind="mistudio.circuit-definition/v1")
        with pytest.raises(ValidationError, match="unknown kind"):
            parse(doc)

    def test_an_unknown_scope_is_refused(self):
        doc = probe_definition(scope="everything")
        with pytest.raises(ValidationError, match="unknown scope"):
            parse(doc)

    @pytest.mark.parametrize("scope", ["all", "prompt", "response"])
    def test_every_contract_scope_is_accepted(self, scope):
        assert parse(probe_definition(scope=scope)).scope == scope

    @pytest.mark.parametrize("scope", ["assistant", "user", "last_assistant"])
    def test_miSTUDIOS_INTERNAL_scope_names_are_NOT_contract_values(self, scope):
        """⚠ These three are 032's internal vocabulary and are mapped away on export.

        The mirror accepted them until 2026-09-27 — and therefore REJECTED `prompt` and
        `response`, which every real document actually carries. The schema-sync test passed
        throughout, because it verified that `scope` existed and was required, never what it
        accepted. `TestEveryEnumMatchesTheContract` now walks every enum in the frozen file.
        """
        with pytest.raises(ValidationError, match="unknown scope"):
            parse(probe_definition(scope=scope))


class TestTestVectors:
    def test_at_least_one_vector_is_required(self):
        """A definition with no vectors cannot be parity-checked, so it could be armed without
        anything ever confirming it scores as miStudio measured."""
        doc = probe_definition()
        doc["test_vectors"]["vectors"] = []
        with pytest.raises(ValidationError):
            parse(doc)

    def test_the_tolerance_must_be_positive(self):
        doc = probe_definition()
        doc["test_vectors"]["tolerance"] = 0.0
        with pytest.raises(ValidationError):
            parse(doc)

    def test_token_ids_is_the_authoritative_input(self):
        assert parse(probe_definition()).test_vectors.authoritative_input == "token_ids"
