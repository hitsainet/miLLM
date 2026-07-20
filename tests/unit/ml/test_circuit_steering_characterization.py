"""Feature 18 task 1.0 — BEHAVIOUR PRESERVATION GATE.

These tests pin the CURRENT serving derivation, written and green BEFORE any
code moves. F18 collapses four independent derivations of the same serving plan
into one engine; this is the contract that the collapse preserves behaviour
rather than quietly changing it.

Why the gate exists: F17's equivalent gate caught three regressions during its
extraction, including one that silently reverted an O(log n) fix and one that
resurrected a defect two earlier rounds had fixed. A refactor described as a
"pure move" is exactly the kind that is not.

**From here on, editing a test in this file is a BEHAVIOUR CHANGE and requires
justification in the review record.** A failure after the move is a regression,
not a stale fixture.
"""

from types import SimpleNamespace

import pytest

from millm.services.circuit_service import CircuitService


# ─────────────────────────────────────────────────────────────────────────
# Fixtures — deliberately built from the real shapes, never from mocks whose
# fields agree with the assertion by construction (the R1-12/R3-06 trap).
# ─────────────────────────────────────────────────────────────────────────


def feat(idx, strength=1.0, sign=1, label=None):
    return SimpleNamespace(
        feature_idx=idx, strength=strength, sign=sign, label=label
    )


def mem(layer, feature=None, expanded=None):
    return SimpleNamespace(
        layer=layer, feature=feature, expanded_members=expanded
    )


def defn(members, sae_by_layer=None, budget=None, edges=()):
    saes = sae_by_layer or {}

    return SimpleNamespace(
        members=members,
        edges=list(edges),
        budget=budget,
        sae_for_layer=lambda layer: saes.get(layer),
        layers=lambda: sorted({m.layer for m in members}),
    )


def sae_ref(sae_id):
    return SimpleNamespace(mistudio_sae_id=sae_id)


# ─────────────────────────────────────────────────────────────────────────
# 1.2 — the flattening rules (EC-18.1, EC-18.2, EC-18.6)
# ─────────────────────────────────────────────────────────────────────────


class TestFlatteningRules:
    def test_a_cluster_ref_contributes_BOTH_sources(self):
        """EC-18.1. Taking only `expanded_members` OR only `feature` silently
        dropped authored members from the intervention — the whole point of
        the rule."""
        d = defn([mem(10, feature=feat(1), expanded=[feat(2), feat(3)])])
        out = CircuitService._serving_members(d)
        assert sorted(m.feature_idx for m in out) == [1, 2, 3]

    def test_expanded_members_come_FIRST(self):
        """Order is observable: dedupe is first-wins, so which source leads
        decides which duplicate survives."""
        d = defn([mem(10, feature=feat(1), expanded=[feat(2)])])
        out = CircuitService._serving_members(d)
        assert [m.feature_idx for m in out] == [2, 1]

    def test_a_duplicate_layer_feature_key_is_collapsed_FIRST_WINS(self):
        """EC-18.2. The serving path rejects a repeated key outright, so the
        collapse is what keeps a legitimate circuit serveable — and the FIRST
        occurrence is the one that survives, carrying ITS strength."""
        d = defn([mem(10, feature=feat(1, strength=9.0),
                      expanded=[feat(1, strength=2.0)])])
        out = CircuitService._serving_members(d)
        assert len(out) == 1
        assert out[0].budget == 2.0, "the later duplicate won"

    def test_the_same_feature_idx_on_DIFFERENT_layers_is_not_a_duplicate(self):
        """The key is (layer, feature_idx). Collapsing on feature_idx alone
        would drop a legitimate member of a multi-layer circuit."""
        d = defn([mem(10, feature=feat(1)), mem(13, feature=feat(1))])
        out = CircuitService._serving_members(d)
        assert len(out) == 2
        assert sorted(m.layer for m in out) == [10, 13]

    def test_an_empty_definition_yields_no_members(self):
        """EC-18.6."""
        assert CircuitService._serving_members(defn([])) == []

    def test_a_member_with_neither_source_contributes_nothing(self):
        d = defn([mem(10, feature=None, expanded=None)])
        assert CircuitService._serving_members(d) == []

    def test_definition_order_is_preserved_across_members(self):
        d = defn([mem(13, feature=feat(5)), mem(10, feature=feat(1))])
        out = CircuitService._serving_members(d)
        assert [(m.layer, m.feature_idx) for m in out] == [(13, 5), (10, 1)]


class TestPerLayerSaeResolution:
    def test_each_member_carries_ITS_OWN_layers_sae_id(self):
        """A feature on layer L must be steered by the SAE whose .layer == L.
        Carrying the wrong one indexes into a different feature space."""
        d = defn(
            [mem(10, feature=feat(1)), mem(13, feature=feat(2))],
            sae_by_layer={10: sae_ref("sae-A"), 13: sae_ref("sae-B")},
        )
        out = CircuitService._serving_members(d)
        by_layer = {m.layer: m.sae_id for m in out}
        assert by_layer == {10: "sae-A", 13: "sae-B"}

    def test_a_layer_with_no_sae_reference_carries_None(self):
        d = defn([mem(10, feature=feat(1))], sae_by_layer={})
        assert CircuitService._serving_members(d)[0].sae_id is None


class TestTheSignRuleIsCarriedNotApplied:
    """EC-18.3. A NEGATIVE strength is already directional — the canonical sign
    rule. The flattening must carry `budget` and `sign` UNTOUCHED and leave the
    combination to `_directional_budget`, or the sign is applied twice."""

    def test_budget_and_sign_are_carried_verbatim(self):
        d = defn([mem(10, feature=feat(1, strength=-3.0, sign=-1))])
        out = CircuitService._serving_members(d)
        assert out[0].budget == -3.0, "the flattening pre-applied the sign"
        assert out[0].sign == -1

    def test_a_positive_strength_is_also_untouched(self):
        d = defn([mem(10, feature=feat(1, strength=2.5, sign=1))])
        out = CircuitService._serving_members(d)
        assert (out[0].budget, out[0].sign) == (2.5, 1)


class TestLabelsRide:
    def test_the_label_is_carried(self):
        d = defn([mem(10, feature=feat(1, label="fear"))])
        assert CircuitService._serving_members(d)[0].label == "fear"


# ─────────────────────────────────────────────────────────────────────────
# 1.3 — intensity resolution (EC-18.4)
# ─────────────────────────────────────────────────────────────────────────


class TestIntensityResolution:
    """The document's budget.intensity WINS over the DB column when both are
    present and differ. F14-R1-01: dialling a circuit authored at 150 to λ=1.0
    must yield 150, not 100 — the DB column is a cache, the document is the
    authored truth."""

    @staticmethod
    def _resolve(definition, circuit):
        # The live expression at circuit_service.py:425-427, characterized here
        # so the engine's `serving_intensity` can be checked against it.
        return (
            definition.budget.intensity if definition.budget else circuit.intensity
        )

    def test_the_document_wins_when_both_are_present(self):
        d = defn([], budget=SimpleNamespace(intensity=150.0))
        c = SimpleNamespace(intensity=100.0)
        assert self._resolve(d, c) == 150.0

    def test_the_db_column_is_used_when_the_document_has_no_budget(self):
        d = defn([], budget=None)
        c = SimpleNamespace(intensity=100.0)
        assert self._resolve(d, c) == 100.0

    def test_a_zero_document_intensity_still_wins(self):
        """0.0 is a legitimate authored value meaning 'off', and a truthiness
        check would silently fall through to the DB column."""
        d = defn([], budget=SimpleNamespace(intensity=0.0))
        c = SimpleNamespace(intensity=100.0)
        assert self._resolve(d, c) == 0.0


# ─────────────────────────────────────────────────────────────────────────
# 1.4 — the four sites agree TODAY (the pre-move witness)
# ─────────────────────────────────────────────────────────────────────────


class TestAllFourSitesAgreeBeforeTheMove:
    """The witness that F18 PRESERVES agreement rather than creating it. If
    these disagree now, the refactor is a behaviour change and the FTASKS'
    premise is wrong."""

    def _definition(self):
        return defn(
            [
                mem(10, feature=feat(1), expanded=[feat(2)]),
                mem(13, feature=feat(3)),
            ],
            sae_by_layer={10: sae_ref("sae-A"), 13: sae_ref("sae-B")},
        )

    def test_circuit_service_and_inference_service_derive_the_same_members(self):
        from millm.services.inference_service import InferenceService

        d = self._definition()
        a = CircuitService._serving_members(d)
        b = InferenceService._circuit_serving_members(d)
        assert [(m.layer, m.feature_idx, m.budget, m.sign, m.sae_id) for m in a] == [
            (m.layer, m.feature_idx, m.budget, m.sign, m.sae_id) for m in b
        ]

    def test_the_participating_LAYER_SET_is_the_same_from_both(self):
        """The claim set. F14-R2-01: the dial's snapshot must cover exactly the
        layers the apply drives, or a member layer missing from the
        `circuits.layers` column is steered and never restored."""
        from millm.services.inference_service import InferenceService

        d = self._definition()
        a = {m.layer for m in CircuitService._serving_members(d)}
        b = {m.layer for m in InferenceService._circuit_serving_members(d)}
        assert a == b == {10, 13}

    def test_the_member_layer_set_can_EXCEED_the_definitions_layers_list(self):
        """The F14-R2-01 shape itself: `definition.layers()` is not the same
        thing as the layers the members actually claim, and the apply follows
        the members. Pinned here because F18 moves `bound_layers` from one to
        the other."""
        d = defn(
            [mem(10, feature=feat(1)), mem(99, feature=feat(2))],
            sae_by_layer={10: sae_ref("sae-A")},
        )
        claimed = {m.layer for m in CircuitService._serving_members(d)}
        assert 99 in claimed


# ─────────────────────────────────────────────────────────────────────────
# 2.5 witness — the construction bypass being retired
# ─────────────────────────────────────────────────────────────────────────


class TestTheDialsServiceConstructionToday:
    """`_sae_service_for_dial` builds an SAEService via `__new__`, bypassing
    `__init__`. It works only because the dial path happens to touch nothing
    but `_sae_state` — a partially-constructed object on the inference hot
    path, one refactor away from an AttributeError in production.

    Pinned as the BEFORE state so the `for_registry` replacement can be shown
    to be total rather than merely different."""

    def test_the_bypass_leaves_four_fields_unset(self):
        from millm.services.inference_service import InferenceService

        svc = InferenceService._sae_service_for_dial()
        unset = [
            f for f in ("_downloader", "_hooker", "_inference_service", "_loader")
            if not hasattr(svc, f)
        ]
        assert unset, "the bypass is gone — retarget this at for_registry"

    def test_it_nonetheless_carries_a_usable_registry(self):
        from millm.services.inference_service import InferenceService

        svc = InferenceService._sae_service_for_dial()
        assert svc._sae_state is not None
        assert hasattr(svc, "set_circuit_steering")
