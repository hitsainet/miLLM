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

from millm.ml.circuit_steering import CircuitSteeringEngine

# F18 task 2.7: the characterization ASSERTIONS are unchanged; only the name
# they call moved, from `CircuitService._serving_members` to the engine. That
# is the whole parity claim — the same tests, the same expectations, a
# different owner. The shim lives HERE, in the test file, and deliberately not
# in production: a forwarding stub in the service would leave two names for one
# thing, which is how four derivations happened in the first place.
class CircuitService:
    _serving_members = staticmethod(CircuitSteeringEngine.serving_members)


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


class TestAllFourSitesShareTheSameDerivation:
    """WAS `TestAllFourSitesAgreeBeforeTheMove` — the pre-move witness that the
    four sites agreed. They did.

    Behaviour change, justified (F18 task 3.5): the two methods it compared no
    longer exist, because that is the feature. Comparing
    `CircuitService._serving_members` with
    `InferenceService._circuit_serving_members` is now impossible, and a test
    that reinstated shims to keep comparing them would be testing the shims.

    What replaces it asserts the STRONGER property: there is one derivation,
    and the four sites reach it."""

    def _definition(self):
        return defn(
            [
                mem(10, feature=feat(1), expanded=[feat(2)]),
                mem(13, feature=feat(3)),
            ],
            sae_by_layer={10: sae_ref("sae-A"), 13: sae_ref("sae-B")},
        )

    def test_there_is_exactly_ONE_flattening_implementation(self):
        """ENG-D4 / task 5.4. A second implementation appearing is the defect
        this whole feature exists to prevent."""
        import subprocess

        # Target the SERVING shape specifically. A bare grep for
        # `expanded_members` also matches two legitimate and DIFFERENT
        # derivations, which this test found on its first run and which are
        # worth naming so nobody "fixes" them into the engine:
        #
        #   millm/api/schemas/circuit.py — counts feature_idx per layer for the
        #     member-cap validator. Different output (a count), different
        #     consumer (validation), and deduping there is about the cap.
        #   circuit_service._slice_members — the slice-fallback PROJECTION,
        #     which emits a per-layer cluster document for the F8 importer.
        #     Deduped on feature_idx alone because a cluster is keyed by index;
        #     the serving path keys on (layer, feature_idx).
        #
        # What must exist exactly once is the construction of `CircuitMember`
        # for the serving path.
        out = subprocess.run(
            ["grep", "-rn", "CircuitMember(", "millm/"],
            capture_output=True, text=True,
        ).stdout
        sites = sorted({
            ln.split(":")[0] for ln in out.splitlines()
            if ln.strip() and "class CircuitMember" not in ln
        })
        assert sites == ["millm/ml/circuit_steering.py"], (
            f"the serving flattening exists in {sites} — a second derivation "
            "has appeared"
        )

    def test_the_deleted_helpers_are_really_gone_with_no_shims(self):
        from millm.services.circuit_service import CircuitService as RealCS
        from millm.services.inference_service import InferenceService

        assert not hasattr(RealCS, "_serving_members")
        assert not hasattr(InferenceService, "_circuit_serving_members")
        assert not hasattr(InferenceService, "_sae_service_for_dial")

    def test_the_claim_set_IS_the_member_layers(self):
        """ENG-K1, the structural fix for F14-R2-01: the layers a request
        snapshots and the layers its apply drives are the same set, not two
        sets that agree."""
        d = self._definition()
        plan = CircuitSteeringEngine().plan_for(d, SimpleNamespace(intensity=1.0))
        assert plan.claimed_layers == frozenset(m.layer for m in plan.members)

    def test_the_member_layer_set_can_EXCEED_the_definitions_layers_list(self):
        """The F14-R2-01 shape itself, preserved from the pre-move gate:
        `definition.layers()` is not the same thing as the layers the members
        claim, and the apply follows the members."""
        d = defn(
            [mem(10, feature=feat(1)), mem(99, feature=feat(2))],
            sae_by_layer={10: sae_ref("sae-A")},
        )
        plan = CircuitSteeringEngine().plan_for(d, SimpleNamespace(intensity=1.0))
        assert 99 in plan.claimed_layers


class TestTheDialsServiceConstructionIsNowTotal:
    """The BEFORE state was: `_sae_service_for_dial` built an SAEService via
    `__new__`, leaving `_downloader`, `_loader`, `_hooker`,
    `_inference_service` and two collections UNSET. It worked only because the
    dial path happened to touch none of them — a partially-constructed object
    on the inference hot path.

    CTX-V2-style behaviour change, justified: the bypass is gone, so the tests
    that pinned its shape are replaced by tests that the replacement is TOTAL.
    Asserting the old unset fields would now be asserting the defect."""

    def test_for_registry_sets_every_field___init___sets(self):
        import inspect
        import re

        from millm.services.sae_service import SAEService

        init_fields = set(
            re.findall(r"self\.(_[a-z_]+)\s*[:=]", inspect.getsource(SAEService.__init__))
        )
        svc = SAEService.for_registry()
        missing = sorted(f for f in init_fields if not hasattr(svc, f))
        assert missing == [], f"partially constructed: {missing} unset"

    def test_it_shares_the_singleton_registry(self):
        from millm.services.sae_service import AttachedSAEState, SAEService

        assert SAEService.for_registry()._sae_state is AttachedSAEState()

    def test_the_new_bypass_is_gone_from_the_tree(self):
        """No shim, no second construction path (ENG-D4)."""
        import subprocess

        # Match a CALL, not prose. The `for_registry` docstring names the
        # retired bypass on purpose, and a grep that cannot tell an explanation
        # from an invocation would either fail forever or be deleted.
        # An ASSIGNMENT from the bypass, not a mention of it. The
        # `for_registry` docstring names the retired pattern deliberately, and
        # a grep that cannot tell an explanation from an invocation would
        # either fail forever or get deleted — which is how a guard stops
        # guarding.
        out = subprocess.run(
            ["grep", "-rnE", r"=\s*SAEService\.__new__\(", "millm/"],
            capture_output=True, text=True,
        ).stdout
        offenders = [ln for ln in out.splitlines() if ln.strip()]
        assert not offenders, f"the __new__ bypass is back: {offenders}"
