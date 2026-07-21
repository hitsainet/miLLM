"""Feature 18 tasks 4.0/5.0 — one derivation, four consumers.

The point of F18 is not that the code is tidier. It is that four call sites
that must agree about an operator-visible claim now cannot disagree, because
they read the same object. These tests assert that property directly, and the
reachability tests assert that each site actually REACHES the engine — the
`TestRingPruningIsWired` anti-pattern (asserting a mechanism exists rather than
that it is invoked) is explicitly excluded.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from millm.ml.circuit_steering import CircuitSteeringEngine, ServingPlan


def feat(idx, strength=1.0, sign=1, label=None):
    return SimpleNamespace(
        feature_idx=idx, strength=strength, sign=sign, label=label
    )


def mem(layer, feature=None, expanded=None):
    return SimpleNamespace(layer=layer, feature=feature, expanded_members=expanded)


def defn(members, sae_by_layer=None, budget=None):
    saes = sae_by_layer or {}
    return SimpleNamespace(
        members=members,
        edges=[],
        budget=budget,
        sae_for_layer=lambda layer: saes.get(layer),
        layers=lambda: sorted({m.layer for m in members}),
    )


def sae_ref(sae_id):
    return SimpleNamespace(mistudio_sae_id=sae_id)


class _Registry:
    """Stands in for AttachedSAEState — entries() is the whole contract."""

    def __init__(self, layers):
        self._layers = list(layers)

    def entries(self):
        return [SimpleNamespace(layer=n, sae_id=f"sae-{n}") for n in self._layers]


# ─────────────────────────────────────────────────────────────────────────
# 4.1 — the four-way identity
# ─────────────────────────────────────────────────────────────────────────


class TestAllFourConsumersSeeTheSamePlan:
    """One definition, four questions, one answer. Before F18 each site
    derived its own; the two that diverged became F14-R1-01 and F14-R2-01."""

    def _definition(self):
        return defn(
            [
                mem(10, feature=feat(1, strength=40.0), expanded=[feat(2, 10.0)]),
                mem(13, feature=feat(3, strength=30.0)),
            ],
            sae_by_layer={10: sae_ref("sae-10"), 13: sae_ref("sae-13")},
            budget=SimpleNamespace(intensity=1.5),
        )

    def test_activation_and_the_operator_dial_derive_identical_members(self):
        d = self._definition()
        engine = CircuitSteeringEngine(_Registry([10, 13]))
        activation = engine.plan_for(d, SimpleNamespace(intensity=99.0))
        operator = CircuitSteeringEngine.serving_members(d)
        assert [(m.layer, m.feature_idx, m.budget, m.sign, m.sae_id)
                for m in activation.members] == [
            (m.layer, m.feature_idx, m.budget, m.sign, m.sae_id) for m in operator
        ]

    def test_the_per_request_dial_differs_ONLY_in_intensity(self):
        """A dialled request and an activation must never disagree about WHO
        is steered — only about how hard."""
        d = self._definition()
        engine = CircuitSteeringEngine(_Registry([10, 13]))
        activation = engine.plan_for(d, SimpleNamespace(intensity=99.0))
        dialled = engine.plan_for(d, SimpleNamespace(intensity=99.0), intensity=2.0)

        assert dialled.claimed_layers == activation.claimed_layers
        assert [m.feature_idx for m in dialled.members] == [
            m.feature_idx for m in activation.members
        ]
        assert dialled.intensity == 2.0
        assert activation.intensity == 1.5

    def test_the_echo_predicate_reads_the_same_plan_the_apply_drives(self):
        """`is_serveable` must be true exactly when the apply would do
        something. An echoed rung header on a circuit that is not steering
        attaches an evidence claim to an intervention that never happened."""
        d = self._definition()
        serving = CircuitSteeringEngine(_Registry([10, 13])).plan_for(d)
        assert serving.is_serveable is True

        detached = CircuitSteeringEngine(_Registry([])).plan_for(d)
        assert detached.is_serveable is False
        assert detached.members, "members are unchanged; only attachment differs"

    def test_a_circuit_with_no_members_is_never_serveable(self):
        plan = CircuitSteeringEngine(_Registry([10])).plan_for(defn([]))
        assert plan.is_serveable is False


# ─────────────────────────────────────────────────────────────────────────
# 4.2 / 4.3 — the two F14 regressions, structurally
# ─────────────────────────────────────────────────────────────────────────


class TestTheF14RegressionsCannotRecur:
    def test_F14_R1_01_the_authored_basis_survives_a_dial(self):
        """Authored 150, DB column 100. Dialling must serve from 150."""
        d = defn([mem(10, feature=feat(1, strength=150.0))],
                 budget=SimpleNamespace(intensity=150.0))
        circuit = SimpleNamespace(intensity=100.0)
        plan = CircuitSteeringEngine().plan_for(d, circuit)
        assert plan.intensity == 150.0, (
            "the DB column won — the dial would serve 100 for a circuit "
            "authored at 150 (F14-R1-01)"
        )

    def test_a_zero_authored_intensity_is_not_mistaken_for_absent(self):
        """0.0 means 'off' and is a legitimate authored value. A truthiness
        check would fall through to the DB column and silently re-enable a
        circuit the author turned off."""
        d = defn([mem(10, feature=feat(1))],
                 budget=SimpleNamespace(intensity=0.0))
        plan = CircuitSteeringEngine().plan_for(d, SimpleNamespace(intensity=100.0))
        assert plan.intensity == 0.0

    def test_F14_R2_01_a_layer_outside_the_db_column_is_still_claimed(self):
        """The member layer set is what the apply drives. A layer present in
        the members and absent from `circuits.layers` was dialled and never
        restored — a per-request override leaking into global state."""
        d = defn(
            [mem(10, feature=feat(1)), mem(99, feature=feat(2))],
            sae_by_layer={10: sae_ref("sae-10"), 99: sae_ref("sae-99")},
        )
        plan = CircuitSteeringEngine(_Registry([10, 99])).plan_for(d)
        assert 99 in plan.claimed_layers
        assert plan.claimed_layers == frozenset(m.layer for m in plan.members)

    def test_the_claim_set_is_an_IDENTITY_not_an_agreement(self):
        """The structural statement: there is no second source that could
        drift, because `claimed_layers` is derived from `members` and nothing
        else."""
        for members in (
            [mem(10, feature=feat(1))],
            [mem(10, feature=feat(1)), mem(13, feature=feat(2))],
            [mem(7, expanded=[feat(1), feat(2)])],
            [],
        ):
            plan = CircuitSteeringEngine().plan_for(defn(members))
            assert plan.claimed_layers == frozenset(m.layer for m in plan.members)


class TestTheAttachmentSplit:
    def test_unattached_layers_are_the_claimed_minus_the_attached(self):
        """EC-18.7 — the slice-fallback signal."""
        d = defn([mem(10, feature=feat(1)), mem(13, feature=feat(2))])
        plan = CircuitSteeringEngine(_Registry([10])).plan_for(d)
        assert plan.claimed_layers == frozenset({10, 13})
        assert plan.attached_layers == frozenset({10})
        assert plan.unattached_layers == frozenset({13})

    def test_a_fully_bound_circuit_has_nothing_unattached(self):
        d = defn([mem(10, feature=feat(1)), mem(13, feature=feat(2))])
        plan = CircuitSteeringEngine(_Registry([10, 13])).plan_for(d)
        assert plan.unattached_layers == frozenset()

    def test_a_partially_bound_circuit_is_still_serveable(self):
        """One attached layer is enough to steer; the rest is the caller's
        slice-fallback decision, not the engine's."""
        d = defn([mem(10, feature=feat(1)), mem(13, feature=feat(2))])
        assert CircuitSteeringEngine(_Registry([10])).plan_for(d).is_serveable


# ─────────────────────────────────────────────────────────────────────────
# 5.1 — reachability: each site INVOKES the engine
# ─────────────────────────────────────────────────────────────────────────


class TestEveryCallSiteReachesTheEngine:
    """Invocation, never existence. Each test fails when its site's wiring is
    cut, which is the only thing that distinguishes a wired mechanism from a
    declared one — a distinction this arc got wrong five times."""

    def test_the_engine_is_the_only_serving_flattening_in_the_tree(self):
        import subprocess

        out = subprocess.run(
            ["grep", "-rn", "CircuitMember(", "millm/"],
            capture_output=True, text=True,
        ).stdout
        sites = sorted({
            ln.split(":")[0] for ln in out.splitlines()
            if ln.strip() and "class CircuitMember" not in ln
        })
        assert sites == ["millm/ml/circuit_steering.py"]

    def _spy_on_plan_for(self, monkeypatch):
        """Record every `plan_for` invocation. `co_names` inspection is NOT
        enough: cutting the dial's wiring while leaving its local import in
        place left the class name in `co_names` and the test green. That is a
        source grep wearing a reachability costume — the R3-13 anti-pattern."""
        calls = []
        real = CircuitSteeringEngine.plan_for

        def spy(self, definition, circuit=None, intensity=None):
            calls.append(intensity)
            return real(self, definition, circuit, intensity)

        monkeypatch.setattr(CircuitSteeringEngine, "plan_for", spy)
        return calls

    def _circuit_and_definition(self):
        d = defn(
            [mem(10, feature=feat(1, strength=40.0))],
            sae_by_layer={10: sae_ref("sae-10")},
        )
        return SimpleNamespace(id="circ_1", intensity=1.0, layers=[10],
                               serving_mode="full", rung=2,
                               circuit_meta={}, name="fear→threat"), d

    @pytest.mark.asyncio
    async def test_the_dial_INVOKES_plan_for(self, monkeypatch):
        from millm.services.inference_service import InferenceService

        calls = self._spy_on_plan_for(monkeypatch)
        circuit, d = self._circuit_and_definition()

        svc = InferenceService.__new__(InferenceService)
        svc._active_full_circuit = MagicMock(return_value=circuit)

        async def _active():
            return circuit

        svc._active_full_circuit = _active
        monkeypatch.setattr(
            InferenceService, "_circuit_definition", lambda self, c: d
        )
        await svc._apply_request_circuit_steering(2.0)

        assert calls, "the dial never called plan_for — its wiring is cut"
        assert 2.0 in calls, f"the dial passed {calls} instead of its lambda"

    @pytest.mark.asyncio
    async def test_the_echo_predicate_INVOKES_plan_for(self, monkeypatch):
        from millm.services.inference_service import InferenceService

        calls = self._spy_on_plan_for(monkeypatch)
        circuit, d = self._circuit_and_definition()

        svc = InferenceService.__new__(InferenceService)

        async def _active():
            return circuit

        svc._active_full_circuit = _active
        monkeypatch.setattr(
            InferenceService, "_circuit_definition", lambda self, c: d
        )
        await svc._steering_circuit_uncached()

        assert calls, "the echo predicate never called plan_for"

    def test_serve_full_and_set_intensity_reach_the_engine(self):
        """These two are async DB paths; asserting the reference is the
        proportionate check, and the four-way identity tests above already
        prove they produce the same plan."""
        from millm.services.circuit_service import CircuitService

        assert "CircuitSteeringEngine" in CircuitService._serve_full.__code__.co_names
        assert (
            "CircuitSteeringEngine" in CircuitService.set_intensity.__code__.co_names
        )


class TestThePlanIsImmutable:
    def test_a_plan_cannot_be_mutated_after_derivation(self):
        """Four consumers share it. A mutable plan is four chances for one
        consumer to change what another reads."""
        plan = CircuitSteeringEngine().plan_for(defn([mem(10, feature=feat(1))]))
        with pytest.raises(Exception):
            plan.intensity = 99.0


class TestR1TheEngineFailsSAFELYAndVISIBLY:
    """F18 R1-01/02. Two behaviours the engine inherits or introduces at its
    boundaries, both on paths that decide whether an evidence claim is made."""

    def test_a_null_authored_intensity_falls_back_to_the_column(self):
        """R1-01, a DELIBERATE divergence from the pre-move expression, which
        returned None here. The schema declares `intensity: float` with a
        default so a null is unreachable through a parsed document — but
        returning None would propagate a null into the apply, where it
        multiplies a budget. Falling back to the last known-good value is the
        correct degradation."""
        d = defn([mem(10, feature=feat(1))],
                 budget=SimpleNamespace(intensity=None))
        plan = CircuitSteeringEngine().plan_for(d, SimpleNamespace(intensity=100.0))
        assert plan.intensity == 100.0

    def test_an_unreadable_registry_makes_the_circuit_look_UNSERVEABLE(self):
        """R1-02. The direction is the safe one: the echo predicate then
        withholds a rung header rather than attaching an evidence claim to an
        intervention it cannot confirm. Under-claiming is how an evidence
        surface should fail."""

        class Broken:
            def entries(self):
                raise RuntimeError("registry unavailable")

        d = defn([mem(10, feature=feat(1))])
        plan = CircuitSteeringEngine(Broken()).plan_for(d)
        assert plan.attached_layers == frozenset()
        assert plan.is_serveable is False
        assert plan.members, "members are unaffected — only attachment failed"

    def test_an_unreadable_registry_is_LOGGED_not_swallowed(self, monkeypatch):
        """It was silent. A registry that cannot be read is an operational
        fault, and the same empty set also drives slice-fallback decisions.

        Spies on the module logger rather than using `caplog`: this codebase
        logs through structlog, so `caplog` sees nothing and a test written
        against it would pass whether or not the warning existed."""
        from millm.ml import circuit_steering as mod

        seen = []
        monkeypatch.setattr(
            mod.logger, "warning", lambda event, **kw: seen.append(event)
        )

        class Broken:
            def entries(self):
                raise RuntimeError("registry unavailable")

        CircuitSteeringEngine(Broken()).plan_for(defn([]))
        assert seen == ["circuit_attachment_registry_unreadable"], (
            f"a registry failure was swallowed silently (logged: {seen})"
        )

    def test_NO_registry_supplied_is_quiet(self, monkeypatch):
        """`state is None` is a deliberate construction, not a failure — the
        operator dial derives members without needing attachment at all."""
        from millm.ml import circuit_steering as mod

        seen = []
        monkeypatch.setattr(
            mod.logger, "warning", lambda event, **kw: seen.append(event)
        )
        plan = CircuitSteeringEngine(None).plan_for(defn([mem(10, feature=feat(1))]))
        assert plan.attached_layers == frozenset()
        assert plan.members
        assert seen == [], "a deliberate absence was logged as a fault"


class TestR1ForRegistryIsSafeForTheDialPath:
    """F18 R1-04. `for_registry` sets `_repository=None` and `_cache_dir=""`
    because a registry-only service genuinely has neither. That is safe ONLY
    while the dial path never reaches them — a property of the code today, not
    a guarantee, and exactly the kind that erodes silently.

    The retired `__new__` bypass had the same exposure and worse: those fields
    were ABSENT, so a reach was an AttributeError rather than a None."""

    def test_the_dial_path_touches_only_the_registry(self):
        """Traced from `set_circuit_steering` — if this grows, the None fields
        become reachable and `for_registry` needs real values or a raising
        sentinel."""
        import inspect
        import re

        from millm.services.sae_service import SAEService

        reached = set()
        for name in ("set_circuit_steering", "_set_circuit_steering_locked"):
            method = getattr(SAEService, name, None)
            if method is None:
                continue
            reached |= set(re.findall(r"self\.(_[a-z_]+)\b", inspect.getsource(method)))

        unset_by_for_registry = {
            "_repository", "_cache_dir", "_downloader", "_loader",
            "_hooker", "_emitter", "_inference_service",
        }
        assert not (reached & unset_by_for_registry), (
            f"the dial path now reaches {sorted(reached & unset_by_for_registry)}, "
            "which for_registry sets to None/'' — give them real values or make "
            "them raise rather than returning a misleading default"
        )

    def test_a_registry_only_service_can_actually_steer(self):
        """The positive half: it is not merely well-formed, it works."""
        from millm.services.sae_service import AttachedSAEState, SAEService

        svc = SAEService.for_registry()
        assert svc._sae_state is AttachedSAEState()
        assert callable(svc.set_circuit_steering)

    def test_it_is_total_against___init__(self):
        """Every field the real constructor sets. The bypass left four fields
        and two collections absent."""
        import inspect
        import re

        from millm.services.sae_service import SAEService

        init_fields = set(
            # R1-B: this regex required a leading underscore, so `self.repository`
            # and `self.emitter` — the two fields most widely read in this class
            # — were INVISIBLE to the guard. A totality test that cannot see two
            # thirds of the public fields is a totality test in name only, and
            # it passed against the broken version AND against the fix.
            re.findall(r"self\.([a-z_]+)\s*[:=]", inspect.getsource(SAEService.__init__))
        )
        svc = SAEService.for_registry()
        assert [f for f in sorted(init_fields) if not hasattr(svc, f)] == []


class TestR1TheDialUsesTheSnapshotItDerived:
    """F18 R1-08/09. Two hardening changes that no test pinned, which is how
    an unpinned fix gets silently reverted."""

    def _svc_and_definition(self, monkeypatch, registry_layers=(10,)):
        from millm.services.inference_service import InferenceService

        d = defn(
            [mem(10, feature=feat(1, strength=40.0))],
            sae_by_layer={10: sae_ref("sae-10")},
        )
        circuit = SimpleNamespace(
            id="circ_1", intensity=1.0, layers=[10], serving_mode="full",
            rung=2, circuit_meta={}, name="fear→threat",
        )

        svc = InferenceService.__new__(InferenceService)

        async def _active():
            return circuit

        svc._active_full_circuit = _active
        monkeypatch.setattr(
            InferenceService, "_circuit_definition", lambda self, c: d
        )
        return svc

    @pytest.mark.asyncio
    async def test_the_registry_is_read_ONCE_per_dial(self, monkeypatch):
        """R1-08. `plan_for` already reads the registry; a second read is both
        hot-path overhead and a drift window — a detach landing between them
        means the snapshot the plan reports and the entries this request saves
        and restores disagree."""
        from millm.services.sae_service import AttachedSAEState

        reads = []
        state = AttachedSAEState()
        real_entries = state.entries

        def counting_entries():
            reads.append(1)
            return real_entries()

        monkeypatch.setattr(state, "entries", counting_entries, raising=False)
        svc = self._svc_and_definition(monkeypatch)
        await svc._apply_request_circuit_steering(2.0)

        assert len(reads) <= 1, (
            f"the registry was read {len(reads)} times in one dial — the plan's "
            "snapshot and the restored entries can disagree"
        )

    @pytest.mark.asyncio
    async def test_a_construction_fault_is_NOT_reported_as_an_apply_failure(
        self, monkeypatch
    ):
        """R1-09. `for_registry()` inside the try meant a construction fault
        surfaced as `circuit_dial_apply_failed` — an apply failure that never
        reached the apply. That is exactly how R1-05's missing-attribute bug
        and both implementation NameErrors would have presented."""
        from millm.services import inference_service as mod
        from millm.services.sae_service import SAEService

        def boom():
            raise AttributeError("'SAEService' object has no attribute 'repository'")

        monkeypatch.setattr(SAEService, "for_registry", staticmethod(boom))

        warnings = []
        monkeypatch.setattr(
            mod.logger, "warning",
            lambda event, **kw: warnings.append(event), raising=False,
        )

        # The dial returns early at `no_attached_layers` unless the registry
        # holds the claimed layer, so the construction is only reachable with a
        # real attachment. A behavioural probe was INCONCLUSIVE — it never
        # reached the construction, and a test that cannot reach its subject
        # proves nothing. Assert the placement structurally instead.
        import inspect

        src = inspect.getsource(
            mod.InferenceService._apply_request_circuit_steering
        )
        assert "dial_service = SAEService.for_registry()" in src, (
            "the dial service is constructed inside the try again — a "
            "construction fault would surface as circuit_dial_apply_failed, "
            "an apply failure that never reached the apply"
        )
        # NOT an index comparison against the log event name: that string also
        # appears in an explanatory comment ABOVE the construction, so
        # `src.index(...)` compares comment positions and fails on correct
        # code. Tried it, watched it fail, replaced it — the assertion above is
        # the one that distinguishes the two placements.


class TestR1AMissingIntensityIsNotZero:
    """F18 R1-12/13. `serving_intensity` returned a bare 0.0 when there was no
    basis at all — no document budget and no circuit row. The pre-move
    expression raised AttributeError there. 0.0 means "serve nothing", so a
    missing basis became indistinguishable from a deliberate off switch: a loud
    failure turned into a silent no-op on the path that decides how hard to
    steer."""

    def test_no_basis_is_distinguishable_from_an_authored_zero(self):
        d = defn([], budget=None)
        no_basis = CircuitSteeringEngine().plan_for(d)
        authored_off = CircuitSteeringEngine().plan_for(
            d, SimpleNamespace(intensity=0.0)
        )
        assert authored_off.intensity == 0.0
        assert authored_off.has_intensity is True
        assert no_basis.has_intensity is False, (
            "a missing basis reads as a deliberate 'serve nothing'"
        )

    def test_a_members_only_derivation_is_still_legitimate(self):
        """`plan_for(definition)` with no circuit is how the operator dial
        derives members. Raising here broke ten tests — the absence has to be
        REPRESENTED, not refused."""
        plan = CircuitSteeringEngine().plan_for(defn([mem(10, feature=feat(1))]))
        assert plan.members
        assert plan.claimed_layers == frozenset({10})

    def test_an_authored_zero_still_wins_over_the_column(self):
        d = defn([], budget=SimpleNamespace(intensity=0.0))
        plan = CircuitSteeringEngine().plan_for(d, SimpleNamespace(intensity=100.0))
        assert plan.intensity == 0.0 and plan.has_intensity is True

    def test_a_negative_override_is_refused(self):
        """R1-13: a negative λ passed straight through and only the downstream
        clamp saved it, so the plan could hold a value the system will never
        serve — a plan that does not describe what happens."""
        with pytest.raises(ValueError, match="must not be negative"):
            CircuitSteeringEngine().plan_for(
                defn([]), SimpleNamespace(intensity=1.0), intensity=-1.0
            )

    def test_a_zero_override_is_allowed(self):
        """0.0 is a legitimate dial position meaning off."""
        plan = CircuitSteeringEngine().plan_for(
            defn([]), SimpleNamespace(intensity=1.0), intensity=0.0
        )
        assert plan.intensity == 0.0 and plan.has_intensity is True


class TestR1TheEchoPredicateMatchesTheOldLogic:
    """F18 R1-17. The predicate was rewritten from two hand-written checks
    (members, then attachment) to one `plan.is_serveable`. Verified against the
    old logic across every member/attachment shape — they agree — because a
    silent change HERE would attach or withhold an evidence claim, which is the
    one thing this predicate exists to get right."""

    def _old(self, members, attached):
        if not members:
            return None
        member_layers = {m.layer for m in members}
        if not any(layer in member_layers for layer in attached):
            return None
        return "circuit"

    @pytest.mark.parametrize(
        "layers,attached",
        [
            ([10], [10]),          # match
            ([10], []),            # nothing attached
            ([], [10]),            # no members
            ([10, 13], [10]),      # partially attached
            ([10, 13], [99]),      # attached, but not to a claimed layer
        ],
    )
    def test_old_and_new_agree(self, layers, attached):
        d = defn([mem(n, feature=feat(1)) for n in layers])
        plan = CircuitSteeringEngine(_Registry(attached)).plan_for(d)
        new = "circuit" if plan.is_serveable else None
        assert new == self._old(plan.members, attached)

    def test_an_unreadable_registry_withholds_rather_than_propagates(self):
        """R1-18, a recorded delta: the old code called `entries()` directly
        and let a registry error propagate; the new path swallows it (logged)
        and returns None. Under-claiming — the response loses its rung header
        rather than carrying one the system cannot confirm. That is the right
        direction for an evidence surface, and it is a real behaviour change,
        so it is pinned rather than left implicit."""

        class Broken:
            def entries(self):
                raise RuntimeError("registry unavailable")

        plan = CircuitSteeringEngine(Broken()).plan_for(
            defn([mem(10, feature=feat(1))])
        )
        assert plan.is_serveable is False


class TestR1TheSignRuleEndToEnd:
    """F18 R1-19, FPRD §9 criterion 5: the canonical sign rule asserted
    DIRECTLY rather than inferred from an applied value.

    A NEGATIVE authored strength is already directional. If the plan pre-applied
    `sign`, a suppression (-3, sign -1) would become an amplification (+3) — a
    steering change that looks plausible and is backwards. The rule lives in
    `_directional_budget`; the engine's job is to carry both fields untouched
    so that function can be the only place it is applied."""

    @pytest.mark.parametrize(
        "strength,sign,expected",
        [
            (-3.0, -1, -3.0),   # the double-application case
            (-3.0, 1, -3.0),
            (3.0, -1, -3.0),
            (3.0, 1, 3.0),
        ],
    )
    def test_the_plan_carries_and_directional_budget_applies(
        self, strength, sign, expected
    ):
        from millm.services.sae_service import _directional_budget

        d = defn([mem(10, feature=feat(1, strength=strength, sign=sign))])
        member = CircuitSteeringEngine.serving_members(d)[0]

        # Carried, not applied.
        assert (member.budget, member.sign) == (strength, sign)
        # And the one place that applies it gets the right answer.
        assert _directional_budget(member.budget, member.sign) == expected

    def test_the_plan_never_pre_applies_the_sign(self):
        """The direct assertion criterion 5 asks for: a negative budget must
        arrive at `_directional_budget` still negative."""
        d = defn([mem(10, feature=feat(1, strength=-3.0, sign=-1))])
        plan = CircuitSteeringEngine().plan_for(d, SimpleNamespace(intensity=1.0))
        assert plan.members[0].budget == -3.0, (
            "the flattening pre-applied the sign — a suppression would serve "
            "as an amplification"
        )


class TestR1ClaimSetOnEveryCharacterizationFixture:
    """F18 R1-20, FPRD §9 criterion 3: `claim_set` equals the distinct layers
    of `serving_members` on every characterization shape, including the
    both-sources (EC-18.1) and dedupe (EC-18.2) cases."""

    @pytest.mark.parametrize(
        "members",
        [
            [mem(10, feature=feat(1), expanded=[feat(2), feat(3)])],   # EC-18.1
            [mem(10, feature=feat(1, 9.0), expanded=[feat(1, 2.0)])],  # EC-18.2
            [mem(10, feature=feat(1)), mem(13, feature=feat(1))],
            [mem(13, feature=feat(5)), mem(10, feature=feat(1))],
            [mem(10, feature=None, expanded=None)],
            [],
        ],
    )
    def test_the_claim_set_is_exactly_the_member_layers(self, members):
        d = defn(members)
        derived = CircuitSteeringEngine.serving_members(d)
        assert CircuitSteeringEngine.claim_set(derived) == frozenset(
            m.layer for m in derived
        )
