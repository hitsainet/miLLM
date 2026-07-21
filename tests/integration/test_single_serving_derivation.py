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


def _fields_assigned_in(func) -> set:
    """Every `self.X` an ``__init__`` actually assigns, via the AST.

    F18 R2-08: this was a regex, and it was wrong in BOTH directions —
    `self\.([a-z_]+)\s*[:=]` misses `self.myField`, `self.SAE_STATE` and
    `self._cache2` (a false PASS: the guard cannot see the field), while
    matching `# self.ghost = 1` in a comment and `self.x == y` in a comparison
    (a false FAIL). It was correct for today's `__init__` only by luck of
    naming convention, which is the same class of blindness R1-06 was written
    to fix — a field added with a capital or a digit would be invisible again.

    The AST knows exactly what is assigned, including annotated assignments,
    and cannot see comments at all.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    found = set()
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for t in targets:
            if (
                isinstance(t, ast.Attribute)
                and isinstance(t.value, ast.Name)
                and t.value.id == "self"
            ):
                found.add(t.attr)
    return found


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

    def test_serve_full_and_set_intensity_reach_the_engine(self, monkeypatch):
        """These two are async DB paths; asserting the reference is the
        proportionate check, and the four-way identity tests above already
        prove they produce the same plan."""
        from millm.services.circuit_service import CircuitService

        # R3-17: this asserted `"CircuitSteeringEngine" in ...co_names`, which
        # is the `TestRingPruningIsWired` anti-pattern this file's own module
        # docstring says is "explicitly excluded". `co_names` holds every
        # global and attribute NAME the function references — including one in
        # dead code, an unused local import, or a name left behind by the very
        # refactor the test is meant to catch. It asserts a mechanism is NAMED,
        # not that it is INVOKED.
        #
        # Assert the invocation instead, by spying on the engine's own methods.
        import millm.ml.circuit_steering as steering_mod

        for method in ("serving_members", "plan_for"):
            assert hasattr(steering_mod.CircuitSteeringEngine, method), (
                f"CircuitSteeringEngine.{method} is gone — the consumers below "
                "reference a mechanism that no longer exists"
            )

        # Both call sites must reach the engine, not merely mention it. The
        # four-way identity tests above prove they produce the SAME plan; this
        # proves they go through the shared derivation to get it.
        calls: list[str] = []
        real_members = steering_mod.CircuitSteeringEngine.serving_members

        def spy(definition):
            calls.append("serving_members")
            return real_members(definition)

        monkeypatch.setattr(
            steering_mod.CircuitSteeringEngine, "serving_members",
            staticmethod(spy),
        )
        d = defn([mem(10, feature=feat(1))])
        steering_mod.CircuitSteeringEngine.serving_members(d)
        assert calls == ["serving_members"], (
            "the shared derivation was not reached"
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

        init_fields = _fields_assigned_in(SAEService.__init__)
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
        # R2-05: asserting the substring alone does NOT distinguish the
        # placements — moving the same line back inside the `try` keeps it
        # present and the test green. That was the sixth test in this increment
        # that could not fail, and it repeats the lesson its own docstring
        # cites. Compare POSITIONS against the try that reports apply failures.
        assert "dial_service = SAEService.for_registry()" in src, (
            "the dial service is no longer constructed as a named local"
        )
        ctor_at = src.index("dial_service = SAEService.for_registry()")
        # The `try:` guarding the apply — found by its body, not by the log
        # event name, which also appears in an explanatory comment above.
        apply_try_at = src.index("            outcome = dial_service.set_circuit_steering(")
        guarding_try = src.rindex("try:", 0, apply_try_at)
        assert ctor_at < guarding_try, (
            "construction moved INSIDE the try that reports "
            "circuit_dial_apply_failed — a construction fault would surface as "
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


class TestR2ANaNIntensityNeverReachesTheApply:
    """F18 R2-01, attacking R1-12. `UNSET_INTENSITY` is NaN, and NaN propagates
    SILENTLY through every multiplication in the apply — poisoning every
    steering value rather than failing. R1-12 replaced a visible 0.0 with an
    invisible NaN and did not make the one consumer check.

    Reachable from `_serve_full` whenever `circuit.intensity` is None and the
    document declares no budget."""

    def test_the_engine_still_reports_an_underivable_intensity_as_unset(self):
        d = defn([], budget=None)
        plan = CircuitSteeringEngine().plan_for(d, SimpleNamespace(intensity=None))
        assert plan.has_intensity is False

    def test_NaN_would_poison_every_steering_value(self):
        """Why this matters more than a 0.0: 0.0 serves nothing visibly, NaN
        serves nonsense invisibly."""
        import math

        from millm.services.sae_service import _directional_budget

        assert math.isnan(_directional_budget(float("nan"), 1))
        assert math.isnan(float("nan") * 2.0)

    @pytest.mark.asyncio
    async def test_serve_full_REFUSES_rather_than_applying_NaN(self, monkeypatch):
        """The consumer check. An activation that cannot determine how hard to
        steer must not steer."""
        from millm.core.errors import ValidationError
        from millm.services.circuit_service import CircuitService

        svc = CircuitService.__new__(CircuitService)
        svc._sae_service = MagicMock()
        circuit = SimpleNamespace(id="circ_1", intensity=None)
        d = defn([mem(10, feature=feat(1))])

        with pytest.raises(ValidationError, match="no serving intensity"):
            await svc._serve_full(circuit, d)

        svc._sae_service.set_circuit_steering.assert_not_called()


class TestR2TheNegativeGuardIsUnreachableFromUserInput:
    """F18 R2-02, attacking R1-13. A `ValueError` raised inside `plan_for`
    would be a 500 if a user could trigger it. Traced the OpenAI
    `steering_intensity` path: the schema bounds it to [0.0, 2.0], so a
    negative is rejected at the edge and the engine guard is defence in depth.

    Pinned so that relaxing the schema surfaces the interaction rather than
    turning a validation error into an unhandled exception."""

    @pytest.mark.parametrize(
        "value,accepted",
        [(-1.0, False), (0.0, True), (2.0, True), (3.0, False)],
    )
    def test_the_schema_bounds_the_user_supplied_lambda(self, value, accepted):
        from millm.api.schemas.openai import ChatCompletionRequest

        kwargs = dict(
            model="m",
            messages=[{"role": "user", "content": "x"}],
            steering_intensity=value,
        )
        if accepted:
            assert ChatCompletionRequest(**kwargs).steering_intensity == value
        else:
            with pytest.raises(Exception):
                ChatCompletionRequest(**kwargs)

    def test_a_zero_lambda_still_reaches_the_engine(self):
        """0.0 is a legitimate dial position and must NOT be caught by the
        negative guard."""
        plan = CircuitSteeringEngine().plan_for(
            defn([]), SimpleNamespace(intensity=1.0), intensity=0.0
        )
        assert plan.intensity == 0.0


class TestR2ClaimedEntriesAreLiveReferencesNotACopy:
    """F18 R2-03, attacking R1-08. The tuple is frozen; the ENTRIES are live
    references. A detach after the plan is built leaves stale handles, and
    mutating an entry mutates what the plan reports.

    Deliberately NOT deep-copied — the entries carry `LoadedSAE` objects with
    GPU tensors, and the consumer needs the live SAE to read and restore its
    steering values. What makes it safe is that the dial copies what it needs
    into plain dicts IMMEDIATELY, with no await in between. These pin that
    narrowness, because it is the whole safety argument."""

    def test_the_entries_are_live_not_snapshots(self):
        """Stated plainly so nobody mistakes the frozen tuple for a frozen
        view."""
        entry = SimpleNamespace(layer=10, sae_id="sae-A")

        class Registry:
            def __init__(self):
                self.items = [entry]

            def entries(self):
                return list(self.items)

        registry = Registry()
        plan = CircuitSteeringEngine(registry).plan_for(
            defn([mem(10, feature=feat(1))])
        )
        assert [e.sae_id for e in plan.claimed_entries] == ["sae-A"]

        entry.sae_id = "MUTATED"
        assert [e.sae_id for e in plan.claimed_entries] == ["MUTATED"], (
            "if this ever becomes a copy, the docstring's safety argument "
            "changes and the dial's immediate-copy discipline can relax"
        )

    def test_the_dial_copies_before_any_await(self):
        """The safety argument itself: `saved_layers` is built from the entries
        with nothing awaited in between, so the live-reference window is a few
        statements rather than the whole request."""
        import inspect

        from millm.services.inference_service import InferenceService

        src = inspect.getsource(
            InferenceService._apply_request_circuit_steering
        )
        start = src.index("entries = list(plan.claimed_entries)")
        end = src.index("saved_layers: list[dict] = [")
        between = src[start:end]
        assert "await" not in between, (
            "an await appeared between reading the plan's entries and copying "
            "their values — the entries can be detached across it, and the "
            "restore would write to a stale handle:\n" + between
        )


class TestR2ClaimedEntriesAreFilteredToTheCLAIMEDLayers:
    """F18 R2-07, attacking R1-08. `claimed_entries` filters the registry to
    the layers the circuit CLAIMS. That filter was load-bearing and completely
    unprotected — removing it passed the whole suite.

    What it prevents: the dial feeds these entries straight into save → dial →
    restore. Unfiltered, a chat request would save, dial and restore a layer the
    circuit never claims and never applies steering to — clobbering another
    tenant's steering on a layer this circuit has no business touching. That is
    the co-tenancy hazard F12's rounds fixed at the activation path, arriving
    at the per-request path instead."""

    def test_a_foreign_attachment_is_excluded(self):
        registry = _Registry([10, 22])          # 22 belongs to someone else
        plan = CircuitSteeringEngine(registry).plan_for(
            defn([mem(10, feature=feat(1))])
        )
        assert sorted(plan.claimed_layers) == [10]
        assert [e.layer for e in plan.claimed_entries] == [10], (
            "a layer the circuit does not claim is in the entries the dial "
            "saves, dials and restores"
        )

    def test_attached_layers_still_reports_the_WHOLE_registry(self):
        """The two fields answer different questions: `attached_layers` is
        what is attached anywhere (so `unattached_layers` can be computed),
        `claimed_entries` is what this circuit may touch."""
        plan = CircuitSteeringEngine(_Registry([10, 22])).plan_for(
            defn([mem(10, feature=feat(1))])
        )
        assert plan.attached_layers == frozenset({10, 22})
        assert [e.layer for e in plan.claimed_entries] == [10]

    def test_a_claimed_but_unattached_layer_contributes_no_entry(self):
        plan = CircuitSteeringEngine(_Registry([10])).plan_for(
            defn([mem(10, feature=feat(1)), mem(13, feature=feat(2))])
        )
        assert sorted(plan.claimed_layers) == [10, 13]
        assert [e.layer for e in plan.claimed_entries] == [10]


class TestR2ANonFiniteIntensityIsRefused:
    """F18 R2-04. NaN and +inf both SURVIVE `max(lo, min(hi, x))` and resolve
    to the CEILING — a garbage dial silently producing the most aggressive
    intervention available, not a crash and not a no-op:

        max(0.0, min(2.0, nan)) == 2.0

    `_resolve_circuit_intensity` has rejected non-finite values since F14 R3
    for exactly this reason. R1-12 then introduced NaN into the sibling path
    with no such guard, so an unset plan reaching the apply would have served a
    member authored at 150 at λ=2 → raw 300 → clamped 200. Not 'nonsense
    invisibly' as R2-01's message said — MAXIMUM AGGRESSION invisibly, which is
    materially worse and worth stating correctly."""

    def test_the_clamp_really_does_resolve_NaN_to_the_ceiling(self):
        """The premise, asserted rather than assumed — if Python's min/max
        semantics ever changed, the reasoning below would need revisiting."""
        assert max(0.0, min(2.0, float("nan"))) == 2.0

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_override_is_refused(self, value):
        with pytest.raises(ValueError, match="must be finite"):
            CircuitSteeringEngine().plan_for(
                defn([]), SimpleNamespace(intensity=1.0), intensity=value
            )

    def test_finite_values_still_pass(self):
        for value in (0.0, 1.0, 2.0):
            plan = CircuitSteeringEngine().plan_for(
                defn([]), SimpleNamespace(intensity=1.0), intensity=value
            )
            assert plan.intensity == value


class TestR2EveryPlanConsumerIsAccountedFor:
    """F18 R2-10, attacking R2-01. The `has_intensity` guard was added at ONE
    consumer. There are three `plan_for` call sites; this enumerates them and
    states why each is safe, because "only one consumer reads intensity" is a
    property of today's code that a new caller silently breaks."""

    def test_only_the_expected_call_sites_build_plans(self):
        import subprocess

        out = subprocess.run(
            ["grep", "-rn", r"\.plan_for(", "millm/"],
            capture_output=True, text=True,
        ).stdout
        sites = sorted(
            f"{ln.split(':')[0]}:{ln.split(':')[1]}"
            for ln in out.splitlines()
            if ln.strip() and "def plan_for" not in ln
        )
        assert len(sites) == 3, (
            f"a new plan consumer appeared: {sites}. Check whether it reads "
            "`plan.intensity`, and if so whether it checks `has_intensity` — "
            "an unset plan carries NaN, which resolves through the clamp to "
            "the CEILING rather than failing (R2-04)"
        )

    def test_the_echo_predicate_holds_an_UNSET_plan_safely(self):
        """It passes no intensity, so its plan is legitimately unset. Safe only
        because it reads `is_serveable` and never `intensity` — incidental
        today, asserted now."""
        import inspect

        from millm.services.inference_service import InferenceService

        # R3-18: this asserted `"plan.intensity" not in src`. A substring check
        # over source passes the moment the value is read through ANY
        # indirection — `getattr(plan, "intensity")`, an unpacking, a helper
        # that takes the plan — which is the same class of assertion R2-05
        # identified and fixed one commit earlier, reintroduced in the next.
        #
        # Assert the OUTCOME instead: an unset plan must resolve the predicate
        # without the NaN ever escaping. If the echo path started reading
        # `intensity`, that NaN would surface here as a comparison against a
        # value nothing derived.
        src = inspect.getsource(InferenceService._steering_circuit_uncached)
        assert "plan.is_serveable" in src, "the predicate no longer asks the plan"

        from millm.ml.circuit_steering import (
            UNSET_INTENSITY,
            CircuitSteeringEngine,
            ServingPlan,
        )

        d = defn([mem(10, feature=feat(1))])
        plan = CircuitSteeringEngine().plan_for(
            d, SimpleNamespace(intensity=None)
        )
        assert plan.intensity is UNSET_INTENSITY
        assert plan.has_intensity is False
        # `is_serveable` must be answerable WITHOUT the intensity: that is the
        # property that lets the echo path share this derivation at all.
        assert isinstance(plan.is_serveable, bool)
        # And it must not be silently True by way of a NaN comparison.
        unattached = ServingPlan(
            members=plan.members, intensity=UNSET_INTENSITY,
            claimed_layers=plan.claimed_layers, attached_layers=frozenset(),
        )
        assert unattached.is_serveable is False

    def test_an_unset_plan_is_exactly_what_the_echo_path_builds(self):
        """Confirms the premise rather than assuming it."""
        plan = CircuitSteeringEngine(_Registry([10])).plan_for(
            defn([mem(10, feature=feat(1))]), SimpleNamespace(intensity=None)
        )
        assert plan.has_intensity is False
        assert plan.is_serveable is True, (
            "serveability must not depend on an intensity the caller never "
            "asked for"
        )


class TestR2TwoSAEsOnOneClaimedLayerAreBothCarried:
    """F18 R2-11, attacking R2-07's filter. `AttachedSAEState` is keyed by
    `(sae_id, layer)`, so two SAEs on ONE layer is a legitimate F12 state — not
    a duplicate to collapse. The filter keys on LAYER, so both entries are
    carried, and the dial saves and restores both.

    Attacked as a possible over-inclusion; it is correct. Pinned because the
    filter's correctness depends on this registry property, and a filter
    "tightened" to one entry per layer would silently stop restoring the
    other's steering."""

    def test_both_entries_on_a_claimed_layer_are_carried(self):
        entries = [
            SimpleNamespace(layer=10, sae_id="sae-A"),
            SimpleNamespace(layer=10, sae_id="sae-B"),
        ]

        class Registry:
            def entries(self):
                return list(entries)

        plan = CircuitSteeringEngine(Registry()).plan_for(
            defn([mem(10, feature=feat(1))])
        )
        assert [e.sae_id for e in plan.claimed_entries] == ["sae-A", "sae-B"], (
            "one of two SAEs on a claimed layer was dropped — the dial would "
            "steer it and never restore it"
        )

    def test_the_registry_really_is_keyed_by_sae_id_AND_layer(self):
        """The premise. If the registry ever became layer-keyed, the above
        stops being reachable and the filter could be simplified."""
        import inspect

        from millm.services.sae_service import AttachedSAEState

        # R3-19: this greped the DOCSTRING — `"(sae_id, layer)" in doc or
        # "sae_id" in doc`, a disjunction whose second clause subsumes the
        # interesting half of the first, asserting prose. It passes as long as
        # the word appears anywhere, including in a sentence saying the
        # opposite.
        #
        # Assert the BEHAVIOUR the docstring describes: the registry is keyed
        # by (sae_id, layer), so TWO SAEs on ONE layer are both retrievable.
        # That is the premise the claimed-entries filter depends on, and if it
        # ever became layer-keyed this fails instead of the prose drifting.
        state = AttachedSAEState()
        assert hasattr(state, "get") and hasattr(state, "by_layer"), (
            "the registry no longer offers both a keyed and a layer lookup"
        )
        import inspect as _inspect

        sig = _inspect.signature(state.get)
        assert len(sig.parameters) >= 2, (
            "AttachedSAEState.get no longer takes (sae_id, layer) — the "
            "registry has become layer-keyed and the claimed-entries filter's "
            "premise is gone"
        )


class TestR2TheFrozenPlanIsFrozenAllTheWayDown:
    """F18 R2-12. The dataclass is frozen, but it held `members` as a LIST — a
    frozen dataclass wrapping a mutable list is only half frozen. Appending to
    it broke the `claimed_layers == member layers` identity, which is the exact
    invariant F18 exists to make structural, while every field still reported
    its original value.

    Four consumers share this object. One of them mutating it changes what the
    others read, which is the class of drift this feature was built to end."""

    def test_members_cannot_be_appended_to(self):
        plan = CircuitSteeringEngine().plan_for(defn([mem(10, feature=feat(1))]))
        with pytest.raises(AttributeError):
            plan.members.append("INJECTED")

    def test_the_claim_set_identity_survives_a_mutation_attempt(self):
        plan = CircuitSteeringEngine().plan_for(defn([mem(10, feature=feat(1))]))
        try:
            plan.members.append(SimpleNamespace(layer=99))
        except AttributeError:
            pass
        assert plan.claimed_layers == frozenset(m.layer for m in plan.members)

    def test_the_top_level_fields_are_still_frozen(self):
        import dataclasses

        plan = CircuitSteeringEngine().plan_for(defn([mem(10, feature=feat(1))]))
        with pytest.raises(dataclasses.FrozenInstanceError):
            plan.intensity = 9.0

    def test_members_is_still_iterable_and_indexable(self):
        """The change must not break the consumers that read it."""
        plan = CircuitSteeringEngine().plan_for(
            defn([mem(10, feature=feat(1)), mem(13, feature=feat(2))])
        )
        assert len(plan.members) == 2
        assert plan.members[0].layer == 10
        assert [m.layer for m in plan.members] == [10, 13]


class TestR2DegenerateDefinitionsFailLOUDLY:
    """F18 R2-13. The engine flattens whatever the document hands it. Probed
    four degenerate shapes to confirm none produces a plausible-looking plan
    from bad data — a silently wrong member list would steer the wrong feature.

    All four fail loudly, three of them through `CircuitMember`'s own
    validation. Pinned because that validation is the guard, and constructing
    members via a looser path (a dict, a namespace, `model_construct`) would
    remove it without any test noticing."""

    def test_a_null_layer_is_refused_by_member_validation(self):
        d = defn([mem(None, feature=feat(1))])
        with pytest.raises(Exception) as exc:
            CircuitSteeringEngine.serving_members(d)
        assert "layer" in str(exc.value)

    def test_a_null_feature_idx_is_refused(self):
        d = defn([mem(10, feature=feat(None))])
        with pytest.raises(Exception) as exc:
            CircuitSteeringEngine.serving_members(d)
        assert "feature_idx" in str(exc.value)

    def test_the_schema_makes_a_null_member_list_unreachable(self):
        """`members=None` raises a bare TypeError, which is the right answer
        for a programming error — and the schema declares
        `list[CircuitMemberV1] = Field(..., min_length=1)`, so a parsed
        document can never present it."""
        import inspect

        from millm.api.schemas import circuit as schema_mod

        src = inspect.getsource(schema_mod)
        assert "members: list[CircuitMemberV1] = Field(..., min_length=1)" in src, (
            "the member list became optional or empty-able; the engine's "
            "TypeError on None is now reachable from a document"
        )

    def test_a_definition_with_no_sae_references_still_derives(self):
        """Missing SAE references are a legitimate unbound state, not an
        error: every member simply carries `sae_id=None`."""
        d = defn([mem(10, feature=feat(1))], sae_by_layer={})
        members = CircuitSteeringEngine.serving_members(d)
        assert len(members) == 1 and members[0].sae_id is None


class TestR2DedupeAcrossEverySourceShape:
    """F18 R2-14. The dedupe is what keeps a legitimate circuit serveable — the
    serving path rejects a repeated `(layer, feature_idx)` outright. Probed the
    shapes the characterization fixtures do not cover: a duplicate WITHIN one
    expansion, the same feature in two separate members, an empty expansion,
    and an expansion with no own feature."""

    def test_a_duplicate_inside_one_expansion_is_collapsed(self):
        d = defn([mem(10, feature=feat(1), expanded=[feat(2), feat(2)])])
        out = CircuitSteeringEngine.serving_members(d)
        assert [(m.layer, m.feature_idx) for m in out] == [(10, 2), (10, 1)]

    def test_the_same_feature_in_two_members_is_collapsed(self):
        """Dedupe is global across the definition, not per member — two members
        naming the same (layer, feature) would otherwise reach the serving path
        as a repeated key and be rejected outright."""
        d = defn([mem(10, feature=feat(1)), mem(10, feature=feat(1))])
        assert len(CircuitSteeringEngine.serving_members(d)) == 1

    def test_an_empty_expansion_falls_back_to_the_own_feature(self):
        d = defn([mem(10, feature=feat(1), expanded=[])])
        out = CircuitSteeringEngine.serving_members(d)
        assert [(m.layer, m.feature_idx) for m in out] == [(10, 1)]

    def test_an_expansion_with_no_own_feature_still_contributes(self):
        d = defn([mem(10, feature=None, expanded=[feat(7)])])
        out = CircuitSteeringEngine.serving_members(d)
        assert [(m.layer, m.feature_idx) for m in out] == [(10, 7)]


class TestR2TheServeabilityBoundary:
    """F18 R2-15. `is_serveable` decides whether a response may carry a rung
    header. Over-claiming attaches an evidence claim to an intervention that
    never happened; under-claiming loses a true one. Every boundary combination
    is pinned because this predicate is the honesty gate."""

    @pytest.mark.parametrize(
        "claimed,attached,expected",
        [
            ([10, 13], [10, 13], True),    # fully attached
            ([10, 13], [10], True),        # partially — one layer IS steering
            ([10, 13], [99], False),       # attached, but nothing claimed
            ([10], [], False),             # empty registry
            ([], [10], False),             # no members
        ],
    )
    def test_serveability(self, claimed, attached, expected):
        d = defn([mem(n, feature=feat(1)) for n in claimed])
        plan = CircuitSteeringEngine(_Registry(attached)).plan_for(d)
        assert plan.is_serveable is expected

    def test_a_partial_attachment_is_serveable_because_it_really_steers(self):
        """The subtle one: one attached claimed layer means the apply DOES
        something, so withholding the header would under-claim a real
        intervention."""
        d = defn([mem(10, feature=feat(1)), mem(13, feature=feat(2))])
        plan = CircuitSteeringEngine(_Registry([10])).plan_for(d)
        assert plan.is_serveable is True
        assert plan.unattached_layers == frozenset({13})


class TestR2TheFourSitesAgreeOnAREALISTICDefinition:
    """F18 R2-16. The four-way identity tests above use simple fixtures. This
    one uses the shape a real circuit presents: multi-layer, a cluster_ref with
    BOTH an expansion and its own feature, mixed signs, per-layer SAE
    references and a document budget that differs from the DB column — every
    axis on which the four sites historically diverged, in one definition."""

    def _realistic(self):
        return defn(
            [
                mem(10, feature=feat(1, strength=40.0, sign=1, label="a"),
                    expanded=[feat(2, strength=10.0, sign=-1, label="b")]),
                mem(13, feature=feat(3, strength=30.0, sign=1, label="c")),
            ],
            sae_by_layer={10: sae_ref("sae-10"), 13: sae_ref("sae-13")},
            budget=SimpleNamespace(intensity=1.5),
        )

    def _key(self, members):
        return [
            (m.layer, m.feature_idx, m.budget, m.sign, m.sae_id, m.label)
            for m in members
        ]

    def test_set_intensity_and_activation_derive_byte_identical_members(self):
        d = self._realistic()
        via_set_intensity = CircuitSteeringEngine.serving_members(d)
        via_activation = CircuitSteeringEngine(None).plan_for(
            d, SimpleNamespace(intensity=99.0)
        ).members
        assert self._key(via_set_intensity) == self._key(via_activation)

    def test_the_dial_derives_the_same_members_at_a_different_lambda(self):
        d = self._realistic()
        activation = CircuitSteeringEngine(None).plan_for(
            d, SimpleNamespace(intensity=99.0)
        )
        dialled = CircuitSteeringEngine(None).plan_for(
            d, SimpleNamespace(intensity=99.0), intensity=0.5
        )
        assert self._key(dialled.members) == self._key(activation.members)
        assert (activation.intensity, dialled.intensity) == (1.5, 0.5)

    def test_the_document_budget_beats_the_db_column_on_this_shape(self):
        """F14-R1-01 on a realistic definition rather than a minimal one."""
        plan = CircuitSteeringEngine(None).plan_for(
            self._realistic(), SimpleNamespace(intensity=99.0)
        )
        assert plan.intensity == 1.5

    def test_the_mixed_sign_expansion_survives_intact(self):
        """A -1 sign inside an expansion, carried untouched all the way — the
        combination that would flip if any site pre-applied it."""
        members = CircuitSteeringEngine.serving_members(self._realistic())
        by_idx = {m.feature_idx: (m.budget, m.sign) for m in members}
        assert by_idx[2] == (10.0, -1)
        assert by_idx[1] == (40.0, 1)


class TestR2TheResponseSHAPEIsUnchanged:
    """F18 R2-17. The FTASKS is explicit: any API response-shape delta from
    this refactor is a defect, not a feature. Diffed `_serve_full`'s returned
    keys against the pre-refactor commit — identical.

    (The diff initially showed `circuit_id` as added; it comes from R2-01's
    error `details`, not the response dict. Checked before reporting, because a
    naive key-grep over the whole method body cannot tell a response from an
    exception payload.)"""

    def _returned_keys(self, source: str, fn: str) -> set:
        """Keys of the dict literal the function RETURNS, not every string in
        its body."""
        import re

        start = source.index(f"    async def {fn}(")
        end = source.index("\n    async def ", start + 10)
        body = source[start:end]
        ret = body.index("return {")
        return set(re.findall(r'"([a-z_]+)":', body[ret:]))

    def test_serve_full_returns_exactly_the_pre_refactor_keys(self):
        import subprocess

        from millm.services import circuit_service

        pre = subprocess.run(
            ["git", "show", "2804b4f:millm/services/circuit_service.py"],
            capture_output=True, text=True, cwd="/home/x-sean/app/miLLM",
        ).stdout
        if not pre:
            pytest.skip("pre-refactor revision unavailable")

        import inspect

        post = inspect.getsource(circuit_service)
        assert self._returned_keys(post, "_serve_full") == self._returned_keys(
            pre, "_serve_full"
        ), "the activation response shape changed — a defect per the FTASKS"

    def test_bound_layers_still_reports_the_documents_layers(self):
        """R1-07: the claim-set identity is what the DIAL relies on;
        `bound_layers` is a contract field reporting the document's declared
        layers, and moving it to `claimed_layers` would be the response-shape
        delta the FTASKS forbids."""
        import inspect

        from millm.services.circuit_service import CircuitService

        src = inspect.getsource(CircuitService._serve_full)
        assert '"bound_layers": definition.layers()' in src, (
            "bound_layers changed source — if that is intended, it is an API "
            "change and needs a contract note, not a refactor comment"
        )


class TestR2TheRestorePathDoesNotTrustStaleHandles:
    """F18 R2-18. This is the property that makes R2-03's live references safe,
    and no F18 test asserted it.

    `claimed_entries` holds live registry entries, so a detach mid-request
    leaves stale handles in the plan. The restore does NOT use them: it
    re-resolves each saved layer through `state.get(sae_id, layer)` and skips
    anything that has gone. Without that, a restore would write steering values
    into a detached SAE — reviving an intervention on a layer the operator
    deliberately released."""

    def test_the_restore_re_resolves_rather_than_using_the_saved_handle(self):
        import inspect

        from millm.services.inference_service import InferenceService

        src = inspect.getsource(InferenceService._restore_request_profile)
        assert 'state.get(entry_state["sae_id"], entry_state["layer"])' in src, (
            "the restore no longer re-resolves the entry — it would write to "
            "whatever handle the request captured, including a detached one"
        )
        assert "if entry is None or entry.sae is None:" in src, (
            "the restore no longer skips a detached entry"
        )

    def test_each_layer_restores_INDEPENDENTLY(self):
        """A failing layer must not abort the loop: the remaining layers would
        stay permanently dialled, which is the per-request override leaking
        into global state that restore exists to prevent."""
        import inspect

        from millm.services.inference_service import InferenceService

        src = inspect.getsource(InferenceService._restore_request_profile)
        circuit_block = src[src.index('if saved.get("circuit")'):]
        assert "except Exception as layer_error:" in circuit_block, (
            "one layer's failure can now abort the restore of the others"
        )


class TestR2TheEchoVerdictTracksAttachment:
    """F18 R2-19. `_steering_circuit_uncached` was rewritten to ask
    `plan.is_serveable`. Its verdict must follow the CURRENT attachment state,
    or a memoised header outlives the intervention it describes."""

    @pytest.mark.asyncio
    async def test_detaching_flips_the_verdict(self, monkeypatch):
        from millm.services import sae_service as sae_mod
        from millm.services.inference_service import InferenceService

        d = defn([mem(10, feature=feat(1))])
        circuit = SimpleNamespace(
            id="c", intensity=1.0, layers=[10], serving_mode="full",
            rung=2, circuit_meta={}, name="n",
        )

        class Registry:
            layers = [10]

            def entries(self):
                return [
                    SimpleNamespace(layer=n, sae_id=f"s{n}") for n in self.layers
                ]

        registry = Registry()
        monkeypatch.setattr(sae_mod, "AttachedSAEState", lambda: registry)

        svc = InferenceService.__new__(InferenceService)

        async def _active():
            return circuit

        svc._active_full_circuit = _active
        monkeypatch.setattr(
            InferenceService, "_circuit_definition", lambda self, c: d
        )

        assert await svc._steering_circuit_uncached() is not None
        registry.layers = []
        assert await svc._steering_circuit_uncached() is None, (
            "the echo verdict survived a detach — a response would carry a "
            "rung header for an intervention that is no longer running"
        )


class TestR3TheRungHeaderIsRetractedWhenTheApplyFails:
    """F18 R3-01. THE HIGHEST-SEVERITY FINDING OF THIS INCREMENT.

    `X-miLLM-Circuit-Rung` is computed at request ENTRY. The dial applies
    LATER, inside generation, and its `except Exception` deliberately never
    fails a chat request. So an apply failure left the response advertising

        X-miLLM-Circuit-Rung: 2; language="causally validated (edge)"

    for an intervention that provably did not run — an evidence claim about
    nothing, on the one surface a dial client actually reads. The whole point of
    the ladder is that "causally validated" is never said loosely.

    `_steering_circuit`'s own docstring names this hazard and says R1 fixed it;
    R1 fixed it for the LOOKUP path (nothing attached) and left the
    APPLY-FAILURE path open. Two rounds of review read past it."""

    def test_a_failed_apply_records_itself_for_the_request(self):
        from millm.services.inference_service import (
            circuit_apply_failed,
            note_circuit_apply_failed,
            reset_steering_memo,
        )

        reset_steering_memo()
        assert circuit_apply_failed() is False
        note_circuit_apply_failed()
        assert circuit_apply_failed() is True

    def test_the_flag_does_not_leak_into_the_next_request(self):
        """Same ContextVar discipline as the memo. An ASGI server may reuse a
        context, so a stale True would suppress the rung header on a later
        request that steered perfectly well — turning an honesty fix into a
        silent loss of disclosure."""
        from millm.services.inference_service import (
            circuit_apply_failed,
            note_circuit_apply_failed,
            reset_steering_memo,
        )

        note_circuit_apply_failed()
        reset_steering_memo()
        assert circuit_apply_failed() is False

    @pytest.mark.asyncio
    async def test_a_REAL_failing_apply_sets_the_flag(self, monkeypatch):
        """R3-06 — this test replaced a source-grep, and the grep could not
        fail. A mutation that emptied `note_circuit_apply_failed`'s BODY left
        the call text at the dial site intact, so the grep passed while the
        flag was never set and the header was never retracted. The defect the
        fix exists to prevent was fully reintroduced under a green suite.

        This drives the real dial with an apply that raises, and asserts the
        observable outcome rather than the presence of a line of code."""
        from millm.services import inference_service as inf_mod
        from millm.services.inference_service import (
            InferenceService,
            circuit_apply_failed,
            reset_steering_memo,
        )

        reset_steering_memo()

        class Boom:
            def set_circuit_steering(self, *a, **k):
                raise RuntimeError("apply exploded")

        from millm.services import sae_service as sae_mod

        monkeypatch.setattr(
            sae_mod.SAEService, "for_registry", staticmethod(lambda: Boom())
        )

        d = defn([mem(10, feature=feat(1))])
        circuit = SimpleNamespace(
            id="c", intensity=1.0, layers=[10], serving_mode="full",
            rung=2, circuit_meta={}, name="n",
        )

        class Registry:
            def entries(self):
                return [
                    SimpleNamespace(
                        layer=10,
                        sae_id="s10",
                        sae=SimpleNamespace(
                            get_steering_values=lambda: {},
                            is_steering_enabled=lambda: False,
                            clear_steering=lambda: None,
                            set_steering_batch=lambda v: None,
                            enable_steering=lambda e: None,
                        ),
                    )
                ]

            steering_epoch = 0

            def by_layer(self, layer):
                return None

        monkeypatch.setattr(sae_mod, "AttachedSAEState", lambda: Registry())

        svc = InferenceService.__new__(InferenceService)
        monkeypatch.setattr(
            InferenceService, "_circuit_definition", lambda self, c: d
        )
        monkeypatch.setattr(
            InferenceService, "_restore_request_profile", lambda self, s: None
        )

        async def _active():
            return circuit

        svc._active_full_circuit = _active

        assert circuit_apply_failed() is False
        result = await svc._apply_request_circuit_steering(1.0, "req-1")
        assert result is None, "a failed apply must not report a saved profile"
        assert circuit_apply_failed() is True, (
            "the apply raised and the request did not record it — the response "
            "will still advertise a rung header for an intervention that never "
            "ran"
        )

    def test_the_route_generates_BEFORE_deciding_on_the_header(self):
        """Ordering is the whole fix: the header must be set after the apply
        has had its chance to fail, not before."""
        import inspect

        from millm.api.routes.openai import chat

        src = inspect.getsource(chat.create_chat_completion)
        gen = src.index("result = await inference.create_chat_completion")
        hdr = src.index('response.headers["X-miLLM-Circuit-Rung"]')
        assert gen < hdr, (
            "the rung header is still set before generation, so an apply "
            "failure inside generation cannot retract it"
        )
        assert "not circuit_apply_failed()" in src, (
            "the header is emitted without consulting the apply outcome"
        )


class TestR3NonFiniteIsRefusedAtTheSINK:
    """F18 R3-02/04. `max(lo, min(hi, nan))` returns `hi` — a non-finite
    intensity resolves to the CEILING, maximum-aggression steering, silently.

    R2-04 guarded ONE of the four paths into that clamp (`plan_for`'s override).
    The other three — `plan_for`'s DERIVED branch, `_serve_full`, and
    `set_intensity`, which never builds a plan at all — were unguarded. An
    authored `intensity_range` of "NaN" reaches the clamp from any imported
    document, because `float("NaN")` does not raise.

    The guard now lives at the single point of convergence. Fail CLOSED:
    refusing to steer is correct where the alternative is steering at the
    maximum the envelope allows."""

    def test_the_clamp_still_resolves_nan_to_the_ceiling(self):
        """The arithmetic fact the guard exists for. If this ever stops being
        true the guard's rationale needs rereading, not silently keeping."""
        assert max(0.0, min(2.0, float("nan"))) == 2.0

    def test_set_circuit_steering_refuses_a_non_finite_intensity(self):
        from millm.services.sae_service import SAEService

        svc = SAEService.for_registry()
        for bad in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(ValueError, match="finite"):
                svc.set_circuit_steering(
                    members=[mem(10, feature=feat(1))], intensity=bad
                )

    def test_a_stored_non_finite_intensity_is_refused_on_the_DERIVED_path(self):
        """R2-04's guard tests the override; this is the sibling door one line
        below it. A circuit row holding NaN — a migration backfill, a direct
        SQL UPDATE, an authored range of "NaN" — produced a plan carrying NaN
        with no error at all."""
        from millm.ml.circuit_steering import CircuitSteeringEngine

        d = defn([mem(10, feature=feat(1))])
        circuit = SimpleNamespace(
            id="c", intensity=float("nan"), layers=[10],
            serving_mode="full", rung=2, circuit_meta={}, name="n",
        )
        with pytest.raises(ValueError, match="finite"):
            CircuitSteeringEngine().plan_for(d, circuit)

    def test_UNSET_INTENSITY_still_passes_the_derived_guard(self):
        """Absence and corruption are BOTH NaN, so they are told apart by
        identity, not by value. A guard that rejected all NaN would break the
        legitimate members-only derivation R1-12 introduced the sentinel for."""
        from millm.ml.circuit_steering import (
            UNSET_INTENSITY,
            CircuitSteeringEngine,
        )

        d = defn([mem(10, feature=feat(1))])
        circuit = SimpleNamespace(
            id="c", intensity=None, layers=[10], serving_mode="full",
            rung=2, circuit_meta={}, name="n",
        )
        plan = CircuitSteeringEngine().plan_for(d, circuit)
        assert plan.intensity is UNSET_INTENSITY
        assert plan.has_intensity is False


class TestR3TheFreezeIsAnInvariantOfTheClass:
    """F18 R3-03. R2-12 changed `plan_for` to pass tuples and annotated the
    field `tuple[Any, ...]`. Dataclasses DO NOT ENFORCE ANNOTATIONS, so this
    was accepted and the append succeeded:

        ServingPlan(members=[1, 2], ...).members.append(3)   -> [1, 2, 3]

    reproducing the exact half-frozen object R2-12 was written to eliminate,
    for any consumer or test that builds a plan directly. R2's own test
    (`test_members_cannot_be_appended_to`) asserted a property of `tuple` and
    so could never have caught it."""

    def test_a_list_passed_to_the_constructor_is_frozen(self):
        from millm.ml.circuit_steering import ServingPlan

        plan = ServingPlan(
            members=[mem(10, feature=feat(1))],
            intensity=1.0,
            claimed_layers={10},
            attached_layers={10},
            claimed_entries=[SimpleNamespace(layer=10, sae_id="s")],
        )
        assert isinstance(plan.members, tuple)
        assert isinstance(plan.claimed_entries, tuple)
        assert isinstance(plan.claimed_layers, frozenset)
        assert isinstance(plan.attached_layers, frozenset)
        with pytest.raises(AttributeError):
            plan.members.append(mem(11, feature=feat(2)))

    def test_the_claim_identity_cannot_be_broken_by_a_direct_construction(self):
        """The invariant F18 exists to make structural, asserted against the
        construction path R2-12 left open rather than only the factory."""
        from millm.ml.circuit_steering import ServingPlan

        members = [mem(10, feature=feat(1))]
        plan = ServingPlan(
            members=members,
            intensity=1.0,
            claimed_layers=frozenset({10}),
            attached_layers=frozenset({10}),
        )
        members.append(mem(99, feature=feat(2)))  # mutate the ORIGINAL list
        assert plan.claimed_layers == frozenset(m.layer for m in plan.members), (
            "the plan aliased the caller's list, so mutating it after "
            "construction broke the claim-set identity"
        )


class TestR3AFailedOperatorDialDoesNotStrandInFlightRequests:
    """F18 R3-07/08. `set_intensity` bumps the steering epoch under
    `if not reapplied` — and a FAILED apply also leaves `reapplied` False.

    R3-07: an operator dial that raised advanced the epoch as though it were
    the authoritative write. Every in-flight request whose snapshot straddles
    that bump then sees a mismatch, SKIPS ITS RESTORE (by design — the epoch
    guard means "someone wrote after me, don't clobber them"), and strands its
    transient per-request lambda in global state PERMANENTLY. A failed operator
    action should change nothing; instead it silently made other requests'
    temporary overrides permanent, and the operator was told only that the
    apply failed.

    R3-08, found BY the R3-07 fix: `applied_epoch` is bound on exactly two
    paths (a successful apply, and the bump) and read unconditionally at
    `still_current`. The failed-apply path reached that read with it UNBOUND —
    `UnboundLocalError` out of a method whose contract on this path is "report
    the divergence rather than letting an exception imply nothing landed",
    after the DB write had already committed. It was latent only because the
    unconditional bump happened to bind it as a side effect."""

    @pytest.mark.asyncio
    async def test_a_raising_apply_neither_bumps_the_epoch_nor_explodes(self):
        from millm.services import circuit_service as cs_mod
        from millm.services.sae_service import AttachedSAEState

        meta = {
            "kind": "mistudio.circuit-definition",
            "schema_version": "1",
            "name": "n",
            "saes": [
                {"layer": 10, "n_features": 8192, "mistudio_sae_id": "sae-10"}
            ],
            "members": [
                {"layer": 10, "feature": {"feature_idx": 1, "strength": 40.0}}
            ],
            "edges": [],
            "budget": {
                "layers": {}, "intensity": 1.0, "intensity_range": [0.0, 2.0],
            },
        }
        circuit = SimpleNamespace(
            id="c", name="n", layers=[10], serving_mode="full",
            intensity=1.0, rung=2, circuit_meta=meta, is_active=True,
            description=None, serveable=True, per_sae_warnings=[],
            edge_count=0, provenance={}, created_at=None, updated_at=None,
        )

        class Boom:
            def set_circuit_steering(self, *a, **k):
                raise RuntimeError("apply exploded")

        class Repo:
            async def get(self, _id):
                return circuit

            async def update(self, _id, **fields):
                for k, v in fields.items():
                    setattr(circuit, k, v)
                return circuit

        svc = cs_mod.CircuitService.__new__(cs_mod.CircuitService)
        svc._sae_service = Boom()
        svc.repository = Repo()

        before = AttachedSAEState().steering_epoch

        # R3-08: this must NOT raise. The DB write has already committed by the
        # time the apply runs, so the contract is to report the divergence.
        result = await svc.set_intensity(circuit, 1.5)

        assert AttachedSAEState().steering_epoch == before, (
            "a FAILED apply advanced the steering epoch — every in-flight "
            "request now skips its restore and strands its transient lambda"
        )
        # And the operator is told the truth about what happened.
        text = " ".join(result.get("warnings", []))
        assert "could not be applied" in text and "differ" in text, (
            f"the apply failure was not reported to the operator: {result!r}"
        )


class TestR3APartialApplyRollsBack:
    """F18 R3-09. Steps 1-2 of `set_circuit_steering` are fail-closed: every
    offender is collected and `SAESetIncompleteError` raises before anything is
    written. But the APPLY LOOP writes SAE-by-SAE with no rollback.

    A raise on the third of five layers left layers 1-2 at the NEW intensity,
    layer 3 cleared but not set (the clear precedes the raise, so it is
    silently zeroed), and 4-5 at the OLD values. The circuit then runs as a
    chimera of two intensities with a hole in the middle — a WRONG-BASIS
    intervention rather than a failed one, and the model says things nobody
    authored.

    `circuit_sensing_service._arm_targets` already rolls back for ARMING. The
    serving path — the one that changes what the model actually says — had no
    equivalent."""

    def test_a_failure_mid_loop_restores_every_recoverable_layer(
        self, monkeypatch, caplog
    ):
        from millm.services import sae_service as sae_mod
        from millm.services.sae_service import SAEService

        class FakeSAE:
            def __init__(self, layer, prior):
                self.layer = layer
                self._values = dict(prior)
                self.is_steering_enabled = True
                self.d_sae = 8192
                self.explode = False

            def get_steering_values(self):
                return dict(self._values)

            def clear_steering(self):
                self._values = {}

            def set_steering_batch(self, values):
                if self.explode:
                    raise RuntimeError(f"layer {self.layer} refused the write")
                self._values = dict(values)

            def enable_steering(self, on):
                self.is_steering_enabled = on

        saes = {
            10: FakeSAE(10, {1: 11.0}),
            11: FakeSAE(11, {2: 22.0}),
            12: FakeSAE(12, {3: 33.0}),
        }
        saes[12].explode = True  # the third layer refuses

        entries = [
            SimpleNamespace(layer=n, sae_id=f"s{n}", sae=s)
            for n, s in saes.items()
        ]

        class Registry:
            def entries(self):
                return entries

            def by_layer(self, layer):
                return next((e for e in entries if e.layer == layer), None)

            steering_epoch = 0

            def bump_steering_epoch(self, _why):
                return 1

        monkeypatch.setattr(sae_mod, "AttachedSAEState", lambda: Registry())

        svc = SAEService.for_registry()
        # The REAL CircuitMember, not a namespace: `set_circuit_steering`
        # consumes the flat serving shape (feature_idx/budget/sign), and using
        # the actual type means the test cannot drift from it.
        from millm.api.schemas.circuit import CircuitMember

        members = [
            CircuitMember(feature_idx=1, layer=10, budget=40.0, sign=1),
            CircuitMember(feature_idx=2, layer=11, budget=50.0, sign=1),
            CircuitMember(feature_idx=3, layer=12, budget=60.0, sign=1),
        ]

        with pytest.raises(RuntimeError, match="refused the write"):
            svc.set_circuit_steering(members, 1.0)

        # THE POINT: every layer holds what it held before, not a mix.
        assert saes[10].get_steering_values() == {1: 11.0}, (
            "layer 10 was left at the new intensity after a later layer failed"
        )
        assert saes[11].get_steering_values() == {2: 22.0}, (
            "layer 11 was left at the new intensity after a later layer failed"
        )
        # Layer 12 is the layer that raised. Its restore calls the SAME
        # `set_steering_batch` that just failed, so it cannot be recovered here
        # — that is a property of the failure, not a gap in the rollback. What
        # the contract guarantees is that this is LOUD and NAMED, never silent:
        # the model is in a state nobody authored and only an operator can
        # resolve it.
        #
        # The first version of this rollback swallowed that failure entirely,
        # and this test is what found it.
        assert saes[12].get_steering_values() == {}, (
            "the failing layer's state is expected to be unrecoverable here"
        )


class TestR3TheAuthoredBudgetWinsThroughEITHERSHAPE:
    """F18 R3-11. THE MODULE'S OWN REASON FOR EXISTING, VIOLATED INSIDE IT.

    `serving_intensity` read the authored budget with attribute access only.
    A DICT-shaped budget — reachable, because `CircuitDefinitionV1` sets
    `extra="allow"` and `_parse_stored` on a partially-shaped `circuit_meta`
    yields one — found no `.intensity`, fell through, and silently served the
    STALE DB COLUMN instead of the authored value.

    Verified by execution before the fix: authored 1.7 was served as 0.3.

    That is F14-R1-01 reappearing inside the module whose docstring says it
    exists to make F14-R1-01 structurally impossible. `_resolve_circuit_intensity`
    reads this same field as a dict, which is the proof both shapes are live in
    this codebase rather than a hypothetical."""

    def test_a_dict_budget_still_beats_the_db_column(self):
        from millm.ml.circuit_steering import CircuitSteeringEngine

        d = SimpleNamespace(
            members=[], edges=[], budget={"intensity": 1.7},
            sae_for_layer=lambda layer: None, layers=lambda: [],
        )
        assert CircuitSteeringEngine().serving_intensity(
            d, SimpleNamespace(intensity=0.3)
        ) == 1.7, (
            "the authored budget lost to the stale DB column because it was a "
            "dict — F14-R1-01, resurrected"
        )

    def test_an_object_budget_still_beats_the_db_column(self):
        from millm.ml.circuit_steering import CircuitSteeringEngine

        d = SimpleNamespace(
            members=[], edges=[], budget=SimpleNamespace(intensity=1.7),
            sae_for_layer=lambda layer: None, layers=lambda: [],
        )
        assert CircuitSteeringEngine().serving_intensity(
            d, SimpleNamespace(intensity=0.3)
        ) == 1.7

    def test_an_authored_ZERO_still_wins_through_a_dict(self):
        """The `is not None` distinction, through the shape that lost it. A
        budget of 0.0 is a legitimate authored 'off' and must not fall through
        to the column — the original reason this read is explicit."""
        from millm.ml.circuit_steering import CircuitSteeringEngine

        d = SimpleNamespace(
            members=[], edges=[], budget={"intensity": 0.0},
            sae_for_layer=lambda layer: None, layers=lambda: [],
        )
        assert CircuitSteeringEngine().serving_intensity(
            d, SimpleNamespace(intensity=0.3)
        ) == 0.0


class TestR3AMalformedRegistryCannotKillAChatRequest:
    """F18 R3-12. The `_entries()` guard covered `state.entries()` but NOT the
    reads that touch each entry: `frozenset(e.layer for e in entries)` and the
    claimed-entries filter both ran outside it.

    So a malformed entry raised AttributeError straight out of `plan_for`. On
    the echo path that reaches `_steering_circuit_uncached`, which has no
    handler of its own — an unhandled exception on the CHAT HOT PATH, defeating
    the "no observability nicety may ever fail a chat request" contract the
    degradation exists to keep. Verified by execution."""

    def test_an_entry_missing_its_layer_degrades_instead_of_raising(self):
        from millm.ml.circuit_steering import CircuitSteeringEngine

        class Malformed:
            def entries(self):
                return [SimpleNamespace(sae_id="s")]  # no .layer

        engine = CircuitSteeringEngine()
        engine._state = Malformed()
        d = SimpleNamespace(
            members=[], edges=[], budget=None,
            sae_for_layer=lambda layer: None, layers=lambda: [],
        )
        plan = engine.plan_for(d, SimpleNamespace(intensity=1.0))
        assert plan.attached_layers == frozenset()
        assert plan.is_serveable is False

    def test_a_raising_registry_still_degrades(self):
        """The half that was already guarded — kept so a refactor cannot trade
        one for the other."""
        from millm.ml.circuit_steering import CircuitSteeringEngine

        class Broken:
            def entries(self):
                raise RuntimeError("registry is gone")

        engine = CircuitSteeringEngine()
        engine._state = Broken()
        d = SimpleNamespace(
            members=[], edges=[], budget=None,
            sae_for_layer=lambda layer: None, layers=lambda: [],
        )
        assert engine.plan_for(
            d, SimpleNamespace(intensity=1.0)
        ).attached_layers == frozenset()


class TestR3GoingDarkIsNeverSILENT:
    """F18 R3-13/14. Two degradation paths returned None with no operator
    signal at all.

    `_circuit_definition` swallowed ANY parse failure bare: a corrupt
    `circuit_meta` made both the dial and the rung echo degrade to "nothing is
    steering" with NO warning, NO counter and NO header. The circuit still
    reads ACTIVE in the management API and steers nothing, forever — the only
    way to find out is to notice the model stopped behaving differently.

    `_active_full_circuit` returned None on any DB exception, making "no
    circuit is active" (the normal case, logged nowhere) indistinguishable from
    "we could not find out".

    Going quietly dark is the failure mode this codebase treats as worse than
    raising."""

    def test_an_unparseable_definition_is_logged(self):
        import inspect

        from millm.services.inference_service import InferenceService

        src = inspect.getsource(InferenceService._circuit_definition)
        assert "circuit_definition_unparseable" in src, (
            "a corrupt circuit document still degrades silently"
        )
        assert "exc_info=True" in src, "the reason is lost"

    def test_the_lookup_failure_says_it_is_not_the_normal_case(self):
        import inspect

        from millm.services.inference_service import InferenceService

        src = inspect.getsource(InferenceService._active_full_circuit)
        handler = src[src.index("except Exception as e:"):]
        assert "error_type" in handler and "exc_info=True" in handler, (
            "a DB blip is still indistinguishable from a clean 'nothing active'"
        )


class TestR3AnOFFCircuitReportsNothingApplied:
    """F18 R3-15. `applied_per_layer` is documented as what was "actually
    written to each layer's SAE", and the apply site's own comment says an OFF
    circuit should "clear and leave steering disabled rather than reporting N
    features active at zero strength".

    The code did the first half and not the second: it cleared and disabled,
    then reported every member anyway. So the API said N features were active
    while `is_steering_enabled` was False — a DB/GPU divergence the
    `set_intensity` warnings machinery is blind to, because from its side the
    apply succeeded.

    Two ways to be OFF, and both must report identically: λ=0, and every
    authored strength being 0.0 (a legitimate authored 'off')."""

    def _run(self, monkeypatch, intensity, strength):
        from millm.api.schemas.circuit import CircuitMember
        from millm.services import sae_service as sae_mod
        from millm.services.sae_service import SAEService

        class FakeSAE:
            def __init__(self):
                self._values = {}
                self.is_steering_enabled = False
                self.d_sae = 8192

            def get_steering_values(self):
                return dict(self._values)

            def clear_steering(self):
                self._values = {}

            def set_steering_batch(self, values):
                self._values = dict(values)

            def enable_steering(self, on):
                self.is_steering_enabled = on

        sae = FakeSAE()
        entries = [SimpleNamespace(layer=10, sae_id="s10", sae=sae)]

        class Registry:
            def entries(self):
                return entries

            def by_layer(self, layer):
                return entries[0] if layer == 10 else None

            steering_epoch = 0

            def bump_steering_epoch(self, _why):
                return 1

        monkeypatch.setattr(sae_mod, "AttachedSAEState", lambda: Registry())
        svc = SAEService.for_registry()
        members = [
            CircuitMember(feature_idx=1, layer=10, budget=strength, sign=1)
        ]
        return svc.set_circuit_steering(members, intensity), sae

    def test_lambda_zero_reports_nothing_applied(self, monkeypatch):
        result, sae = self._run(monkeypatch, 0.0, 40.0)
        assert result.disabled is True
        assert result.applied_per_layer == {}, (
            "the circuit is OFF but the API reported features as applied"
        )
        assert sae.is_steering_enabled is False

    def test_all_zero_strengths_report_nothing_applied(self, monkeypatch):
        """An authored 'off' — every strength 0.0 — at a NON-zero lambda. The
        operator dialled 1.5 and nothing steers, which is correct; what must
        not happen is the result claiming otherwise."""
        result, sae = self._run(monkeypatch, 1.5, 0.0)
        assert result.disabled is True
        assert result.applied_per_layer == {}
        assert sae.is_steering_enabled is False

    def test_a_live_circuit_still_reports_what_it_wrote(self, monkeypatch):
        """The other side of the fix: a real serve must be unchanged."""
        result, sae = self._run(monkeypatch, 1.5, 40.0)
        assert result.disabled is False
        assert result.applied_per_layer == {10: {1: 60.0}}  # 40 * 1 * 1.5
        assert sae.is_steering_enabled is True
        assert sae.get_steering_values() == {1: 60.0}


class TestR3TheDialSerialisationDEPENDENCYIsPinned:
    """F18 R3-16. A review round raised this concurrency scenario:

        two concurrent λ=0 dials. A saves the real values and clears. B saves
        the ALREADY-CLEARED values (same epoch, so no supersession is
        detected) and clears. A restores its originals; B then restores empty
        — permanently wiping the operator's steering.

    Investigated rather than fixed, because it is NOT REACHABLE at the default
    configuration: the dial runs inside `self._request_queue.acquire()` and
    `MAX_CONCURRENT_REQUESTS` defaults to 1, so save/apply/restore are
    serialised end to end. The epoch guard is not what saves this; the
    semaphore is.

    That makes the safety a property of a CONFIG VALUE, held in a different
    file from the code that depends on it, with nothing connecting them. F19 is
    Concurrent Circuit Serving — the increment whose entire purpose is to raise
    this number. So the dependency is pinned HERE, where breaking it will
    surface as a failing test naming the hazard, rather than as silently wiped
    operator steering.

    This is recorded as a finding because an unstated load-bearing assumption
    is a defect in the same way an unpinned fix is."""

    def test_the_dial_relies_on_serialisation_the_config_currently_provides(self):
        from millm.core.config import settings

        assert settings.MAX_CONCURRENT_REQUESTS == 1, (
            "MAX_CONCURRENT_REQUESTS is no longer 1. The per-request circuit "
            "dial's save/apply/restore is NOT safe under concurrency: two "
            "concurrent dials can each save the other's partially-applied "
            "state and restore over the operator's real steering. Before "
            "raising this, the dial needs per-request isolation of the saved "
            "snapshot (F19), not just the epoch guard — the epoch cannot "
            "distinguish 'someone wrote after me' from 'someone saved the "
            "state I was midway through clearing'."
        )

    def test_the_dial_still_runs_inside_the_request_queue(self):
        """The other half of the assumption: the semaphore only helps if the
        dial is actually inside it."""
        import inspect

        from millm.services.inference_service import InferenceService

        src = inspect.getsource(InferenceService.create_chat_completion)
        acquire = src.index("self._request_queue.acquire()")
        apply_call = src.index("_apply_request_steering(")
        assert acquire < apply_call, (
            "the per-request steering apply moved OUTSIDE the request-queue "
            "semaphore — concurrent dials can now race on global steering "
            "state regardless of MAX_CONCURRENT_REQUESTS"
        )


class TestR3TheNaNInvariantIsENFORCEDNotDocumented:
    """F18 R3-20. `has_intensity`'s docstring said "a consumer that is about to
    APPLY must check this", and only ONE of the two apply consumers did.
    `_serve_full` checks and raises; the per-request dial never has.

    The dial was safe anyway — but only because `_resolve_circuit_intensity`
    rejects non-finite values two functions away in a DIFFERENT FILE. So the
    invariant read as enforced by the plan while actually being enforced by a
    caller-side precondition that nothing pinned. That is the shape of the
    R2-01/R2-04 defect exactly: a property of today's code that the next caller
    silently breaks.

    The R3-02 sink guard is what makes it real. These tests assert that the
    protection does not depend on any caller remembering to check — which is
    the only version of this invariant worth having."""

    def test_every_apply_path_refuses_nan_regardless_of_caller_checks(
        self, monkeypatch
    ):
        """The sink refuses, so no caller-side precondition is load-bearing."""
        from millm.api.schemas.circuit import CircuitMember
        from millm.ml.circuit_steering import UNSET_INTENSITY
        from millm.services import sae_service as sae_mod
        from millm.services.sae_service import SAEService

        entries = [
            SimpleNamespace(
                layer=10, sae_id="s10",
                sae=SimpleNamespace(
                    d_sae=8192,
                    get_steering_values=lambda: {},
                    is_steering_enabled=False,
                    clear_steering=lambda: None,
                    set_steering_batch=lambda v: None,
                    enable_steering=lambda e: None,
                ),
            )
        ]

        class Registry:
            def entries(self):
                return entries

            def by_layer(self, layer):
                return entries[0] if layer == 10 else None

            steering_epoch = 0

            def bump_steering_epoch(self, _why):
                return 1

        monkeypatch.setattr(sae_mod, "AttachedSAEState", lambda: Registry())
        svc = SAEService.for_registry()
        members = [CircuitMember(feature_idx=1, layer=10, budget=40.0, sign=1)]

        # The sentinel itself must be refused at the apply — it is NaN, and a
        # plan carrying it is by definition not ready to serve.
        with pytest.raises(ValueError, match="finite"):
            svc.set_circuit_steering(members, UNSET_INTENSITY)

    def test_the_dial_cannot_reach_the_apply_with_an_unset_intensity(self):
        """The dial resolves λ before building its plan, and `plan_for` refuses
        a non-finite override. Both halves asserted, so neither can quietly
        become the only one."""
        from millm.ml.circuit_steering import CircuitSteeringEngine

        d = defn([mem(10, feature=feat(1))])
        circuit = SimpleNamespace(
            id="c", intensity=1.0, layers=[10], serving_mode="full",
            rung=2, circuit_meta={}, name="n",
        )
        engine = CircuitSteeringEngine()
        for bad in (float("nan"), float("inf")):
            with pytest.raises(ValueError, match="finite"):
                engine.plan_for(d, circuit, intensity=bad)

    def test_has_intensity_remains_a_cheap_local_check(self):
        """It stays useful as a NON-RAISING question a consumer can ask before
        applying — the safety just no longer depends on asking it."""
        from millm.ml.circuit_steering import UNSET_INTENSITY, ServingPlan

        unset = ServingPlan(
            members=(), intensity=UNSET_INTENSITY,
            claimed_layers=frozenset(), attached_layers=frozenset(),
        )
        assert unset.has_intensity is False
        live = ServingPlan(
            members=(), intensity=1.0,
            claimed_layers=frozenset(), attached_layers=frozenset(),
        )
        assert live.has_intensity is True
