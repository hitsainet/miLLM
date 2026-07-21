"""Feature 19 task 2.6 — the claim registry.

The distinction these tests exist to protect is CONTENTION vs COLLISION:

  * contention — two circuits want the same LAYER. Overridable.
  * collision  — two circuits name the same (LAYER, FEATURE_IDX). NEVER
    overridable, because `set_steering_batch` merges into one dict and one
    strength silently overwrites the other, so the served value belongs to
    neither author.

Collapsing those two into one severity field is the defect this design most
wants to avoid, so they are asserted separately and the collision case is
asserted WITH the override explicitly set.
"""

import pytest
import sqlalchemy as sa

from millm.db.models.circuit import Circuit
from millm.db.models.circuit_layer_claim import CircuitLayerClaim
from millm.services.circuit_claim_registry import CircuitClaimRegistry

pytestmark = pytest.mark.asyncio


async def _circuit(session, cid, name=None, layers=(10,), active=True):
    session.add(
        Circuit(
            id=cid,
            name=name or cid,
            circuit_meta={"kind": "mistudio.circuit-definition", "schema_version": "1"},
            rung=2,
            edge_count=0,
            layers=list(layers),
            per_sae_warnings=[],
            serveable=True,
            is_active=active,
            provenance={},
        )
    )
    await session.flush()


class TestAssess:
    async def test_disjoint_layers_are_clear(self, test_session):
        await _circuit(test_session, "cA")
        await _circuit(test_session, "cB")
        reg = CircuitClaimRegistry(test_session)
        await reg.claim("cA", {10})

        verdict = await reg.assess("cB", {13})
        assert verdict.is_clear
        assert not verdict.has_contention
        assert not verdict.has_collision

    async def test_an_overlapping_layer_contends_and_names_the_incumbent(
        self, test_session
    ):
        await _circuit(test_session, "cA", name="fear→threat")
        await _circuit(test_session, "cB", name="hedging")
        reg = CircuitClaimRegistry(test_session)
        await reg.claim("cA", {10, 13})

        verdict = await reg.assess("cB", {13, 20})
        assert verdict.has_contention
        assert verdict.contended_layers == (13,)
        # Naming the incumbent is what makes the operator's next action obvious.
        assert "cA" in verdict.incumbents
        assert verdict.incumbents["cA"][0] == "fear→threat"

    async def test_assess_is_SELF_EXCLUDING(self, test_session):
        """EC-19.3. A circuit re-activating, or extending its own claim set,
        must not contend with ITSELF — otherwise an idempotent re-activation
        refuses against its own incumbent claim."""
        await _circuit(test_session, "cA")
        reg = CircuitClaimRegistry(test_session)
        await reg.claim("cA", {10})

        verdict = await reg.assess("cA", {10, 13})
        assert verdict.is_clear, "the circuit contended with its own claim"

    async def test_a_released_claim_does_not_contend(self, test_session):
        await _circuit(test_session, "cA")
        await _circuit(test_session, "cB")
        reg = CircuitClaimRegistry(test_session)
        await reg.claim("cA", {10})
        await reg.release("cA")

        assert (await reg.assess("cB", {10})).is_clear


class TestCollisionIsNotContention:
    """The two are structurally distinct, and only one is overridable."""

    async def test_the_same_feature_on_a_shared_layer_COLLIDES(self, test_session):
        await _circuit(test_session, "cA", name="fear→threat")
        await _circuit(test_session, "cB", name="hedging")
        reg = CircuitClaimRegistry(test_session)
        await reg.claim("cA", {10}, steering_keys={10: {42, 43}})

        verdict = await reg.assess("cB", {10}, steering_keys={10: {42, 99}})
        assert verdict.has_collision
        assert verdict.colliding_keys == ((10, 42, "cA"),), (
            "only the SHARED key collides — 43 and 99 belong to one author each"
        )

    async def test_different_features_on_a_shared_layer_only_CONTEND(
        self, test_session
    ):
        """Still contention — the layer sums either way — but composable."""
        await _circuit(test_session, "cA")
        await _circuit(test_session, "cB")
        reg = CircuitClaimRegistry(test_session)
        await reg.claim("cA", {10}, steering_keys={10: {42}})

        verdict = await reg.assess("cB", {10}, steering_keys={10: {99}})
        assert verdict.has_contention
        assert not verdict.has_collision, (
            "distinct features are composable; refusing them as a collision "
            "would remove a legitimate override"
        )

    async def test_a_collision_error_offers_NO_override_route(self):
        """A refusal that named `allow_layer_overlap` on a collision would
        invite the operator to try a parameter that cannot help — and if it
        ever did help, one author's strength would silently win."""
        from millm.core.errors import CircuitLayerContentionError

        err = CircuitLayerContentionError(
            contended_layers=[10],
            colliding_keys=[(10, 42, "cA")],
            incumbent_id="cA",
            incumbent_name="fear→threat",
        )
        assert err.details["overridable"] is False
        assert "override_param" not in err.details
        assert "cannot be overridden" in err.message

    async def test_a_contention_error_CARRIES_THE_MEASUREMENT(self):
        """BR-011 binding condition: a refusal stating only the fact of
        contention fails the task. The operator overriding this must have been
        told what happened last time — including that it is one model and one
        fixture."""
        from millm.core.errors import CircuitLayerContentionError

        err = CircuitLayerContentionError(
            contended_layers=[13],
            incumbent_id="circ_abc",
            incumbent_name="fear→threat",
            requested_id="circ_xyz",
        )
        assert err.details["overridable"] is True
        assert err.details["override_param"] == "allow_layer_overlap"
        assert err.details["rung_header_suppressed_if_overridden"] is True

        hazard = err.details["measured_hazard"]
        assert "degenerate" in hazard["two_layers_at_strength_5"]
        assert "indicative, not exhaustive" in hazard["note"], (
            "the caveat is part of the data, not a footnote — stating the "
            "measurement as more than it is would be the same overclaim the "
            "evidence ladder exists to prevent"
        )
        assert "fear→threat" in err.message and "13" in err.message


class TestClaimAndRelease:
    async def test_release_touches_ONLY_the_callers_rows(self, test_session):
        """The highest-consequence defect available in this feature: releasing
        A clearing B's claim means a circuit the operator never touched stops
        serving while its row still reads active."""
        await _circuit(test_session, "cA")
        await _circuit(test_session, "cB")
        reg = CircuitClaimRegistry(test_session)
        await reg.claim("cA", {10})
        await reg.claim("cB", {13})

        released = await reg.release("cA")
        assert released == [10]

        live = {c.circuit_id: c.layer for c in await reg.live_claims()}
        assert live == {"cB": 13}, "releasing cA disturbed cB's claim"

    async def test_the_database_index_refuses_a_duplicate_exclusive_claim(
        self, test_session
    ):
        """EC-19.7. `assess` and the INSERT are a check-then-act pair, so the
        INDEX is what actually decides a race. Asserted directly, with the
        service-level pre-check bypassed."""
        await _circuit(test_session, "cA")
        await _circuit(test_session, "cB")
        reg = CircuitClaimRegistry(test_session)
        await reg.claim("cA", {10})

        from millm.core.errors import CircuitLayerContentionError

        with pytest.raises(CircuitLayerContentionError):
            await reg.claim("cB", {10})  # no assess() first — the index decides

    async def test_composed_rows_bypass_the_exclusive_index(self, test_session):
        await _circuit(test_session, "cA")
        await _circuit(test_session, "cB")
        reg = CircuitClaimRegistry(test_session)
        await reg.claim("cA", {10}, composed=True)
        await reg.claim("cB", {10}, composed=True)

        assert len([c for c in await reg.live_claims() if c.layer == 10]) == 2

    async def test_mark_composed_flips_the_INCUMBENT_too(self, test_session):
        """Both sides must be marked: the rung header is suppressed for any
        circuit sitting on a composed layer, which cannot be determined from
        the requester's rows alone."""
        await _circuit(test_session, "cA")
        await _circuit(test_session, "cB")
        reg = CircuitClaimRegistry(test_session)
        await reg.claim("cA", {10})
        # R1-19: the requester must actually HOLD the layer before it can be
        # composed. The original version of this test called `mark_composed`
        # with only the incumbent present, which the scoping guard now
        # correctly refuses — marking an unshared layer composed would strip
        # its exclusivity protection for no reason. The real gate claims first,
        # so this now mirrors it.
        await reg.claim("cB", {10}, composed=True)
        await reg.mark_composed("cB", {10})

        live = {c.circuit_id: c.composed for c in await reg.live_claims()}
        assert live["cA"] is True, "the incumbent's row was not marked composed"
        assert live["cB"] is True

    async def test_steering_keys_round_trip(self, test_session):
        await _circuit(test_session, "cA")
        reg = CircuitClaimRegistry(test_session)
        await reg.claim("cA", {10}, steering_keys={10: {42, 7}})

        claim = next(c for c in await reg.live_claims() if c.circuit_id == "cA")
        assert sorted(claim.steering_keys) == [7, 42]


class TestReconcile:
    async def test_orphan_claims_are_released(self, test_session):
        """EC-19.4. A claim outliving its circuit's activation refuses every
        future activation on that layer, for a circuit nobody can deactivate —
        deactivating an inactive circuit is a no-op."""
        await _circuit(test_session, "cA", active=False)
        reg = CircuitClaimRegistry(test_session)
        await reg.claim("cA", {10})

        result = await reg.reconcile(allow_concurrent=True)
        assert result["orphans_released"] == [{"circuit_id": "cA", "layers": [10]}]
        assert await reg.live_claims() == []

    async def test_the_flag_being_false_DEMOTES_to_a_single_active(
        self, test_session
    ):
        """EC-19.5. A database written while the flag was true must not keep
        serving two circuits after an operator turns it off — the flag would
        be a lie."""
        await _circuit(test_session, "cA", layers=(10,))
        await _circuit(test_session, "cB", layers=(13,))
        reg = CircuitClaimRegistry(test_session)
        await reg.claim("cA", {10})
        await reg.claim("cB", {13})

        result = await reg.reconcile(allow_concurrent=False)
        assert len(result["demoted"]) == 1

        actives = (
            await test_session.execute(
                sa.select(Circuit.id).where(Circuit.is_active.is_(True))
            )
        ).scalars().all()
        assert len(actives) == 1, "the flag was false and two circuits still serve"

    async def test_the_flag_being_true_demotes_NOTHING(self, test_session):
        await _circuit(test_session, "cA", layers=(10,))
        await _circuit(test_session, "cB", layers=(13,))
        reg = CircuitClaimRegistry(test_session)
        await reg.claim("cA", {10})
        await reg.claim("cB", {13})

        result = await reg.reconcile(allow_concurrent=True)
        assert result["demoted"] == []
        assert len(await reg.live_claims()) == 2

    async def test_reconcile_leaves_a_healthy_single_active_alone(
        self, test_session
    ):
        await _circuit(test_session, "cA")
        reg = CircuitClaimRegistry(test_session)
        await reg.claim("cA", {10})

        result = await reg.reconcile(allow_concurrent=False)
        assert result == {"orphans_released": [], "demoted": []}
        assert len(await reg.live_claims()) == 1


class TestR1ComposedIsNotPermanent:
    """F19 R1-08. `mark_composed` flips both sides — correct — but nothing ever
    un-flipped it.

    When the composing circuit left, the incumbent's row stayed
    `composed=True` FOREVER: a validated rung-2 circuit serving alone,
    permanently badged composed in the UI and permanently stripped of its
    `X-miLLM-Circuit-Rung` disclosure because something once composed onto it.

    That UNDER-claims rather than over-claims, so no honesty test would have
    caught it — it just quietly deletes a disclosure the feature exists to
    protect. It also leaves the row outside the exclusive index, so a third
    circuit could claim the layer unopposed.
    """

    async def test_the_survivor_stops_being_composed(self, test_session):
        await _circuit(test_session, "cA")
        await _circuit(test_session, "cB")
        reg = CircuitClaimRegistry(test_session)
        await reg.claim("cA", {10})
        await reg.claim("cB", {10}, composed=True)
        await reg.mark_composed("cB", {10})

        assert all(c.composed for c in await reg.live_claims())

        await reg.release("cB")

        survivors = await reg.live_claims()
        assert len(survivors) == 1
        assert survivors[0].composed is False, (
            "the surviving circuit is still flagged composed, so its rung "
            "header stays suppressed forever and it sits outside the "
            "exclusive index"
        )

    async def test_a_STILL_shared_layer_stays_composed(self, test_session):
        """Un-flipping must be specific: three circuits composed, one leaves,
        the remaining two are still genuinely composed."""
        for cid in ("cA", "cB", "cC"):
            await _circuit(test_session, cid)
        reg = CircuitClaimRegistry(test_session)
        for cid in ("cA", "cB", "cC"):
            await reg.claim(cid, {10}, composed=True)
        await reg.mark_composed("cC", {10})

        await reg.release("cC")

        survivors = await reg.live_claims()
        assert len(survivors) == 2
        assert all(c.composed for c in survivors), (
            "the layer is still shared by two circuits — un-flagging it would "
            "restore a rung header that no single circuit's evidence supports"
        )


class TestR1MarkComposedIsSCOPED:
    """F19 R1-19. `mark_composed` updated EVERY live claim on every layer
    passed in, with no check that the layer is genuinely shared.

    Composed rows are EXCLUDED from the partial unique index, so marking a row
    composed permanently removes its exclusivity protection. A circuit that
    never consented to composition could lose its exclusive hold, and a third
    circuit could then claim its layer unopposed.

    The gate only ever passes contended layers, so this was safe by the
    CALLER's discipline rather than by construction.
    """

    async def test_an_UNSHARED_layer_is_not_marked(self, test_session):
        await _circuit(test_session, "cA")
        reg = CircuitClaimRegistry(test_session)
        await reg.claim("cA", {10})

        await reg.mark_composed("cA", {10})

        claim = next(c for c in await reg.live_claims() if c.layer == 10)
        assert claim.composed is False, (
            "a layer with ONE holder was marked composed, stripping its "
            "exclusivity protection so a third circuit could take it unopposed"
        )

    async def test_a_genuinely_shared_layer_IS_marked(self, test_session):
        await _circuit(test_session, "cA")
        await _circuit(test_session, "cB")
        reg = CircuitClaimRegistry(test_session)
        await reg.claim("cA", {10})
        await reg.claim("cB", {10}, composed=True)

        await reg.mark_composed("cB", {10})

        assert all(c.composed for c in await reg.live_claims())


class TestR1ClaimsCarryTheirNAME:
    """F19 R1-20. `LayerClaim.circuit_name` was declared and never populated,
    so every consumer got claims identifiable only by opaque id — and the one
    place that needed names reached through the API boundary to call the
    PRIVATE `_names_for`."""

    async def test_live_claims_populate_the_name(self, test_session):
        await _circuit(test_session, "cA", name="fear→threat")
        reg = CircuitClaimRegistry(test_session)
        await reg.claim("cA", {10})

        claim = next(c for c in await reg.live_claims() if c.circuit_id == "cA")
        assert claim.circuit_name == "fear→threat", (
            "the field that makes a claim intelligible to an operator is "
            "still empty"
        )


class TestR2ReconcileRefusesAtRUNTIME:
    """F19 R2-18. `reconcile()` demotes circuits in the DATABASE without
    touching the in-memory owner map.

    That is safe at startup, where nothing is attached and nothing is steering,
    and actively harmful at runtime: it would mark a circuit inactive while its
    SAEs keep steering it — the "row inactive, model still steering"
    divergence F19 exists to remove, produced by the method meant to repair it.

    The only guard was convention. The method is public and its docstring said
    "runs UNCONDITIONALLY", so a future caller reading that had no reason to
    hesitate.
    """

    async def test_calling_it_at_runtime_REFUSES(self, test_session):
        await _circuit(test_session, "cA")
        reg = CircuitClaimRegistry(test_session)
        await reg.claim("cA", {10})

        with pytest.raises(RuntimeError, match="safe only at startup"):
            await reg.reconcile(allow_concurrent=False, at_startup=False)

        # And it changed nothing on the way out.
        assert len(await reg.live_claims()) == 1

    async def test_the_refusal_names_the_alternative(self, test_session):
        reg = CircuitClaimRegistry(test_session)
        with pytest.raises(RuntimeError) as exc:
            await reg.reconcile(allow_concurrent=True, at_startup=False)
        assert "CircuitService" in str(exc.value), (
            "the refusal does not tell the caller what to do instead"
        )

    async def test_startup_still_works(self, test_session):
        await _circuit(test_session, "cA", active=False)
        reg = CircuitClaimRegistry(test_session)
        await reg.claim("cA", {10})

        result = await reg.reconcile(allow_concurrent=True, at_startup=True)
        assert result["orphans_released"]
