"""Startup stale-state reset (Feature 13 R3 regression).

The reset clears in-memory-derived state that cannot survive a restart: loaded
models, attached SAEs, active attachments, and active circuits.

The regression this pins: all four UPDATEs originally shared ONE transaction
with a single commit, so a table that does not exist yet (a migration not run
on this deployment) aborted the transaction and silently rolled back the resets
that HAD succeeded — leaving exactly the stale state the block exists to clear,
reported only as a warning. Each reset now owns its transaction.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MAIN = REPO / "millm" / "main.py"


def _reset_block() -> str:
    """The stale-reset region of the lifespan handler."""
    text = MAIN.read_text()
    start = text.index("resets: list[tuple[str, str, str]]")
    # F19: stop at the claim-reconciliation block, which now sits between the
    # resets and the SAE-state clear. Without this the helper swallowed the new
    # block and the "one session per reset" count went from 1 to 2.
    end = text.index("F19 R2-09: RELEASE ALL CLAIMS at startup")
    return text[start:end]


class TestResetsAreIndependent:
    def test_each_reset_gets_its_own_session_and_commit(self):
        """One shared transaction meant one failure rolled back the others."""
        block = _reset_block()
        # The loop opens a session per reset and commits inside it.
        assert "for event, sql, table in resets:" in block
        assert block.count("async with async_session_factory() as session:") == 1
        assert "await session.commit()" in block
        # And each iteration is individually guarded.
        assert "except Exception" in block
        assert "stale_reset_failed" in block

    def test_a_failing_reset_cannot_abort_the_others(self):
        """The try/except is INSIDE the loop, so one bad table is isolated."""
        block = _reset_block()
        loop_at = block.index("for event, sql, table in resets:")
        try_at = block.index("try:", loop_at)
        session_at = block.index("async with async_session_factory()", loop_at)
        # try: comes after the loop header and before the session — i.e. the
        # guard wraps each iteration's own transaction.
        assert loop_at < try_at < session_at

    def test_all_four_tables_are_reset(self):
        block = _reset_block()
        for table in ("models", "saes", "sae_attachments", "circuits"):
            assert f'"{table}",' in block, f"{table} reset missing"

    def test_circuits_reset_clears_serving_mode_too(self):
        """A stale is_active row with a serving_mode would keep claiming to
        serve — and would let set_intensity re-arm an unvalidated circuit."""
        block = _reset_block()
        m = re.search(r"UPDATE circuits SET[^\"]*", block)
        assert m, "circuits reset SQL not found"
        sql = m.group(0)
        assert "is_active = false" in sql
        assert "serving_mode = NULL" in sql

    def test_model_reset_still_clears_loaded_and_loading(self):
        """The pre-existing behaviour must survive the restructure."""
        block = _reset_block()
        assert "status IN ('loaded', 'loading')" in block
        assert "loaded_at = NULL" in block


class TestF19ReconcileIsDELIBERATELYNotCalledAtStartup:
    """F19 R3-15. R1-01 wired `reconcile()` at startup because nothing called
    it. R2-09 then made startup release EVERY claim outright — correct, since
    nothing is steering after a restart — and that left reconcile dead in BOTH
    directions:

      * the ORPHAN branch computes {live claims} - {active circuits}, and the
        release empties the first set;
      * the DEMOTION branch reads active circuits, and the bulk
        `UPDATE circuits SET is_active=false` empties that set.

    Calling it anyway is theatre: a green log line implying a check ran when
    both its inputs are empty by construction. And the previous test here
    asserted reconcile was "WIRED" by MOCKING it — so it would have kept
    passing after the method body was deleted, which is the dead-mechanism trap
    this increment has hit four times.

    The method is KEPT (recovery for a database written by an older build,
    gated by `at_startup=True`), so these tests pin the DECISION rather than
    the absence.
    """

    def test_startup_does_not_call_reconcile(self):
        # Strip comments before asserting: the word appears in the block
        # explaining WHY it is not called, and matching prose would make this
        # test fail on its own documentation.
        text = MAIN.read_text()
        code = "\n".join(
            line for line in text.splitlines() if not line.strip().startswith("#")
        )
        assert ".reconcile(" not in code, (
            "reconcile is called at startup again — both its inputs are "
            "emptied by the steps above it, so it can only report an empty "
            "result while implying a check ran"
        )

    def test_the_reasoning_is_recorded_where_the_call_was(self):
        """So the next reader does not re-add it as a missing safety step."""
        text = MAIN.read_text()
        assert "F19 R3-15" in text
        assert "dead in" in text

    def test_the_method_still_exists_for_older_databases(self):
        from millm.services.circuit_claim_registry import CircuitClaimRegistry

        assert hasattr(CircuitClaimRegistry, "reconcile"), (
            "reconcile was deleted — it is the recovery path for a database "
            "written by a build that did not release claims at startup"
        )


class TestF19R2StartupReleasesClaimsPLAINLY:
    """F19 R2-09. Reconcile ran AFTER the bulk
    `UPDATE circuits SET is_active=false`, which empties the active set — so
    its orphan branch fired for EVERY claim on EVERY restart and logged
    `circuit_claim_orphan_released` ("claims outlived their circuit's
    activation") as an anomaly.

    It is not an anomaly. It is the guaranteed steady state of a restart. A
    permanently false-positive warning trains operators to ignore the one
    signal that would matter when a genuine orphan appears, and reconcile's
    demotion branch was unreachable for the same reason.

    Nothing is steering after a restart — the in-memory owner map is empty — so
    no claim can be valid. Startup now releases them plainly and says so as
    routine.
    """

    def test_startup_releases_claims_as_a_restart_consequence(self):
        text = MAIN.read_text()
        assert "circuit_claims_released_on_startup" in text
        assert "nothing is steering after a restart" in text, (
            "the log does not say WHY the claims went, so it reads as an "
            "anomaly rather than as the expected outcome"
        )

    def test_it_runs_AFTER_the_circuits_are_deactivated(self):
        """Ordering still matters, just against a different neighbour.

        (This asserted the release ran BEFORE reconcile; R3-15 removed the
        reconcile call, so the meaningful ordering is now against the bulk
        deactivation — the release must see the post-reset state.)
        """
        text = MAIN.read_text()
        deactivate = text.index("deactivated_stale_circuits")
        release = text.index("circuit_claims_released_on_startup")
        assert deactivate < release

    def test_a_failed_release_is_an_ERROR_not_a_warning(self):
        """Stale claims refuse every activation on those layers for the life of
        the process, and there is no runtime remedy — that is not a warning."""
        text = MAIN.read_text()
        start = text.index("circuit_claim_startup_release_failed")
        # The handler is the logger.error call this event name sits inside.
        block = text[start - 60 : start + 700]
        assert "logger.error" in block, "a failed release is only a warning"
        # NB: the source wraps the sentence across concatenated string
        # literals, so match a fragment that cannot straddle a line break.
        assert "runtime remedy short of another restart" in block, (
            "the log does not say that stale claims cannot be cleared without "
            "another restart"
        )


class TestF19R2ClaimsDegradationIsVISIBLE:
    """F19 R2-19. A failed reconcile logged a line and the app served anyway.

    That is the right call — refusing to start over a bookkeeping table would
    be worse — but stale claims REFUSE every activation on their layers for the
    life of the process, with no runtime remedy. The only clue was one WARNING
    in startup logs that has already scrolled away, while `/health` reported a
    fully healthy system and a readiness probe could not tell the difference.
    """

    def test_health_reports_DEGRADED_when_claims_are_unreconciled(self):
        import asyncio
        from unittest.mock import MagicMock

        from millm.api.routes.system import health as health_mod

        health_mod.CIRCUIT_CLAIMS_DEGRADED["degraded"] = False
        health_mod.CIRCUIT_CLAIMS_DEGRADED["reason"] = None
        try:
            health_mod.note_claims_degraded("reconcile failed: boom")
            result = asyncio.run(
                health_mod.detailed_health_check(model_loader=MagicMock(is_loaded=False))
            )
            claims = next(
                c for c in result.components if c.name == "circuit_claims"
            )
            assert claims.status == health_mod.HealthStatus.DEGRADED
            assert "restarts" in claims.message
            assert result.status == health_mod.HealthStatus.DEGRADED
        finally:
            health_mod.CIRCUIT_CLAIMS_DEGRADED["degraded"] = False
            health_mod.CIRCUIT_CLAIMS_DEGRADED["reason"] = None

    def test_health_reports_HEALTHY_when_they_reconciled(self):
        import asyncio
        from unittest.mock import MagicMock

        from millm.api.routes.system import health as health_mod

        health_mod.CIRCUIT_CLAIMS_DEGRADED["degraded"] = False
        result = asyncio.run(
            health_mod.detailed_health_check(model_loader=MagicMock(is_loaded=False))
        )
        claims = next(c for c in result.components if c.name == "circuit_claims")
        assert claims.status == health_mod.HealthStatus.HEALTHY

    def test_the_release_failure_path_flags_degraded(self):
        """R3-15 removed the reconcile call, so there is one startup claim path
        left — and its failure must still reach `/health/detailed`."""
        text = MAIN.read_text()
        idx = text.index("circuit_claim_startup_release_failed")
        window = text[max(0, idx - 400) : idx]
        assert "_note_claims_degraded" in window, (
            "the release failure path does not flag claims as degraded, so "
            "/health stays green while activations are refused"
        )
