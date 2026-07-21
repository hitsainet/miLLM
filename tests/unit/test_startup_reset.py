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


class TestF19ClaimReconciliationIsWIRED:
    """F19 R1-01. `CircuitClaimRegistry.reconcile()` had ZERO production
    callers — written, unit-tested, and never invoked.

    A grep-the-source test would pass the moment the call APPEARS in the file,
    including inside dead code or a branch that never runs. So this EXECUTES
    the lifespan handler and asserts `reconcile` was actually awaited, which is
    the only form of this assertion that distinguishes a wired mechanism from a
    declared one.
    """

    def test_the_lifespan_actually_calls_reconcile(self):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch

        from millm.main import lifespan

        reconcile = AsyncMock(
            return_value={"orphans_released": [], "demoted": []}
        )

        class FakeRegistry:
            def __init__(self, _session):
                pass

            async def reconcile(self, **kwargs):
                return await reconcile(**kwargs)

        async def drive():
            with patch(
                "millm.services.circuit_claim_registry.CircuitClaimRegistry",
                FakeRegistry,
            ), patch("millm.main.setup_logging"), patch(
                "millm.db.base.async_session_factory"
            ) as factory:
                session = MagicMock()
                session.execute = AsyncMock(
                    return_value=MagicMock(rowcount=0)
                )
                session.commit = AsyncMock()
                factory.return_value.__aenter__ = AsyncMock(return_value=session)
                factory.return_value.__aexit__ = AsyncMock(return_value=False)
                async with lifespan(MagicMock()):
                    pass

        asyncio.run(drive())

        assert reconcile.await_count >= 1, (
            "startup never reconciled layer claims — orphaned claims survive "
            "a restart and refuse every future activation on their layers, "
            "for circuits nobody can deactivate"
        )

    def test_reconcile_runs_AFTER_the_circuits_are_deactivated(self):
        """Ordering matters: reconcile computes orphans against the ACTIVE set,
        so running it before the deactivation would see every circuit as
        legitimately active and release nothing."""
        text = MAIN.read_text()
        deactivate = text.index("deactivated_stale_circuits")
        reconcile = text.index("CircuitClaimRegistry(session).reconcile")
        assert deactivate < reconcile, (
            "reconcile runs before the stale-circuit deactivation, so it sees "
            "the pre-reset active set and releases no orphans"
        )

    def test_the_flag_is_passed_through(self):
        """With CIRCUIT_ALLOW_CONCURRENT false, a database written while it was
        true must be demoted to a single active circuit — otherwise the flag
        is a lie about what the server is doing."""
        text = MAIN.read_text()
        assert "allow_concurrent=settings.CIRCUIT_ALLOW_CONCURRENT" in text


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

    def test_it_runs_BEFORE_reconcile(self):
        """Ordering: the plain release is what actually frees the layers, so
        reconcile's orphan branch is left for genuine orphans."""
        text = MAIN.read_text()
        release = text.index("circuit_claims_released_on_startup")
        reconcile = text.index("CircuitClaimRegistry(session).reconcile")
        assert release < reconcile

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
