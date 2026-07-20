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
    end = text.index("Clear any stale in-memory SAE attachment state")
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
