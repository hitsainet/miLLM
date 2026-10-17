"""An armed probe does not survive a restart, and nothing may claim it did.

⚠ OBSERVED LIVE, NOT REASONED ABOUT (2026-09-28). A rollout landed on the node under an armed
probe. Afterwards the database row still said `armed=True`, `/api/probes/status` still listed it
under `armed`, and a real `/v1/chat/completions` call came back with **no verdict header at all** —
because the hook lives in `ProbeRuntimeState`, which the new process starts empty.

`probe_event_service.status()`'s own docstring names the invariant this broke: *"A probe never goes
silently quiet"*, and *"a status block that lists a probe as armed with no further comment is
exactly that silence."*

Two independent guards, because one is not enough:

* `main.py` reconciles at startup — the state becomes wrong at boot, so that is where it is fixed.
* `status()` consults the live registry on every read, so a hook lost for any OTHER reason is still
  reported. A read that trusted the column would go quiet again the first time something new
  dropped a hook.

This is the third time this estate has shipped "a new in-memory flag was left out of startup
reconciliation" — miLLM's steering lock hid thirteen models for three months.
"""

from __future__ import annotations

import ast
import inspect
from unittest.mock import MagicMock
from contextlib import asynccontextmanager

import pytest

pytestmark = pytest.mark.asyncio


class _Row:
    """⚠ A STAND-IN MUST NEVER BE MORE FORGIVING THAN THE THING IT STANDS IN FOR.

    This raised `AttributeError: '_Row' object has no attribute 'threshold'` the moment `status()`
    began reporting the bar in force, and that is the stub working as intended: the real `Probe`
    row always has these columns, so `status()` reads them DIRECTLY rather than through a `getattr`
    default. A forgiving read would turn a renamed column into a silent `None` — a probe reported
    as having no threshold when it has one.
    """

    def __init__(
        self,
        probe_id: str,
        *,
        armed: bool,
        paused_reason=None,
        threshold: float | None = 2.5,
        threshold_revision: int = 1,
        armed_threshold_revision: int | None = None,
    ):
        self.id = probe_id
        self.name = f"name-{probe_id}"
        self.layer = 11
        self.rule = "mean"
        self.rung = 2
        self.streamable = True
        self.basis = "residual"
        self.armed = armed
        self.paused_reason = paused_reason
        self.threshold = threshold
        self.threshold_revision = threshold_revision
        self.armed_threshold_revision = armed_threshold_revision


def _probe_row(probe_id: str, *, armed: bool, paused_reason=None):
    from millm.db.models.probe import Probe

    return Probe(
        id=probe_id,
        name=probe_id,
        layer=11,
        rule="mean",
        scope="all",
        basis="residual",
        rung=2,
        armed=armed,
        paused_reason=paused_reason,
        definition={},
        hf_id="LiquidAI/LFM2.5-1.2B-Instruct",
        d_model=2048,
        n_layers=16,
    )


def _factory(session):
    """A session factory returning the test's own session, without closing it."""

    @asynccontextmanager
    async def _cm():
        yield session

    return _cm


def _service(rows):
    from millm.services.probe_event_service import ProbeEventService

    repo = MagicMock()

    async def _list():
        return rows

    repo.list = _list
    events = MagicMock()

    async def _count():
        return 0

    events.count = _count
    return ProbeEventService(repo, events)


class TestStatusDoesNotBelieveTheColumn:
    async def test_an_armed_row_with_no_hook_is_reported_as_not_scoring(self, monkeypatch):
        """The exact live state: row says armed, registry is empty."""
        import millm.services.probe_event_service as mod

        monkeypatch.setattr(mod, "_live_armed_ids", lambda: [])

        status = await _service([_Row("pr_ghost", armed=True)]).status()

        assert status["armed_rows"] == 1
        assert status["armed_count"] == 0, (
            "armed_count counted rows, so a restarted process reported a monitor that is not "
            "monitoring"
        )
        assert status["stale_armed"] == ["pr_ghost"]
        entry = status["armed"][0]
        assert entry["hook_installed"] is False
        assert entry["paused_reason"], "an armed-but-not-scoring probe must say why"
        assert "restart" in entry["paused_reason"]
        assert status["paused_reasons"], "the top-level summary must surface it too"

    async def test_a_genuinely_hooked_probe_is_unaffected(self, monkeypatch):
        """⚠ Specificity. If every armed probe read as stale, the field would be noise."""
        import millm.services.probe_event_service as mod

        monkeypatch.setattr(mod, "_live_armed_ids", lambda: ["pr_real"])

        status = await _service([_Row("pr_real", armed=True)]).status()

        assert status["armed_count"] == 1
        assert status["armed_rows"] == 1
        assert status["stale_armed"] == []
        assert status["armed"][0]["hook_installed"] is True
        assert status["armed"][0]["paused_reason"] is None
        assert status["paused_reasons"] == []

    async def test_an_existing_paused_reason_is_not_overwritten(self, monkeypatch):
        """A real reason outranks the stale-hook one: the operator paused it on purpose."""
        import millm.services.probe_event_service as mod

        monkeypatch.setattr(mod, "_live_armed_ids", lambda: [])

        status = await _service([_Row("pr_p", armed=True, paused_reason="operator")]).status()
        assert status["armed"][0]["paused_reason"] == "operator"

    async def test_it_reads_the_real_registry_when_not_patched(self):
        """⚠ The two tests above patch `_live_armed_ids`, so they would both pass against a
        helper that returns a constant. Prove the real one reaches ProbeRuntimeState."""
        from millm.services.probe_event_service import _live_armed_ids
        from millm.services.probe_runtime import ProbeRuntimeState

        ProbeRuntimeState.reset_for_tests()
        assert _live_armed_ids() == []

        probe = MagicMock()
        probe.probe_id = "pr_live"
        ProbeRuntimeState()._armed["pr_live"] = probe
        try:
            assert _live_armed_ids() == ["pr_live"]
        finally:
            ProbeRuntimeState.reset_for_tests()


class TestStartupDisarmsThem:
    """The reconciliation is tested by RUNNING it against a real session.

    ⚠ My first version of this class read `lifespan`'s AST for a `.values(armed=False)` call and
    passed — then survived a mutation that wrapped the whole block in `if False:`. A test that
    reads source cannot tell a statement that runs from one that is merely present, which is this
    repo's oldest recurring defect wearing my own handwriting. So the decision moved into
    `disarm_probes_on_startup`, which is exercised for real below, and the AST is used only for the
    one thing it is good for: proving `lifespan` CALLS it.
    """

    async def test_it_disarms_an_armed_row_and_says_why(self, test_session):
        from millm.db.models.probe import Probe
        from millm.main import disarm_probes_on_startup

        test_session.add(_probe_row("pr_ghost", armed=True))
        await test_session.commit()

        disarmed = await disarm_probes_on_startup(_factory(test_session))

        assert disarmed == ["pr_ghost"]
        row = await test_session.get(Probe, "pr_ghost")
        await test_session.refresh(row)
        assert row.armed is False, "the row still claims to be armed after a restart"
        assert row.paused_reason and "restart" in row.paused_reason

    async def test_it_leaves_a_disarmed_row_alone(self, test_session):
        """⚠ Specificity, and it matters: stamping every row would overwrite an operator's own
        paused_reason with a restart notice they never caused.

        ⚠ AND THE ARMED ROW BESIDE IT IS LOAD-BEARING. My first version of this test had only the
        idle row, so the function returned early on an empty armed set and the UPDATE never ran —
        a mutation deleting the UPDATE's `WHERE armed IS TRUE` survived it. A fixture that agrees
        with the code by construction is the usual reason a suite stays green over a real bug, and
        this one agreed by *not reaching the line at all*.
        """
        from millm.db.models.probe import Probe
        from millm.main import disarm_probes_on_startup

        test_session.add_all(
            [
                _probe_row("pr_idle", armed=False, paused_reason="operator"),
                _probe_row("pr_hot", armed=True),
            ]
        )
        await test_session.commit()

        assert await disarm_probes_on_startup(_factory(test_session)) == ["pr_hot"]

        idle = await test_session.get(Probe, "pr_idle")
        await test_session.refresh(idle)
        assert idle.armed is False
        assert idle.paused_reason == "operator", (
            "the UPDATE reached a row it was not meant to touch and rewrote its reason"
        )

        hot = await test_session.get(Probe, "pr_hot")
        await test_session.refresh(hot)
        assert hot.armed is False and "restart" in hot.paused_reason

    async def test_a_failure_does_not_stop_the_server_starting(self):
        """It runs in `lifespan`. Probe bookkeeping must never be why miLLM will not boot."""
        from millm.main import disarm_probes_on_startup

        def broken():
            raise RuntimeError("no database")

        assert await disarm_probes_on_startup(broken) == []

    def test_lifespan_actually_calls_it(self):
        """The wiring half, and the only thing the AST is used for here.

        ⚠ Asserts a CALL, not the name: a text search matches the comment above the call site.
        """
        from millm import main

        tree = ast.parse(inspect.getsource(main.lifespan))
        called = [
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        assert "disarm_probes_on_startup" in called, (
            f"lifespan does not call it, so armed rows survive every restart; calls seen: "
            f"{sorted(set(called))}"
        )

    def test_the_ast_scan_can_see_a_known_call(self):
        """⚠ A source scan that matches nothing asserts nothing. `_run_migrations` is definitely
        awaited in `lifespan`; if the scanner cannot see a call at all, the test above is void."""
        from millm import main

        tree = ast.parse(inspect.getsource(main.lifespan))
        calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert calls, "the scanner found no calls in lifespan, so it is not reading it"
