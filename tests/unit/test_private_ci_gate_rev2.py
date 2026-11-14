"""Rev2 review of 040 P0 (2026-11-08): with Backend Tests unfiltered, the gate's exact-commit rule
is the ONLY way a backend verdict reaches the mirror. Three lines of that rule survived mutation:

* G1: `if concl != "success"` widened to accept `neutral`/`None` - only failure, cancelled and
  timed_out were tried.
* G5: the `all-shards` job accepted as `skipped`.
* G6: `max(...)` turned into `min(...)` - the OLDEST run for the commit decided. A commit pushed a
  second time (a force-push back to an earlier head, or a dispatch) gets a new run; an old green
  run must not vouch while the new one is running or red.
"""

from __future__ import annotations

import pytest

from tests.unit.test_private_ci_trust import _Api, _gate, _gate_module, _run, history_repo  # noqa: F401


@pytest.mark.parametrize("conclusion", ["neutral", "skipped", "action_required", "stale",
                                        "startup_failure", None])
def test_gate_refuses_every_completed_conclusion_but_success(history_repo, conclusion):  # noqa: F811
    mod, h = _gate_module(), history_repo
    ok, msg = _gate(mod, h.repo, h.c, _Api(by_sha=[[_run(1, h.c, conclusion=conclusion)]])).check(
        "filtered.yml")
    assert not ok, msg


def test_gate_refuses_a_skipped_all_shards_job(history_repo):  # noqa: F811
    mod, h = _gate_module(), history_repo
    api = _Api(by_sha=[[_run(7, h.c)]], jobs={7: [{"name": "all shards", "conclusion": "skipped"}]})
    assert not _gate(mod, h.repo, h.c, api).check("sharded.yml")[0]


def test_the_newest_run_for_the_commit_decides(history_repo):  # noqa: F811
    mod, h = _gate_module(), history_repo
    old_green = _run(1, h.c)
    new_red = _run(2, h.c, conclusion="failure")
    ok, msg = _gate(mod, h.repo, h.c, _Api(by_sha=[[old_green, new_red]])).check("filtered.yml")
    assert not ok and "concluded failure" in msg, msg
    # Listed newest first (the API's order): still the newest decides.
    ok, msg = _gate(mod, h.repo, h.c, _Api(by_sha=[[new_red, old_green]])).check("filtered.yml")
    assert not ok, msg
    # The newest still running: wait for it, never pass on the old green.
    running = _run(3, h.c, status="in_progress", conclusion=None)
    api = _Api(by_sha=[[old_green, running], [old_green, running], [old_green, _run(3, h.c, conclusion="failure")]])
    ok, msg = _gate(mod, h.repo, h.c, api).check("filtered.yml")
    assert not ok and api.calls == 3, msg
