"""AC 1 (ENH-006 G19): with a verdict definition attached, EVERY generation path that
`test_probe_paths_discovered.py` discovers carries exactly one `kind: "behaviour"` result per
(behaviour, window), whose `members_fired` agrees with the member's own result on the same request,
and the builder runs once per request.

Driven through the discovered paths' own scenarios (the shared harness, `generation_entry_points`), on
the tiny real model where the path has a hook. The member probe carries a descriptor (behaviour key and
detector digest) and the definition is installed in the live verdict registry.

Controls: delete `behaviour_verdict.build_results(...)` in `_probe_publish_results` → every site red;
count members from every window instead of this one → `test_two_windows…` red.
"""

from __future__ import annotations

import dataclasses
from unittest.mock import patch

import pytest

from miserver.services import behaviour_verdict as bv
from miserver.services.probe_runtime import ProbeDescriptor, ProbeRuntimeState
from tests.support import verdict_factory as vf
from tests.support.generation_entry_points import SCENARIOS
from tests.unit.services import test_probe_paths_discovered as paths

pytestmark = pytest.mark.asyncio

KEY = "g19-path"
DIGEST = "d" * 64


@pytest.fixture
def harness(monkeypatch):
    log = None
    for log in paths.harness.__wrapped__(monkeypatch):
        break
    bv.forget_for_tests()
    yield log
    bv.forget_for_tests()
    ProbeRuntimeState.reset_for_tests()


def _member(windows=("all",)):
    probe = paths._probe()
    return dataclasses.replace(probe, windows=windows,
                               descriptor=ProbeDescriptor(behaviour_key=KEY, detector_digest=DIGEST))


def _build(log, scenario, monkeypatch, probe):
    real_probe = paths._probe
    monkeypatch.setattr(paths, "_probe", lambda d=16: probe)
    try:
        svc = paths._build(log, scenario, monkeypatch)
    finally:
        monkeypatch.setattr(paths, "_probe", real_probe)
    captured: list = []
    real_record = svc._probe_record

    async def record(context, verdicts=None, full_ids=None, **kwargs):
        out = await real_record(context, verdicts, full_ids=full_ids, **kwargs)
        if context is not None:
            captured.append(list(getattr(context, "results", None) or ()))
        return out

    svc._probe_record = record
    return svc, captured


def _count_builds(monkeypatch):
    calls = []
    real = bv.build_results

    def spy(verdicts, context, **kw):
        out = real(verdicts, context, **kw)
        calls.append(len(out))
        return out

    monkeypatch.setattr(bv, "build_results", spy)
    return calls


@pytest.mark.parametrize("site", sorted(SCENARIOS))
async def test_every_path_carries_one_verdict_per_window(site, harness, monkeypatch):
    bv.install([bv.attached_from(KEY, vf.definition(KEY, [DIGEST], candidate="any_of"))])
    calls = _count_builds(monkeypatch)
    scenario = SCENARIOS[site]
    svc, captured = _build(harness, scenario, monkeypatch, _member())
    with patch("miserver.services.inference_service.torch.cuda.is_available", return_value=False):
        await paths._drive(svc, scenario)
    assert len(captured) == 1, f"{site}: {len(captured)} records"
    results = captured[0]
    verdicts = [r for r in results if r.kind == "behaviour"]
    assert [(v.behaviour.key, v.window) for v in verdicts] == [(KEY, "all")], f"{site}: {verdicts}"
    assert calls == [1], f"{site}: the builder ran {calls}"
    (v,) = verdicts
    (member,) = [r for r in results if r.kind == "probe" and r.window == "all"]
    if member.outcome == "completed":
        assert v.outcome == "completed"
        fired = [m.probe_id for m in v.members_fired]
        assert fired == (["pr_guard"] if member.verdict else []), f"{site}: {fired} vs {member.verdict}"
        assert v.fires is bool(member.verdict)
    else:
        # a path with no hook says so on the verdict too: never a clean answer
        assert v.fires is None and v.decided_by == "not_evaluated", f"{site}: {v}"
        assert [m.detector_digest for m in v.members_missing] == [DIGEST]
    # it leads its behaviour's group
    assert results.index(v) == min(i for i, r in enumerate(results)
                                   if getattr(getattr(r, "behaviour", None), "key", None) == KEY)


async def test_no_definition_adds_nothing_on_any_path(harness, monkeypatch):
    """AC 4 at the builder: an armed behaviour with no definition gets no verdict and no build."""
    scenario = SCENARIOS["create_chat_completion"]
    svc, captured = _build(harness, scenario, monkeypatch, _member())
    with patch("miserver.services.inference_service.torch.cuda.is_available", return_value=False):
        await paths._drive(svc, scenario)
    assert [r.kind for r in captured[0]] == ["probe"]


async def test_two_windows_count_each_windows_member_verdict(harness, monkeypatch):
    """Members are read on THIS window: a probe reporting `all` and `prompt` gives one verdict per
    window, each from that window's member verdict (bars forced apart so the two differ)."""
    definition = vf.definition(KEY, [DIGEST], candidate="any_of", windows=("all", "prompt"))
    bv.install([bv.attached_from(KEY, definition)])
    member = dataclasses.replace(_member(windows=("all", "prompt")), threshold=None,
                                 window_thresholds={"all": -1e9, "prompt": 1e9})
    scenario = SCENARIOS["create_chat_completion"]
    svc, captured = _build(harness, scenario, monkeypatch, member)
    with patch("miserver.services.inference_service.torch.cuda.is_available", return_value=False):
        await paths._drive(svc, scenario)
    results = captured[0]
    by_window = {r.window: r for r in results if r.kind == "behaviour"}
    members = {r.window: r for r in results if r.kind == "probe"}
    assert set(by_window) == {"all", "prompt"}
    assert members["all"].verdict is True and members["prompt"].verdict is False
    assert by_window["all"].fires is True and by_window["prompt"].fires is False
