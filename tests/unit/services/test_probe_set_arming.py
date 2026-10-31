"""Arming a behaviour's set, all or nothing (039 FR-39.2, FR-39.4 – FR-39.7, FR-39.9; D14, TD39-7 – 9).

Real SQLite, the real repositories and arming service, the tiny real Llama, and probe definitions
whose test vectors are computed on it per run (`tests/support/behaviour_factory.py`), so parity
runs FOR REAL on every probe. Only the cost timer is injected where a test is about the budget.

Mutation controls (recorded in `0xcc/reviews/039_implementation_controls_2026-10-24.md`):
* `check` sets `armed=True`                         → `test_check_writes_only_evidence` red
* rollback leaves the first probe armed              → `test_one_identity_failure_arms_none` red
* `commit` skips the undo                            → `test_a_failed_write_removes_every_hook` red
* `commit` skips the pin re-read                     → `test_a_model_swap_between_check_and_commit` red
* the count gate reverts to `count_armed()` (rows)   → `test_the_count_gate_counts_behaviours` red
* one probe's cost dropped from the projection       → `test_the_whole_set_is_costed` red
* the (layer, device) dedup off in `probe_cost.project` → `test_a_shared_layer_is_charged_once` red
"""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from millm.core.config import settings
from millm.core.errors import (
    ProbeBudgetError,
    ProbeLimitError,
    ProbeModelChangedError,
    ProbeModelMismatchError,
    UnvalidatedProbeError,
)
from millm.db.repositories.probe_repository import ProbeRepository
from millm.services import probe_cost, probe_selection
from millm.services.probe_arm_bridge import build_parity_forward, build_probe_encoder
from millm.services.probe_arming import (
    INVOLUNTARY_REASONS,
    UNLOAD_REASON,
    ProbeArmingService,
    mark_armed_rows_disarmed,
)
from millm.services.probe_limits import global_limits
from millm.services.probe_runtime import ProbeRuntimeState
from tests.support import behaviour_factory as bf
from tests.unit.f25_fixtures import clear_loaded, make_service, word_model, word_tokenizer
from tests.unit.score_fixtures import tiny_identity

pytestmark = pytest.mark.asyncio


async def run_inline(fn):
    return fn()


@pytest.fixture
def model():
    ProbeRuntimeState.reset_for_tests()
    probe_selection.forget_for_tests()
    m = word_model()
    make_service(m, word_tokenizer())  # sets the resident model (the pin reads it)
    yield m
    ProbeRuntimeState.reset_for_tests()
    probe_selection.forget_for_tests()
    clear_loaded()


@pytest.fixture
def ctx(test_session, model):
    repo = ProbeRepository(test_session)
    return repo, ProbeArmingService(repo, ProbeRuntimeState()), model


def _kwargs(model, **over):
    base = dict(
        loaded=tiny_identity(), executor=run_inline, tokenizer=None,
        acknowledge_below_rung2=False, reason="", windows=None,
        forward_for=lambda p: build_parity_forward(model, p.layer),
        encoder_for=build_probe_encoder,
    )
    base.update(over)
    return base


async def _behaviour(session, key, n, *, layers=(0, 1, 1), thresholds=(0.0, 0.5, -0.5),
                     version=1, **over):
    rows = []
    for i in range(n):
        rows.append(await bf.import_probe(session, bf.definition(
            f"{key}-p{i}", key=key, version=version, layer=layers[i % len(layers)],
            threshold=thresholds[i % len(thresholds)], scale=0.2 + 0.07 * i, **over)))
    return rows


async def _select(repo, key, **kw):
    return await probe_selection.select_behaviour(repo, key, loaded=tiny_identity(), **kw)


def _snapshot(state):
    return (sorted(p.probe_id for p in state.armed()), sorted(state._handles))


async def test_a_three_probe_set_arms_with_its_own_thresholds_and_windows(ctx, test_session):
    repo, arming, model = ctx
    key, other = bf.random_key(), bf.random_key()
    rows = await _behaviour(test_session, key, 3)
    stranger = (await _behaviour(test_session, other, 1))[0]
    result = await arming.arm_set(await _select(repo, key), limits=global_limits(), model=model,
                                  **_kwargs(model))
    state = ProbeRuntimeState()
    assert sorted(result["armed"]) == sorted(r.id for r in rows)
    for row in rows:
        live = state.get(row.id)
        assert live is not None and live.threshold == row.threshold
        assert live.windows == ("all", "prompt", "response")
        assert (await repo.get(row.id)).armed is True
    assert state.get(stranger.id) is None, "a probe of another behaviour was armed"
    assert {p.threshold for p in state.armed()} == {0.0, 0.5, -0.5}


async def test_check_writes_only_evidence(ctx, test_session):
    """`check` writes the parity report and `last_cost`, and NOTHING that says armed (TD39-8)."""
    repo, arming, model = ctx
    row = (await _behaviour(test_session, bf.random_key(), 1))[0]
    before = _snapshot(ProbeRuntimeState())
    plan = await arming.check(row, model=model, forward=build_parity_forward(model, row.layer),
                              loaded=tiny_identity(), executor=run_inline)
    stored = await repo.get(row.id)
    assert _snapshot(ProbeRuntimeState()) == before
    assert stored.armed is False and stored.paused_reason is None
    assert stored.arm_acknowledgement is None and stored.cost is None
    assert stored.parity and stored.parity["passed"] is True
    assert stored.last_cost and stored.last_cost["fingerprint"]["repo_id"] == "tiny/llama"
    assert plan.armed.cost is not None


async def test_one_identity_failure_arms_none(ctx, test_session):
    """D14: one probe of the set fails identity → NONE armed: registry, rows and hook handles
    are exactly as before the call."""
    repo, arming, model = ctx
    key = bf.random_key()
    rows = await _behaviour(test_session, key, 2)
    wrong = await bf.import_probe(test_session, bf.definition(
        f"{key}-wrong", key=key, layer=1, hf_id="other/model"))
    selection = dataclasses.replace(await _select(repo, key), probes=tuple(rows + [wrong]))
    before = _snapshot(ProbeRuntimeState())
    with pytest.raises(ProbeModelMismatchError) as refused:
        await arming.arm_set(selection, limits=global_limits(), model=model, **_kwargs(model))
    assert _snapshot(ProbeRuntimeState()) == before
    stored = [await repo.get(r.id) for r in rows + [wrong]]
    assert not any(row.armed for row in stored)
    outcomes = {o["probe_id"]: o for o in refused.value.details["probes"]}
    assert outcomes[wrong.id]["outcome"] == "refused" and outcomes[wrong.id]["gate"] == "identity"
    assert all(outcomes[r.id]["outcome"] == "passed" for r in rows)


async def test_a_failed_write_removes_every_hook(ctx, test_session, monkeypatch):
    """D14: the DB write raises after the hooks are installed → every hook is removed."""
    repo, arming, model = ctx
    key = bf.random_key()
    await _behaviour(test_session, key, 3)
    before = _snapshot(ProbeRuntimeState())

    async def boom(_updates):
        raise RuntimeError("the database went away")

    monkeypatch.setattr(repo, "update_many", boom)
    with pytest.raises(RuntimeError):
        await arming.arm_set(await _select(repo, key), limits=global_limits(), model=model,
                             **_kwargs(model))
    assert _snapshot(ProbeRuntimeState()) == before, "a hook outlived the failed commit"


async def test_a_model_swap_between_check_and_commit(ctx, test_session, monkeypatch):
    """TD39-9: the pin taken before the first gate is re-read under the lock."""
    repo, arming, model = ctx
    key = bf.random_key()
    await _behaviour(test_session, key, 2)
    real_check = arming.check
    calls = []

    async def swapping_check(probe, **kw):
        plan = await real_check(probe, **kw)
        if not calls:
            make_service(model, word_tokenizer())  # a reload: a new `loaded_at`
            from millm.ml.model_loader import LoadedModelState
            from datetime import datetime

            LoadedModelState().current.loaded_at = datetime(2026, 10, 24, 9, 0)
        calls.append(probe.id)
        return plan

    monkeypatch.setattr(arming, "check", swapping_check)
    with pytest.raises(ProbeModelChangedError) as refused:
        await arming.arm_set(await _select(repo, key), limits=global_limits(), model=model,
                             **_kwargs(model))
    assert refused.value.details["pin_before"] != refused.value.details["pin_after"]
    assert ProbeRuntimeState().armed() == []


async def test_two_concurrent_set_arms_at_a_limit_of_one(ctx, test_session):
    """The arm lock (TD39-9): two concurrent set arms against a limit only one fits → exactly one.
    D83: the limit is the probe backstop (two 2-probe sets, `max_armed` 2), not a behaviour count."""
    from sqlalchemy.ext.asyncio import AsyncSession

    repo, _arming, model = ctx
    a, b = bf.random_key(), bf.random_key()
    await _behaviour(test_session, a, 2)
    await _behaviour(test_session, b, 2)
    limits = dataclasses.replace(global_limits(), max_armed=2)
    # Two requests: two sessions, two services, one registry (as two concurrent route calls).
    engine = test_session.bind
    async with AsyncSession(engine, expire_on_commit=False) as s1, \
            AsyncSession(engine, expire_on_commit=False) as s2:
        r1, r2 = ProbeRepository(s1), ProbeRepository(s2)
        sel_a, sel_b = await _select(r1, a), await _select(r2, b)
        outcomes = await asyncio.gather(
            ProbeArmingService(r1).arm_set(sel_a, limits=limits, model=model, **_kwargs(model)),
            ProbeArmingService(r2).arm_set(sel_b, limits=limits, model=model, **_kwargs(model)),
            return_exceptions=True,
        )
    assert sum(isinstance(o, dict) for o in outcomes) == 1, outcomes
    assert sum(isinstance(o, ProbeLimitError) for o in outcomes) == 1, outcomes


async def test_behaviours_are_not_counted(ctx, test_session):
    """ENH-006 D83: three 9-probe sets arm when their cost fits — the old count (2 behaviours)
    refused the third with BEHAVIOUR_LIMIT. The cost gate still charges all 27 probes (opened
    wide here; it has its own tests)."""
    repo, arming, model = ctx
    limits = dataclasses.replace(global_limits(), decode_ms_per_pass=10.0,
                                 prefill_ms_per_1k=10.0)
    keys = [bf.random_key() for _ in range(3)]
    for key in keys:
        await _behaviour(test_session, key, 9)
        await arming.arm_set(await _select(repo, key), limits=limits, model=model,
                             **_kwargs(model))
    assert len(ProbeRuntimeState().armed()) == 27


async def test_a_stored_behaviour_limit_binds_nothing(ctx, test_session):
    """The 8B's approval of 2026-10-29 stored `max_behaviours` 2; resolved, it limits nothing."""
    from types import SimpleNamespace

    from millm.services.probe_limits import effective

    repo, arming, model = ctx
    row = SimpleNamespace(max_armed=None, decode_budget_value=10.0, decode_budget_unit="ms",
                          prefill_budget_value=10.0, prefill_budget_unit="ms",
                          max_behaviours=1, source="operator")
    limits = effective(row, None)
    for _ in range(3):
        key = bf.random_key()
        await _behaviour(test_session, key, 1)
        await arming.arm_set(await _select(repo, key), limits=limits, model=model,
                             **_kwargs(model))
    assert len(ProbeRuntimeState().armed()) == 3


def _fixed_costs(monkeypatch, own: float, fixed: float):
    def fake(armed, *, device, d_model, k, dtype, first_on_layer=True):
        return probe_cost.ProbeCost(decode_ms=own, prefill_ms_per_1k=own,
                                    layer_fixed_ms=fixed, layer_fixed_prefill_ms_per_1k=fixed,
                                    layer=armed.layer, device="cpu", k=k, measured_at="t")
    monkeypatch.setattr(probe_cost, "measure", fake)


async def test_the_whole_set_is_costed(ctx, test_session, monkeypatch):
    """FR-39.9: the first probe alone fits, the whole set does not → refused, every probe listed;
    nothing armed."""
    repo, arming, model = ctx
    _fixed_costs(monkeypatch, own=0.05, fixed=0.02)
    key = bf.random_key()
    rows = await _behaviour(test_session, key, 4, layers=(1,))
    # 0.25 × 0.9 = 0.225: one probe 0.07 fits; four 0.22 fit; five would not. Lower the budget so
    # the first fits and the set does not: 0.1 × 0.9 = 0.09 → one (0.07) fits, four (0.22) do not.
    limits = dataclasses.replace(global_limits(), decode_ms_per_pass=0.1, prefill_ms_per_1k=0.1)
    with pytest.raises(ProbeBudgetError) as refused:
        await arming.arm_set(await _select(repo, key), limits=limits, model=model,
                             **_kwargs(model))
    details = refused.value.details
    assert details["projected"]["decode"] == pytest.approx(4 * 0.05 + 0.02)
    assert sorted(p["probe_id"] for p in details["probes"]) == sorted(r.id for r in rows)
    assert ProbeRuntimeState().armed() == []


async def test_a_shared_layer_is_charged_once(ctx, test_session, monkeypatch):
    repo, arming, model = ctx
    _fixed_costs(monkeypatch, own=0.01, fixed=0.03)
    key = bf.random_key()
    await _behaviour(test_session, key, 3, layers=(1,))
    await arming.arm_set(await _select(repo, key), limits=global_limits(), model=model,
                         **_kwargs(model))
    assert ProbeRuntimeState().projected_cost()["decode"] == pytest.approx(3 * 0.01 + 0.03)


async def test_a_partial_set_rearms_only_its_remainder(ctx, test_session):
    """FR-39.4: a fourth probe imported into an armed three-probe set stays disarmed; re-arming
    arms only it and leaves the other three exactly as they were."""
    repo, arming, model = ctx
    key = bf.random_key()
    rows = await _behaviour(test_session, key, 3)
    await arming.arm_set(await _select(repo, key), limits=global_limits(), model=model,
                         **_kwargs(model))
    live_before = {r.id: ProbeRuntimeState().get(r.id) for r in rows}
    fourth = await bf.import_probe(test_session, bf.definition(f"{key}-p3", key=key, layer=0,
                                                               threshold=0.25, scale=0.9))
    assert ProbeRuntimeState().get(fourth.id) is None
    from millm.services import probe_behaviours

    summary = probe_behaviours.summarise(key, list((await _select(repo, key)).probes))
    assert summary["set"]["text"] == "partial: 3 of 4 armed"
    assert summary["set"]["missing_probe_ids"] == [fourth.id]
    result = await arming.arm_set(await _select(repo, key), limits=global_limits(),
                                  model=model, **_kwargs(model))
    assert result["armed"] == [fourth.id]
    assert sorted(result["already_armed"]) == sorted(r.id for r in rows)
    for row in rows:
        assert ProbeRuntimeState().get(row.id) is live_before[row.id], "an armed member was redone"


async def test_a_remainder_that_fails_parity_leaves_the_set_partial(ctx, test_session):
    repo, arming, model = ctx
    key = bf.random_key()
    rows = await _behaviour(test_session, key, 3)
    await arming.arm_set(await _select(repo, key), limits=global_limits(), model=model,
                         **_kwargs(model))
    await bf.import_probe(test_session, bf.definition(f"{key}-bad", key=key, layer=0,
                                                      break_parity=True))
    with pytest.raises(Exception) as refused:
        await arming.arm_set(await _select(repo, key), limits=global_limits(), model=model,
                             **_kwargs(model))
    assert getattr(refused.value, "code", None) == "PROBE_PARITY_FAILED"
    assert sorted(p.probe_id for p in ProbeRuntimeState().armed()) == sorted(r.id for r in rows)


async def test_dry_run_arms_nothing_and_reports_every_probe(ctx, test_session):
    """FR-39.6: one probe fails rung → its failure and the others' passes; nothing armed; the
    projection equals the one the real gate would compute (one function)."""
    repo, arming, model = ctx
    key = bf.random_key()
    rows = await _behaviour(test_session, key, 2)
    weak = await bf.import_probe(test_session, bf.definition(f"{key}-weak", key=key, layer=1,
                                                             rung=1, scale=0.6))
    before = _snapshot(ProbeRuntimeState())
    result = await arming.arm_set(await _select(repo, key), limits=global_limits(), model=model,
                                  dry_run=True, **_kwargs(model))
    assert _snapshot(ProbeRuntimeState()) == before
    stored = [await repo.get(r.id) for r in rows + [weak]]
    assert not any(row.armed for row in stored)
    outcomes = {o["probe_id"]: o for o in result["probes"]}
    assert outcomes[weak.id]["gate"] == "rung" and outcomes[weak.id]["outcome"] == "refused"
    assert all(outcomes[r.id]["outcome"] == "passed" for r in rows)
    assert result["would_refuse"]["code"] == UnvalidatedProbeError.code
    assert result["armed"] == []


async def test_disarming_one_behaviour_keeps_a_shared_layer_scoring(ctx, test_session):
    """FR-39.3: two behaviours share layer 1; disarming one leaves the other's hook and verdicts."""
    import torch

    from millm.services.probe_runtime import ProbeRequestContext

    repo, arming, model = ctx
    a, b = bf.random_key(), bf.random_key()
    await _behaviour(test_session, a, 1, layers=(1,))
    kept = (await _behaviour(test_session, b, 1, layers=(1,)))[0]
    for key in (a, b):
        await arming.arm_set(await _select(repo, key), limits=global_limits(), model=model,
                             **_kwargs(model))
    await arming.disarm_set(await _select(repo, a))
    state = ProbeRuntimeState()
    assert [p.probe_id for p in state.armed()] == [kept.id] and 1 in state._handles
    context = state.begin_request("req-live")
    try:
        with torch.inference_mode():
            model(input_ids=torch.tensor([bf.VECTOR_IDS]), use_cache=False)
    finally:
        state.end_request()
    verdicts = [v for v in context.finish() if v.window == "all"]
    assert {v.probe_id for v in verdicts} == {kept.id} and all(v.scored for v in verdicts)
    assert isinstance(context, ProbeRequestContext)


async def test_armed_before_unload_is_named_and_cleared_by_rearming(ctx, test_session):
    """FR-39.5 (TD39-12): two behaviours armed, the model unloaded → both named, nothing armed;
    re-arming one takes it off the list."""
    from millm.db.base import async_session_factory  # noqa: F401 - the real helper's import path
    from millm.services import probe_behaviours

    repo, arming, model = ctx
    a, b = bf.random_key(), bf.random_key()
    for key in (a, b):
        await _behaviour(test_session, key, 2)
        await arming.arm_set(await _select(repo, key), limits=global_limits(), model=model,
                             **_kwargs(model))
    ProbeRuntimeState().disarm_all("unload")

    async def factory():
        return test_session

    class _Factory:
        def __call__(self):
            return self

        async def __aenter__(self):
            return test_session

        async def __aexit__(self, *exc):
            return False

    await mark_armed_rows_disarmed(_Factory(), UNLOAD_REASON, event="probes_disarmed_on_unload")
    assert UNLOAD_REASON in INVOLUNTARY_REASONS
    summary = await probe_behaviours.armed_summary(test_session)
    assert summary["behaviours"] == []
    assert sorted(s["key"] for s in summary["armed_before_unload"]) == sorted([a, b])
    await arming.arm_set(await _select(repo, a), limits=global_limits(), model=model,
                         **_kwargs(model))
    summary = await probe_behaviours.armed_summary(test_session)
    assert [s["key"] for s in summary["armed_before_unload"]] == [b]
    assert [s["key"] for s in summary["behaviours"]] == [a]


async def test_a_disarm_waits_for_an_arm_in_flight(ctx, test_session):
    """The arm lock (TD39-9) keeps the ROWS and the REGISTRY in agreement when an arm and a disarm
    of one behaviour race. The commit installs the hooks, then awaits its one transaction; a
    disarm that ran in that gap would remove the hooks and write `armed=False`, and the arm's
    transaction would then commit `armed=True` over it — rows claiming armed with nothing hooked.

    Control S1-08 (the lock removed) SURVIVED the count-gate test alone (no await sits between
    that gate and the registry write); this test is its regression, re-run as a negative control.
    """
    from sqlalchemy.ext.asyncio import AsyncSession

    _repo, _arming, model = ctx
    key = bf.random_key()
    rows = await _behaviour(test_session, key, 2)
    engine = test_session.bind
    in_commit = asyncio.Event()
    async with AsyncSession(engine, expire_on_commit=False) as s1, \
            AsyncSession(engine, expire_on_commit=False) as s2:
        r1, r2 = ProbeRepository(s1), ProbeRepository(s2)
        real = r1.update_many

        async def slow_update_many(updates):
            in_commit.set()  # the hooks are installed; the transaction is about to run
            await asyncio.sleep(0.05)
            await real(updates)

        r1.update_many = slow_update_many
        arming_a, arming_b = ProbeArmingService(r1), ProbeArmingService(r2)
        sel_a, sel_b = await _select(r1, key), await _select(r2, key)

        async def disarm_when_committing():
            await in_commit.wait()
            await arming_b.disarm_set(sel_b)

        await asyncio.gather(
            arming_a.arm_set(sel_a, limits=global_limits(), model=model, **_kwargs(model)),
            disarm_when_committing(),
        )
    stored = {r.id: (await ProbeRepository(test_session).get(r.id)) for r in rows}
    for row in rows:
        await test_session.refresh(stored[row.id])
        in_registry = ProbeRuntimeState().get(row.id) is not None
        assert stored[row.id].armed == in_registry, (
            f"{row.id}: row armed={stored[row.id].armed} but registry armed={in_registry}"
        )
