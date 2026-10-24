"""A per-request dial changes nothing while several circuits serve (D19 hardware run, 2026-10-16).

On the 3090 with two disjoint circuits serving `full` (Qwen2.5-7B, L20+L25 and L4), `"off"` came
back `X-miLLM-Steering-Intensity: 0` on a response `X-miLLM-Steering` reported both circuits
steering at 1.0, and `"max"` echoed 1.5 — the CLUSTER envelope. `_active_full_circuit` answers
None for "several" by design (F19 R3-06, whose log line says the dial is "suppressed"), and both
the echo and the apply then fell through to the single-SAE profile path, which dialled whichever
SAE `attached_sae` returned. Real SAEs hooked on a real tiny model, real circuit rows in SQLite.
"""

from __future__ import annotations

import pytest

from millm.db.models.circuit import Circuit
from tests.unit.f28_fixtures import SAE_A, SAE_B, build, chat, clean_state, db  # noqa: F401


async def _add_circuits(factory, n_active: int) -> None:
    async with factory() as session:
        for i in range(2):
            session.add(Circuit(
                id=f"circ_{i}", name=f"c{i}", description=None, rung=0, edge_count=0,
                layers=[i], serveable=True, is_active=i < n_active, serving_mode="full",
                intensity=1.0, per_sae_warnings=[], circuit_meta={"budget": {}},
            ))
        await session.commit()


@pytest.fixture
def served(clean_state):  # noqa: F811
    served = build([(SAE_A, 0, 11), (SAE_B, 1, 12)])
    for (sae_id, layer), sae in served.saes.items():
        sae.set_steering_batch({3: 5.0 + layer})
        sae.enable_steering(True)
    yield served
    for h in served.handles:
        h.remove()


@pytest.mark.parametrize("dial", ["off", "max", 0.5])
async def test_two_serving_circuits_leave_steering_untouched(served, db, dial):  # noqa: F811
    await _add_circuits(db, n_active=2)
    before = served.state()
    saved = await served.svc._apply_request_steering(None, dial)
    assert saved is None
    assert served.state() == before


@pytest.mark.parametrize("dial", ["off", "max", 0.5])
async def test_two_serving_circuits_echo_no_intensity(served, db, dial):  # noqa: F811
    await _add_circuits(db, n_active=2)
    assert await served.svc.resolve_request_intensity(chat(steering_intensity=dial)) is None


async def test_control_without_circuits_the_dial_still_applies(served, db):  # noqa: F811
    """The control the two tests above need: with no circuit serving, the same call DOES dial the
    live values (Feature 10), so an untouched state above is the guard, not the fixture."""
    await _add_circuits(db, n_active=0)
    before = served.state()
    saved = await served.svc._apply_request_steering(None, "off")
    assert saved is not None
    assert served.state() != before
    served.svc._restore_request_profile(saved)
    assert await served.svc.resolve_request_intensity(chat(steering_intensity="off")) == 0.0
