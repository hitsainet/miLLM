"""One flow, any behaviour (039 FR-39.16, FR-39.32; FTASKS 8.1).

The same route flow — import, arm the set through the route, one real chat, read every surface —
runs over three behaviour keys with NO branch on which one it is: a key generated per run, the
pilot's first key and the second behaviour's key, both READ from `tests/fixtures/behaviour_packs/`
(the files 038's generality guard derives its denylist from, so no key is typed here). A key swap
arms the pilot and the second behaviour together and requires every result under its own key.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.support import behaviour_factory as bf
from tests.support import capture_pre038_goldens as cap

pytestmark = pytest.mark.asyncio
PACKS = Path(__file__).resolve().parents[1] / "fixtures" / "behaviour_packs"


def _pack_key(pack: str, index: int) -> str:
    return json.loads((PACKS / f"{pack}.json").read_text())["behaviours"][index]["key"]


KEYS = {
    "generated": bf.random_key(),
    "pilot": _pack_key("pilot", 1),
    "second": _pack_key("seeded", 0),
}


async def _arm(h, key, n=2, start=0):
    async with h.factory() as session:
        rows = [await bf.import_probe(session, bf.definition(
            f"{key}-gen-{i}", key=key, layer=i % 2, scale=0.13 + 0.2 * i))
            for i in range(start, start + n)]
    response = h.client.post(f"/api/probes/behaviours/{key}/arm", json={})
    assert response.status_code == 200, response.text
    return rows


@pytest.mark.parametrize("which", sorted(KEYS))
async def test_the_flow_is_the_same_for_any_behaviour(which):
    key = KEYS[which]
    async with cap.Harness() as h:
        rows = await _arm(h, key)
        response = h.post_chat(cap.BODY, {})
        assert response.status_code == 200, response.text
        body = response.json()["millm_probe_verdicts"]
        assert {e["probe_id"] for e in body} == {r.id for r in rows}
        assert {e["behaviour"]["key"] for e in body} == {key}
        assert all(e["set"]["complete"] for e in body)
        listed = h.client.get("/api/probes/behaviours", params={"key": key}).json()["data"]
        assert [b["key"] for b in listed] == [key] and listed[0]["armed"] == len(rows)


async def test_two_named_behaviours_never_swap_results():
    pilot, second = KEYS["pilot"], KEYS["second"]
    async with cap.Harness() as h:
        p_rows = await _arm(h, pilot)
        s_rows = await _arm(h, second, start=2)
        response = h.post_chat(cap.BODY, {})
        owner = {**{r.id: pilot for r in p_rows}, **{r.id: second for r in s_rows}}
        for entry in response.json()["millm_probe_verdicts"]:
            assert entry["behaviour"]["key"] == owner[entry["probe_id"]]
