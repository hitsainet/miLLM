"""`X-miLLM-Probe-Verdicts` is bounded, grouped and valid RFC 8941 (039 FR-39.12, TD39-5, TD39-30;
FTASKS 4.3, 4.4), parsed by an independent structured-field parser (`http-sfv`, a dev dependency).

Controls (039 controls record):
* the bound never truncates                      → `test_forty_probes_three_windows_stay_within_the_bound` red
* truncation cuts mid-group                      → `test_whole_behaviour_groups_or_none` red
* the truncation member is `"…"` (U+2026)        → `test_forty_probes_three_windows_stay_within_the_bound` red (no parse)
* `sf_string` stops escaping `\\`                 → `test_a_backslash_and_a_non_ascii_name_parse` red
* the served header ignores the setting          → `test_the_served_header_obeys_the_configured_bound` red
  (review R1, D43: the bound is raised later by one settings change, so it must be the ONE input)
"""

from __future__ import annotations

from types import SimpleNamespace

import http_sfv
import pytest
import torch

from millm.api.routes.openai.chat import build_probe_verdicts_header
from millm.core.config import settings
from millm.ml.probe_head import ProbeHead
from millm.services.probe_result import build_probe_results
from millm.services.probe_result_order import order_results
from millm.services.probe_runtime import ArmedProbe, ProbeDescriptor, Verdict
from tests.support import behaviour_factory as bf
from tests.support import capture_pre038_goldens as cap

WINDOWS = ("all", "prompt", "response")


def _results(n_probes, per_behaviour=5, name=lambda i: f"probe-{i:03d}"):
    keys = [bf.random_key() for _ in range((n_probes + per_behaviour - 1) // per_behaviour)]
    probes = [ArmedProbe(
        probe_id=f"pr_{i:012x}", name=name(i), head=ProbeHead(weight=torch.ones(2)), rule="mean",
        scope="all", layer=0, rung=2, rung_language="detects on unseen tasks", threshold=0.5,
        descriptor=ProbeDescriptor(behaviour_key=keys[i // per_behaviour], behaviour_version=1),
    ) for i in range(n_probes)]
    verdicts = [Verdict(probe_id=p.probe_id, name=p.name, rung=2,
                        rung_language="detects on unseen tasks", scored=True,
                        score=0.123456 * (i + 1), threshold=0.5, fires=i % 2 == 0,
                        n_scored_tokens=4, window=w)
                for i, p in enumerate(probes) for w in WINDOWS]
    context = SimpleNamespace(request_id="req-bound", probes=probes, session=None, trace=None,
                              interrupted=None)
    return order_results(build_probe_results(verdicts, context), probes)


def _parse(header):
    parsed = http_sfv.List()
    parsed.parse(header.encode("ascii"))
    return parsed


def test_forty_probes_three_windows_stay_within_the_bound():
    results = _results(40)
    assert len(results) == 120
    header = build_probe_verdicts_header(results, 2048)
    assert len(header.encode("ascii")) <= 2048
    parsed = _parse(header)
    tail = parsed[-1]
    assert tail.value == "..." and tail.params["truncated"] is True
    assert tail.params["total"] == 120


def test_whole_behaviour_groups_or_none():
    results = _results(40)
    parsed = list(_parse(build_probe_verdicts_header(results, 2048)))[:-1]
    kept: dict[str, int] = {}
    for member in parsed:
        kept[member.params["behaviour"]] = kept.get(member.params["behaviour"], 0) + 1
    assert kept and all(n == 5 * len(WINDOWS) for n in kept.values()), kept


def test_under_the_bound_there_is_no_truncation_member():
    results = _results(2)
    header = build_probe_verdicts_header(results, 2048)
    parsed = _parse(header)
    assert len(parsed) == 6 and all(m.value != "..." for m in parsed)
    for member, result in zip(parsed, results, strict=True):
        assert member.params["probe"] == result.probe_id
        assert member.params["behaviour"] == result.behaviour.key


def test_a_backslash_and_a_non_ascii_name_parse():
    results = _results(1, name=lambda i: 'C:\\probe "\u00e9"')
    parsed = _parse(build_probe_verdicts_header(results, 2048))
    assert parsed[0].value == 'C:\\probe "?"'
    assert parsed[0].params["probe"] == results[0].probe_id


def test_scores_are_rfc8941_decimals():
    """Pre-039 defect fixed: `:.6g` wrote 7 fractional digits, which no RFC 8941 parser accepts."""
    member = _parse(build_probe_verdicts_header(_results(1), 2048))[0]
    assert float(member.params["score"]) == round(0.123456, 3)


def test_the_bound_is_the_setting_read_at_request_time(monkeypatch):
    """`PROBE_VERDICT_HEADER_MAX_BYTES` is what the header builder obeys by default (a runtime
    setting: read on every call, never captured)."""
    from millm.core.config import settings

    results = _results(10)
    monkeypatch.setattr(settings, "PROBE_VERDICT_HEADER_MAX_BYTES", 600)
    header = build_probe_verdicts_header(results)
    assert len(header.encode("ascii")) <= 600
    assert _parse(header)[-1].params["total"] == 30


# ── 039 review R1: the served header obeys the ONE setting (D43, X-18) ─────────────────────────
#
# The operator chose (D43, 2026-10-24) to RAISE the header bound rather than compact rule members,
# at a value set from the deployed chain's measurement (038 task 5.5). That is only a one-line
# settings change if the served response reads `PROBE_VERDICT_HEADER_MAX_BYTES` — the runtime
# setting the operator writes through `PUT /api/environment/{key}` — and nothing else. This drives
# the REAL chat route at a small and at the maximum value. Control (R1-15b): pin the provenance
# caller to 2,048 → red.

KEY_SETTING = "PROBE_VERDICT_HEADER_MAX_BYTES"


@pytest.mark.asyncio
async def test_the_served_header_obeys_the_configured_bound():
    original = settings.PROBE_VERDICT_HEADER_MAX_BYTES
    async with cap.Harness() as h:
        key = bf.random_key()
        async with h.factory() as session:
            for i in range(6):
                await bf.import_probe(session, bf.definition(f"{key}-h{i}", key=key, layer=i % 2,
                                                             scale=0.11 + 0.1 * i))
        assert h.client.post(f"/api/probes/behaviours/{key}/arm", json={}).status_code == 200
        try:
            sizes = {}
            for value in (256, 65536):
                r = h.client.put(f"/api/environment/{KEY_SETTING}",
                                 json={"value": value, "reason": "review R1 bound test"})
                assert r.status_code == 200, r.text
                response = h.post_chat(dict(cap.BODY), {})
                header = response.headers.get("x-millm-probe-verdicts", "")
                body = response.json()["millm_probe_verdicts"]
                members = _parse(header)
                sizes[value] = (len(header.encode("ascii")), members, len(body))
            small, members_small, n_body = sizes[256]
            assert small <= 256
            assert members_small[-1].params["total"] == n_body  # truncated, and says so
            large, members_large, n_body_large = sizes[65536]
            assert large > 256 and large <= 65536
            assert len(members_large) == n_body_large  # the whole list, no truncation member
            assert all("truncated" not in m.params for m in members_large)
        finally:
            h.client.delete(f"/api/environment/{KEY_SETTING}")
            settings.PROBE_VERDICT_HEADER_MAX_BYTES = original
