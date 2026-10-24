"""Every surface is a projection of ONE finalized result (FR-38.3, FR-38.4; FTASKS 3.4 – 3.9).

Through the real app (`create_app`), a tiny real Llama with two probes armed on two layers (one
with a threshold, one ranking without; three windows between them), a real SQLite database and the
socket captured at `progress_emitter.emit_probe_event` — the pre-038 goldens' own harness
(`tests/support/capture_pre038_goldens.py`), so a golden is compared with the identical request.

* **Agreement.** The header, the terminal chunk, the body field, the stored row, the socket emit,
  the events list, the event detail, the session cell, the score response and a Batch line carry
  the same `result_id` per probe × window and equal values on every shared field, and every
  object validates against the strict schema.
* **Sentinel (wiring).** The builder is wrapped to stamp `SENTINEL-<n>` (one `n` per build) into
  the probe's name; each surface must show the stamp of ITS request's one build. Controls
  M3.1 – M3.10 re-point one surface at a time to its old serializer; each turns one test red.
* **Goldens.** Every key a surface carried before 038 is still there with the same value, except
  the documented changes listed in `ALLOWED`.
* **Body field (FR-38.4).** `millm_probe_verdicts` equals the stream chunk's list for the same
  request; an unarmed body is byte-identical to its golden; Batch lines carry it; a packed
  scoring row never carries another row's results (X-07). Control M4: delete the assignment.
* **Reset (X-11, FTASKS 2.9).** Two Batch rows in one task, the first armed and the second not:
  the second body has no field. Mutation: delete the reset in `reset_steering_memo`.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

import jsonschema
import pytest

from millm.api.schemas.probe_result import ProbeResult, strict_schema
from tests.support import capture_pre038_goldens as cap

from tests.support.telemetry_mode import _stub_exports, telemetry_mode  # noqa: E402,F401 - fixtures

# 038 FR-38.23: every test here runs with export off AND with every signal on against the stub.
pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("telemetry_mode")]
STRICT = jsonschema.Draft202012Validator(strict_schema())
CONTRACT = {f.alias or name for name, f in ProbeResult.model_fields.items()}
LEGACY_ONLY = {"name", "provisional", "not_scored_reason"}


def contract_part(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if k in CONTRACT}


def finalized(payload: dict) -> dict:
    return {k: v for k, v in contract_part(payload).items()
            if k not in ("persistence", "source_event_id")}


def _header_members(header: str) -> list[dict]:
    out = []
    for member in header.split(", "):
        name, *params = member.split(";")
        entry = {"name": name.strip('"')}
        for p in params:
            key, _, value = p.partition("=")
            entry[key] = value.strip('"') if value else True
        out.append(entry)
    return out


async def _armed(h: cap.Harness, headers=None) -> dict[str, Any]:
    """One non-streaming armed request, every surface it reaches, unnormalised."""
    rows = await h.arm([cap.ALPHA, cap.BETA])
    response = h.post_chat(cap.BODY, headers or {"X-miLLM-Session": "agree"})
    assert response.status_code == 200, response.text
    rid = response.json()["id"]
    listed = h.client.get("/api/probes/events", params={"request_id": rid}).json()["data"]
    session_id = response.headers.get("X-miLLM-Session-Id")
    detail = h.client.get(f"/api/probes/sessions/{session_id}").json()["data"]
    return {
        "rows": rows, "response": response, "body": response.json(),
        "header": response.headers.get("X-miLLM-Probe-Verdicts"),
        "event_rows": await h.event_rows(rid), "socket": list(h.socket), "list": listed,
        "detail": [h.client.get(f"/api/probes/events/{e['id']}").json()["data"] for e in listed],
        "cells": [c for t in detail["turns"] for c in t["verdicts"]],
    }


async def test_every_surface_agrees_and_validates():
    async with cap.Harness() as h:
        s = await _armed(h)
        body = s["body"]["millm_probe_verdicts"]
        assert len(body) == 3, "two probes, three windows"
        by_id = {e["result_id"]: finalized(e) for e in body}
        for entry in body:
            STRICT.validate(contract_part(entry))
            # the legacy keys are the one projection of the entry's own contract part
            from millm.services.probe_result import legacy_projection

            legacy = legacy_projection(contract_part(entry))
            for key in ("score", "threshold", "verdict", "window", "rung", "reason"):
                assert entry[key] == legacy[key], key
            assert entry["not_scored_reason"] == legacy["not_scored_reason"]
        for surface in ("socket", "list", "detail", "cells"):
            items = s[surface]
            assert {i["result_id"] for i in items} == set(by_id), surface
            for item in items:
                STRICT.validate(contract_part(item))
                assert contract_part(item)["persistence"] == "persisted"
                assert finalized(item) == by_id[item["result_id"]], surface
        for row in s["event_rows"]:
            stored = {**row["result"], "source_event_id": row["id"]}
            STRICT.validate(stored)
            assert finalized(stored) == by_id[row["result_id"]]
        members = _header_members(s["header"])
        assert sorted((m["name"], m["window"]) for m in members) == \
            sorted((e["name"], e["window"]) for e in body)
        for m in members:
            e = next(x for x in body if (x["name"], x["window"]) == (m["name"], m["window"]))
            # 039: the header's number is an RFC 8941 Decimal (3 fractional digits).
            assert math.isclose(float(m["score"]), e["score"], abs_tol=HEADER_DECIMAL_TOL)
            assert m["probe"] == e["probe_id"], "039: each member names its probe"
            assert "result_id" not in m, "the header carries no result_id (D26, OQ-11)"

        # Streaming: the terminal chunk is the same object as that request's stored rows.
        text = h.stream_chat(cap.BODY, {})
        chunk = cap._stream_chunk(text)["millm_probe_verdicts"]
        rid = next(json.loads(line[6:])["id"] for line in text.splitlines()
                   if line.startswith("data: {"))
        stored = {r["result_id"]: r for r in await h.event_rows(rid)}
        for entry in chunk:
            STRICT.validate(contract_part(entry))
            row = stored[entry["result_id"]]
            assert finalized({**row["result"]}) == finalized(entry)

        # Score route: stateless objects, never stored.
        scored = h.client.post("/api/probes/score", json={
            "probe_ids": [r.id for r in s["rows"]],
            "inputs": [{"token_ids": [0, 2, 6, 7, 8], "prompt_tokens": 3}], "windows": ["all"],
        }).json()["data"]
        for v in scored["results"][0]["verdicts"]:
            STRICT.validate(contract_part(v))
            assert v["origin"] == "stateless" and v["persistence"] == "not_persisted"
            assert v["input_index"] == 0 and "source_event_id" not in v

        # A Batch line: its body field is that row's stored objects (origin batch).
        line = await h.batch_line(cap.BODY)
        entries = line["response"]["body"]["millm_probe_verdicts"]
        rows = {r["result_id"]: r for r in await h.event_rows(line["response"]["body"]["id"])}
        assert set(rows) == {e["result_id"] for e in entries}
        for entry in entries:
            STRICT.validate(contract_part(entry))
            assert entry["origin"] == "batch" and entry["batch_line"] == 1
            assert finalized(rows[entry["result_id"]]["result"]) == finalized(entry)
        assert "X-miLLM-Probe-Verdicts" in line["response"]["millm"]["headers"]


# ── sentinel: each surface takes its object from the builder ─────────────────────────


@pytest.fixture
def stamped(monkeypatch):
    import millm.services.probe_result as pr

    real = pr.build_probe_results
    calls = {"n": 0}

    def stamp(*args, **kwargs):
        results = real(*args, **kwargs)
        calls["n"] += 1
        for r in results:
            r.probe_name = f"SENTINEL-{calls['n']}"
            r._legacy_name = f"SENTINEL-{calls['n']}"
        return results

    monkeypatch.setattr(pr, "build_probe_results", stamp)
    return calls


async def _stamp_of(text: Any) -> set[str]:
    return set(re.findall(r"SENTINEL-\d+", json.dumps(text)))


async def test_sentinel_reaches_header_body_row_socket_list_detail_cell(stamped):
    async with cap.Harness() as h:
        s = await _armed(h)
        one = await _stamp_of(s["body"]["millm_probe_verdicts"])
        assert len(one) == 1, one
        assert await _stamp_of(s["header"]) == one, "header"
        assert await _stamp_of([r["result"] for r in s["event_rows"]]) == one, "stored row"
        assert await _stamp_of(s["socket"]) == one, "socket"
        assert await _stamp_of(s["list"]) == one, "events list"
        assert await _stamp_of(s["detail"]) == one, "event detail"
        assert await _stamp_of(s["cells"]) == one, "session cell"


async def test_sentinel_reaches_the_stream_chunk(stamped):
    async with cap.Harness() as h:
        await h.arm([cap.ALPHA, cap.BETA])
        chunk = cap._stream_chunk(h.stream_chat(cap.BODY, {}))
        assert len(await _stamp_of(chunk)) == 1


async def test_sentinel_reaches_the_score_response(stamped):
    async with cap.Harness() as h:
        rows = await h.arm([cap.ALPHA])
        scored = h.client.post("/api/probes/score", json={
            "probe_ids": [r.id for r in rows], "inputs": [{"token_ids": [0, 2, 6]}],
            "windows": ["all"]}).json()["data"]
        assert len(await _stamp_of(scored["results"])) == 1


async def test_sentinel_reaches_the_batch_line(stamped):
    async with cap.Harness() as h:
        await h.arm([cap.ALPHA])
        line = await h.batch_line(cap.BODY)
        assert len(await _stamp_of(line["response"]["body"]["millm_probe_verdicts"])) == 1
        assert len(await _stamp_of(line["response"]["millm"]["headers"])) == 1


# ── the body field (FR-38.4) ───────────────────────────────────────────────────────


def _strip_volatile(entries: list[dict]) -> list[dict]:
    drop = {"result_id", "event_time", "completed_at", "evaluation_ms", "request_id",
            "session_id"}  # each request opens its own session
    return [{k: v for k, v in e.items() if k not in drop} for e in entries]


async def test_body_equals_chunk():
    async with cap.Harness() as h:
        await h.arm([cap.ALPHA, cap.BETA])
        body = h.post_chat(cap.BODY, {}).json()["millm_probe_verdicts"]
        chunk = cap._stream_chunk(h.stream_chat(cap.BODY, {}))["millm_probe_verdicts"]
        # 038 S2: the two requests' trace ids are their own (null with export off; minted per
        # request with it on). Within one request every result carries the same pair.
        ids = [{(e.get("trace_id"), e.get("span_id")) for e in entries} for entries in (body, chunk)]
        assert all(len(pair) == 1 for pair in ids), ids
        assert ids[0] == ids[1] == {(None, None)} or ids[0] != ids[1], ids
        per_request = {"trace_id", "span_id"}
        assert _strip_volatile([{k: v for k, v in e.items() if k not in per_request}
                                for e in body]) == \
            _strip_volatile([{k: v for k, v in e.items() if k not in per_request} for e in chunk])


async def test_the_unarmed_body_is_byte_identical_to_its_golden():
    golden = json.loads(cap.golden_path("unarmed").read_text())
    now = await cap.capture(next(s for s in cap.SCENARIOS if s.name == "unarmed"))
    assert now["body"] == golden["body"] and "millm_probe_verdicts" not in now["body"]
    assert now["stream_chunk"] == golden["stream_chunk"]
    assert now["header"] is None and now["batch_line"] == golden["batch_line"]


async def test_two_batch_rows_never_share_results():
    """X-11 / FTASKS 2.9: the second row (nothing armed) must not carry the first row's field."""
    from millm.services.probe_runtime import ProbeRuntimeState

    async with cap.Harness() as h:
        await h.arm([cap.ALPHA])
        from types import SimpleNamespace

        from millm.services.batch.executors import RowExecutor
        from tests.unit.f25_fixtures import model_row

        line = json.dumps({"custom_id": "c", "method": "POST", "url": "/v1/chat/completions",
                           "body": cap.BODY}).encode()

        async def read_line(_row):
            return line

        executor = RowExecutor(batch_id="batch_0123456789abcdef", endpoint="/v1/chat/completions",
                               inference=h.svc, model_row=model_row(), read_line=read_line,
                               session_factory=h.factory)
        first = await executor.run_one(SimpleNamespace(line_no=1, custom_id="c1", kind="chat"))
        ProbeRuntimeState().disarm_all("test")
        second = await executor.run_one(SimpleNamespace(line_no=2, custom_id="c2", kind="chat"))
        assert "millm_probe_verdicts" in first.result["response"]["body"]
        assert "millm_probe_verdicts" not in second.result["response"]["body"], (
            "row 2 carried row 1's results: the results ContextVar was not reset per row"
        )


async def test_a_packed_scoring_row_carries_none_of_another_rows_results():
    """X-07: packed scoring rows reach `finish_body` too; they carry a list or nothing."""
    from types import SimpleNamespace

    from millm.services.batch.executors import RowExecutor
    from millm.services.batch.packing import run_packed_scoring
    from tests.unit.f25_fixtures import model_row

    async with cap.Harness() as h:
        await h.arm([cap.ALPHA])
        h.post_chat(cap.BODY, {})  # a live request publishes results in this context first
        lines = {
            i: json.dumps({"custom_id": f"s{i}", "method": "POST", "url": "/v1/completions",
                           "body": {"model": "tiny", "prompt": "w1 w2", "max_tokens": 1,
                                    "logprobs": 1}}).encode()
            for i in (1, 2)
        }

        async def read_line(row):
            return lines[row.line_no]

        executor = RowExecutor(batch_id="batch_0123456789abcdee", endpoint="/v1/completions",
                               inference=h.svc, model_row=model_row(), read_line=read_line,
                               session_factory=h.factory)
        rows = [SimpleNamespace(line_no=i, custom_id=f"s{i}", kind="scoring") for i in (1, 2)]
        results = await run_packed_scoring(executor, rows)
        assert results, "the packed path did not run; this test would check nothing"
        assert all(r.result["response"]["status_code"] == 200 for r in results), results
        for r in results:
            body = r.result["response"]["body"]
            assert body.get("millm_probe_verdicts") in (None, []) or all(
                e["batch_line"] == r.line_no for e in body["millm_probe_verdicts"]
            )


# ── goldens (FR-38.3) ─────────────────────────────────────────────────────────────

#: FR-38.5: the three free-text reason forms are codes; the detail is in the local log. Applied
#: to the GOLDEN before comparing, so the rest of each value must still match exactly (review R1:
#: allowing any change to a path ending in `header` let a whole header drift unnoticed).
_FREE_TEXT_REASON = re.compile(
    r"\b(encoder_failed|prompt_boundary_invalid|prompt_boundary_conflict): [^\";]*")


def _codes_only(value: Any) -> Any:
    if isinstance(value, str):
        return _FREE_TEXT_REASON.sub(r"\1", value)
    if isinstance(value, dict):
        return {k: _codes_only(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_codes_only(v) for v in value]
    return value


#: Documented changes from the pre-038 surfaces: (scenario, path regex, why).
ALLOWED = [
    (r".*", r"^events_list\[\d+\]\.behaviour\.name$",
     "FTASKS 4.9: the events list's behaviour is the stored result's {key, version}; today's "
     "catalogue name is no longer joined into it"),
    (r".*", r"\.set$",
     "039 FR-39.10: a result of a behaviour carries its `set` block (038 sent `set: null`)"),
]


#: 039 (FR-39.10, FR-39.12): each header member gains `behaviour="<key>"` and `probe="<id>"`
#: parameters, and its numbers are RFC 8941 Decimals (3 fractional digits; the old `:.6g` form did
#: not parse). Applied to the NOW header, so every other byte must still match; numbers then
#: compare at half a unit of the third decimal.
_039_PARAMS = re.compile(r';(?:behaviour|probe)="(?:[^"\\]|\\.)*"')
HEADER_DECIMAL_TOL = 5e-4


def _pre039_header(value: Any) -> Any:
    return _039_PARAMS.sub("", value) if isinstance(value, str) else value


def _allowed(scenario: str, path: str) -> bool:
    return any(re.fullmatch(s, scenario) and re.search(p, path) for s, p, _why in ALLOWED)


def _compare(golden: Any, now: Any, path: str, out: list) -> None:
    if isinstance(golden, dict):
        if not isinstance(now, dict):
            out.append((path, "not an object"))
            return
        for key, value in golden.items():
            if key not in now:
                out.append((f"{path}.{key}", "missing"))
            else:
                _compare(value, now[key], f"{path}.{key}", out)
    elif isinstance(golden, list):
        if not isinstance(now, list) or len(golden) != len(now):
            out.append((path, "length"))
            return
        for i, (a, b) in enumerate(zip(golden, now, strict=True)):
            _compare(a, b, f"{path}[{i}]", out)
    elif isinstance(golden, float) and isinstance(now, float):
        if not _close(golden, now):
            out.append((path, f"{golden} != {now}"))
    elif isinstance(golden, str) and isinstance(now, str) and (
            path.endswith("header") or path.endswith("X-miLLM-Probe-Verdicts")):
        # 039: the header's numbers are RFC 8941 Decimals now (3 fractional digits), so they are
        # compared at half a unit of the third decimal; every other byte stays exact. The Batch
        # line carries the same header under its own name.
        now = _pre039_header(now)
        g_nums, n_nums = _HEADER_NUMBER.findall(golden), _HEADER_NUMBER.findall(now)
        same_text = _HEADER_NUMBER.sub("#", golden) == _HEADER_NUMBER.sub("#", now)
        if not (same_text and len(g_nums) == len(n_nums)
                and all(math.isclose(float(a), float(b), abs_tol=HEADER_DECIMAL_TOL)
                        for a, b in zip(g_nums, n_nums, strict=True))):
            out.append((path, f"{golden!r} != {now!r}"))
    elif isinstance(golden, str) and isinstance(now, str) and _NUMBER.search(golden):
        # A header carries scores as text. The goldens were captured on one CPU and CI runs on
        # another, where the tiny model's forward differs around the 8th significant digit
        # (2026-10-23: -0.0105607556 vs -0.0105607565). Everything but the numbers stays exact.
        g_nums, n_nums = _NUMBER.findall(golden), _NUMBER.findall(now)
        same_text = _NUMBER.sub("#", golden) == _NUMBER.sub("#", now)
        if not (same_text and len(g_nums) == len(n_nums)
                and all(_close(float(a), float(b)) for a, b in zip(g_nums, n_nums, strict=True))):
            out.append((path, f"{golden!r} != {now!r}"))
    elif golden != now:
        out.append((path, f"{golden!r} != {now!r}"))


#: A signed decimal number embedded in a string (header score, threshold).
_NUMBER = re.compile(r"(?<![\w.])-?\d+\.\d+(?:[eE][-+]?\d+)?")
#: A header parameter's number, integer or decimal (`threshold=0` was an integer before 039).
_HEADER_NUMBER = re.compile(r"(?<==)-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?=[;,]|$)")


def _close(a: float, b: float) -> bool:
    """Equal up to cross-CPU float noise; the same tolerance the surface-agreement test uses."""
    return math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-9)


@pytest.mark.parametrize("scenario", [s.name for s in cap.SCENARIOS])
async def test_every_pre038_key_is_unchanged(scenario):
    golden = json.loads(cap.golden_path(scenario).read_text())
    now = await cap.capture(next(s for s in cap.SCENARIOS if s.name == scenario))
    diffs: list = []
    for surface, value in golden.items():
        current = _pre039_header(now.get(surface)) if surface == "header" else now.get(surface)
        _compare(_codes_only(value), current, surface, diffs)
    unexplained = [d for d in diffs if not _allowed(scenario, d[0])]
    assert not unexplained, unexplained
