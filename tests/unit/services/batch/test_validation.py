"""Feature 26 task 4.6: validation before any row runs — one test per edge case, end to end.

Everything goes through the REAL app (`POST /v1/files`, `POST /v1/batches`), a real BatchRunner,
a real ModelService (leases) and a tiny real model. A refusal's code and message are compared with
what the SYNCHRONOUS endpoint answers for the same body, not with a literal copied into the test.
"""

from __future__ import annotations

import pytest

from millm.core.config import settings
from tests.unit.batch_fixtures import (  # noqa: F401
    BatchHarness,
    batch_db,
    batch_dir,
    client_for,
    completion_line,
    harness,
    jsonl,
    upload,
)

CHAT_URL = "/v1/chat/completions"


def chat_line(custom_id: str, **body) -> dict:
    return {
        "custom_id": custom_id, "method": "POST", "url": CHAT_URL,
        "body": {"model": "tiny", "messages": [{"role": "user", "content": "w1 w2"}],
                 "max_tokens": 2, **body},
    }


async def _run_create(harness, lines, **kw):
    async with client_for(harness.app()) as client:
        created = await harness.create(client, lines, **kw)
        assert created.status_code == 200, created.text
        await harness.settle_validation()
        batch = await harness.batch(created.json()["id"])
        errors = await harness.lines(client, batch.error_file_id)
        got = (await client.get(f"/v1/batches/{batch.id}")).json()
    return batch, errors, got


async def test_malformed_lines_go_to_the_error_file_and_valid_lines_run(harness):
    lines = [completion_line("ok-1"), b"{not json", [1, 2], completion_line("ok-2")]
    batch, _errors, got = await _run_create(harness, lines)
    assert batch.status == "in_progress"
    assert got["request_counts"] == {"total": 4, "completed": 0, "failed": 2}
    codes = [(e["line"], e["code"]) for e in got["errors"]["data"]]
    assert codes == [(2, "invalid_json"), (3, "invalid_line")]


async def test_wrong_url_duplicate_id_and_stream_are_each_invalid(harness):
    wrong_url = {**completion_line("u"), "url": CHAT_URL}
    dup = completion_line("ok")
    streaming = completion_line("s", stream=True)
    batch, _, got = await _run_create(harness, [completion_line("ok"), wrong_url, dup, streaming])
    codes = {e["line"]: e["code"] for e in got["errors"]["data"]}
    assert codes == {2: "invalid_url", 3: "duplicate_custom_id", 4: "stream_not_supported"}
    assert batch.status == "in_progress"


async def test_a_schema_error_names_the_field(harness):
    bad = completion_line("bad", max_tokens=0)
    _, _, got = await _run_create(harness, [completion_line("ok"), bad])
    error = got["errors"]["data"][0]
    assert error["code"] == "invalid_request" and error["param"] == "body.max_tokens"


async def test_strict_mode_applies_to_every_line_and_names_the_unused_field(harness):
    """FR-26.2.3: whatever the create request's headers, an unused field invalidates the line."""
    line = completion_line("x", made_up_field=1)
    _, _, got = await _run_create(harness, [completion_line("ok"), line])
    error = got["errors"]["data"][0]
    assert error["code"] == "unused_fields_refused"
    assert "made_up_field" in error["message"]


async def test_an_output_changing_refusal_carries_the_synchronous_message(harness):
    body = {"model": "tiny", "messages": [{"role": "user", "content": "w1"}],
            "tools": [{"type": "function", "function": {"name": "f"}}]}
    async with client_for(harness.app()) as client:
        sync = (await client.post(CHAT_URL, json=body)).json()["error"]
    line = {"custom_id": "t", "method": "POST", "url": CHAT_URL, "body": body}
    _, _, got = await _run_create(harness, [chat_line("ok"), line], endpoint=CHAT_URL)
    error = got["errors"]["data"][0]
    assert error["code"] == sync["code"] == "field_not_honoured"
    assert error["message"] == f"Line 2: {sync['message']}"
    await harness.drain()
    async with client_for(harness.app()) as client:
        errors = await harness.lines(client, (await harness.batch(got["id"])).error_file_id)
    assert errors[0]["custom_id"] == "t" and errors[0]["line"] == 2
    assert errors[0]["error"]["code"] == "field_not_honoured"


async def test_scoring_on_a_gguf_row_is_refused_as_the_route_refuses_it(batch_db, batch_dir):
    from tests.unit.f25_fixtures import clear_loaded

    h = BatchHarness(batch_db, model_name="gg")
    await h.seed_model(name="gg", gguf_files=["m.gguf"])
    try:
        body = completion_line("x", model="gg")["body"]
        async with client_for(h.app()) as client:
            sync = (await client.post("/v1/completions", json=body)).json()["error"]
        _, _, got = await _run_create(h, [completion_line("x", model="gg")])
        assert got["status"] == "failed"  # its only line was invalid (T-67)
        assert got["errors"]["data"][0]["code"] == sync["code"]
        assert sync["message"] in got["errors"]["data"][0]["message"]
    finally:
        await h.runner.stop()
        clear_loaded()


async def test_two_models_fail_the_batch_listing_each_with_its_count(harness):
    await harness.seed_model(model_id=2, name="other", repo_id="acme/other")
    lines = [completion_line("a"), completion_line("b"), completion_line("c", model="other")]
    batch, _, got = await _run_create(harness, lines)
    assert batch.status == "failed" and got["failed_at"] is not None
    message = got["errors"]["data"][0]["message"]
    assert got["errors"]["data"][0]["code"] == "multiple_models"
    assert "tiny (2 lines)" in message and "other (1 lines)" in message


async def test_a_file_with_no_valid_line_fails_takes_no_lease_and_runs_nothing(harness):
    """T-67 / M-T67's target."""
    from millm.services.model_lease import get_lease_registry

    batch, errors, got = await _run_create(harness, [b"nope", completion_line("x", url="/v1/x")])
    assert batch.status == "failed"
    assert got["request_counts"] == {"total": 2, "completed": 0, "failed": 2}
    assert get_lease_registry().current(1) is None
    assert [e["line"] for e in errors] == [1, 2]
    assert batch.output_file_id is None


async def test_the_probe_endpoint_is_accepted_only_while_its_route_is_served(harness):
    async with client_for(harness.app()) as client:
        file_id = (await upload(client, jsonl([{"custom_id": "p"}]))).json()["id"]
        accepted = await client.post("/v1/batches", json={
            "input_file_id": file_id, "endpoint": "/api/probes/score", "completion_window": "24h"})
    assert accepted.status_code == 200, accepted.text

    # Remove the route from the document the server derives its endpoint set from: with Feature
    # 27's route not served, the same request must be refused (FR-26.1.4).
    import copy

    app = harness.app()
    document = copy.deepcopy(app.openapi())
    del document["paths"]["/api/probes/score"]
    app.openapi = lambda: document
    async with client_for(app) as client:
        file_id = (await upload(client, jsonl([{"custom_id": "p"}]))).json()["id"]
        refused = await client.post("/v1/batches", json={
            "input_file_id": file_id, "endpoint": "/api/probes/score", "completion_window": "24h"})
    assert refused.status_code == 400
    assert refused.json()["error"]["param"] == "endpoint"
    assert "/api/probes/score" not in refused.json()["error"]["message"].split("got")[0]


@pytest.mark.parametrize("window,ok", [
    ("24h", True), ("72h", True), ("168h", True), ("0h", False), ("200h", False), ("1d", False),
    ("24", False), ("01h", False),
])
async def test_completion_window(harness, window, ok):
    async with client_for(harness.app()) as client:
        response = await harness.create(client, [completion_line("a")], completion_window=window)
    if ok:
        assert response.status_code == 200, response.text
        body = response.json()
        hours = int(window[:-1])
        assert body["expires_at"] - body["created_at"] == hours * 3600
        assert body["millm"]["completion_window_extension"] is (window != "24h")
    else:
        assert response.status_code == 400
        assert response.json()["error"]["param"] == "completion_window"


@pytest.mark.parametrize("seconds,ok", [(3600, True), (2_592_000, True), (3599, False),
                                        (2_592_001, False)])
async def test_output_expires_after_bounds(harness, seconds, ok):
    async with client_for(harness.app()) as client:
        response = await harness.create(
            client, [completion_line("a")],
            output_expires_after={"anchor": "created_at", "seconds": seconds},
        )
    assert (response.status_code == 200) is ok, response.text


async def test_a_lease_held_by_another_refuses_the_create_and_its_own_holder_is_accepted(harness):
    async with harness.factory() as session:
        grant = await harness.model_service(session).acquire_lease(1, "midataworks", "labels", 600)
    async with client_for(harness.app()) as client:
        refused = await harness.create(client, [completion_line("a")])
        accepted = await harness.create(
            client, [completion_line("a")], headers={"X-miLLM-Lease": grant.lease_id}
        )
    assert refused.status_code == 409
    error = refused.json()["error"]
    assert error["code"] == "model_leased" and "midataworks" in error["message"]
    assert grant.lease_id not in refused.text
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["millm"]["lease_mode"] == "caller"


async def test_validation_takes_no_slot(harness, monkeypatch):
    """FR-26.2.9: validation is CPU work; not one queue acquisition while it runs."""
    queue = harness.inference.request_queue
    calls: list[str] = []
    real_acquire, real_background = queue.acquire, queue.acquire_background
    monkeypatch.setattr(queue, "acquire", lambda *a, **k: calls.append("acquire") or real_acquire(*a, **k))
    monkeypatch.setattr(
        queue, "acquire_background",
        lambda *a, **k: calls.append("background") or real_background(*a, **k),
    )
    batch, _, _ = await _run_create(harness, [completion_line(str(i)) for i in range(20)])
    assert batch.status == "in_progress"
    assert calls == []


async def test_unknown_create_fields_are_reported_or_refused_under_strict(harness):
    async with client_for(harness.app()) as client:
        reported = await harness.create(client, [completion_line("a")], bogus=1)
        refused = await harness.create(
            client, [completion_line("a")], bogus=1, headers={"X-miLLM-Strict": "true"}
        )
    assert reported.status_code == 200
    assert reported.headers["X-miLLM-Ignored-Fields"] == '"bogus"'
    assert refused.status_code == 400
    assert refused.json()["error"]["code"] == "unused_fields_refused"


async def test_pack_defaults_from_the_setting(harness, monkeypatch):
    async with client_for(harness.app()) as client:
        default_on = (await harness.create(client, [completion_line("a")])).json()
        monkeypatch.setattr(settings, "BATCH_PACK_DEFAULT", False)
        default_off = (await harness.create(client, [completion_line("a")])).json()
        explicit = (await harness.create(client, [completion_line("a")], pack=True)).json()
    assert default_on["millm"]["pack"] is True
    assert default_off["millm"]["pack"] is False
    assert explicit["millm"]["pack"] is True


async def test_validate_functions_are_what_the_routes_answer_with(harness):
    """4.2's snapshot: the route's response for a refusal IS the validate_* refusal's response."""
    from millm.api.routes.openai.completions import validate_completions
    from millm.api.routes.openai.errors import OpenAIRefusal
    from millm.api.schemas.openai import TextCompletionRequest
    from tests.unit.f25_fixtures import model_row

    body = {"model": "tiny", "prompt": "w1", "stream": True}
    async with client_for(harness.app()) as client:
        sync = await client.post("/v1/completions", json=body)
    with pytest.raises(OpenAIRefusal) as caught:
        validate_completions(TextCompletionRequest(**body), model_row(), strict=False)
    assert sync.status_code == caught.value.status_code
    assert sync.json() == caught.value.body()
