"""An import refused by a pydantic `ValueError` validator answers 422, never 500 (038 12.5a F4).

Found by the 12.5 run: `ProbeDefinitionV1`'s validators (`unknown rule`, a `streamable` flag that
contradicts the rule) raise `ValueError`, and pydantic puts the exception OBJECT in each error's
`ctx.error`. `ProbeService` copied `exc.errors()` into the refusal's `details`, the management
handler could not serialise it, and the operator got a bare 500 for a document that was simply
wrong. Two fixes, each its own control:

* F4a — `probe_service`: the details come from pydantic's own JSON (`exc.json()`), so the refusal
  carries the validator's message as text;
* F4b — `exception_handlers.millm_error_handler`: any `details` value that is an exception is
  written as its class and message, so no other refusal can render as a 500 the same way.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from millm.core.errors import ValidationError
from tests.support import capture_pre038_goldens as cap


def _bad_rule_definition() -> dict:
    doc = cap._definition(cap.ALPHA)
    doc["aggregation"]["rule"] = "no-such-rule"
    return doc


async def test_f4_an_import_refused_by_a_value_error_validator_is_422_through_the_route():
    async with cap.Harness() as h:
        response = h.client.post("/api/probes/import", json=_bad_rule_definition())
    assert response.status_code == 422, response.text
    error = response.json()["error"]
    assert error["code"] == "VALIDATION_ERROR"
    errors = error["details"]["errors"]
    assert len(errors) == 1, errors
    assert "unknown rule 'no-such-rule'" in errors[0]["msg"]
    assert "no-such-rule" in str(errors[0]["ctx"]["error"])


async def test_f4a_the_service_refusal_details_are_plain_json():
    from millm.services.probe_service import ProbeService

    service = ProbeService(SimpleNamespace())
    with pytest.raises(ValidationError) as caught:
        await service._import_definition(_bad_rule_definition())
    details = caught.value.details
    # Round-trips through the standard library unchanged: nothing in it is an object.
    assert json.loads(json.dumps(details)) == details
    ctx_error = details["errors"][0]["ctx"]["error"]
    assert isinstance(ctx_error, str) and ctx_error.startswith("unknown rule 'no-such-rule'")


async def test_f4b_the_handler_writes_an_exception_in_details_as_text():
    from millm.api.exception_handlers import millm_error_handler

    exc = ValidationError("bad document", details={"errors": [{"ctx": {"error": ValueError("x1")}}]})
    request = SimpleNamespace(url=SimpleNamespace(path="/api/probes/import"), method="POST")
    response = await millm_error_handler(request, exc)
    assert response.status_code == 422
    error = json.loads(response.body)["error"]
    assert error["code"] == "VALIDATION_ERROR"
    assert error["details"]["errors"] == [{"ctx": {"error": "ValueError: x1"}}]


async def test_f4b_a_server_faults_exception_in_details_is_its_class_only():
    """The 5xx rule holds on the new path: an internal error's text is never sent."""
    from millm.api.exception_handlers import millm_error_handler
    from millm.core.errors import ModelLoadError

    exc = ModelLoadError("load failed", details={"cause": RuntimeError("SENTINEL /secret/path")})
    request = SimpleNamespace(url=SimpleNamespace(path="/api/models/1/load"), method="POST")
    response = await millm_error_handler(request, exc)
    assert response.status_code == 500
    assert b"SENTINEL" not in response.body
    assert json.loads(response.body)["error"]["details"]["cause"] == "RuntimeError"


async def test_r1_a_refused_import_echoes_none_of_the_document(caplog, capsys):
    """12.5a R1: once F4 made these refusals render, a MODEL-level validator's `input` — the
    whole definition, test-vector `messages` text included — went back in the 422 body and into
    the `api_error` log line. Control R1-I: drop `include_input=False` in `ProbeService`."""
    import logging

    doc = cap._definition(cap.ALPHA)
    doc["scope"] = "nope"  # refused by ProbeDefinitionV1's own model validator: `loc` is []
    sentinel = "SENTINEL-VECTOR-TEXT"
    doc["test_vectors"]["vectors"][0]["messages"][0]["content"] = sentinel
    caplog.set_level(logging.DEBUG)
    async with cap.Harness() as h:
        response = h.client.post("/api/probes/import", json=doc)
    assert response.status_code == 422, response.text
    errors = response.json()["error"]["details"]["errors"]
    assert [(e["loc"], "unknown scope 'nope'" in e["msg"]) for e in errors] == [([], True)]
    assert all("input" not in e for e in errors), errors
    assert sentinel not in response.text
    # The `api_error` line goes through structlog to stdout; stdlib records to caplog.
    out = capsys.readouterr()
    assert "api_error" in out.out + out.err + caplog.text  # the log line was written ...
    assert sentinel not in out.out + out.err + caplog.text  # ... without the document
