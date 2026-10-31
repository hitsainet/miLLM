"""The probe-session read API's names, pinned from the LIVE `app.openapi()` (036 FPRD §7).

⚠ A CROSS-REPO CONTRACT. miStudio 038's spot-check runner calls these routes and pins the same
names on its side (`miStudio/backend/tests/unit/test_millm_session_api_names.py`). Renaming a path,
a parameter or a header here without miStudio moving with it breaks spot-checks silently — so the
names are a SET read from the served schema, never a list beside the code.
"""

from __future__ import annotations

import pytest

from millm.api import probe_request_meta as meta


@pytest.fixture(scope="module")
def openapi():
    from millm.main import create_app

    return create_app().openapi()


def _params(openapi, path, method, where="query"):
    op = openapi["paths"][path][method]
    return {p["name"] for p in op.get("parameters", []) if p["in"] == where}


#: path -> {method: query parameters}
SESSION_API = {
    "/api/probes/sessions": {
        "get": {"since", "until", "served_model", "behaviour", "fired_only", "source", "limit",
                "offset", "cursor"},  # 039 FR-39.15: keyset pagination
    },
    "/api/probes/sessions/{session_id}": {"get": set(), "delete": {"events"}},
    "/api/probes/sessions/{session_id}/transcript": {"get": {"turn_index"}},
    "/api/probes/sessions/sample": {"post": set()},
}

#: The sample request body's field names (phase 4), read from the served schema.
SAMPLE_BODY = {"behaviour", "probe_id", "window", "strata", "near_band", "since", "until", "seed",
               "require_transcript"}
SAMPLE_STRATA = {"fired", "near_threshold", "not_fired"}

EVENT_FILTERS = {"probe_id", "request_id", "session_id", "since", "until", "verdict", "window",
                 "origin", "behaviour", "score_min", "score_max", "limit", "offset",
                 # 038 FR-38.11: the join back from a result or a trace.
                 "result_id", "trace_id",
                 # 039 FR-39.15: keyset pagination (`offset` kept, deprecated, one version).
                 "cursor",
                 # 039 S2 (FR-39.23): the union of probe and rule results, narrowed by kind.
                 "kind"}


def test_every_session_route_and_parameter_is_served(openapi):
    for path, methods in SESSION_API.items():
        assert path in openapi["paths"], f"{path} is not served"
        for method, params in methods.items():
            assert method in openapi["paths"][path], f"{method.upper()} {path} is not served"
            assert _params(openapi, path, method) == params, (method, path)


def test_the_event_filters_are_served(openapi):
    assert _params(openapi, "/api/probes/events", "get") == EVENT_FILTERS


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/completions"])
def test_the_session_request_header_is_declared_on_both_v1_routes(openapi, path):
    assert {meta.SESSION_HEADER, meta.ORIGIN_HEADER} <= _params(openapi, path, "post", where="header")


def test_the_sample_body_names_are_served(openapi):
    schemas = openapi["components"]["schemas"]
    ref = openapi["paths"]["/api/probes/sessions/sample"]["post"]["requestBody"]["content"][
        "application/json"]["schema"]["$ref"].rsplit("/", 1)[-1]
    assert set(schemas[ref]["properties"]) == SAMPLE_BODY
    assert set(schemas["SampleStrata"]["properties"]) == SAMPLE_STRATA


def test_the_header_names_are_the_contract_names():
    assert (meta.SESSION_HEADER, meta.SESSION_ID_HEADER, meta.TRANSCRIPT_HEADER,
            meta.ORIGIN_HEADER) == ("X-miLLM-Session", "X-miLLM-Session-Id", "X-miLLM-Transcript",
                                    "X-miLLM-Origin")
