"""Every output-changing field, on every `/v1` POST path, on both engines, through HTTP
(Feature 25, FR-25.3, SC-2).

The endpoint set is READ FROM THE LIVE APP (`app.openapi()["paths"]`; FastAPI 0.141 wraps
included routers, so `app.routes` is not a reliable route list here). A `/v1` POST path that the
policy table has no column for fails — a new endpoint cannot ship without deciding every listed
field.

For each (field, path, engine): a non-neutral value is POSTed through the real app. Where the
table says *refused*, the answer must be 400 `field_not_honoured` naming the field, with AND
without `X-miLLM-Strict`, and `load_model_and_wait` must not have been called (the refusal is
decidable from the row, FR-25.3.8). Where it says *honoured*, the answer must not be that
refusal.

MUTATION CONTROLS: M2 (an outcome check removed in `evaluate`), M5 (the policy call deleted from
one route).
"""

from __future__ import annotations

import pytest

from millm.api.request_policy import ENDPOINT_PATHS, OUTPUT_CHANGING, Engine, Honoured
from tests.unit.f25_fixtures import make_client, model_row, unloaded_inference

BASE = {
    "/v1/chat/completions": {"model": "tiny", "messages": [{"role": "user", "content": "hi"}],
                             "max_tokens": 1},
    "/v1/completions": {"model": "tiny", "prompt": "hi", "max_tokens": 1},
    "/v1/embeddings": {"model": "tiny", "input": "hi"},
}

#: A non-neutral value per field, valid for the field's declared type wherever it is declared.
VALUES = {
    "logprobs": {"/v1/chat/completions": True, "/v1/completions": 2, "/v1/embeddings": True},
    "top_logprobs": {"/v1/chat/completions": 2, "/v1/completions": 2, "/v1/embeddings": 2},
    "allowed_token_ids": [3, 4],
    "response_format": {"type": "json_object"},
    "seed": 7,
    "n": 2,
    "dimensions": 8,
    "steering": {"clusters": []},
    "tools": [{"type": "function", "function": {"name": "f"}}],
    "tool_choice": "auto",
    "logit_bias": {"1": 1},
    "max_completion_tokens": 1,
    "profile": "some-profile",
    "steering_intensity": 1.0,
    "return_sae_activations": {"top_k": 2, "positions": "last"},
    "pooling": "last",
    "normalize": True,
}

#: Companion fields a value needs to pass schema validation (chat `top_logprobs` needs logprobs).
COMPANIONS = {
    ("top_logprobs", "/v1/chat/completions"): {"logprobs": True},
}


def _openapi_v1_post_paths() -> set[str]:
    from millm.main import create_app

    paths = create_app().openapi()["paths"]
    return {p for p, ops in paths.items() if p.startswith("/v1/") and "post" in ops}


def test_every_v1_post_path_has_a_table_column():
    live = _openapi_v1_post_paths()
    assert live, "the OpenAPI document listed no /v1 POST path; the guard would assert nothing"
    unmapped = live - set(ENDPOINT_PATHS)
    assert not unmapped, f"/v1 POST paths with no request-policy column: {sorted(unmapped)}"
    assert set(ENDPOINT_PATHS) <= live, "the table names a path the app does not serve"


def _value(field: str, path: str):
    v = VALUES[field]
    return v[path] if isinstance(v, dict) and path in v and field in ("logprobs", "top_logprobs") else v


CASES = [
    (field, path, engine)
    for field in OUTPUT_CHANGING
    for path in sorted(ENDPOINT_PATHS)
    for engine in Engine
]


def test_every_field_has_a_test_value():
    assert set(VALUES) == set(OUTPUT_CHANGING)


@pytest.mark.parametrize("strict", [False, True], ids=["lenient", "strict"])
@pytest.mark.parametrize("field, path, engine", CASES,
                         ids=[f"{f}-{p.rsplit('/', 1)[-1]}-{e.value}" for f, p, e in CASES])
def test_the_table_is_enforced_over_http(field, path, engine, strict):
    endpoint = ENDPOINT_PATHS[path]
    outcome = OUTPUT_CHANGING[field][(endpoint, engine)]
    row = model_row(gguf_files=["m.gguf"] if engine is Engine.LLAMACPP else None)
    client, svc = make_client(unloaded_inference(), row)
    body = dict(BASE[path])
    body.update(COMPANIONS.get((field, path), {}))
    body[field] = _value(field, path)
    headers = {"X-miLLM-Strict": "true"} if strict else {}
    response = client.post(path, json=body, headers=headers)
    error = (response.json() or {}).get("error") if response.status_code >= 400 else None

    refused_here = (
        response.status_code == 400
        and error is not None
        # The table refuses (field_not_honoured), or a schema validator refuses first for a rule
        # that needs no row (e.g. n > 1 on completions, T-56: invalid_parameter). Both name it.
        and error.get("code") in ("field_not_honoured", "invalid_parameter")
        and error.get("type") == "invalid_request_error"
        and (error.get("param") == field or f"'{field}'" in error["message"])
    )
    if isinstance(outcome, Honoured) and outcome.refuse_if is None:
        assert not refused_here, (field, path, engine, response.text)
    elif not isinstance(outcome, Honoured):
        assert refused_here, (field, path, engine, response.status_code, response.text)
        assert svc.load_model_and_wait.call_count == 0, "refused only after an auto-load"


@pytest.mark.parametrize("field, body", [
    ("logprobs", {"logprobs": 2}),
    ("allowed_token_ids", {"allowed_token_ids": [3]}),
])
def test_the_table_answers_gguf_completion_scoring_as_the_route_check_did(field, body):
    """8.4 / FTID I9: the route-level GGUF scoring refusal in completions.py was deleted once the
    table produced the same answer — status 400, type invalid_request_error, param naming the
    field, the reason naming GGUF, and no load. (Deliberate difference: the old check named
    `logprobs` even when only `allowed_token_ids` was sent; the table names the field sent.)"""
    client, svc = make_client(unloaded_inference(), model_row(gguf_files=["m.gguf"]))
    r = client.post("/v1/completions",
                    json={"model": "tiny", "prompt": "x", "max_tokens": 1, **body})
    assert r.status_code == 400
    error = r.json()["error"]
    assert error["type"] == "invalid_request_error" and error["param"] == field
    assert "GGUF" in error["message"]
    assert svc.load_model_and_wait.call_count == 0
