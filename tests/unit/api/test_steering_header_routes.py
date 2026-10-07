"""`X-miLLM-Steering` and inline steering through the REAL app (Feature 28; FTASKS 2.5, 4.2, 4.3,
6.1 – 6.5).

Placed under `tests/unit/` rather than the FTASKS' `tests/integration/api/`: CI runs
`tests/unit` only, and a reachability guard that CI never runs guards nothing.

The service is a real `InferenceService` over a tiny real Llama with a real hooked SAE; the model
service is the `make_client` spy, whose `load_model_and_wait` call count is the "before auto-load"
evidence.
"""

from __future__ import annotations

import json

import pytest

from tests.unit.f25_fixtures import make_client, model_row, unloaded_inference
from tests.unit.f28_fixtures import (
    SAE_A,
    SAE_B,
    add_profile,
    build,
    clean_state,  # noqa: F401 - fixture
    db,  # noqa: F401 - fixture
    independent_hash,
    no_cuda,
)

CHAT = {"model": "tiny", "messages": [{"role": "user", "content": "w1 w2"}], "max_tokens": 3,
        "temperature": 0.0}
TEXT = {"model": "tiny", "prompt": "w1 w2", "max_tokens": 3, "temperature": 0.0}
SET = {"features": [{"index": 3, "strength": 500.0}]}


@pytest.fixture
def one(clean_state, db):  # noqa: F811
    served = build([(SAE_A, 0, 11)])
    client, spy = make_client(served.svc, model_row())
    yield served, client, spy
    for h in served.handles:
        h.remove()


def _post(client, path, body):
    with no_cuda():
        return client.post(path, json=body)


# ── 6.1 / 6.2 the header on every non-streaming route ───────────────────────────────────────


class TestHeaderPerKind:
    @pytest.mark.parametrize("path, base", [("/v1/chat/completions", CHAT),
                                            ("/v1/completions", TEXT)])
    def test_none(self, one, path, base):
        _served, client, _ = one
        r = _post(client, path, base)
        assert r.status_code == 200, r.text
        assert r.headers["X-miLLM-Steering"] == "none"

    @pytest.mark.parametrize("path, base", [("/v1/chat/completions", CHAT),
                                            ("/v1/completions", TEXT)])
    def test_inline_with_the_clamp_reported(self, one, path, base):
        served, client, _ = one
        r = _post(client, path, {**base, "steering": SET})
        assert r.status_code == 200, r.text
        assert r.headers["X-miLLM-Steering"] == (
            f'inline;sae="{SAE_A}";layer=0;features=1;'
            f'hash="{independent_hash(SAE_A, {3: 200.0})}";clamped=1'
        )
        assert served.sae(SAE_A, 0).get_steering_values() == {}, "restored after the request"

    @pytest.mark.parametrize("path, base", [("/v1/chat/completions", CHAT),
                                            ("/v1/completions", TEXT)])
    async def test_profile_named_by_the_request(self, one, db, path, base):  # noqa: F811
        _served, client, _ = one
        await add_profile(db, "humor", {1: 2.0, 2: 0.0})
        r = _post(client, path, {**base, "profile": "humor"})
        assert r.status_code == 200, r.text
        assert r.headers["X-miLLM-Steering"] == (
            f'profile;name="humor";source=request;intensity="1.0";sae="{SAE_A}";layer=0;'
            f'features=1;hash="{independent_hash(SAE_A, {1: 2.0})}"'
        )

    def test_a_dial_over_live_values_keeps_its_echo_and_reports_manual(self, one):
        """FR-28.3.11: `X-miLLM-Steering-Intensity` unchanged (a pre-generation echo); the
        authoritative header reports what ran — the live values × λ, claimed by nothing."""
        served, client, _ = one
        sae = served.sae(SAE_A, 0)
        sae.set_steering_batch({2: 8.0})
        sae.enable_steering(True)
        r = _post(client, "/v1/chat/completions", {**CHAT, "steering_intensity": 0.5})
        assert r.status_code == 200, r.text
        assert r.headers["X-miLLM-Steering-Intensity"] == "0.5"
        assert r.headers["X-miLLM-Steering"] == (
            f'manual;sae="{SAE_A}";layer=0;features=1;'
            f'hash="{independent_hash(SAE_A, {2: 4.0})}"'
        )

    def test_batched_extra_messages_carry_exactly_one_report(self, one):
        _served, client, _ = one
        body = {**CHAT, "steering": SET,
                "extra_messages": [[{"role": "user", "content": "w5"}]]}
        r = _post(client, "/v1/chat/completions", body)
        assert r.status_code == 200, r.text
        assert len(r.json()["choices"]) == 2
        assert r.headers.get_list("X-miLLM-Steering") == [
            f'inline;sae="{SAE_A}";layer=0;features=1;'
            f'hash="{independent_hash(SAE_A, {3: 200.0})}";clamped=1'
        ]

    def test_multi_prompt_completion_carries_exactly_one_report(self, one):
        _served, client, _ = one
        r = _post(client, "/v1/completions", {**TEXT, "prompt": ["w1", "w2 w3"], "steering": SET})
        assert r.status_code == 200, r.text
        assert len(r.json()["choices"]) == 2
        assert len(r.headers.get_list("X-miLLM-Steering")) == 1

    def test_a_scoring_response_says_none(self, one):
        served, client, _ = one
        served.sae(SAE_A, 0).set_steering_batch({2: 8.0})
        served.sae(SAE_A, 0).enable_steering(True)
        r = _post(client, "/v1/completions", {**TEXT, "max_tokens": 1, "logprobs": 2})
        assert r.status_code == 200, r.text
        assert r.headers["X-miLLM-Steering"] == "none"

    def test_a_report_that_cannot_be_read_is_unknown_and_the_request_succeeds(self, one,
                                                                               monkeypatch):
        from millm.services.steering_report import SteeringStateReader

        async def boom(self, *a, **k):
            raise RuntimeError("reader exploded")

        monkeypatch.setattr(SteeringStateReader, "_describe", boom)
        _served, client, _ = one
        r = _post(client, "/v1/chat/completions", {**CHAT, "steering": SET})
        assert r.status_code == 200
        assert r.headers["X-miLLM-Steering"] == "unknown;reason=read_failed"


# ── 6.3 streaming ───────────────────────────────────────────────────────────────────────────


class TestStreaming:
    def _events(self, r) -> list[str]:
        return [line for line in r.text.split("\n\n") if line.startswith("data: ")]

    @pytest.mark.parametrize("body, expected", [
        ({}, "none"),
        ({"steering": SET}, None),
    ])
    def test_the_stream_ends_with_one_steering_chunk_and_no_header(self, one, body, expected):
        _served, client, _ = one
        r = _post(client, "/v1/chat/completions", {**CHAT, **body, "stream": True})
        assert r.status_code == 200, r.text
        assert "X-miLLM-Steering" not in r.headers, "no statement of intent before the body"
        events = self._events(r)
        assert events[-1] == "data: [DONE]"
        steering = [e for e in events if '"millm_steering"' in e]
        assert len(steering) == 1 and events[-2] == steering[0]
        payload = json.loads(steering[0][len("data: "):])
        assert payload["choices"] == []
        want = expected or (f'inline;sae="{SAE_A}";layer=0;features=1;'
                            f'hash="{independent_hash(SAE_A, {3: 200.0})}";clamped=1')
        assert payload["millm_steering"] == want

    @pytest.mark.parametrize("steering, code", [
        ({"sae_id": "sae_gamma", "features": [{"index": 1, "strength": 1.0}]},
         "sae_not_attached"),
        ({"features": [{"index": 99, "strength": 1.0}]}, "invalid_feature_index"),
    ])
    def test_a_bad_inline_set_is_a_400_before_the_stream_commits(self, one, steering, code):
        """FTASKS 6.4: the in-slot checks run as a dry run before the 200."""
        _served, client, _ = one
        r = _post(client, "/v1/chat/completions", {**CHAT, "steering": steering, "stream": True})
        assert r.status_code == 400, r.text
        assert r.json()["error"]["code"] == code


# ── 4.2 / 4.3 / 2.5 refusals before any auto-load ──────────────────────────────────────────


class TestRefusedBeforeLoad:
    @pytest.mark.parametrize("path, base", [("/v1/chat/completions", CHAT),
                                            ("/v1/completions", TEXT)])
    def test_a_non_empty_set_on_a_non_resident_model(self, path, base):
        client, spy = make_client(unloaded_inference(), model_row())
        r = client.post(path, json={**base, "steering": {"sae_id": "sae_z", **SET}})
        assert r.status_code == 400, r.text
        assert r.json()["error"]["code"] == "sae_not_attached"
        assert "sae_z" in r.json()["error"]["message"]
        assert spy.load_model_and_wait.call_count == 0

    @pytest.mark.parametrize("path, base", [("/v1/chat/completions", CHAT),
                                            ("/v1/completions", TEXT)])
    def test_the_empty_set_on_a_non_resident_model_is_not_refused(self, path, base):
        client, spy = make_client(unloaded_inference(), model_row())
        client.post(path, json={**base, "steering": {"features": []}})
        assert spy.load_model_and_wait.call_count == 1, "the auto-load must still run"

    @pytest.mark.parametrize("path, base", [("/v1/chat/completions", CHAT),
                                            ("/v1/completions", TEXT)])
    @pytest.mark.parametrize("field, value", [
        ("steering", SET), ("steering", {"features": []}), ("profile", "p"),
        ("steering_intensity", 1.0),
    ])
    def test_steering_on_a_gguf_row(self, path, base, field, value):
        client, spy = make_client(unloaded_inference(), model_row(gguf_files=["m.gguf"]))
        r = client.post(path, json={**base, field: value})
        assert r.status_code == 400, r.text
        assert r.json()["error"]["param"] == field
        assert spy.load_model_and_wait.call_count == 0

    @pytest.mark.parametrize("field, value", [
        ("steering", SET), ("steering", {"features": []}), ("profile", "p"),
        ("steering_intensity", 1.0),
    ])
    def test_any_steering_field_on_completion_scoring(self, field, value):
        """FR-25.7.2 / X-09 on /v1/completions, which gained the fields in Feature 28."""
        client, spy = make_client(unloaded_inference(), model_row())
        r = client.post("/v1/completions",
                        json={**TEXT, "max_tokens": 1, "logprobs": 2, field: value})
        assert r.status_code == 400, r.text
        assert r.json()["error"]["param"] == field
        assert "X-09" in r.json()["error"]["message"]
        assert spy.load_model_and_wait.call_count == 0

    @pytest.mark.parametrize("path, base", [("/v1/chat/completions", CHAT),
                                            ("/v1/completions", TEXT)])
    @pytest.mark.parametrize("extra", [{"profile": "p"}, {"steering_intensity": 0.5}])
    def test_steering_with_profile_or_dial_is_400_naming_both(self, path, base, extra):
        client, spy = make_client(unloaded_inference(), model_row())
        r = client.post(path, json={**base, "steering": SET, **extra})
        assert r.status_code == 400, r.text
        message = r.json()["error"]["message"]
        assert "'steering'" in message and f"'{next(iter(extra))}'" in message
        assert spy.load_model_and_wait.call_count == 0


def test_selection_ambiguity_over_http(clean_state, db):  # noqa: F811
    served = build([(SAE_A, 0, 11), (SAE_B, 1, 22)])
    try:
        client, _ = make_client(served.svc, model_row())
        r = _post(client, "/v1/chat/completions", {**CHAT, "steering": SET})
        assert r.status_code == 400, r.text
        assert SAE_A in r.json()["error"]["message"] and SAE_B in r.json()["error"]["message"]
    finally:
        for h in served.handles:
            h.remove()
