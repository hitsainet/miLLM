"""`return_sae_activations` through the LIVE `/v1` routes (Feature 27, task 6.4, 6.1, X-09).

Every refusal arrives as a 400 before any generation — asserted by counting the generation
primitive, not by reading the route. A body that asked for nothing carries no `millm` key at all.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from millm.core.config import settings
from millm.ml.sae_hooker import SAEHooker
from millm.services.probe_runtime import ProbeRuntimeState
from tests.unit.f25_fixtures import (
    clear_loaded,
    make_client,
    make_service,
    model_row,
    word_model,
    word_tokenizer,
)
from tests.unit.services.test_request_activations import make_sae

CHAT = {"model": "tiny", "messages": [{"role": "user", "content": "w1 w2"}], "max_tokens": 2,
        "temperature": 0}
TEXT = {"model": "tiny", "prompt": "w1 w2", "max_tokens": 2, "temperature": 0}
ACT = {"top_k": 2, "positions": "all"}


@pytest.fixture(autouse=True)
def clean():
    ProbeRuntimeState.reset_for_tests()
    yield
    ProbeRuntimeState.reset_for_tests()
    clear_loaded()


@pytest.fixture
def live(monkeypatch):
    model = word_model()
    svc = make_service(model, word_tokenizer())
    sae = make_sae()
    handle = SAEHooker().install(model, 0, sae)
    entries = [SimpleNamespace(sae=sae, sae_id="sae_a", layer=0)]
    monkeypatch.setattr("millm.services.sae_service.AttachedSAEState.entries",
                        lambda self: list(entries))
    generated = {"n": 0}
    real = svc._generate_sync

    def counting(*a, **k):
        generated["n"] += 1
        return real(*a, **k)

    svc._generate_sync = counting
    client, _ = make_client(svc, model_row())
    with patch("millm.services.inference_service.torch.cuda.is_available", return_value=False):
        yield client, entries, generated, sae
    handle.remove()


def post(client, path, body):
    return client.post(path, json=body)


class TestRefusedBeforeGeneration:
    @pytest.mark.parametrize(("path", "body", "code", "fragment"), [
        ("/v1/chat/completions", {**CHAT, "n": 2}, "sae_activations_refused", "n > 1"),
        ("/v1/chat/completions",
         {**CHAT, "extra_messages": [[{"role": "user", "content": "w3"}]]},
         "sae_activations_refused", "extra_messages"),
        ("/v1/completions", {**TEXT, "prompt": ["w1", "w2"]}, "sae_activations_refused",
         "several prompts"),
        ("/v1/chat/completions", {**CHAT, "return_sae_activations": {**ACT, "sae_id": "sae_x"}},
         "sae_not_attached", "sae_x"),
        ("/v1/chat/completions", {**CHAT, "return_sae_activations": {**ACT, "features": [99]}},
         "sae_activations_refused", "outside SAE"),
        ("/v1/chat/completions", {**CHAT, "return_sae_activations": {**ACT, "top_k": 1000}},
         "sae_activations_refused", "top_k"),
    ])
    def test_each_shape(self, live, path, body, code, fragment):
        client, _entries, generated, _sae = live
        body = dict(body)
        body.setdefault("return_sae_activations", ACT)
        response = post(client, path, body)
        assert response.status_code == 400, response.text
        error = response.json()["error"]
        assert error["code"] == code and fragment in error["message"]
        assert generated["n"] == 0, "a refused request must never generate"

    def test_over_the_worst_case_entry_cap(self, live, monkeypatch):
        client, _entries, generated, _sae = live
        monkeypatch.setattr(settings, "SAE_ACTIVATIONS_MAX_ENTRIES", 5)
        response = post(client, "/v1/chat/completions",
                        {**CHAT, "max_tokens": 50, "return_sae_activations": ACT})
        assert response.status_code == 400
        assert "activation entries" in response.json()["error"]["message"]
        assert generated["n"] == 0

    def test_ambiguous_sae_names_the_candidates(self, live):
        client, entries, generated, sae = live
        entries.append(SimpleNamespace(sae=sae, sae_id="sae_b", layer=1))
        response = post(client, "/v1/chat/completions", {**CHAT, "return_sae_activations": ACT})
        assert response.status_code == 400
        message = response.json()["error"]["message"]
        assert "sae_a (layer 0)" in message and "sae_b (layer 1)" in message
        assert generated["n"] == 0

    def test_no_sae_attached(self, live):
        client, entries, generated, _sae = live
        entries.clear()
        response = post(client, "/v1/chat/completions", {**CHAT, "return_sae_activations": ACT})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "sae_not_attached"
        assert generated["n"] == 0

    def test_a_gguf_row_is_refused_before_any_load(self):
        from tests.unit.f25_fixtures import unloaded_inference

        client, svc = make_client(unloaded_inference(), model_row(gguf_files=["m.gguf"]))
        response = client.post("/v1/chat/completions", json={**CHAT,
                                                              "return_sae_activations": ACT})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "field_not_honoured"
        assert svc.load_model_and_wait.call_count == 0


class TestServed:
    def test_the_body_carries_millm_and_the_note(self, live):
        client, *_ = live
        response = post(client, "/v1/chat/completions", {**CHAT, "return_sae_activations": ACT})
        assert response.status_code == 200, response.text
        block = response.json()["millm"]["sae_activations"]
        assert block["sae_id"] == "sae_a" and block["read_point"] == "post_steering"
        assert "not an unsteered counterfactual" in block["note"]
        assert "X-miLLM-Ignored-Fields" not in response.headers, "the field is a known field"

    def test_without_the_field_the_body_has_no_millm_key(self, live):
        client, *_ = live
        response = post(client, "/v1/chat/completions", CHAT)
        assert response.status_code == 200
        assert "millm" not in response.json()
        response = post(client, "/v1/completions", TEXT)
        assert "millm" not in response.json()

    def test_a_scoring_response_with_activations_says_steering_none(self, live):
        """X-09: scoring is always unsteered, and its activations response says so."""
        client, *_ = live
        response = post(client, "/v1/completions", {**TEXT, "max_tokens": 1, "logprobs": 1,
                                                     "return_sae_activations": ACT})
        assert response.status_code == 200, response.text
        assert response.headers["X-miLLM-Steering"] == "none"
        assert response.json()["millm"]["sae_activations"]["read_point"] == "unsteered"
