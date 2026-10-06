"""Seed and system fingerprint (Feature 25, FR-25.13, FR-25.14; SC-7).

A seeded sampled request is reproducible on the serial path, the response states the scope of
that promise (`X-miLLM-Seed`), and an unseeded request is unaffected by a seeded one before it.
Every chat and text completion carries a `system_fingerprint` naming model, revision, precision
and engine, with unknown parts written `unrecorded`.

TINY REAL Llama, sampling at temperature 1 — a mocked generate() would agree with any seed.

MUTATION CONTROLS: M12 (`fork_rng` removed), M13 (`manual_seed` removed), plus the seed wiring
on each generation path and the route's header and fingerprint lines.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from millm.api.schemas.openai import ChatCompletionRequest, TextCompletionRequest
from millm.services.system_fingerprint import build_system_fingerprint
from tests.unit.f25_fixtures import (
    clear_loaded,
    make_client,
    make_service,
    model_row,
    unloaded_inference,
    word_model,
    word_tokenizer,
)

MESSAGES = [{"role": "user", "content": "w1 w2 w3"}]


@pytest.fixture(autouse=True)
def _clean():
    clear_loaded()
    yield
    clear_loaded()


@pytest.fixture
def svc():
    return make_service(word_model(seed=3), word_tokenizer())


def chat(**over) -> ChatCompletionRequest:
    body = {"model": "tiny", "messages": MESSAGES, "max_tokens": 12, "temperature": 1.0}
    body.update(over)
    return ChatCompletionRequest(**body)


def _text(resp):
    return [(c.message.content, c.finish_reason) for c in resp.choices]


class TestRepeatability:
    async def test_same_seed_same_bytes(self, svc):
        """SC-7 / M13."""
        a = await svc.create_chat_completion(chat(seed=7))
        torch.manual_seed(999)  # disturb the global stream between the two
        torch.rand(10)
        b = await svc.create_chat_completion(chat(seed=7))
        assert _text(a) == _text(b)

    async def test_a_different_seed_differs(self, svc):
        a = await svc.create_chat_completion(chat(seed=7))
        b = await svc.create_chat_completion(chat(seed=8))
        assert _text(a) != _text(b)

    async def test_n_choices_are_reproducible_and_not_all_identical(self, svc):
        a = await svc.create_chat_completion(chat(seed=7, n=3))
        b = await svc.create_chat_completion(chat(seed=7, n=3))
        assert _text(a) == _text(b)
        assert len(set(_text(a))) > 1, "per-call seeding would make every choice identical"

    async def test_text_completion_is_seeded_per_prompt(self, svc):
        req = TextCompletionRequest(model="tiny", prompt=["w1 w2", "w5"], max_tokens=10,
                                    temperature=1.0, seed=7)
        a = await svc.create_text_completion(req)
        b = await svc.create_text_completion(req)
        assert [c.text for c in a.choices] == [c.text for c in b.choices]
        single = await svc.create_text_completion(req.model_copy(update={"prompt": ["w5"]}))
        assert single.choices[0].text == a.choices[1].text, (
            "prompt 1's output must not depend on prompt 0 (FTID I7)")

    async def test_batched_rows_repeat_at_the_same_shape(self, svc):
        req = chat(seed=7, extra_messages=[[{"role": "user", "content": "w9"}]])
        a = await svc.create_chat_completion(req)
        b = await svc.create_chat_completion(req)
        assert _text(a) == _text(b)

    async def test_streaming_is_seeded(self, svc):
        async def collect(req):
            return "".join([c async for c in svc.stream_chat_completion(req)])

        def content(sse: str) -> str:
            import json

            out = []
            for line in sse.split("\n\n"):
                if line.startswith("data: {"):
                    for ch in json.loads(line[6:]).get("choices", []):
                        out.append(ch.get("delta", {}).get("content") or "")
            return "".join(out)

        a = content(await collect(chat(seed=7, stream=True)))
        b = content(await collect(chat(seed=7, stream=True)))
        c = content(await collect(chat(seed=8, stream=True)))
        assert a == b and a != c


class TestIsolation:
    async def test_the_global_generator_is_restored(self, svc):
        """M12: an unseeded request after a seeded one is not made deterministic."""
        torch.manual_seed(1234)
        before = torch.get_rng_state()
        await svc.create_chat_completion(chat(seed=7))
        assert torch.equal(torch.get_rng_state(), before)

    async def test_an_unseeded_request_after_a_seeded_one_is_not_replayed(self, svc):
        torch.manual_seed(1234)
        await svc.create_chat_completion(chat(seed=7))
        first = await svc.create_chat_completion(chat())
        await svc.create_chat_completion(chat(seed=7))
        second = await svc.create_chat_completion(chat())
        assert _text(first) != _text(second)


class TestScope:
    async def test_serial_greedy_and_scoring_report_request(self, svc):
        from millm.services.inference_service import get_request_outcome, reset_request_outcome

        for req in (chat(seed=7), chat(seed=7, temperature=0),
                    chat(seed=7, max_tokens=1, logprobs=True)):
            reset_request_outcome()
            await svc.create_chat_completion(req)
            assert get_request_outcome()["seed_scope"] == "request", req

    async def test_batched_reports_batch_shape(self, svc):
        from millm.services.inference_service import get_request_outcome, reset_request_outcome

        reset_request_outcome()
        await svc.create_chat_completion(chat(seed=7, extra_messages=[[MESSAGES[0]]]))
        assert get_request_outcome()["seed_scope"] == "batch-shape"

    async def test_best_effort_while_continuous_batching_runs(self, svc):
        from millm.services.inference_service import get_request_outcome, reset_request_outcome

        svc._cbm_backend = SimpleNamespace(is_running=True,
                                           sampling_params_match=lambda t, p: True)
        assert svc.seed_scope_for(chat(seed=7)) == "best-effort"

        async def no_cbm(*a, **k):
            raise AssertionError("a seeded request was served by the manager")

        svc._cbm_chat_completion = no_cbm
        reset_request_outcome()
        await svc.create_chat_completion(chat(seed=7))
        assert get_request_outcome()["seed_scope"] == "best-effort"

    async def test_no_scope_recorded_without_a_seed(self, svc):
        from millm.services.inference_service import get_request_outcome, reset_request_outcome

        reset_request_outcome()
        await svc.create_chat_completion(chat())
        assert "seed_scope" not in get_request_outcome()


class TestOverHttp:
    @pytest.fixture
    def client(self):
        c, _ = make_client(make_service(word_model(seed=3), word_tokenizer()), model_row())
        return c

    def test_the_seed_is_echoed_with_its_scope(self, client):
        body = {"model": "tiny", "messages": MESSAGES, "max_tokens": 6, "seed": 7}
        a = client.post("/v1/chat/completions", json=body)
        b = client.post("/v1/chat/completions", json=body)
        assert a.status_code == 200, a.text
        assert a.headers["X-miLLM-Seed"] == '7;scope="request"'
        assert a.json()["choices"] == b.json()["choices"]

    def test_batched_rows_say_batch_shape(self, client):
        body = {"model": "tiny", "messages": MESSAGES, "max_tokens": 4, "seed": 7,
                "extra_messages": [[{"role": "user", "content": "w9"}]]}
        r = client.post("/v1/chat/completions", json=body)
        assert r.headers["X-miLLM-Seed"] == '7;scope="batch-shape"'

    def test_streaming_echoes_the_seed(self, client):
        body = {"model": "tiny", "messages": MESSAGES, "max_tokens": 4, "seed": 9,
                "stream": True}
        with client.stream("POST", "/v1/chat/completions", json=body) as r:
            assert r.headers["X-miLLM-Seed"] == '9;scope="request"'
            "".join(r.iter_text())

    def test_completions_and_scoring_echo_the_seed(self, client):
        r = client.post("/v1/completions", json={"model": "tiny", "prompt": "w1", "seed": 3,
                                                 "max_tokens": 3})
        assert r.headers["X-miLLM-Seed"] == '3;scope="request"'
        r = client.post("/v1/chat/completions",
                        json={"model": "tiny", "messages": MESSAGES, "max_tokens": 1,
                              "logprobs": True, "seed": 4})
        assert r.headers["X-miLLM-Seed"] == '4;scope="request"'

    def test_no_header_without_a_seed(self, client):
        r = client.post("/v1/chat/completions",
                        json={"model": "tiny", "messages": MESSAGES, "max_tokens": 2})
        assert "X-miLLM-Seed" not in r.headers

    @pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5])
    def test_an_out_of_range_seed_is_400(self, client, seed):
        r = client.post("/v1/chat/completions",
                        json={"model": "tiny", "messages": MESSAGES, "seed": seed})
        assert r.status_code == 400 and r.json()["error"]["param"] == "seed"

    def test_the_largest_seed_is_accepted(self, client):
        r = client.post("/v1/chat/completions",
                        json={"model": "tiny", "messages": MESSAGES, "max_tokens": 2,
                              "seed": 2**32 - 1})
        assert r.status_code == 200, r.text

    @pytest.mark.parametrize("path, body", [
        ("/v1/chat/completions", {"model": "tiny", "messages": MESSAGES, "seed": 7}),
        ("/v1/completions", {"model": "tiny", "prompt": "w1", "seed": 7}),
    ])
    def test_a_gguf_seed_is_refused_before_load(self, path, body):
        client, svc = make_client(unloaded_inference(), model_row(gguf_files=["m.gguf"]))
        r = client.post(path, json=body)
        assert r.status_code == 400 and r.json()["error"]["param"] == "seed"
        assert "T-61" in r.json()["error"]["message"]
        assert svc.load_model_and_wait.call_count == 0

    def test_a_fingerprint_is_on_every_chat_and_completion_response(self, client):
        for path, body in (
            ("/v1/chat/completions", {"model": "tiny", "messages": MESSAGES, "max_tokens": 2}),
            ("/v1/completions", {"model": "tiny", "prompt": "w1", "max_tokens": 2}),
            ("/v1/completions", {"model": "tiny", "prompt": "w1", "max_tokens": 1,
                                 "logprobs": 1}),
        ):
            r = client.post(path, json=body)
            assert r.status_code == 200, r.text
            assert r.json()["system_fingerprint"] == "millm:tiny@abc123:float32/FP16:transformers"


async def test_llamacpp_refuses_a_seed_in_the_service_too(svc):
    """Defence in depth (a direct caller, or a row/engine mismatch)."""
    from millm.core.errors import FieldNotHonouredError

    svc._engine_is_llamacpp = lambda: True
    with pytest.raises(FieldNotHonouredError) as exc:
        await svc.create_chat_completion(chat(seed=7))
    assert exc.value.details["param"] == "seed"
    with pytest.raises(FieldNotHonouredError):
        await svc.create_text_completion(
            TextCompletionRequest(model="tiny", prompt="w1", seed=7))


class TestFingerprint:
    def _loaded(self, dtype="bfloat16", engine="transformers"):
        return SimpleNamespace(dtype=dtype, engine=engine)

    def test_names_model_revision_precision_engine(self):
        fp = build_system_fingerprint(model_row(name="LFM", revision="0f604ada"), self._loaded())
        assert fp == "millm:LFM@0f604ada:bfloat16/FP16:transformers"

    def test_a_null_revision_is_unrecorded(self):
        fp = build_system_fingerprint(model_row(revision=None), self._loaded())
        assert "@unrecorded:" in fp

    def test_the_unknown_dtype_sentinel_is_unrecorded(self):
        fp = build_system_fingerprint(model_row(), self._loaded(dtype="unknown"))
        assert ":unrecorded/" in fp

    def test_it_changes_when_the_loaded_dtype_changes(self):
        row = model_row()
        assert build_system_fingerprint(row, self._loaded("float16")) != \
            build_system_fingerprint(row, self._loaded("bfloat16"))
        assert build_system_fingerprint(row, self._loaded()) == \
            build_system_fingerprint(row, self._loaded())

    def test_a_gguf_label_is_the_exact_quantization(self):
        row = model_row(gguf_files=["m.gguf"], quantization="Q4")
        row.gguf_label = "Q4_K_M"
        fp = build_system_fingerprint(row, self._loaded(dtype="unknown", engine="llamacpp"))
        assert fp.endswith(":unrecorded/Q4_K_M:llamacpp")

    def test_never_guesses_from_a_non_string(self):
        fp = build_system_fingerprint(SimpleNamespace(name=None, revision=3), None)
        assert fp == "millm:unrecorded@unrecorded:unrecorded/unrecorded:transformers"
