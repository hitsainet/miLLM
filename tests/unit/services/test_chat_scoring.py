"""Chat scoring (Feature 25, FR-25.5 – FR-25.9; SC-4 unit form, SC-5).

`/v1/chat/completions` renders the chat template and scores through `_score_prompts` — THE scorer
`/v1/completions` uses — with `add_special_tokens=False`. These tests run a TINY REAL Llama and
a real tokenizer whose chat template begins with BOS while its post-processor ALSO adds one, so a
double BOS is a visible difference rather than a fixture that agrees by construction.

MUTATION CONTROLS: M6 (`_unsteered` removed), M7 (scoring branch after the batched branch), M8
(`add_special_tokens=False` -> True), M9 (`_score_chat_completion` stops calling `_score_prompts`).
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from pydantic import ValidationError

from millm.api.schemas.openai import ChatCompletionRequest, TextCompletionRequest
from millm.core.errors import (
    EngineUnsupportedError,
    InvalidScoringRequestError,
    NoChatTemplateError,
)
from tests.unit.f25_fixtures import (
    WORDS,
    clear_loaded,
    make_client,
    make_service,
    model_row,
    unloaded_inference,
    word_model,
    word_tokenizer,
)

ALLOWED = [6, 9, 12]
MESSAGES = [{"role": "system", "content": "w1 w2"}, {"role": "user", "content": "w3 w4 w5"}]


@pytest.fixture(autouse=True)
def _clean():
    clear_loaded()
    yield
    clear_loaded()


def chat(**over) -> ChatCompletionRequest:
    body = {"model": "tiny", "messages": MESSAGES, "max_tokens": 1, "temperature": 1.0,
            "logprobs": True, "top_logprobs": 3, "allowed_token_ids": ALLOWED,
            "return_tokens_as_token_ids": True}
    body.update(over)
    return ChatCompletionRequest(**body)


def _capture_inputs(svc):
    """Wrap the worker-thread forward to record the exact token ids it scored."""
    seen: list[torch.Tensor] = []
    real = svc._unsteered_next_token_logits

    def spy(inputs):
        seen.append(inputs["input_ids"].clone())
        return real(inputs)

    svc._unsteered_next_token_logits = spy
    return seen


class TestParityWithCompletionScoring:
    async def test_identical_token_ids_and_logprobs(self):
        """SC-4 (unit form): chat scoring of `messages` equals completion scoring of the rendered
        prompt with add_special_tokens=False — exact on CPU, not approximately."""
        tok = word_tokenizer(bos=True)
        svc = make_service(word_model(), tok)
        seen = _capture_inputs(svc)
        chat_resp = await svc.create_chat_completion(chat())
        rendered = svc._format_chat_messages(chat().messages)
        text_resp = await svc.create_text_completion(TextCompletionRequest(
            model="tiny", prompt=rendered, max_tokens=1, temperature=1.0, logprobs=3,
            allowed_token_ids=ALLOWED, add_special_tokens=False, return_tokens_as_token_ids=True))

        assert len(seen) == 2 and torch.equal(seen[0], seen[1]), "different token ids scored"
        assert int((seen[0] == 0).sum()) == 1, "the rendered prompt must carry exactly one BOS"
        c = chat_resp.choices[0].logprobs.content[0]
        t = text_resp.choices[0].logprobs
        assert c.token == t.tokens[0]
        assert c.logprob == t.token_logprobs[0]
        assert {a.token: a.logprob for a in c.top_logprobs} == t.top_logprobs[0]
        assert chat_resp.usage.prompt_tokens == text_resp.usage.prompt_tokens

    async def test_it_calls_the_shared_scorer_once_with_the_rendered_texts(self):
        """M9: the call, its payload and its count — not a copy of the arithmetic."""
        svc = make_service(word_model(), word_tokenizer())
        calls = []
        real = svc._score_prompts

        async def spy(texts, **kwargs):
            calls.append((list(texts), kwargs))
            return await real(texts, **kwargs)

        svc._score_prompts = spy
        req = chat(extra_messages=[[{"role": "user", "content": "w7"}]])
        await svc.create_chat_completion(req)
        assert len(calls) == 1
        texts, kwargs = calls[0]
        assert texts == [svc._format_chat_messages(req.messages),
                         svc._format_chat_messages(req.extra_messages[0])]
        assert kwargs["add_special_tokens"] is False
        assert kwargs["allowed"] == ALLOWED and kwargs["top_k"] == 3
        assert kwargs["temperature"] == 1.0


class TestRouting:
    async def test_extra_messages_score_one_choice_per_conversation_in_order(self):
        """M7: routed BEFORE the batched path, which would generate and drop the scores."""
        svc = make_service(word_model(), word_tokenizer())

        def no_generate(*a, **k):
            raise AssertionError("a scoring request reached generation")

        svc._generate_sync = no_generate
        extra = [[{"role": "user", "content": "w7"}], [{"role": "user", "content": "w8 w9"}]]
        resp = await svc.create_chat_completion(chat(extra_messages=extra))
        assert [c.index for c in resp.choices] == [0, 1, 2]
        assert all(c.logprobs is not None for c in resp.choices)
        singles = [await svc.create_chat_completion(chat(messages=m))
                   for m in [MESSAGES] + extra]
        for got, single in zip(resp.choices, singles):
            assert got.logprobs == single.choices[0].logprobs
        assert resp.usage.prompt_tokens == sum(s.usage.prompt_tokens for s in singles)
        assert resp.usage.completion_tokens == 3

    async def test_the_cbm_and_llamacpp_never_take_it(self):
        svc = make_service(word_model(), word_tokenizer())
        svc._use_cbm_for_request = lambda **k: True

        async def cbm(*a, **k):
            raise AssertionError("scoring went to continuous batching")

        svc._cbm_chat_completion = cbm
        assert (await svc.create_chat_completion(chat())).choices[0].logprobs is not None
        svc._engine_is_llamacpp = lambda: True
        with pytest.raises(EngineUnsupportedError):
            await svc.create_chat_completion(chat())

    async def test_logprobs_false_alone_is_ordinary_generation(self):
        svc = make_service(word_model(), word_tokenizer())
        called = []
        svc._score_chat_completion = lambda r: called.append(r)
        resp = await svc.create_chat_completion(ChatCompletionRequest(
            model="tiny", messages=MESSAGES, logprobs=False, max_tokens=2, temperature=0))
        assert called == [] and resp.choices[0].logprobs is None


class _ThreadLocalSteer:
    """A real forward hook that steers unless suppressed IN THE THREAD running the forward —
    the property `LoadedSAE._suppressed` has. An entry like AttachedSAEState's."""

    def __init__(self, model):
        self.local = threading.local()
        layer = model.model.layers[0]
        self.handle = layer.register_forward_hook(self._hook)
        self.sae = SimpleNamespace(suppressed=self._suppressed)

    @contextmanager
    def _suppressed(self):
        self.local.on = True
        try:
            yield
        finally:
            self.local.on = False

    def _hook(self, module, args, output):
        if getattr(self.local, "on", False):
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        steered = hidden + 5.0
        return (steered,) + tuple(output[1:]) if isinstance(output, tuple) else steered


class TestUnsteered:
    @pytest.mark.parametrize("endpoint", ["chat", "completions"])
    async def test_scoring_with_steering_active_equals_scoring_with_none(self, endpoint):
        """SC-5 / M6, on both endpoints: suppression entered in the worker thread."""
        model = word_model()
        svc = make_service(model, word_tokenizer())

        async def score():
            if endpoint == "chat":
                return (await svc.create_chat_completion(chat())).choices[0].logprobs.content[0]
            rendered = svc._format_chat_messages(chat().messages)
            r = await svc.create_text_completion(TextCompletionRequest(
                model="tiny", prompt=rendered, max_tokens=1, logprobs=3,
                allowed_token_ids=ALLOWED, add_special_tokens=False))
            return r.choices[0].logprobs

        plain = await score()
        steer = _ThreadLocalSteer(model)
        try:
            with patch("millm.services.sae_service.AttachedSAEState.entries",
                       return_value=[steer]):
                steered_scoring = await score()
            # Control: the hook really does change the model when not suppressed.
            ids = word_tokenizer()(svc._format_chat_messages(chat().messages),
                                   return_tensors="pt", add_special_tokens=False)
            with torch.no_grad():
                hooked = model(**ids).logits[0, -1]
        finally:
            steer.handle.remove()
        with torch.no_grad():
            unhooked = model(**ids).logits[0, -1]
        assert not torch.allclose(hooked, unhooked), "precondition: the hook steers"
        assert steered_scoring == plain

    async def test_no_monitor_opens_and_generate_never_runs(self):
        """FR-25.7.3/7.4: no probe, sensing or circuit-sensing context; no generate()."""
        svc = make_service(word_model(), word_tokenizer())

        def refuse(*a, **k):
            raise AssertionError("a monitor or generation was begun for chat scoring")

        svc._probe_begin = svc._sensing_begin = svc._circuit_sensing_begin = refuse
        svc._apply_request_steering = refuse
        svc._generate_sync = svc._generate_in_thread = refuse
        resp = await svc.create_chat_completion(chat(extra_messages=[[MESSAGES[1]]]))
        assert len(resp.choices) == 2


class TestRefusals:
    @pytest.mark.parametrize("over, message", [
        ({"max_tokens": 2}, "max_tokens=1"),
        ({"max_tokens": None}, "max_tokens=1"),
        ({"n": 2}, "n=1"),
        ({"stream": True}, "stream=false"),
        ({"temperature": 1e-4}, "temperature 0 or at least"),
        ({"logprobs": None, "top_logprobs": 2}, "top_logprobs requires logprobs"),
        ({"logprobs": False, "top_logprobs": 2}, "top_logprobs requires logprobs"),
        ({"allowed_token_ids": [-1]}, "non-negative"),
        ({"top_logprobs": 21}, "less than or equal to 20"),
    ])
    def test_schema_refusals(self, over, message):
        with pytest.raises(ValidationError, match=message):
            chat(**over)

    @pytest.mark.parametrize("over, param", [
        ({"profile": "p"}, "profile"),
        ({"steering_intensity": 1.0}, "steering_intensity"),
        ({"steering": {"clusters": []}}, "steering"),
        ({"response_format": {"type": "json_object"}}, "response_format"),
    ])
    def test_steering_and_format_fields_on_a_scoring_request_are_400_before_load(self, over,
                                                                               param):
        client, svc = make_client(unloaded_inference(), model_row())
        body = chat(**{}).model_dump(exclude_none=True, exclude_unset=True)
        body.update(over)
        r = client.post("/v1/chat/completions", json=body)
        assert r.status_code == 400, r.text
        assert r.json()["error"]["param"] == param
        assert svc.load_model_and_wait.call_count == 0

    def test_a_gguf_row_is_refused_before_load(self):
        client, svc = make_client(unloaded_inference(), model_row(gguf_files=["m.gguf"]))
        body = chat().model_dump(exclude_none=True, exclude_unset=True)
        r = client.post("/v1/chat/completions", json=body)
        assert r.status_code == 400 and r.json()["error"]["param"] == "logprobs"
        assert "GGUF" in r.json()["error"]["message"]
        assert svc.load_model_and_wait.call_count == 0

    async def test_no_chat_template_is_refused(self):
        svc = make_service(word_model(), word_tokenizer(template=False))
        with pytest.raises(NoChatTemplateError) as exc:
            await svc.create_chat_completion(chat())
        assert exc.value.status_code == 400 and "'tiny'" in exc.value.message

    def test_no_chat_template_over_http(self):
        client, _ = make_client(make_service(word_model(), word_tokenizer(template=False)),
                                model_row())
        r = client.post("/v1/chat/completions",
                        json=chat().model_dump(exclude_none=True, exclude_unset=True))
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "no_chat_template"
        assert r.json()["error"]["type"] == "invalid_request_error"

    def test_generation_keeps_its_fallback_without_a_template(self):
        """Only scoring refuses (FR-25.5.9)."""
        client, _ = make_client(make_service(word_model(), word_tokenizer(template=False)),
                                model_row())
        r = client.post("/v1/chat/completions", json={"model": "tiny", "messages": MESSAGES,
                                                      "max_tokens": 2, "temperature": 0})
        assert r.status_code == 200, r.text

    async def test_an_out_of_vocabulary_id_names_the_conversation(self):
        svc = make_service(word_model(), word_tokenizer())
        with pytest.raises(InvalidScoringRequestError) as exc:
            await svc.create_chat_completion(chat(allowed_token_ids=[6, 999]))
        assert exc.value.details["index"] == 0

    async def test_an_empty_render_fails_the_request_naming_its_index(self):
        tok = word_tokenizer(bos=False)
        tok.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
        svc = make_service(word_model(), tok)
        extra = [[{"role": "user", "content": "w1"}], [{"role": "user", "content": ""}]]
        with pytest.raises(InvalidScoringRequestError) as exc:
            await svc.create_chat_completion(chat(messages=[{"role": "user", "content": "w2"}],
                                                  extra_messages=extra))
        assert exc.value.details["index"] == 2
        assert "conversation 2" in exc.value.message


class TestShape:
    async def test_token_id_form_with_bytes_of_the_decoded_text(self):
        svc = make_service(word_model(), word_tokenizer())
        resp = await svc.create_chat_completion(chat())
        entry = resp.choices[0].logprobs.content[0]
        chosen = int(entry.token.split(":")[1])
        assert entry.token == f"token_id:{chosen}" and chosen in ALLOWED
        assert entry.bytes == list(WORDS[chosen].encode("utf-8"))
        assert resp.choices[0].message.content == WORDS[chosen]
        assert resp.choices[0].finish_reason == "length"
        assert len(entry.top_logprobs) == 3
        for alt in entry.top_logprobs:
            assert alt.bytes == list(WORDS[int(alt.token.split(":")[1])].encode("utf-8"))
        assert [a.logprob for a in entry.top_logprobs] == sorted(
            [a.logprob for a in entry.top_logprobs], reverse=True)

    async def test_decoded_token_form(self):
        svc = make_service(word_model(), word_tokenizer())
        entry = (await svc.create_chat_completion(
            chat(return_tokens_as_token_ids=False))).choices[0].logprobs.content[0]
        assert entry.token in {WORDS[i] for i in ALLOWED}
        assert entry.bytes == list(entry.token.encode("utf-8"))

    async def test_top_logprobs_zero_and_absent_give_an_empty_list(self):
        svc = make_service(word_model(), word_tokenizer())
        for over in ({"top_logprobs": 0}, {"top_logprobs": None}):
            entry = (await svc.create_chat_completion(chat(**over))).choices[0].logprobs.content[0]
            assert entry.top_logprobs == []

    async def test_allowed_ids_alone_constrain_and_return_null_logprobs(self):
        svc = make_service(word_model(), word_tokenizer())
        resp = await svc.create_chat_completion(chat(logprobs=None, top_logprobs=None))
        assert resp.choices[0].logprobs is None
        assert resp.choices[0].message.content in {WORDS[i] for i in ALLOWED}

    def test_over_http_the_shape_survives_response_model_filtering(self):
        client, _ = make_client(make_service(word_model(), word_tokenizer()), model_row())
        body = chat(extra_messages=[[MESSAGES[1]]]).model_dump(exclude_none=True,
                                                                exclude_unset=True)
        r = client.post("/v1/chat/completions", json=body)
        assert r.status_code == 200, r.text
        choice = r.json()["choices"][0]
        entry = choice["logprobs"]["content"][0]
        assert set(entry) == {"token", "logprob", "bytes", "top_logprobs"}
        assert set(entry["top_logprobs"][0]) == {"token", "logprob", "bytes"}
        assert r.headers["X-miLLM-Batch"] == "2"
        assert len(r.json()["choices"]) == 2

    def test_a_generation_response_carries_no_logprobs_key(self):
        """FPRD §8 compatibility: a client sending no new field gets the body it got before."""
        client, _ = make_client(make_service(word_model(), word_tokenizer()), model_row())
        r = client.post("/v1/chat/completions",
                        json={"model": "tiny", "messages": MESSAGES, "max_tokens": 2,
                              "temperature": 0})
        assert r.status_code == 200
        assert "logprobs" not in r.json()["choices"][0]
