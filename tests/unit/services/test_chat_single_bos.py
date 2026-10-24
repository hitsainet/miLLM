"""A chat request carries exactly ONE begin-of-text token (operator-approved fix, 2026-10-08).

THE DEFECT. miLLM rendered a chat with the tokenizer's chat template — which on Llama 3, gemma and
LFM2.5 already BEGINS with the BOS text — then tokenized it with the default
`add_special_tokens=True`, which prepends another. Every live chat on those models started
`128000, 128000`. Probe scoring copied the live path on purpose, so it carried the same duplicate,
and miStudio, which trained the probes, never had it (`add_special_tokens=False`). The other way
round, chat SCORING hard-coded `add_special_tokens=False`, so a model whose template writes no BOS
but whose tokenizer adds one was scored with NONE.

THE FIXTURES ARE REAL TOKENIZERS, one per behaviour the tokenizers in use exhibit (measured on the
real ones, `0xcc/reviews/chat_double_bos_2026-10-08.md` §2):

* `template_bos`  — template writes `<s>`, tokenizer adds `<s>`   (Llama 3, gemma 2/3/4, LFM2.5)
* `tokenizer_bos` — template writes none, tokenizer adds `<s>`    (TinyLlama / Llama 2 chat)
* `no_bos`        — no `bos_token` at all, nothing added           (Qwen2.5)
* `bos_unused`    — a `bos_token` nobody writes or adds            (granite 4.x, Phi-4-mini)

Each is a `tokenizers` WordLevel model under a real `PreTrainedTokenizerFast` with a real Jinja
template, driving a tiny REAL Llama. The ids are read off the model's FIRST forward (a pre-hook),
so what is asserted is what the model actually ran — not what a helper says it would.

MUTATION CONTROLS (each must turn this file red; recorded in the review):
  * `encode_rendered_chat` with `add_special_tokens=True`            -> the duplicate returns
  * the no-BOS-template branch dropped (always False)                -> `tokenizer_bos` loses its BOS
  * one live site routed around the helper                           -> that path's test
  * probe scoring encoding differently from live serving             -> the equality test
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from millm.api.schemas.openai import ChatCompletionRequest, ChatMessage, TextCompletionRequest
from millm.api.schemas.probe_scoring import ProbeScoreInput
from millm.services.probe_scoring import ProbeInputPreparer, ProbeScoringService
from millm.services.probe_turns import last_user_token_span
from millm.services.prompt_encoding import (
    encode_prompt,
    encode_rendered_chat,
    encode_rendered_chats,
    llamacpp_completion_prompt,
    rendered_chat_ids,
)
from tests.unit.f25_fixtures import WORDS, clear_loaded, make_service, word_model

BOS_ID = WORDS.index("<s>")
PAD_ID = WORDS.index("[UNK]")
_BODY = (
    "{% for m in messages %} {{ m['role'] }} {{ m['content'] }}{% endfor %}"
    "{% if add_generation_prompt %} assistant{% endif %}"
)

#: name -> (template writes `<s>`, tokenizer adds `<s>`, tokenizer has a bos_token, BOS expected)
FAMILIES = {
    "template_bos": (True, True, True, 1),
    "tokenizer_bos": (False, True, True, 1),
    "no_bos": (False, False, False, 0),
    "bos_unused": (False, False, True, 0),
}


def family_tokenizer(name: str) -> PreTrainedTokenizerFast:
    writes, adds, has_bos, _ = FAMILIES[name]
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    if adds:
        tok.post_processor = processors.TemplateProcessing(
            single="<s> $A", special_tokens=[("<s>", BOS_ID)]
        )
    kwargs: dict[str, Any] = {"unk_token": "[UNK]", "eos_token": "</s>"}
    if has_bos:
        kwargs["bos_token"] = "<s>"
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, **kwargs)
    fast.pad_token = "[UNK]"
    fast.chat_template = ("<s>" if writes else "") + _BODY
    return fast


CHAT = [ChatMessage(role="system", content="w1 w2"), ChatMessage(role="user", content="w3 w4 w5")]
PLAIN = [{"role": m.role, "content": m.content} for m in CHAT]


def expected_ids(tok: Any, family: str, messages=PLAIN, generation_prompt=True) -> list[int]:
    """Written out from the family's declared behaviour, NOT from the helper: the render's own
    tokens with every leading BOS removed, then exactly the number of BOS the family uses."""
    render = tok.apply_chat_template(messages, tokenize=False,
                                     add_generation_prompt=generation_prompt)
    body = list(tok(render, add_special_tokens=False)["input_ids"])
    while body and body[0] == BOS_ID:
        body = body[1:]
    return [BOS_ID] * FAMILIES[family][3] + body


def leading_bos(ids: list[int]) -> int:
    n = 0
    while n < len(ids) and ids[n] == BOS_ID:
        n += 1
    return n


class FirstForward:
    """Records the input ids of the model's forwards; the first is the prompt prefill."""

    def __init__(self, model: Any) -> None:
        self.calls: list[tuple[torch.Tensor, Any]] = []

        def hook(_module, args, kwargs):
            ids = kwargs.get("input_ids", args[0] if args else None)
            if ids is not None:
                self.calls.append((ids.detach().clone(), kwargs.get("attention_mask")))

        self.handle = model.register_forward_pre_hook(hook, with_kwargs=True)

    def rows(self) -> list[list[int]]:
        ids, mask = self.calls[0]
        out = []
        for r in range(ids.shape[0]):
            row = ids[r].tolist()
            if mask is not None:
                row = [t for t, m in zip(row, mask[r].tolist()) if m]
            out.append(row)
        return out

    def first(self) -> list[int]:
        return self.rows()[0]


@pytest.fixture(params=sorted(FAMILIES))
def family(request):
    return request.param


@pytest.fixture
def served(family):
    """(service, model, tokenizer, family, recorder) over a tiny real Llama."""
    tok = family_tokenizer(family)
    model = word_model()
    svc = make_service(model, tok)
    rec = FirstForward(model)
    yield svc, model, tok, family, rec
    rec.handle.remove()
    clear_loaded()


def chat_request(**over) -> ChatCompletionRequest:
    base = dict(model="tiny", messages=CHAT, max_tokens=2, temperature=0.0)
    base.update(over)
    return ChatCompletionRequest(**base)


# ── the rule itself, on every family ─────────────────────────────────────────────


class TestTheRule:
    def test_a_render_gets_exactly_the_bos_its_family_uses(self, family):
        tok = family_tokenizer(family)
        render = tok.apply_chat_template(PLAIN, tokenize=False, add_generation_prompt=True)
        ids = rendered_chat_ids(tok, render)
        assert ids == expected_ids(tok, family)
        assert leading_bos(ids) == FAMILIES[family][3]

    def test_the_defect_is_real_on_the_fixture(self):
        """Precondition: on `template_bos` the tokenizer's default DOES duplicate the BOS, so the
        tests below can tell the two apart."""
        tok = family_tokenizer("template_bos")
        render = tok.apply_chat_template(PLAIN, tokenize=False, add_generation_prompt=True)
        assert leading_bos(list(tok(render)["input_ids"])) == 2

    def test_a_padded_batch_applies_the_rule_per_row(self, family):
        tok = family_tokenizer(family)
        renders = [tok.apply_chat_template(PLAIN, tokenize=False, add_generation_prompt=True),
                   tok.apply_chat_template(PLAIN[1:], tokenize=False, add_generation_prompt=True)]
        enc = encode_rendered_chats(tok, renders, padding=True, padding_side="left",
                                    return_tensors="pt")
        rows = [[t for t, m in zip(r.tolist(), mk.tolist()) if m]
                for r, mk in zip(enc["input_ids"], enc["attention_mask"])]
        assert rows == [expected_ids(tok, family), expected_ids(tok, family, PLAIN[1:])]

    def test_rows_that_disagree_are_each_encoded_on_their_own(self):
        tok = family_tokenizer("tokenizer_bos")
        with_bos = "<s> user w3 assistant"
        without = "user w4 w5 assistant"
        enc = encode_rendered_chats(tok, [with_bos, without], padding=True,
                                    padding_side="left", return_tensors="pt")
        rows = [[t for t, m in zip(r.tolist(), mk.tolist()) if m]
                for r, mk in zip(enc["input_ids"], enc["attention_mask"])]
        assert [leading_bos(r) for r in rows] == [1, 1]
        assert enc["input_ids"][1].tolist()[-1] == WORDS.index("assistant")  # left-padded

    def test_raw_text_honours_the_callers_choice(self):
        tok = family_tokenizer("template_bos")
        assert leading_bos(list(encode_prompt(tok, "w1 w2", rendered_chat=False)["input_ids"])) == 1
        assert leading_bos(list(encode_prompt(
            tok, "w1 w2", rendered_chat=False, add_special_tokens=False)["input_ids"])) == 0

    def test_the_caller_cannot_override_the_decision(self):
        tok = family_tokenizer("template_bos")
        with pytest.raises(TypeError):
            encode_rendered_chat(tok, "<s> user w1", add_special_tokens=True)
        with pytest.raises(TypeError):
            encode_rendered_chats(tok, ["<s> user w1"], add_special_tokens=True)


# ── live generation: what the model RAN ──────────────────────────────────────────


class TestLiveGeneration:
    async def test_non_streaming_chat(self, served):
        svc, _, tok, family, rec = served
        response = await svc.create_chat_completion(chat_request())
        assert rec.first() == expected_ids(tok, family)
        assert response.usage.prompt_tokens == len(rec.first())

    async def test_streaming_chat(self, served):
        svc, _, tok, family, rec = served
        async for _chunk in svc.stream_chat_completion(chat_request(stream=True)):
            pass
        assert rec.first() == expected_ids(tok, family)

    async def test_n_greater_than_one(self, served):
        svc, _, tok, family, rec = served
        await svc.create_chat_completion(chat_request(n=2))
        assert rec.first() == expected_ids(tok, family)

    async def test_batched_chat_every_row(self, served):
        svc, _, tok, family, rec = served
        second = [{"role": "user", "content": "w6"}]
        await svc.create_chat_completion(chat_request(extra_messages=[second]))
        assert rec.rows() == [expected_ids(tok, family), expected_ids(tok, family, second)]

    async def test_the_admission_check_and_the_activation_cap_count_the_same_ids(
        self, served, monkeypatch
    ):
        svc, _, tok, family, rec = served
        seen = []
        real = svc._check_context_length
        monkeypatch.setattr(svc, "_check_context_length",
                            lambda n, m: (seen.append(n), real(n, m))[1])
        req = chat_request()
        svc.check_stream_admission(req)
        assert seen == [len(expected_ids(tok, family))]
        assert svc.count_prompt_tokens(req, chat=True) == len(expected_ids(tok, family))

    async def test_continuous_batching_chat(self, served, monkeypatch):
        svc, _, tok, family, _ = served
        captured = _route_to_fake_cbm(svc, monkeypatch)
        await svc.create_chat_completion(chat_request())
        assert captured == [expected_ids(tok, family)]

    async def test_continuous_batching_stream(self, served, monkeypatch):
        svc, _, tok, family, _ = served
        captured = _route_to_fake_cbm(svc, monkeypatch)
        async for _chunk in svc.stream_chat_completion(chat_request(stream=True)):
            pass
        assert captured == [expected_ids(tok, family)]


def _route_to_fake_cbm(svc: Any, monkeypatch) -> list[list[int]]:
    from millm.core.config import settings

    captured: list[list[int]] = []

    class CapturingCBM:
        is_running = True
        _default_temperature = 1.0
        _default_top_p = 1.0

        def sampling_params_match(self, temperature, top_p):
            return True

        async def generate(self, input_ids, max_new_tokens, request_id):
            captured.append(list(input_ids))
            return [7], "stop"

        async def generate_stream(self, input_ids, max_new_tokens, request_id):
            captured.append(list(input_ids))
            yield [7]

    monkeypatch.setattr(settings, "PROBE_FORCE_SERIAL", False)
    svc._cbm_backend = CapturingCBM()
    monkeypatch.setattr(svc, "_use_cbm", lambda: True)
    return captured


# ── raw text keeps its own setting ───────────────────────────────────────────────


class TestTextCompletion:
    @pytest.mark.parametrize("special,bos", [(True, 1), (False, 0)])
    async def test_generation_honours_add_special_tokens(self, special, bos):
        """Raw text is not a template render. Generation ignored the field before this fix (only
        scoring read it), so a template-exact prompt sent for GENERATION still got a duplicate."""
        tok = family_tokenizer("template_bos")
        model = word_model()
        svc = make_service(model, tok)
        rec = FirstForward(model)
        try:
            await svc.create_text_completion(TextCompletionRequest(
                model="tiny", prompt="w1 w2", max_tokens=2, temperature=0.0,
                add_special_tokens=special))
            assert leading_bos(rec.first()) == bos
            assert svc.count_prompt_tokens(TextCompletionRequest(
                model="tiny", prompt="w1 w2", add_special_tokens=special), chat=False
            ) == len(rec.first())
        finally:
            rec.handle.remove()
            clear_loaded()


class TestContinuousBatchingText:
    @pytest.mark.parametrize("special,bos", [(True, 1), (False, 0)])
    async def test_cbm_text_honours_add_special_tokens(self, special, bos, monkeypatch):
        """Review control M13 survived without this: the CBM text path had no test at all."""
        tok = family_tokenizer("template_bos")
        svc = make_service(word_model(), tok)
        try:
            captured = _route_to_fake_cbm(svc, monkeypatch)
            await svc.create_text_completion(TextCompletionRequest(
                model="tiny", prompt="w1 w2", max_tokens=2, temperature=1.0,
                add_special_tokens=special))
            assert len(captured) == 1, "the request did not reach continuous batching"
            assert leading_bos(captured[0]) == bos
        finally:
            clear_loaded()


# ── chat scoring ─────────────────────────────────────────────────────────────────


class TestChatScoring:
    async def test_scoring_reads_the_ids_generation_reads(self, served):
        """Was `add_special_tokens=False` unconditionally: right for `template_bos`, and NO BOS
        at all for `tokenizer_bos`."""
        svc, _, tok, family, rec = served
        await svc.create_chat_completion(chat_request(logprobs=True, top_logprobs=2,
                                                      max_tokens=1))
        assert rec.first() == expected_ids(tok, family)


# ── probe scoring equals live serving ────────────────────────────────────────────


class TestProbeScoringEqualsLive:
    async def test_probe_scoring_ids_are_the_live_ids(self, served):
        svc, _, tok, family, rec = served
        await svc.create_chat_completion(chat_request())
        renderer = ProbeScoringService(None, svc)._renderer(tok)
        prepared = ProbeInputPreparer(tok, renderer).prepare(
            0, ProbeScoreInput(messages=[{"role": m.role, "content": m.content} for m in CHAT])
        )
        assert prepared.ids == rec.first()
        assert prepared.prompt_tokens == len(rec.first())

    def test_an_assistant_ended_input_has_one_bos_too(self, family):
        tok = family_tokenizer(family)
        svc = make_service(word_model(), tok)
        try:
            msgs = PLAIN + [{"role": "assistant", "content": "w6 w7"}]
            prepared = ProbeInputPreparer(tok, ProbeScoringService(None, svc)._renderer(tok)
                                          ).prepare(0, ProbeScoreInput(messages=msgs))
            assert prepared.ids == expected_ids(tok, family, msgs, generation_prompt=False)
            assert prepared.prompt_tokens == len(expected_ids(tok, family))
        finally:
            clear_loaded()


# ── the windows sit where the ids are ────────────────────────────────────────────


class TestWindowBoundaries:
    async def test_last_user_resolves_on_the_live_ids(self, served):
        """`probe_turns` renders without special tokens and places the span by offset, so it must
        resolve — and land on the user turn — on whatever the live path now serves."""
        svc, _, tok, family, rec = served
        await svc.create_chat_completion(chat_request())
        span, reason = last_user_token_span(tok, PLAIN, rec.first(), generation_prompt=True)
        assert reason is None
        assert tok.convert_ids_to_tokens(rec.first()[span[0]: span[1]]) == [
            "user", "w3", "w4", "w5",
        ]


# ── llama.cpp's continuation path ────────────────────────────────────────────────


class _Llama:
    def __init__(self, adds_bos: bool) -> None:
        self.adds_bos = adds_bos

    def token_bos(self) -> int:
        return 2

    def tokenize(self, text: bytes, add_bos: bool = True, special: bool = False) -> list[int]:
        return [2] if (add_bos and self.adds_bos) else []


class TestLlamaCppContinuation:
    def test_a_render_with_bos_loses_it_when_llama_cpp_adds_one(self):
        assert llamacpp_completion_prompt(_Llama(True), "<bos>turn", "<bos>") == "turn"

    def test_it_keeps_it_when_llama_cpp_adds_none(self):
        assert llamacpp_completion_prompt(_Llama(False), "<bos>turn", "<bos>") == "<bos>turn"

    def test_a_render_without_bos_is_untouched(self):
        assert llamacpp_completion_prompt(_Llama(True), "turn", "<bos>") == "turn"

    def test_it_is_wired_into_the_continuation_prompt(self):
        from unittest.mock import MagicMock

        from millm.services.inference_service import InferenceService

        svc = InferenceService.__new__(InferenceService)
        model = MagicMock()
        model.metadata = {"tokenizer.chat_template": (
            "{{ bos_token }}{% for m in messages %}<t>{{ m.role }}\n{{ m.content }}</t>\n"
            "{% endfor %}{% if add_generation_prompt %}<t>assistant\n{% endif %}")}
        model.token_eos.return_value = 1
        model.token_bos.return_value = 2
        model.tokenize.side_effect = lambda text, add_bos=True, special=False: (
            [2] if add_bos else [])
        model._model.token_get_text.side_effect = lambda t: "<eos>" if t == 1 else "<bos>"
        state = MagicMock()
        state.is_loaded = True
        state.current.model = model
        svc._model_state = state
        prompt = svc._llamacpp_continuation_prompt(
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "PARTIAL"}])
        assert prompt.startswith("<t>user") and prompt.endswith("PARTIAL")


# ── the real tokenizers, when this machine has them ─────────────────────────────


def _real_tokenizer_dirs() -> list[Path]:
    """`MILLM_REAL_TOKENIZERS` (os.pathsep-separated directories), else the HuggingFace cache's
    snapshots of the families named in the review record. CI has none; this then skips LOUDLY,
    and the fixtures above carry the guarantee."""
    configured = os.environ.get("MILLM_REAL_TOKENIZERS")
    if configured:
        return [Path(p) for p in configured.split(os.pathsep) if p]
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    found = []
    for repo in ("models--TinyLlama--TinyLlama-1.1B-Chat-v1.0", "models--google--gemma-2-2b-it",
                 "models--google--gemma-4-12b-it", "models--microsoft--Phi-4-mini-instruct"):
        found += sorted((hub / repo / "snapshots").glob("*"))
    return [p for p in found if (p / "tokenizer_config.json").exists()]


def test_real_tokenizers_get_at_most_one_bos():
    dirs = _real_tokenizer_dirs()
    if not dirs:
        pytest.skip("NO REAL TOKENIZER ON THIS MACHINE — set MILLM_REAL_TOKENIZERS to check the "
                    "rule against Llama 3 / gemma / LFM2.5 / Qwen; the fixture families still ran")
    from transformers import AutoTokenizer

    checked = 0
    for path in dirs:
        try:
            tok = AutoTokenizer.from_pretrained(str(path))
        except Exception:  # noqa: BLE001 - a snapshot this transformers cannot read
            continue
        if not getattr(tok, "chat_template", None):
            continue
        render = tok.apply_chat_template([{"role": "user", "content": "Hello there"}],
                                         tokenize=False, add_generation_prompt=True)
        ids = rendered_chat_ids(tok, render)
        bos_id = tok.bos_token_id
        uses_bos = bos_id is not None and (
            render.startswith(tok.bos_token) or list(tok("x")["input_ids"])[:1] == [bos_id])
        assert ids[:2] != [bos_id, bos_id], f"{path}: duplicate BOS"
        if uses_bos:
            assert ids[0] == bos_id, f"{path}: the model's BOS is missing"
        checked += 1
    assert checked, "no readable tokenizer among " + ", ".join(map(str, dirs))


# ── packed (Batch API) chat scoring ──────────────────────────────────────────────


class TestPackedChatScoring:
    async def test_packed_rows_follow_the_rule(self, served, monkeypatch):
        """The Batch API's packed scorer: a chat spec is a template render (`rendered_chat`)."""
        from millm.services.inference_service import ScoreSpec

        svc, _, tok, family, _ = served
        rows_seen: list[list[list[int]]] = []
        real = svc._packed_next_token_logits
        monkeypatch.setattr(svc, "_packed_next_token_logits",
                            lambda rows: rows_seen.append([list(r) for r in rows]) or real(rows))
        renders = [svc._format_chat_messages(CHAT),
                   svc._format_chat_messages([ChatMessage(role="user", content="w6")])]
        async with svc._admit():
            await svc._score_specs_packed(
                [ScoreSpec(r, False, None, 1.0, 2, rendered_chat=True) for r in renders],
                max_rows=4)
        assert sorted(map(tuple, rows_seen[0])) == sorted([
            tuple(expected_ids(tok, family)),
            tuple(expected_ids(tok, family, [{"role": "user", "content": "w6"}])),
        ])
