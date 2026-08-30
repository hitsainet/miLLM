"""Batched chat completion: one forward pass for N conversations.

The fixtures here are deliberately INPUT-SENSITIVE, unlike the shared
`mock_tokenizer` in test_inference_service.py which returns a hard-coded
one-row tensor regardless of its arguments. A fixture that cannot pad cannot
observe a padding bug, and a model mock that returns a fixed single row cannot
observe a per-row bug — the batch defects this module guards against are
exactly the ones such fixtures agree away.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
import torch

from millm.api.schemas.openai import ChatCompletionRequest, ChatMessage
from millm.services.inference_service import InferenceService
from millm.ml.model_loader import LoadedModel, LoadedModelState

PAD = 0          # gemma's real pad id, and falsy
EOS = 2
GEN_LEN = 6


@pytest.fixture(autouse=True)
def _reset_state():
    state = LoadedModelState()
    state._loaded = None
    yield
    state._loaded = None


class FakeTokenizer:
    """Tokenizes by word count and pads for real, honouring padding_side."""

    def __init__(self):
        self.pad_token_id = PAD
        self.eos_token_id = EOS
        self.chat_template = None
        self.last_padding_side = None
        self.decoded: list[list[int]] = []

    def encode(self, text):
        return [5] * len(text.split())

    def __call__(self, prompts, return_tensors=None, padding=False,
                 padding_side=None, **kwargs):
        self.last_padding_side = padding_side
        if isinstance(prompts, str):
            prompts = [prompts]
        rows = [self.encode(p) for p in prompts]
        width = max(len(r) for r in rows)
        ids, mask = [], []
        for r in rows:
            pad_n = width - len(r)
            if padding_side == "left":
                ids.append([PAD] * pad_n + r)
                mask.append([0] * pad_n + [1] * len(r))
            else:
                ids.append(r + [PAD] * pad_n)
                mask.append([1] * len(r) + [0] * pad_n)
        enc = {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(mask),
        }
        return _Encoded(enc)

    def decode(self, ids, skip_special_tokens=True):
        toks = [int(t) for t in ids]
        self.decoded.append(toks)
        return " ".join(str(t) for t in toks if t not in (PAD, EOS))

    def apply_chat_template(self, *a, **k):
        raise AssertionError("not used in these tests")


class _Encoded(dict):
    """Supports both `enc["input_ids"]` and `enc.input_ids`.

    The batch path subscripts; the single path uses attribute access. A fixture
    that offered only one would make the other untestable here.
    """

    def to(self, device):
        return self

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


@pytest.fixture
def model():
    m = MagicMock()
    m.config = MagicMock()
    m.config.max_position_embeddings = 4096
    m.config.num_hidden_layers = 4
    m.config.hidden_size = 64
    m.config.num_attention_heads = 4
    m.config.num_key_value_heads = 4
    m.config.head_dim = 16
    m.device = "cpu"

    def _generate(**kwargs):
        ids = kwargs["input_ids"]
        b, w = ids.shape
        # Row 0 stops early (EOS then pad filler); the others run to the cap.
        tail = []
        for i in range(b):
            if i == 0:
                tail.append([21, 22, EOS] + [PAD] * (GEN_LEN - 3))
            else:
                tail.append([31, 32, 33, 34, 35, 36][:GEN_LEN])
        return torch.cat([ids, torch.tensor(tail)], dim=1)

    m.generate = MagicMock(side_effect=_generate)
    return m


@pytest.fixture
def service(model, tokenizer):
    LoadedModelState().set(
        LoadedModel(
            model_id=1, model_name="test-model", model=model,
            tokenizer=tokenizer, loaded_at=datetime(2026, 1, 1),
            memory_used_mb=1024, num_parameters=1, device="cpu",
            dtype="float16",
        )
    )
    with patch("millm.services.inference_service.torch") as t:
        t.cuda.is_available.return_value = False
        t.no_grad.return_value = MagicMock(__enter__=MagicMock(),
                                           __exit__=MagicMock())
        svc = InferenceService(model_service=None)
    svc._device = "cpu"
    svc._format_chat_messages = lambda msgs: msgs[0].content
    svc._use_cbm_for_request = MagicMock(return_value=False)
    svc._get_draft_model = MagicMock(return_value=None)
    svc._notify_monitoring = MagicMock()
    return svc


def _req(*prompts, max_tokens=GEN_LEN):
    def conv(p):
        return [ChatMessage(role="user", content=p)]
    return ChatCompletionRequest(
        model="test-model",
        messages=conv(prompts[0]),
        extra_messages=[conv(p) for p in prompts[1:]] or None,
        max_tokens=max_tokens,
    )


class TestBatchedChatCompletion:
    @pytest.mark.asyncio
    async def test_one_choice_per_conversation_in_input_order(self, service):
        resp = await service.create_chat_completion(
            _req("one", "two two", "three three three")
        )
        assert len(resp.choices) == 3
        assert [c.index for c in resp.choices] == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_one_forward_pass_for_the_whole_batch(self, service, model):
        """The point of the feature: weights read once, not N times."""
        await service.create_chat_completion(_req("a", "b b", "c c c"))
        assert model.generate.call_count == 1, (
            "the batch was generated row-by-row; the throughput gain is gone"
        )
        assert model.generate.call_args.kwargs["input_ids"].shape[0] == 3

    @pytest.mark.asyncio
    async def test_padding_is_left_side(self, service, tokenizer):
        """The highest-risk item, and its failure is silent.

        Right padding puts pads BETWEEN prompt and first generated token, so
        every row but the longest continues from padding and produces fluent
        garbage while raising nothing.
        """
        await service.create_chat_completion(_req("a", "b b", "c c c"))
        assert tokenizer.last_padding_side == "left", (
            "batch tokenized with right padding: short rows generate from pad "
            "tokens and silently return garbage"
        )

    @pytest.mark.asyncio
    async def test_short_row_decodes_only_its_own_generation(self, service,
                                                             tokenizer):
        """Left padding means the generated slice starts at the padded width.

        With right padding the same slice would carry another row's pads.
        """
        await service.create_chat_completion(_req("a", "b b", "c c c"))
        for toks in tokenizer.decoded:
            assert PAD not in toks[:1], "generated slice began at a pad token"

    @pytest.mark.asyncio
    async def test_early_finisher_reports_its_own_length_not_the_batch(
        self, service
    ):
        """generate() pads finished rows to the batch's length.

        Untrimmed, row 0 would report GEN_LEN tokens and inherit the batch's
        "length" finish_reason.
        """
        resp = await service.create_chat_completion(_req("a", "b b", "c c c"))
        row0 = next(c for c in resp.choices if c.index == 0)
        row1 = next(c for c in resp.choices if c.index == 1)
        assert row0.finish_reason == "stop", (
            "a row that emitted EOS reported the batch's finish_reason"
        )
        assert row1.finish_reason == "length"

    @pytest.mark.asyncio
    async def test_usage_counts_real_tokens_not_padding(self, service):
        """prompt_tokens is the sum of true lengths, not batch x padded width."""
        resp = await service.create_chat_completion(
            _req("a", "b b", "c c c")  # true lengths 1 + 2 + 3
        )
        assert resp.usage.prompt_tokens == 6, (
            f"expected 6 real prompt tokens, got {resp.usage.prompt_tokens} "
            "(padded width x batch is 9)"
        )
        # row0 stops at 3 (incl. EOS), rows 1 and 2 run to GEN_LEN.
        assert resp.usage.completion_tokens == 3 + GEN_LEN + GEN_LEN

    @pytest.mark.asyncio
    async def test_pad_id_of_zero_is_passed_through_not_replaced_by_eos(
        self, service, model
    ):
        await service.create_chat_completion(_req("a", "b b"))
        assert model.generate.call_args.kwargs["pad_token_id"] == PAD

    @pytest.mark.asyncio
    async def test_single_conversation_is_unaffected(self, service, model):
        """No extra_messages -> the ordinary path, not the batch path."""
        req = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="a")],
        )
        assert req.extra_messages is None
        service._create_batched_chat_completion = MagicMock(
            side_effect=AssertionError("batch path taken for a single request")
        )
        await service.create_chat_completion(req)


class TestBatchSafety:
    @pytest.mark.asyncio
    async def test_speculative_decoding_falls_back_rather_than_raising(
        self, service, model
    ):
        """transformers rejects assisted generation for batch_size > 1."""
        service._get_draft_model = MagicMock(return_value=MagicMock())
        resp = await service.create_chat_completion(_req("a", "b b"))
        assert len(resp.choices) == 2
        assert "assistant_model" not in model.generate.call_args.kwargs

    @pytest.mark.asyncio
    async def test_batch_larger_than_cap_is_chunked_not_refused(
        self, service, model
    ):
        """The caller asked for N conversations and gets N back."""
        service.MAX_BATCH_ROWS = 2
        resp = await service.create_chat_completion(_req("a", "b", "c", "d", "e"))
        assert len(resp.choices) == 5
        assert [c.index for c in resp.choices] == [0, 1, 2, 3, 4]
        assert model.generate.call_count == 3  # 2 + 2 + 1

    def test_unmeasurable_config_keeps_the_row_cap(self, service):
        """An unmeasurable batch must not become an unbounded one.

        CUDA must be patched TRUE here. The projection block is guarded by
        is_cuda_available(), which is False under this fixture, so without the
        patch the code under test never executes and this assertion passes
        against any implementation at all — which is exactly how the first
        version of this test let a mutation survive.
        """
        service._model.config.num_hidden_layers = None
        assert service._project_kv_bytes(8, 100) is None

        with patch("millm.ml.memory_utils.is_cuda_available",
                   return_value=True), \
             patch("millm.ml.memory_utils.verify_memory_available",
                   return_value=(False, 0)):
            chunks = service._chunk_batch_for_memory(["a"] * 20, 10)

        assert all(len(c) <= service.MAX_BATCH_ROWS for _, c in chunks), (
            "an unmeasurable KV projection grew the batch past the row cap"
        )

    def test_projection_shrinks_the_batch_when_memory_is_short(self, service):
        """And the guard must actually bite when the projection IS available."""
        with patch("millm.ml.memory_utils.is_cuda_available",
                   return_value=True), \
             patch("millm.ml.memory_utils.verify_memory_available",
                   return_value=(False, 1)):
            chunks = service._chunk_batch_for_memory(["a b c"] * 8, 64)
        assert all(len(c) == 1 for _, c in chunks), (
            "memory was exhausted at every size yet the batch was not reduced"
        )

    @pytest.mark.asyncio
    async def test_sensing_is_refused_out_loud_for_a_batch(self, service):
        """Going dark silently is the failure this project has shipped before."""
        armed = MagicMock()
        armed.is_sensing_armed = True
        state = MagicMock()
        state.attached_sae = armed
        with patch("millm.services.sae_service.AttachedSAEState",
                   return_value=state):
            with patch("millm.services.inference_service.logger") as log:
                await service.create_chat_completion(_req("a", "b"))
        reasons = [
            c.kwargs.get("reason") for c in log.info.call_args_list
        ]
        assert "batched_request" in reasons, (
            "sensing was skipped for a batch with no log line, while "
            "/api/sensing/status still reports armed"
        )

    @pytest.mark.asyncio
    async def test_batch_holds_exactly_one_queue_slot(self, service):
        """MAX_CONCURRENT_REQUESTS=1 is what isolates steering.

        A batch must not become N slots by another name.
        """
        acquires = []
        real = service._request_queue.acquire

        def _tracked(*a, **k):
            acquires.append(1)
            return real(*a, **k)

        service._request_queue.acquire = _tracked
        await service.create_chat_completion(_req("a", "b", "c"))
        assert len(acquires) == 1
