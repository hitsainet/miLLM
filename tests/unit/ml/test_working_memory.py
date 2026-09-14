"""A card's working memory — beyond its weights, KV cache and CUDA context — is traced from the model, not guessed.

Hardware acceptance, 2026-09-14, items 4 and 10 (0xcc/reviews/multi_gpu_phase2_acceptance_2026-09-14.md):
OLMo-2-13B at FP16 was admitted on cuda:0 with 11 MiB to spare and a 3,879 + 217-token request ran
it out of memory ("Tried to allocate 76.00 MiB ... 11.22 GiB allocated, 285 MiB reserved but
unallocated"); the Q8 split did the same. millm/ml/working_memory.py sizes a card's prefill
activations by running the model's own forward on the meta device and following every storage,
and adds the caching allocator's share, calibrated by replaying traced requests through
tests/support/caching_allocator.py.

Shapes are the real configs' fields (config.json), built on the meta device only.

What is alive at a decoder layer's peak (read off the trace, derived here by hand):
  OLMo-2-13B (hidden 5,120, head_dim 128), a token:
      2 x 5,120  the embedding output (transformers keeps `inputs_embeds` for the whole forward)
    + 2 x 5,120  the layer's input
    + 6 x 5,120  q and k after their RMSNorms and v, bfloat16
    + 16 x 5,120 q's rotary sum and k's two products and sum: OLMo-2's RoPE returns float32
                 cos/sin, so all four are float32
    + 2 x 4 x 128 cos and sin (float32)   + 16 the input and position ids (int64)
    = 134,160 bytes
  Qwen2.5-7B (hidden 3,584, intermediate 18,944, head_dim 128), a token:
      8 x 3,584  the embedding output, the layer input, the residual and the MLP's normed input
    + 6 x 18,944 act_fn(gate), up and their product (bfloat16)
    + 2 x 2 x 128 cos and sin (bfloat16) + 16 ids
    = 142,864 bytes

MUTATION CONTROLS for the working-memory fit (millm-p2-accept-fix/mutate.py; each restored and its
sha256 verified, no backup left behind). Run over this file, test_per_card_fit, test_fit_rebalance,
test_split_preflight, test_fit_admitted_context, test_sae_attach_kv_reserve,
test_idle_cache_release and test_unload_admission. First run, by the session that wrote the fix
(its per-test output was not kept):
  AF1-M1  a card's need leaves out its working memory                         -> RED
  AF1-M2  a card's need leaves out the bitsandbytes staging                   -> RED
  AF1-M3  the fit is built with no working memory (working=None)              -> RED
  AF1-M4  ALLOCATOR_OVERHEAD_FRACTION = 0.0                                   -> RED
  AF1-M5  ALLOCATOR_OVERHEAD_FRACTION = 0.30                                  -> RED
  AF1-M6  flash attention's output traced as a (batch, length, heads, dim) buffer -> RED
  AF1-M7  bitsandbytes' int8 kernels not modelled (kernel = None)             -> RED
  AF1-M8  a split's card limits keep back only the CUDA context               -> RED
  AF1-M9  every card of a split charged for the input and output phases       -> SURVIVED
  AF1-M10 a split's card counts no bitsandbytes staging (quantized_bytes=0)   -> RED
  AF1-M11 SAE attachment reads no working reserve                             -> RED
  AF1-M12 a split's placement does not carry working_mb_by_device             -> RED
  AF1-M13 a named card's placement does not carry working_mb_by_device        -> SURVIVED
  AF1-M14 Auto's one-card placement does not carry working_mb_by_device       -> RED
  AF1-M15 an estimated working memory is not logged as an error               -> RED
  AF1-M16 the prefill's KV cache and logits counted in the transient peak     -> RED
  AF1-M17 BNB_LOAD_OVERHEAD_FRACTION = 0.04                                   -> RED
  AF1-M18 a split's plan need leaves out the allocator's share of the KV cache -> RED
The two survivors were test gaps. In every real model traced here the decoder layer's peak is
the largest, so charging a card for phases it does not run changed no figure; and no test
accepted a named card. Closed, and re-run (millm-p2-accept-fix2, the same eight files, 140 tests):
  AF1-M9  -> 2 red: test_per_card_fit::TestACardOfASplitIsChargedOnlyThePhasesItRuns, both tests
  AF1-M13 -> 1 red: test_per_card_fit::test_a_named_card_that_holds_the_model_carries_its_working_memory
Added with them:
  AF1-M19 a split's input and output devices swapped in TransformersFit.layout
          -> 2 red, the same two tests as AF1-M9
  AF1-M5  (0.30) re-run on this file alone -> 4 red, among them
          test_the_fits_allowance_covers_what_the_node_measured[2077-943] and [3083-1129]: the
          node's own figures now pin the allocator's share
"""

from __future__ import annotations

import functools
import math
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("transformers")
from transformers import Olmo2Config, Qwen2Config  # noqa: E402
from transformers.quantizers.auto import get_hf_quantizer  # noqa: E402

from millm.ml import working_memory as wm  # noqa: E402
from millm.ml.model_loader import _bitsandbytes_config, _meta_model, kv_cache_spec  # noqa: E402
from tests.support.caching_allocator import MIB, CachingAllocator, replay  # noqa: E402

#: allenai/OLMo-2-1124-13B-Instruct config.json.
OLMO2_13B = dict(
    vocab_size=100_352, hidden_size=5_120, intermediate_size=13_824, num_hidden_layers=40,
    num_attention_heads=40, num_key_value_heads=40, max_position_embeddings=4_096,
    rope_theta=500_000, tie_word_embeddings=False,
)
#: Qwen/Qwen2.5-7B-Instruct config.json.
QWEN25_7B = dict(
    vocab_size=152_064, hidden_size=3_584, intermediate_size=18_944, num_hidden_layers=28,
    num_attention_heads=28, num_key_value_heads=4, max_position_embeddings=32_768,
    rope_theta=1_000_000.0, sliding_window=131_072, use_sliding_window=False, max_window_layers=28,
    tie_word_embeddings=False,
)
#: Qwen/Qwen2.5-14B-Instruct config.json: the calibration's worst case.
QWEN25_14B = dict(
    vocab_size=152_064, hidden_size=5_120, intermediate_size=13_824, num_hidden_layers=48,
    num_attention_heads=40, num_key_value_heads=8, max_window_layers=70, sliding_window=131_072,
    use_sliding_window=False, max_position_embeddings=32_768, tie_word_embeddings=False,
)
CONFIGS = {"olmo2-13b": (Olmo2Config, OLMO2_13B), "qwen2.5-7b": (Qwen2Config, QWEN25_7B),
           "qwen2.5-14b": (Qwen2Config, QWEN25_14B)}


@functools.lru_cache(maxsize=None)
def _meta(name: str, quantization: str = "FP16"):
    config_class, fields = CONFIGS[name]
    config = config_class(**fields)
    quantizer = None
    bnb = _bitsandbytes_config(quantization)
    if bnb is not None:
        quantizer, config, device_map = get_hf_quantizer(config, bnb, "sequential", True, {})
    model = _meta_model(config, False)
    if quantizer is not None:
        quantizer.preprocess_model(
            model=model, dtype=torch.bfloat16, device_map=device_map, checkpoint_files=None,
            use_kernels=False,
        )
    names = [f"model.layers.{index}" for index in range(config.num_hidden_layers)]
    return config, model, names, quantizer


def _traced(name: str, tokens: int, quantization: str = "FP16") -> wm.WorkingMemory:
    config, model, names, _ = _meta(name, quantization)
    return wm.trace_working_memory(model, names, tokens, config.vocab_size)


class TestWhatALayerHoldsAtItsPeak:
    @pytest.mark.parametrize("tokens", [1_000, 2_000])
    def test_olmo2_13b_peaks_at_its_float32_rotary_step(self, tokens):
        """Layer 0's input IS the embedding output, so it holds one hidden-wide tensor fewer."""
        working = _traced("olmo2-13b", tokens)
        per_token = 26 * 5_120 + 2 * 4 * 128 + 16
        assert working.layer_bytes[0] == tokens * (per_token - 2 * 5_120)
        assert set(working.layer_bytes[1:]) == {tokens * per_token}

    def test_qwen25_7b_peaks_in_its_mlp(self):
        working = _traced("qwen2.5-7b", 1_000)
        per_token = 8 * 3_584 + 6 * 18_944 + 2 * 2 * 128 + 16
        assert working.layer_bytes[0] == 1_000 * (per_token - 2 * 3_584)
        assert set(working.layer_bytes[1:]) == {1_000 * per_token}

    def test_an_int8_prefill_runs_through_the_modelled_bitsandbytes_kernels(self):
        """On the meta device bitsandbytes' own fake kernels cannot run (their outlier columns
        are a dynamic size), so the int8 matmul must go through the modelled CUDA kernels."""
        calls: list[str] = []
        kernels = {
            name: (lambda kernel, name: lambda *a, **k: (calls.append(name), kernel(*a, **k))[1])(kernel, name)
            for name, kernel in wm._KERNELS.items()
        }
        with patch.dict(wm._KERNELS, kernels):
            q8 = _traced("olmo2-13b", 1_000, "Q8")
        fp16 = _traced("olmo2-13b", 1_000)

        assert calls.count("bitsandbytes::int8_vectorwise_quant") == 7 * 40
        assert calls.count("bitsandbytes::int8_mixed_scaled_mm") == 7 * 40
        assert max(q8.layer_bytes) > max(fp16.layer_bytes), "the int32 product and dequantisation add to the peak"


class TestACardsWorkingMemory:
    WORKING = wm.WorkingMemory(
        tokens=10, layer_bytes=(100 * MIB, 300 * MIB, 200 * MIB), input_bytes=50 * MIB,
        output_bytes=400 * MIB, sampling_bytes=MIB, method="traced",
    )

    def test_its_peak_is_the_largest_phase_it_runs_plus_the_allocators_share(self):
        assert self.WORKING.card_mb([0, 1], holds_input=True, holds_output=False, kv_mb=1_000) == (
            300 + math.ceil(0.40 * (300 + 1_000))
        )
        assert self.WORKING.card_mb([2], holds_input=False, holds_output=True, kv_mb=0) == 400 + 160
        assert self.WORKING.transient_mb([0], holds_input=True, holds_output=False) == 100

    @pytest.mark.parametrize(
        "name, layers, tokens",
        [("olmo2-13b", 15, 4_096), ("qwen2.5-14b", 16, 4_096)],
    )
    def test_the_allowance_serves_a_request_the_allocator_model_replays(self, name, layers, tokens):
        """OLMo-2-13B's cuda:0 on the node, and the calibration's worst case (Qwen2.5-14B's first
        16 of 48 layers, excess 0.338 x (T + KV)): a prefill of S-16 tokens and 16 decoded, on a
        card holding exactly its KV cache at S and its working memory, never runs out. Without
        the allocator's share it does."""
        config, model, names, _ = _meta(name)
        working = wm.trace_working_memory(model, names, tokens, config.vocab_size)
        kv_mb = kv_cache_spec(config)[0].mb(range(layers), tokens)
        card = {wm.INPUT, *range(layers)}
        events, _ = wm.trace_events(model, names, tokens - 16, decode_steps=16)
        need = working.card_mb(range(layers), holds_input=True, holds_output=False, kv_mb=kv_mb)
        transient = working.transient_mb(range(layers), holds_input=True, holds_output=False)

        assert replay(events, CachingAllocator((kv_mb + need) * MIB), card.__contains__) is None
        assert replay(events, CachingAllocator((kv_mb + transient) * MIB), card.__contains__) is not None


def test_the_trace_reproduces_the_nodes_out_of_memory_error():
    """17:14:11Z: OLMo-2-13B FP16, cuda:0 holding layers 0-14 and 10,074 MiB of weights, a 3,879-token
    prefill. torch: "Tried to allocate 76.00 MiB ... 11.22 GiB is allocated by PyTorch, and 285.00 MiB
    is reserved by PyTorch but unallocated"; the card had 52.62 MiB free, so it could reserve 11,826."""
    config, model, names, _ = _meta("olmo2-13b")
    events, _ = wm.trace_events(model, names, 3_879)

    oom = replay(events, CachingAllocator(11_826 * MIB, reserved=10_074 * MIB), {wm.INPUT, *range(15)}.__contains__)

    assert oom is not None
    assert 2 * math.ceil(oom.size / (2 * MIB)) == 76, "torch rounds the failing request to 2 MiB"
    assert abs(oom.allocated / MIB - 11.22 * 1024) <= 16
    assert 0.85 * 285 <= (oom.reserved - oom.allocated) / MIB <= 1.15 * 285


@pytest.mark.parametrize(
    "tokens, overhead_mb", [(977 + 96, 536), (1_981 + 96, 943), (2_987 + 96, 1_129)]
)
def test_the_fits_allowance_covers_what_the_node_measured(tokens, overhead_mb):
    """The three OLMo-2-13B FP16 requests that succeeded on the node (1:18-1:20 PM ET), on cuda:0's 15
    layers: peak minus the weights (10,074 MiB) minus the request's KV cache — the working memory and
    the CUDA context together. The fit's figure for a card holding the input at that length, plus the
    smallest context the node measured after a generation (330 MiB), covers each, and by no more than
    a quarter: 650 / 946 / 1,245 MiB against 536 / 943 / 1,129 (+21%, +0.3%, +10%). With the
    allocator's share at 0.30 it no longer covers the 2,077-token request."""
    config, model, names, _ = _meta("olmo2-13b")
    working = wm.trace_working_memory(model, names, tokens, config.vocab_size)
    kv_mb = kv_cache_spec(config)[0].mb(range(15), tokens)

    allowance = working.card_mb(range(15), holds_input=True, holds_output=False, kv_mb=kv_mb) + 330

    assert overhead_mb <= allowance <= 1.25 * overhead_mb


@pytest.mark.parametrize("name, quantization", [("olmo2-13b", "FP16"), ("qwen2.5-7b", "FP16"), ("olmo2-13b", "Q8")])
def test_the_estimate_bounds_every_traced_model(name, quantization):
    config, _, _, _ = _meta(name, quantization)
    traced = _traced(name, 1_000, quantization)
    estimated = wm.estimate_working_memory(config, config.num_hidden_layers, 1_000, quantization, "test")

    assert min(estimated.layer_bytes) >= max(traced.layer_bytes)
    assert estimated.input_bytes + estimated.sampling_bytes >= traced.input_bytes + traced.sampling_bytes
    assert estimated.output_bytes >= traced.output_bytes


class TestAModelTheTraceCannotRun:
    def test_it_is_estimated_and_the_error_names_the_architecture(self):
        config, model, names, _ = _meta("olmo2-13b")
        original = model.forward

        def data_dependent(*args, **kwargs):
            raise RuntimeError("Cannot copy out of meta tensor; no data!")

        model.forward = data_dependent
        try:
            with patch.object(wm, "logger") as logger:
                working = wm.size_working_memory(model, config, names, 4_096, "FP16", "Olmo2ForCausalLM")
        finally:
            del model.forward
        assert model.forward.__func__ is original.__func__

        assert working.method == "estimated"
        assert working == wm.estimate_working_memory(config, 40, 4_096, "FP16", working.reason)
        [call] = logger.error.call_args_list
        assert call.args == ("transformers_fit_working_memory_estimated",)
        assert call.kwargs["architecture"] == "Olmo2ForCausalLM"
        assert call.kwargs["reason"] == "its forward pass could not be traced on the meta device"
        assert call.kwargs["error_type"] == "RuntimeError"
        assert not logger.info.called

    def test_one_whose_layers_cannot_be_found_is_estimated_too(self):
        config, model, _, _ = _meta("olmo2-13b")
        with patch.object(wm, "logger") as logger:
            working = wm.size_working_memory(model, config, None, 4_096, "FP16", "Olmo2ForCausalLM")

        assert working.method == "estimated"
        [call] = logger.error.call_args_list
        assert call.kwargs["reason"] == "its decoder layers could not be found"

    def test_a_traced_model_says_so_once(self):
        config, model, names, _ = _meta("qwen2.5-7b")
        with patch.object(wm, "logger") as logger:
            working = wm.size_working_memory(model, config, names, 1_000, "FP16", "Qwen2ForCausalLM")

        assert working.method == "traced"
        [call] = logger.info.call_args_list
        assert call.args == ("transformers_fit_working_memory_traced",)
        assert call.kwargs["layer_peak_mb"] == math.ceil(max(working.layer_bytes) / MIB)
        assert not logger.error.called


@pytest.mark.parametrize("quantization", ["Q8", "Q4"])
def test_the_meta_model_is_left_as_it_was(quantization):
    config, model, names, _ = _meta("olmo2-13b", quantization)
    wm.trace_working_memory(model, names, 64, config.vocab_size)

    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())
    for module in model.modules():
        kind = type(module).__name__
        if kind == "Linear8bitLt":
            assert module.state.CB is None and module.state.SCB is None
        if kind == "Linear4bit":
            assert "forward" not in vars(module)


def test_real_work_in_another_thread_is_neither_intercepted_nor_counted():
    """The fit runs in a worker thread while generation runs in others: the trace's modes are thread-local."""
    config, model, names, _ = _meta("qwen2.5-7b")
    alone = wm.trace_working_memory(model, names, 512, config.vocab_size)
    q = torch.randn(1, 4, 64, 32)
    reference = torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
    started, done = threading.Event(), threading.Event()
    seen: list[bool] = []

    def attend():
        started.set()
        while not done.is_set() or not seen:
            out = torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
            seen.append(out.device.type == "cpu" and torch.equal(out, reference))

    worker = threading.Thread(target=attend)
    worker.start()
    assert started.wait(10)
    try:
        beside = wm.trace_working_memory(model, names, 512, config.vocab_size)
    finally:
        done.set()
        worker.join(10)

    assert seen and all(seen)
    assert beside == alone


class TestTheAttentionKernel:
    @staticmethod
    def _peak(trace: wm._StorageTrace) -> int:
        live = peak = 0
        for allocated, _, nbytes, _ in trace.events:
            live += nbytes if allocated else -nbytes
            peak = max(peak, live)
        return peak

    def test_flash_attention_keeps_only_its_logsumexp(self):
        trace = wm._StorageTrace()
        q = torch.empty((1, 40, 1_001, 128), dtype=torch.bfloat16, device="meta")
        out = wm._sdpa_kernel(trace, (q, q, q), {"attn_mask": None, "is_causal": True})

        assert tuple(out.shape) == (1, 40, 1_001, 128)
        assert self._peak(trace) == 40 * 1_001 * 4

    def test_a_boolean_mask_is_materialised_at_every_head(self):
        """gemma-3's sliding layers past their window: the memory-efficient kernel's bias is a
        per-head copy of the mask, its last dimension padded to a multiple of 16."""
        trace = wm._StorageTrace()
        q = torch.empty((1, 16, 1_001, 256), dtype=torch.bfloat16, device="meta")
        mask = torch.empty((1, 1, 1_001, 1_001), dtype=torch.bool, device="meta")
        wm._sdpa_kernel(trace, (q, q, q), {"attn_mask": mask})

        assert self._peak(trace) == 1_001 * 1_001 * 2 + 16 * 1_001 * 1_008 * 2 + 16 * 1_024 * 4


class TestBitsandbytesStaging:
    QUANTIZED_PER_LAYER = 2 * (4 * 5_120 * 5_120 + 3 * 5_120 * 13_824)  # bfloat16 bytes

    def test_the_quantized_weights_are_the_linear_layers_and_nothing_else(self):
        _, model, _, quantizer = _meta("olmo2-13b", "Q8")
        found = wm.bnb_quantized_bytes_by_module(model, quantizer)

        assert sum(found.values()) == 40 * self.QUANTIZED_PER_LAYER
        assert "lm_head" not in found and not any("embed" in name for name in found)

    def test_the_staging_covers_what_the_node_landed_above_the_map(self):
        """OLMo-2-13B Q8, 26 / 14 layers: the load landed 653 / 444 MiB above its map."""
        assert wm.bnb_staging_mb([26 * self.QUANTIZED_PER_LAYER]) >= 653
        assert wm.bnb_staging_mb([14 * self.QUANTIZED_PER_LAYER]) >= 444

    def test_a_checkpoint_already_quantized_is_not_staged(self):
        _, model, _, _ = _meta("olmo2-13b", "Q8")
        assert wm.bnb_quantized_bytes_by_module(model, SimpleNamespace(pre_quantized=True)) == {}
        assert wm.bnb_quantized_bytes_by_module(model, None) == {}
