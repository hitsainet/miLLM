"""A composite (vision/audio + text) checkpoint loads when one tower has no SDPA.

Defect, 2026-10-18: `kushalpatil/jevify-gemma4-e4b` (Gemma4ForConditionalGeneration)
refused to load at FP16 and Q8 with "Gemma4AudioModel does not support an attention
implementation through torch.nn.functional.scaled_dot_product_attention". The loader
passed `attn_implementation="sdpa"` as a plain string, which transformers applies to
EVERY sub-config as a hard requirement; in transformers 5.20.1 (the serving image's
pin) Gemma 4's audio tower declares no SDPA, flash or flex path. 5.15.1 (an older dev
venv) did not refuse, which is why the defect was invisible there.

The fix asks for the implementation on the text stack only (`{"text_config": ...}`)
and leaves every other tower to transformers' default, which picks SDPA where the tower
supports it and eager where it does not. A refusal that still reaches
`from_pretrained` is retried ONCE without an explicit choice.

Driven through the REAL ModelLoadContext.load with only the HuggingFace factories
replaced, and with REAL transformers config objects (Gemma4Config, Gemma3Config,
LlamaConfig) rather than namespaces, so `sub_configs` is what transformers declares.
The payload and the call count of `from_pretrained` are asserted, not a helper's
return value.

MUTATION CONTROLS (2026-10-18; each restored from saved bytes, `git diff` checked):
  A1 load_kwargs["attn_implementation"] = attn_impl  (the plain string again)
       -> TestWhatFromPretrainedReceives gemma4/gemma3/flash cases red
  A2 _attn_implementation_for returns `requested` for every config
       -> the same, plus TestTheHelper composite cases
  A3 the retry re-raises instead of popping attn_implementation
       -> TestTheRetry::test_a_refused_request_is_retried_once_without_it
  A4 the retry swallows every ValueError (marker check removed)
       -> TestTheRetry::test_another_value_error_is_not_retried
  A5 the reported implementation stays the requested one
       -> TestTheRetry::test_the_report_says_what_the_text_stack_resolved_to
  A6 the float32 skeleton's flash->sdpa looks at the top level only (old code)
       -> TestFloat32SkeletonOfAComposite
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import Gemma3Config, LlamaConfig

from millm.ml import model_loader
from millm.ml.gpu_placement import choose_gpu
from millm.ml.model_loader import (
    LoadedModelState,
    ModelLoadContext,
    _attn_implementation_for,
    _served_attn_implementation,
)
from tests.support.fake_gpus import RTX_3090, TI_3080, fake_gpus

gemma4_config = pytest.importorskip("transformers.models.gemma4.configuration_gemma4")

NODE = ((TI_3080, 11_000, 12_288), (RTX_3090, 23_000, 24_576))
CUDA1 = torch.device("cuda", 1)
AUDIO_REFUSAL = (
    "Gemma4AudioModel does not support an attention implementation through "
    "torch.nn.functional.scaled_dot_product_attention yet. Please request the support"
)


def tiny_gemma4() -> object:
    """A real Gemma4Config with all three towers (text, vision, audio), tiny."""
    g = gemma4_config
    return g.Gemma4Config(
        text_config=g.Gemma4TextConfig(
            hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=2,
            num_key_value_heads=1, head_dim=32, global_head_dim=32, vocab_size=128,
            hidden_size_per_layer_input=16, vocab_size_per_layer_input=128,
            layer_types=["sliding_attention", "full_attention"], num_kv_shared_layers=0,
        ),
        vision_config=g.Gemma4VisionConfig(
            hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, head_dim=16, global_head_dim=16,
        ),
        audio_config=g.Gemma4AudioConfig(
            hidden_size=32, num_hidden_layers=1, num_attention_heads=2, output_proj_dims=64,
        ),
    )


class FakeModel:
    """Just enough of a PreTrainedModel for the load path."""

    def __init__(self, config=None):
        self._tensors = [SimpleNamespace(device=CUDA1)]
        self.dtype = torch.bfloat16
        self.forward = MagicMock(name="forward")
        self.generate = MagicMock(side_effect=lambda **_: torch.ones(1, 12, dtype=torch.long))
        if config is not None:
            self.config = config

    def parameters(self):
        return iter(self._tensors)

    def buffers(self):
        return iter([])

    def num_parameters(self):
        return 1_000


@pytest.fixture(autouse=True)
def _reset_state():
    state = LoadedModelState()
    state._loaded = None
    yield
    state._loaded = None


@pytest.fixture
def no_flash(monkeypatch):
    """The serving image installs no flash_attn package; make that true here too."""
    monkeypatch.setitem(sys.modules, "flash_attn", None)


@pytest.fixture
def with_flash(monkeypatch):
    monkeypatch.setitem(sys.modules, "flash_attn", types.ModuleType("flash_attn"))


def _load(config, factory=None, model=None):
    """Run ModelLoadContext.load against `config`; return (loaded, factory)."""
    factory = factory or MagicMock()
    if factory.from_pretrained.side_effect is None:
        factory.from_pretrained.return_value = model or FakeModel()
    tokenizer = MagicMock(eos_token="</s>", pad_token="</s>")
    with fake_gpus(*NODE), \
            patch.object(model_loader, "AutoConfig") as auto_config, \
            patch.object(model_loader, "AutoTokenizer") as auto_tokenizer, \
            patch.object(model_loader, "AutoModelForCausalLM", factory), \
            patch.object(model_loader, "_get_auto_model_class", lambda _config: factory):
        auto_config.from_pretrained.return_value = config
        auto_tokenizer.from_pretrained.return_value = tokenizer
        with ModelLoadContext(1, "m") as ctx:
            loaded = ctx.load(
                cache_path="/nonexistent/model",
                quantization="FP16",
                placement=choose_gpu(8_000),
                torch_compile=False,
            )
    return loaded, factory


class TestWhatFromPretrainedReceives:
    def test_gemma4_asks_for_sdpa_on_the_text_stack_only(self, no_flash):
        _, factory = _load(tiny_gemma4())
        assert factory.from_pretrained.call_count == 1
        assert factory.from_pretrained.call_args.kwargs["attn_implementation"] == {"text_config": "sdpa"}

    def test_gemma3_vision_text_asks_for_sdpa_on_the_text_stack_only(self, no_flash):
        _, factory = _load(Gemma3Config())
        assert factory.from_pretrained.call_count == 1
        assert factory.from_pretrained.call_args.kwargs["attn_implementation"] == {"text_config": "sdpa"}

    def test_a_text_only_model_keeps_the_plain_string(self, no_flash):
        """Llama, Qwen, Gemma 2: no sub-configs, so nothing changes for them."""
        _, factory = _load(LlamaConfig())
        assert factory.from_pretrained.call_count == 1
        assert factory.from_pretrained.call_args.kwargs["attn_implementation"] == "sdpa"

    def test_with_flash_attn_installed_the_text_stack_gets_flash(self, with_flash):
        _, factory = _load(tiny_gemma4())
        assert factory.from_pretrained.call_args.kwargs["attn_implementation"] == {
            "text_config": "flash_attention_2"
        }
        _, factory = _load(LlamaConfig())
        assert factory.from_pretrained.call_args.kwargs["attn_implementation"] == "flash_attention_2"

    def test_no_config_keeps_the_plain_string(self, no_flash):
        _, factory = _load(None)
        assert factory.from_pretrained.call_args.kwargs["attn_implementation"] == "sdpa"


class TestTheRetry:
    def test_a_refused_request_is_retried_once_without_it(self, no_flash):
        resolved = FakeModel(config=LlamaConfig(attn_implementation="eager"))
        factory = MagicMock()
        factory.from_pretrained.side_effect = [ValueError(AUDIO_REFUSAL), resolved]
        loaded, _ = _load(LlamaConfig(), factory=factory)
        calls = factory.from_pretrained.call_args_list
        assert len(calls) == 2
        assert calls[0].kwargs["attn_implementation"] == "sdpa"
        assert "attn_implementation" not in calls[1].kwargs
        assert {k: v for k, v in calls[0].kwargs.items() if k != "attn_implementation"} == calls[1].kwargs
        assert loaded.model is resolved

    def test_the_report_says_what_the_text_stack_resolved_to(self, no_flash):
        """After a retry the model may run eager; the report must not say sdpa."""
        factory = MagicMock()
        factory.from_pretrained.side_effect = [
            ValueError(AUDIO_REFUSAL),
            FakeModel(config=LlamaConfig(attn_implementation="eager")),
        ]
        loaded, _ = _load(LlamaConfig(), factory=factory)
        assert loaded.attn_implementation == "eager"

    def test_another_value_error_is_not_retried(self, no_flash):
        factory = MagicMock()
        factory.from_pretrained.side_effect = ValueError("some other construction failure")
        with pytest.raises(Exception, match="some other construction failure"):
            _load(LlamaConfig(), factory=factory)
        assert factory.from_pretrained.call_count == 1


class TestTheHelper:
    def test_real_transformers_builds_gemma4_with_the_request(self):
        """The dict is accepted by the installed transformers and builds every tower."""
        from transformers import AutoModelForCausalLM

        request = _attn_implementation_for(tiny_gemma4(), "sdpa")
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(tiny_gemma4(), attn_implementation=request)
        config = model.config
        assert config.text_config._attn_implementation == "sdpa"
        assert config.vision_config._attn_implementation == "sdpa"
        assert config.audio_config._attn_implementation in ("sdpa", "eager")
        assert _served_attn_implementation(model, "?") == "sdpa"

    def test_on_this_transformers_the_plain_string_is_what_refused(self):
        """Pins the cause: where the audio tower lacks SDPA, the old request refuses."""
        from transformers import AutoModelForCausalLM
        from transformers.models.gemma4 import modeling_gemma4

        if modeling_gemma4.Gemma4AudioModel._supports_sdpa:
            pytest.skip("this transformers gives Gemma 4's audio tower an SDPA path (5.15.x)")
        with torch.device("meta"), pytest.raises(ValueError, match="does not support an attention"):
            AutoModelForCausalLM.from_config(tiny_gemma4(), attn_implementation="sdpa")

    def test_a_composite_without_a_text_config_leaves_every_tower_to_the_default(self):
        config = SimpleNamespace(sub_configs={"vision_config": object, "audio_config": object})
        assert _attn_implementation_for(config, "sdpa") == {}


class TestFloat32SkeletonOfAComposite:
    def test_flash_on_the_text_stack_becomes_sdpa_and_the_audio_tower_stays_eager(self):
        from transformers import AutoModelForCausalLM

        from millm.services.float32_scoring import _clone_skeleton

        with torch.device("meta"):
            served = AutoModelForCausalLM.from_config(
                tiny_gemma4(), attn_implementation={"text_config": "sdpa"}
            )
        # As if flash_attn were installed: the text stack resolved to flash.
        served.config.text_config._attn_implementation_internal = "flash_attention_2"
        audio_before = served.config.audio_config._attn_implementation

        skeleton = _clone_skeleton(served)

        assert skeleton.config.text_config._attn_implementation == "sdpa"
        assert skeleton.config.audio_config._attn_implementation == audio_before
        assert served.config.text_config._attn_implementation == "flash_attention_2"  # untouched
