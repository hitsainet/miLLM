"""The Auto class comes from transformers' registry, not the architecture's name suffix.

`_get_auto_model_class` treated every `*ForConditionalGeneration` as seq-to-seq. Gemma 3/4,
Mllama, LLaVA, PaliGemma and Qwen2-VL are all named that way and are decoder-only, so every load
of them first called `AutoModelForSeq2SeqLM.from_pretrained`, which raised "Unrecognized
configuration class", and `ModelLoadContext.load` logged `model_class_fallback_to_causal_lm`
before trying AutoModelForCausalLM (and AutoModel after it). The class each model ENDS UP with is
unchanged by the fix; only the refused attempt and its misleading warning are gone.

Configs are REAL transformers config classes. The stand-in Auto classes accept or refuse a config
exactly as the real ones do — by `type(config) in RealAutoClass._model_mapping`, which is the test
`from_pretrained` itself applies — so a wrong choice is refused here as it is in production.

MUTATION CONTROLS (2026-10-18, restored from saved bytes, sha256-verified):
  N1  revert the fix (original bytes; the name suffix decides)            -> 12 red
  M1  a type no causal mapping takes goes to Causal instead of AutoModel  -> 4 red
  M2  drop "in the seq-to-seq mapping keeps Seq2Seq"                       -> 5 red
  M3  drop the remote-code (auto_map) guard                                -> 1 red
  M4  drop the unknown-model_type guard                                    -> 1 red
  M5  the caller ignores the registry answer                               -> 12 red
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import structlog
import torch
import transformers
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM

from millm.ml import model_loader
from millm.ml.gpu_placement import choose_gpu
from millm.ml.model_loader import ModelLoadContext, _get_auto_model_class
from tests.support.fake_gpus import fake_gpus
from tests.unit.ml.test_model_load_placement import NODE, FakeModel, _reset_state  # noqa: F401

CUDA1 = torch.device("cuda", 1)


def _config(name: str, arch: str) -> transformers.PretrainedConfig:
    config = getattr(transformers, name)()
    config.architectures = [arch]
    return config


# (config class, architecture recorded in the checkpoint, the class the load ends up with)
CASES = [
    ("Gemma4Config", "Gemma4ForConditionalGeneration", AutoModelForCausalLM),
    ("Gemma4UnifiedConfig", "Gemma4UnifiedForConditionalGeneration", AutoModelForCausalLM),
    ("Gemma3Config", "Gemma3ForConditionalGeneration", AutoModelForCausalLM),
    ("MllamaConfig", "MllamaForConditionalGeneration", AutoModelForCausalLM),
    ("LlamaConfig", "LlamaForCausalLM", AutoModelForCausalLM),
    ("Qwen2Config", "Qwen2ForCausalLM", AutoModelForCausalLM),
    ("Qwen3Config", "Qwen3ForCausalLM", AutoModelForCausalLM),
    ("T5Config", "T5ForConditionalGeneration", AutoModelForSeq2SeqLM),
    ("MT5Config", "MT5ForConditionalGeneration", AutoModelForSeq2SeqLM),
    ("BartConfig", "BartForConditionalGeneration", AutoModelForSeq2SeqLM),
    # Image-text-to-text models no causal-LM mapping takes: the old chain ended at AutoModel.
    ("LlavaConfig", "LlavaForConditionalGeneration", AutoModel),
    ("PaliGemmaConfig", "PaliGemmaForConditionalGeneration", AutoModel),
    ("Qwen2VLConfig", "Qwen2VLForConditionalGeneration", AutoModel),
]
IDS = [c[0] for c in CASES]


def _old_chain_landing(config, name_says_seq2seq: bool):
    """Where the pre-fix load landed: the first class of its chain the registry accepts."""
    chain = ([AutoModelForSeq2SeqLM] if name_says_seq2seq else []) + [AutoModelForCausalLM, AutoModel]
    return next(cls for cls in chain if type(config) in cls._model_mapping)


class TestTheClassIsTheOneTheRegistryAccepts:
    @pytest.mark.parametrize("name,arch,expected", CASES, ids=IDS)
    def test_class_choice(self, name, arch, expected):
        config = _config(name, arch)
        chosen = _get_auto_model_class(config)
        assert chosen is expected
        assert type(config) in chosen._model_mapping, "the first attempt must be one that loads"

    @pytest.mark.parametrize("name,arch,expected", CASES, ids=IDS)
    def test_the_final_class_is_unchanged_from_the_fallback_chain(self, name, arch, expected):
        """The fix removes the refused attempt; it must not move any model to another class."""
        config = _config(name, arch)
        assert _old_chain_landing(config, "ConditionalGeneration" in arch) is expected

    def test_gemma4_is_refused_by_seq2seq_the_way_production_saw_it(self):
        with pytest.raises(ValueError, match="Unrecognized configuration class"):
            AutoModelForSeq2SeqLM.from_config(_config("Gemma4Config", "Gemma4ForConditionalGeneration"))

    def test_an_unknown_model_type_keeps_the_name_rule(self):
        """Custom code: the registry cannot speak for it, so nothing changes."""
        config = SimpleNamespace(architectures=["NovelForConditionalGeneration"], model_type="novel_x")
        assert _get_auto_model_class(config) is AutoModelForSeq2SeqLM

    def test_remote_code_mapping_an_auto_class_keeps_the_name_rule(self):
        config = _config("Gemma3Config", "Gemma3ForConditionalGeneration")
        config.auto_map = {"AutoModelForSeq2SeqLM": "modeling_x.XForConditionalGeneration"}
        assert _get_auto_model_class(config) is AutoModelForSeq2SeqLM


def _registry_faithful(real_cls, name):
    """A stand-in for `real_cls` that refuses what the real class refuses, and counts calls."""
    stand_in = MagicMock(name=name)
    stand_in.__name__ = name

    def _from_pretrained(path, **kwargs):
        config = model_loader.AutoConfig.from_pretrained.return_value
        if type(config) not in real_cls._model_mapping:
            raise ValueError(f"Unrecognized configuration class {type(config)} for {name}")
        return FakeModel([CUDA1])

    def _from_config(config, **kwargs):
        if type(config) not in real_cls._model_mapping:
            raise ValueError(f"Unrecognized configuration class {type(config)} for {name}")
        return MagicMock(name=f"{name}-meta")

    stand_in.from_pretrained.side_effect = _from_pretrained
    stand_in.from_config.side_effect = _from_config
    return stand_in


@pytest.fixture
def auto_classes():
    seq2seq = _registry_faithful(AutoModelForSeq2SeqLM, "AutoModelForSeq2SeqLM")
    causal = _registry_faithful(AutoModelForCausalLM, "AutoModelForCausalLM")
    auto = _registry_faithful(AutoModel, "AutoModel")
    with patch("transformers.AutoModelForSeq2SeqLM", seq2seq), \
            patch("transformers.AutoModelForCausalLM", causal), \
            patch("transformers.AutoModel", auto), \
            patch.object(model_loader, "AutoModelForCausalLM", causal):
        yield SimpleNamespace(seq2seq=seq2seq, causal=causal, auto=auto)


def _load_with(config):
    """Run the real ModelLoadContext.load with `config` as the checkpoint's config."""
    with fake_gpus(*NODE), \
            patch.object(model_loader, "AutoConfig") as auto_config, \
            patch.object(model_loader, "AutoTokenizer") as auto_tokenizer, \
            patch.object(model_loader, "BitsAndBytesConfig", MagicMock()):
        auto_config.from_pretrained.return_value = config
        auto_tokenizer.from_pretrained.return_value = MagicMock(eos_token="</s>", pad_token="</s>")
        with ModelLoadContext(1, "m") as ctx:
            return ctx.load(cache_path="/nonexistent/model", quantization="FP16",
                            placement=choose_gpu(8_000), torch_compile=False)


LOAD_CASES = [
    ("Gemma4Config", "Gemma4ForConditionalGeneration", "causal"),
    ("Gemma3Config", "Gemma3ForConditionalGeneration", "causal"),
    ("LlamaConfig", "LlamaForCausalLM", "causal"),
    ("T5Config", "T5ForConditionalGeneration", "seq2seq"),
    ("LlavaConfig", "LlavaForConditionalGeneration", "auto"),
]


class TestTheLoadMakesOneAttempt:
    """Through the REAL ModelLoadContext.load: which from_pretrained runs, and how often."""

    @pytest.mark.parametrize("name,arch,loads_with", LOAD_CASES, ids=[c[0] for c in LOAD_CASES])
    def test_one_from_pretrained_and_no_fallback_warning(self, auto_classes, name, arch, loads_with):
        with structlog.testing.capture_logs() as logs:
            _load_with(_config(name, arch))
        counts = {k: getattr(auto_classes, k).from_pretrained.call_count
                  for k in ("seq2seq", "causal", "auto")}
        assert counts == {k: int(k == loads_with) for k in counts}
        events = [entry["event"] for entry in logs]
        assert "model_class_fallback_to_causal_lm" not in events
        assert "model_class_fallback_to_auto_model" not in events

    def test_gemma4_loads_with_the_causal_lm_class_and_its_kwargs(self, auto_classes):
        _load_with(_config("Gemma4Config", "Gemma4ForConditionalGeneration"))
        assert auto_classes.causal.from_pretrained.call_count == 1
        call = auto_classes.causal.from_pretrained.call_args
        assert call.args == ("/nonexistent/model",)
        assert call.kwargs["device_map"] == {"": "cuda:1"}


class TestTheMetaModelBuildsOnce:
    """`_meta_model` (preflight sizing) shares the class choice."""

    def test_gemma4_meta_skeleton_is_built_by_causal_lm_first_time(self, auto_classes):
        config = _config("Gemma4Config", "Gemma4ForConditionalGeneration")
        model_loader._meta_model(config, trust_remote_code=False, dtype=torch.bfloat16)
        assert auto_classes.seq2seq.from_config.call_count == 0
        assert auto_classes.causal.from_config.call_count == 1
        assert auto_classes.causal.from_config.call_args.args == (config,)
        assert auto_classes.auto.from_config.call_count == 0
