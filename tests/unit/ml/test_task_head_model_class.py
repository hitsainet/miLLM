"""A task-head checkpoint loads with the Auto class that builds ITS head, and serves.

facebook/bart-large-mnli says architectures=["BartForSequenceClassification"], model_type
"bart". `_get_auto_model_class` checked the seq-to-seq model_type rule before any head suffix,
so it loaded as AutoModelForSeq2SeqLM: the load SUCCEEDED, the classification head's weights
were dropped as unused, and an LM head tied to the embeddings stood in its place. Same shape:
T5/mT5/MBart/PLBart classifiers, and BART/T5 QA heads. Token classification and multiple choice
went to AutoModelForSequenceClassification (a different, randomly initialised head) and QA fell
through to AutoModelForCausalLM (RobertaForCausalLM with a random LM head).

Downstream of the load, a classifier also broke or wasted work three ways, fixed here:
  * `/v1/embeddings` on any encoder-decoder (BART-MNLI, T5) read `outputs.hidden_states`, which a
    seq-to-seq output does not have -> AttributeError (and T5 refused the forward outright);
    it now embeds with the encoder.
  * the torch.compile soak called `generate` on a model with none, logging an ERROR on every
    classifier load; it is skipped for a model transformers says cannot generate.
  * continuous batching started a generation manager around it; skipped likewise.
  * "question-answering" rows were not refused for chat.

Configs are REAL transformers configs; the head is proven present by a forward pass on a tiny
random model (logits [batch, num_labels]) and by a save/load round trip of its weights.

MUTATION CONTROLS (2026-10-18; each restored from saved bytes, sha256-verified, never git;
run over this file + test_model_class_dispatch/registry + test_embedding_options +
test_gguf_refusals, 149 tests):
  N1  model_loader.py back to its pre-fix bytes (seq2seq rule before any head)   -> 18 red
  M1  the task-head loop moved after the model_type seq2seq rule                  -> 12 red
  M2  ForTokenClassification -> AutoModelForSequenceClassification (old mapping)  ->  4 red
  M3  the encoder branch in `_hidden_states_forward` disabled                     ->  2 red
  M4  the compile skip for non-generative models disabled                         ->  1 red
  M5  the continuous-batching skip for non-generative models disabled             ->  1 red
  M6  "question-answering" dropped from EMBEDDING_ARCHITECTURES                    ->  1 red
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import structlog
import torch
import transformers
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    BartConfig,
    BartForSequenceClassification,
    PreTrainedTokenizerFast,
)

from millm.ml import model_loader
from millm.ml.gpu_placement import choose_gpu
from millm.ml.model_loader import (
    LoadedModel,
    LoadedModelState,
    ModelLoadContext,
    _get_auto_model_class,
    model_can_generate,
)
from tests.support.fake_gpus import fake_gpus
from tests.unit.ml.test_model_class_registry import _registry_faithful
from tests.unit.ml.test_model_load_placement import NODE, FakeModel, _reset_state  # noqa: F401

CUDA1 = torch.device("cuda", 1)
MNLI_LABELS = {0: "contradiction", 1: "neutral", 2: "entailment"}

TINY = {
    "BartConfig": dict(vocab_size=40, d_model=16, encoder_layers=1, decoder_layers=1,
                       encoder_attention_heads=2, decoder_attention_heads=2,
                       encoder_ffn_dim=32, decoder_ffn_dim=32, max_position_embeddings=32),
    "T5Config": dict(vocab_size=40, d_model=16, d_kv=8, d_ff=32, num_layers=1,
                     num_decoder_layers=1, num_heads=2),
    "MT5Config": dict(vocab_size=40, d_model=16, d_kv=8, d_ff=32, num_layers=1,
                      num_decoder_layers=1, num_heads=2),
    "MBartConfig": dict(vocab_size=40, d_model=16, encoder_layers=1, decoder_layers=1,
                        encoder_attention_heads=2, decoder_attention_heads=2,
                        encoder_ffn_dim=32, decoder_ffn_dim=32, max_position_embeddings=32),
    "PLBartConfig": dict(vocab_size=40, d_model=16, encoder_layers=1, decoder_layers=1,
                         encoder_attention_heads=2, decoder_attention_heads=2,
                         encoder_ffn_dim=32, decoder_ffn_dim=32, max_position_embeddings=32),
    "BertConfig": dict(vocab_size=40, hidden_size=16, num_hidden_layers=1,
                       num_attention_heads=2, intermediate_size=32, max_position_embeddings=32),
    "RobertaConfig": dict(vocab_size=40, hidden_size=16, num_hidden_layers=1,
                          num_attention_heads=2, intermediate_size=32,
                          max_position_embeddings=40),
    "ModernBertConfig": dict(vocab_size=40, hidden_size=16, num_hidden_layers=1,
                             num_attention_heads=2, intermediate_size=32,
                             max_position_embeddings=32, pad_token_id=1, eos_token_id=2,
                             bos_token_id=0, cls_token_id=0, sep_token_id=2),
}


def _config(name: str, arch: str, **extra) -> transformers.PretrainedConfig:
    config = getattr(transformers, name)(**{**TINY.get(name, {}), **extra})
    config.architectures = [arch]
    return config


def _mnli_config() -> BartConfig:
    return _config("BartConfig", "BartForSequenceClassification", num_labels=3,
                   id2label=MNLI_LABELS, label2id={v: k for k, v in MNLI_LABELS.items()})


# (config class, architecture recorded in the checkpoint, the class that builds that head)
HEAD_CASES = [
    ("BartConfig", "BartForSequenceClassification", AutoModelForSequenceClassification),
    ("T5Config", "T5ForSequenceClassification", AutoModelForSequenceClassification),
    ("MT5Config", "MT5ForSequenceClassification", AutoModelForSequenceClassification),
    ("MBartConfig", "MBartForSequenceClassification", AutoModelForSequenceClassification),
    ("PLBartConfig", "PLBartForSequenceClassification", AutoModelForSequenceClassification),
    ("ModernBertConfig", "ModernBertForSequenceClassification",
     AutoModelForSequenceClassification),
    ("BartConfig", "BartForQuestionAnswering", AutoModelForQuestionAnswering),
    ("T5Config", "T5ForQuestionAnswering", AutoModelForQuestionAnswering),
    ("RobertaConfig", "RobertaForQuestionAnswering", AutoModelForQuestionAnswering),
    ("BertConfig", "BertForTokenClassification", AutoModelForTokenClassification),
    ("T5Config", "T5ForTokenClassification", AutoModelForTokenClassification),
    ("BertConfig", "BertForMultipleChoice", AutoModelForMultipleChoice),
]
HEAD_IDS = [c[1] for c in HEAD_CASES]


class TestTheHeadDecidesTheClass:
    @pytest.mark.parametrize("name,arch,expected", HEAD_CASES, ids=HEAD_IDS)
    def test_class_choice_builds_the_named_head(self, name, arch, expected):
        config = _config(name, arch)
        chosen = _get_auto_model_class(config)
        assert chosen is expected
        assert type(config) in chosen._model_mapping, "the first attempt must be one that loads"
        built = chosen.from_config(config)
        assert type(built).__name__ == arch, "the class built is the one the checkpoint names"

    @pytest.mark.parametrize("name,arch", [
        ("BartConfig", "BartForConditionalGeneration"),
        ("T5Config", "T5ForConditionalGeneration"),
        ("MBartConfig", "MBartForConditionalGeneration"),
        ("PegasusConfig", "PegasusForConditionalGeneration"),
    ])
    def test_generative_seq2seq_is_unchanged(self, name, arch):
        assert _get_auto_model_class(_config(name, arch)) is AutoModelForSeq2SeqLM

    def test_bart_with_no_head_named_still_takes_the_model_type_rule(self):
        config = _config("BartConfig", "BartModel")
        assert _get_auto_model_class(config) is AutoModelForSeq2SeqLM


class TestBartMnliKeepsItsHead:
    """The defect: the load SUCCEEDED and was not the model. Prove the head is there."""

    def test_forward_gives_one_logit_per_label(self):
        config = _mnli_config()
        torch.manual_seed(0)
        model = _get_auto_model_class(config).from_config(config).eval()
        assert model.config.id2label == MNLI_LABELS
        ids = torch.tensor([[0, 5, 6, 7, 2], [0, 8, 9, 10, 2]])
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=torch.ones_like(ids))
        assert out.logits.shape == (2, 3)
        assert not model_can_generate(model)

    def test_a_saved_checkpoint_round_trips_its_classification_head(self, tmp_path):
        torch.manual_seed(0)
        original = BartForSequenceClassification(_mnli_config()).eval()
        original.save_pretrained(tmp_path)

        config = AutoConfig.from_pretrained(tmp_path)
        loaded = _get_auto_model_class(config).from_pretrained(tmp_path).eval()

        assert type(loaded) is BartForSequenceClassification
        for name in ("dense.weight", "dense.bias", "out_proj.weight", "out_proj.bias"):
            assert torch.equal(
                loaded.classification_head.state_dict()[name],
                original.classification_head.state_dict()[name],
            ), f"classification_head.{name} was not loaded from the checkpoint"
        ids = torch.tensor([[0, 5, 6, 7, 2]])
        with torch.no_grad():
            assert torch.equal(loaded(input_ids=ids).logits, original(input_ids=ids).logits)

    def test_the_old_class_would_have_dropped_the_head(self, tmp_path):
        """What the pre-fix class did with the same checkpoint: no classification head."""
        torch.manual_seed(0)
        BartForSequenceClassification(_mnli_config()).save_pretrained(tmp_path)
        wrong = AutoModelForSeq2SeqLM.from_pretrained(tmp_path)
        assert not hasattr(wrong, "classification_head")


@pytest.fixture
def auto_classes():
    stand_ins = {
        "seq2seq": _registry_faithful(AutoModelForSeq2SeqLM, "AutoModelForSeq2SeqLM"),
        "causal": _registry_faithful(AutoModelForCausalLM, "AutoModelForCausalLM"),
        "auto": _registry_faithful(AutoModel, "AutoModel"),
        "seqcls": _registry_faithful(AutoModelForSequenceClassification,
                                     "AutoModelForSequenceClassification"),
        "tokcls": _registry_faithful(AutoModelForTokenClassification,
                                     "AutoModelForTokenClassification"),
        "qa": _registry_faithful(AutoModelForQuestionAnswering, "AutoModelForQuestionAnswering"),
    }
    with patch("transformers.AutoModelForSeq2SeqLM", stand_ins["seq2seq"]), \
            patch("transformers.AutoModelForCausalLM", stand_ins["causal"]), \
            patch("transformers.AutoModel", stand_ins["auto"]), \
            patch("transformers.AutoModelForSequenceClassification", stand_ins["seqcls"]), \
            patch("transformers.AutoModelForTokenClassification", stand_ins["tokcls"]), \
            patch("transformers.AutoModelForQuestionAnswering", stand_ins["qa"]), \
            patch.object(model_loader, "AutoModelForCausalLM", stand_ins["causal"]):
        yield SimpleNamespace(**stand_ins)


def _load_with(config, torch_compile=False):
    """Run the real ModelLoadContext.load with `config` as the checkpoint's config."""
    with fake_gpus(*NODE), \
            patch.object(model_loader, "AutoConfig") as auto_config, \
            patch.object(model_loader, "AutoTokenizer") as auto_tokenizer, \
            patch.object(model_loader, "BitsAndBytesConfig", MagicMock()):
        auto_config.from_pretrained.return_value = config
        auto_tokenizer.from_pretrained.return_value = MagicMock(eos_token="</s>", pad_token="<pad>")
        with ModelLoadContext(1, "m") as ctx:
            return ctx.load(cache_path="/nonexistent/model", quantization="FP16",
                            placement=choose_gpu(8_000), torch_compile=torch_compile)


LOAD_CASES = [
    (_mnli_config, "seqcls"),
    (lambda: _config("T5Config", "T5ForSequenceClassification"), "seqcls"),
    (lambda: _config("BertConfig", "BertForTokenClassification"), "tokcls"),
    (lambda: _config("RobertaConfig", "RobertaForQuestionAnswering"), "qa"),
    (lambda: _config("BartConfig", "BartForConditionalGeneration"), "seq2seq"),
]


class TestTheLoadMakesOneAttemptWithTheHeadClass:
    """Through the REAL ModelLoadContext.load: which from_pretrained runs, how often, with what."""

    @pytest.mark.parametrize("make,loads_with", LOAD_CASES,
                             ids=["bart-mnli", "t5-cls", "bert-tok", "roberta-qa", "bart-gen"])
    def test_one_from_pretrained_on_the_head_class(self, auto_classes, make, loads_with):
        with structlog.testing.capture_logs() as logs:
            _load_with(make())
        counts = {k: v.from_pretrained.call_count for k, v in vars(auto_classes).items()}
        assert counts == {k: int(k == loads_with) for k in counts}
        call = getattr(auto_classes, loads_with).from_pretrained.call_args
        assert call.args == ("/nonexistent/model",)
        assert call.kwargs["device_map"] == {"": "cuda:1"}
        events = [entry["event"] for entry in logs]
        assert "model_class_fallback_to_causal_lm" not in events

    def test_preflight_meta_model_builds_the_classifier(self, auto_classes):
        config = _mnli_config()
        model_loader._meta_model(config, trust_remote_code=False, dtype=torch.float32)
        assert auto_classes.seqcls.from_config.call_count == 1
        assert auto_classes.seqcls.from_config.call_args.args == (config,)
        assert auto_classes.seq2seq.from_config.call_count == 0


class TestNoGenerationMachineryForAClassifier:
    def _classifier_fake(self):
        fake = FakeModel([CUDA1])
        fake.can_generate = lambda: False
        return fake

    def test_compile_soak_is_skipped(self, auto_classes):
        fake = self._classifier_fake()
        auto_classes.seqcls.from_pretrained.side_effect = lambda path, **kw: fake
        with patch.object(model_loader.torch, "compile") as compile_, \
                structlog.testing.capture_logs() as logs:
            _load_with(_mnli_config(), torch_compile=True)
        assert compile_.call_count == 0
        assert fake.generate.call_count == 0
        events = [entry["event"] for entry in logs]
        assert "torch_compile_skipped_not_generative" in events
        assert "torch_compile_soak_failed_reverted_to_eager" not in events

    def test_a_generative_model_is_still_compiled(self, auto_classes):
        with patch.object(model_loader.torch, "compile", side_effect=lambda fn, **kw: fn) \
                as compile_:
            _load_with(_config("BartConfig", "BartForConditionalGeneration"), torch_compile=True)
        assert compile_.call_count == 1

    def _service_with(self, model):
        from millm.services.inference_service import InferenceService

        svc = InferenceService.__new__(InferenceService)
        svc._cbm_backend = MagicMock(name="cbm")
        svc._draft_model = None
        state = MagicMock()
        state.is_loaded = True
        state.current = SimpleNamespace(model=model, tokenizer=MagicMock(), supports_hooks=True,
                                        engine="transformers", gpu_indices=[1])
        svc._model_state = state
        svc._release_draft_model = MagicMock()
        return svc

    def test_continuous_batching_is_not_started(self):
        svc = self._service_with(self._classifier_fake())
        svc.on_model_loaded()
        assert svc._cbm_backend.start.call_count == 0

    def test_continuous_batching_still_starts_for_a_generative_model(self):
        model = FakeModel([CUDA1])
        svc = self._service_with(model)
        svc.on_model_loaded()
        assert svc._cbm_backend.start.call_count == 1
        assert svc._cbm_backend.start.call_args.args[0] is model


class TestQuestionAnsweringIsRefusedForChat:
    def test_question_answering_is_non_generative(self):
        from millm.api.routes.openai.errors import is_embedding_only

        assert is_embedding_only("question-answering")
        assert not is_embedding_only("text2text-generation")


# --- /v1/embeddings on a REAL tiny BART-MNLI ------------------------------------------------

WORDS = ["<s>", "<pad>", "</s>", "<unk>"] + [f"w{i}" for i in range(36)]


def _bart_tokenizer() -> PreTrainedTokenizerFast:
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)]
    )
    return PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>", eos_token="</s>",
                                   pad_token="<pad>", unk_token="<unk>", model_max_length=32)


@pytest.fixture
def mnli():
    torch.manual_seed(0)
    return BartForSequenceClassification(_mnli_config()).eval()


@pytest.fixture
def service(mnli):
    from millm.services.inference_service import InferenceService

    tokenizer = _bart_tokenizer()
    LoadedModelState().set(LoadedModel(
        model_id=1, model_name="bart-mnli", model=mnli, tokenizer=tokenizer,
        loaded_at=datetime(2026, 10, 18), memory_used_mb=1, num_parameters=1, device="cpu",
        dtype="float32",
    ))
    with patch("millm.services.inference_service.torch.cuda.is_available", return_value=False):
        svc = InferenceService(model_service=None)
    svc._device = "cpu"
    return svc


def _encoder_mean(model, tokenizer, text: str) -> torch.Tensor:
    enc = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        hidden = model.get_encoder()(**enc).last_hidden_state[0]
    return hidden.mean(dim=0)


class TestEmbeddingsOnAnEncoderDecoderClassifier:
    async def test_embeddings_are_the_encoders_mean_pooled_last_layer(self, service, mnli):
        from millm.api.schemas.openai import EmbeddingRequest

        texts = ["w1 w2 w3", "w4 w5"]
        response = await service.create_embeddings(
            EmbeddingRequest(model="bart-mnli", input=texts, normalize=False)
        )
        assert len(response.data) == 2
        tokenizer = _bart_tokenizer()
        for item, text in zip(response.data, texts, strict=True):
            expected = _encoder_mean(mnli, tokenizer, text)
            assert torch.allclose(torch.tensor(item.embedding), expected, atol=1e-5)
        assert response.usage.prompt_tokens == 5 + 4

    def test_packed_rows_use_the_encoder_and_pad_with_bart_pad(self, service, mnli):
        tokenizer = _bart_tokenizer()
        rows = [tokenizer("w1 w2 w3")["input_ids"], tokenizer("w4")["input_ids"]]
        calls: list[torch.Tensor] = []
        mnli.get_encoder().register_forward_pre_hook(
            lambda _m, args, kwargs: calls.append(kwargs["input_ids"].clone()), with_kwargs=True
        )
        pooled = service._packed_pooled_hidden(rows, ["mean", "mean"])
        assert len(calls) == 1, "one forward over the pack, through the encoder"
        assert calls[0][1].tolist() == rows[1] + [tokenizer.pad_token_id] * 2
        assert tokenizer.pad_token_id == 1
        for vector, text in zip(pooled, ["w1 w2 w3", "w4"], strict=True):
            assert torch.allclose(vector, _encoder_mean(mnli, tokenizer, text), atol=1e-5)
