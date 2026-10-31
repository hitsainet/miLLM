"""A fill-mask checkpoint loads with its masked-LM head and is refused for generation.

Operator decision 2026-10-18: `*ForMaskedLM` joins the task-head table (`millm.ml.task_heads`)
as `AutoModelForMaskedLM` / `fill-mask`. Before it, `BertForMaskedLM` fell through to
`AutoModelForCausalLM`, which builds `BertLMHeadModel` — a decoder the checkpoint never was —
and the endpoints served chats from it; `ModernBertForMaskedLM` has no causal class at all.

One table, three consumers, each tested through its real code:
  * the loader's class choice and the REAL `ModelLoadContext.load` (registry-faithful Auto
    stand-ins: which `from_pretrained` runs, how often, with what);
  * the `/v1` refusal, through the real app (`POST /api/models` with a local folder, then chat and
    completions), before any load;
  * `/v1/embeddings` on a REAL tiny `BertForMaskedLM`: the encoder's last hidden state, pooled.

NEGATIVE CONTROLS (2026-10-18, restored from saved bytes; this file + the local-classifier route
file, 29 tests): N1 drop the `ForMaskedLM` row from TASK_HEADS -> 9 red (class choice, load
count, registration tag, chat and completions refusal); N2 map it to AutoModelForCausalLM -> 6 red.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
import transformers
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    BertForMaskedLM,
    PreTrainedTokenizerFast,
)

from millm.ml import model_loader
from millm.ml.model_loader import (
    LoadedModel,
    LoadedModelState,
    _get_auto_model_class,
    model_can_generate,
)
from millm.services.model_service import ModelService
from tests.unit.api.test_local_classifier_refused_for_chat import (  # noqa: F401
    _cache_dir,
    _chat,
    _folder,
    _register,
    _row,
    model_root,
)
from tests.unit.batch_fixtures import app_with_db, batch_db, client_for  # noqa: F401
from tests.unit.ml.test_model_class_registry import _registry_faithful
from tests.unit.ml.test_task_head_model_class import TINY, _load_with

MASKED_CASES = [
    ("BertConfig", "BertForMaskedLM"),
    ("RobertaConfig", "RobertaForMaskedLM"),
    ("ModernBertConfig", "ModernBertForMaskedLM"),
]
IDS = [arch for _, arch in MASKED_CASES]


def _config(name: str, arch: str) -> transformers.PretrainedConfig:
    config = getattr(transformers, name)(**TINY[name])
    config.architectures = [arch]
    return config


class TestTheMaskedLmHeadDecidesTheClass:
    @pytest.mark.parametrize("name,arch", MASKED_CASES, ids=IDS)
    def test_class_choice_builds_the_masked_lm(self, name, arch):
        config = _config(name, arch)
        chosen = _get_auto_model_class(config)
        assert chosen is AutoModelForMaskedLM
        built = chosen.from_config(config)
        assert type(built).__name__ == arch
        assert not model_can_generate(built), "no compile soak, no continuous batching"


@pytest.fixture
def stand_ins():
    made = {
        "masked": _registry_faithful(AutoModelForMaskedLM, "AutoModelForMaskedLM"),
        "causal": _registry_faithful(AutoModelForCausalLM, "AutoModelForCausalLM"),
        "seq2seq": _registry_faithful(AutoModelForSeq2SeqLM, "AutoModelForSeq2SeqLM"),
        "auto": _registry_faithful(AutoModel, "AutoModel"),
    }
    with patch("transformers.AutoModelForMaskedLM", made["masked"]), \
            patch("transformers.AutoModelForCausalLM", made["causal"]), \
            patch("transformers.AutoModelForSeq2SeqLM", made["seq2seq"]), \
            patch("transformers.AutoModel", made["auto"]), \
            patch.object(model_loader, "AutoModelForCausalLM", made["causal"]):
        yield SimpleNamespace(**made)


class TestTheLoadMakesOneAttemptWithTheMaskedLmClass:
    @pytest.mark.parametrize("name,arch", MASKED_CASES, ids=IDS)
    def test_one_from_pretrained_on_auto_model_for_masked_lm(self, stand_ins, name, arch):
        _load_with(_config(name, arch))
        counts = {k: v.from_pretrained.call_count for k, v in vars(stand_ins).items()}
        assert counts == {"masked": 1, "causal": 0, "seq2seq": 0, "auto": 0}
        call = stand_ins.masked.from_pretrained.call_args
        assert call.args == ("/nonexistent/model",)
        assert call.kwargs["device_map"] == {"": "cuda:1"}


# --- the refusal, through the real app ---------------------------------------------------------

BERT_MLM = {"architectures": ["BertForMaskedLM"], "model_type": "bert", "hidden_size": 768}
ROBERTA_MLM = {"architectures": ["RobertaForMaskedLM"], "model_type": "roberta"}
MODERNBERT_MLM = {"architectures": ["ModernBertForMaskedLM"], "model_type": "modernbert"}


class TestFillMaskIsRefusedForGeneration:
    @pytest.mark.parametrize("config", [BERT_MLM, ROBERTA_MLM, MODERNBERT_MLM],
                             ids=["bert", "roberta", "modernbert"])
    async def test_registration_stores_fill_mask_and_chat_is_refused(
        self, batch_db, model_root, config  # noqa: F811
    ):
        folder = _folder(model_root, "mlm", config)
        load = AsyncMock()
        async with client_for(app_with_db(batch_db)) as c:
            data = await _register(c, folder)
            with patch.object(ModelService, "load_model_and_wait", load):
                chat = await c.post("/v1/chat/completions", json=_chat(data["name"]))
                text = await c.post("/v1/completions", json={"model": data["name"],
                                                             "prompt": "hi", "max_tokens": 4})
        assert (await _row(batch_db, data["id"])).architecture == "fill-mask"
        for response in (chat, text):
            assert response.status_code == 400, response.text
            assert response.json()["error"]["code"] == "model_not_generative"
        load.assert_not_awaited()


# --- /v1/embeddings on a REAL tiny BertForMaskedLM ----------------------------------------------

WORDS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"w{i}" for i in range(35)]


def _bert_tokenizer() -> PreTrainedTokenizerFast:
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    return PreTrainedTokenizerFast(tokenizer_object=tok, cls_token="[CLS]", sep_token="[SEP]",
                                   pad_token="[PAD]", unk_token="[UNK]", mask_token="[MASK]",
                                   model_max_length=32)


@pytest.fixture
def mlm():
    torch.manual_seed(0)
    return BertForMaskedLM(_config("BertConfig", "BertForMaskedLM")).eval()


@pytest.fixture
def service(mlm):
    from millm.services.inference_service import InferenceService

    LoadedModelState().set(LoadedModel(
        model_id=1, model_name="bert-mlm", model=mlm, tokenizer=_bert_tokenizer(),
        loaded_at=datetime(2026, 10, 18), memory_used_mb=1, num_parameters=1, device="cpu",
        dtype="float32",
    ))
    with patch("millm.services.inference_service.torch.cuda.is_available", return_value=False):
        svc = InferenceService(model_service=None)
    svc._device = "cpu"
    yield svc
    LoadedModelState()._loaded = None


class TestEmbeddingsOnAMaskedLm:
    async def test_embeddings_are_the_encoders_mean_pooled_last_layer(self, service, mlm):
        from millm.api.schemas.openai import EmbeddingRequest

        texts = ["w1 w2 w3", "w4 w5"]
        response = await service.create_embeddings(
            EmbeddingRequest(model="bert-mlm", input=texts, normalize=False)
        )
        tokenizer = _bert_tokenizer()
        assert len(response.data) == 2
        for item, text in zip(response.data, texts, strict=True):
            enc = tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                expected = mlm.bert(**enc).last_hidden_state[0].mean(dim=0)
            assert torch.allclose(torch.tensor(item.embedding), expected, atol=1e-5)
        assert response.usage.prompt_tokens == 5 + 4
