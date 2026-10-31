"""A classifier registered from a LOCAL folder is refused for chat, exactly as a Hub one is.

The generation endpoints refuse non-generative models (`model_not_generative`, 400) by the tag
stored on the model row — the Hub `pipeline_tag`. A model registered from a local path has no Hub
tag, so a `BartForSequenceClassification` or `DebertaV2ForSequenceClassification` folder was not
refused before the load, reached `generate()` and failed with a 500.

Now the task is derived from the folder's `config.json` with the loader's task-head table
(`millm.ml.task_heads`): at registration (stored on the row) and at check time (so rows registered
before the derivation existed are covered without a migration).

Everything here goes through the REAL app: `POST /api/models` with `source: local`, then
`POST /v1/chat/completions` / `/v1/completions` naming the registered model. Only the load itself is
stubbed — the generative case asserts the request got PAST the refusal to the load, with the row's
id, exactly once.
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from millm.core.errors import ModelBusyError
from millm.db.models.model import Model, ModelSource, ModelStatus, QuantizationType
from millm.ml.task_heads import TASK_HEADS
from millm.services.model_service import ModelService
from tests.unit.batch_fixtures import app_with_db, batch_db, client_for  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[3]

# Real configs, trimmed to the fields that decide the class (facebook/bart-large-mnli,
# microsoft/deberta-v3-base fine-tuned for NLI, a Llama-architecture causal LM, BART's generator).
BART_MNLI = {
    "architectures": ["BartForSequenceClassification"],
    "model_type": "bart",
    "id2label": {"0": "contradiction", "1": "neutral", "2": "entailment"},
    "d_model": 1024,
    "vocab_size": 50265,
}
DEBERTA_NLI = {
    "architectures": ["DebertaV2ForSequenceClassification"],
    "model_type": "deberta-v2",
    "id2label": {"0": "entailment", "1": "neutral", "2": "contradiction"},
    "hidden_size": 768,
}
ROBERTA_QA = {"architectures": ["RobertaForQuestionAnswering"], "model_type": "roberta"}
BERT_NER = {"architectures": ["BertForTokenClassification"], "model_type": "bert"}
ROBERTA_MC = {"architectures": ["RobertaForMultipleChoice"], "model_type": "roberta"}
LLAMA_CAUSAL = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 2048,
    "num_hidden_layers": 16,
    "vocab_size": 128256,
}
BART_GENERATOR = {"architectures": ["BartForConditionalGeneration"], "model_type": "bart"}


@pytest.fixture(autouse=True)
def _cache_dir(tmp_path, monkeypatch):
    from millm.core.config import settings

    monkeypatch.setattr(settings, "MODEL_CACHE_DIR", str(tmp_path / "models"))


@pytest.fixture
def model_root():
    """A writable directory OUTSIDE /tmp — the local_path validator refuses /tmp."""
    root = REPO_ROOT / ".pytest_cache" / "local-models" / uuid.uuid4().hex
    root.mkdir(parents=True)
    yield root
    shutil.rmtree(root, ignore_errors=True)


def _folder(root: Path, name: str, config: dict) -> Path:
    d = root / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps(config))
    (d / "model.safetensors").write_bytes(b"\0" * 16)
    return d


async def _register(client, folder: Path) -> dict:
    r = await client.post("/api/models", json={"source": "local", "local_path": str(folder),
                                               "quantization": "FP16"})
    assert r.status_code == 202, r.text
    return r.json()["data"]


def _chat(name: str) -> dict:
    return {"model": name, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}


async def _row(factory, model_id: int) -> Model:
    async with factory() as s:
        return await s.get(Model, model_id)


class TestRegistrationDerivesTheTask:
    @pytest.mark.parametrize("config, tag", [
        (BART_MNLI, "text-classification"),
        (DEBERTA_NLI, "text-classification"),
        (ROBERTA_QA, "question-answering"),
        (BERT_NER, "token-classification"),
        (ROBERTA_MC, "multiple-choice"),
    ])
    async def test_a_local_task_head_is_stored_with_its_tag(self, batch_db, model_root, config, tag):
        folder = _folder(model_root, "clf", config)
        async with client_for(app_with_db(batch_db)) as c:
            data = await _register(c, folder)
        assert data["status"] == "ready"
        assert (await _row(batch_db, data["id"])).architecture == tag

    async def test_a_local_causal_lm_gets_no_task_tag(self, batch_db, model_root):
        folder = _folder(model_root, "llama", LLAMA_CAUSAL)
        async with client_for(app_with_db(batch_db)) as c:
            data = await _register(c, folder)
        assert (await _row(batch_db, data["id"])).architecture is None


class TestChatRefusesALocalClassifier:
    @pytest.mark.parametrize("config", [BART_MNLI, DEBERTA_NLI, ROBERTA_QA, BERT_NER, ROBERTA_MC])
    async def test_chat_is_refused_before_any_load(self, batch_db, model_root, config):
        folder = _folder(model_root, "local-clf", config)
        load = AsyncMock()
        async with client_for(app_with_db(batch_db)) as c:
            data = await _register(c, folder)
            with patch.object(ModelService, "load_model_and_wait", load):
                r = await c.post("/v1/chat/completions", json=_chat(data["name"]))
        assert r.status_code == 400, r.text
        err = r.json()["error"]
        assert err["code"] == "model_not_generative"
        assert data["name"] in err["message"]
        load.assert_not_awaited()

    async def test_completions_is_refused_too(self, batch_db, model_root):
        folder = _folder(model_root, "bart-mnli", BART_MNLI)
        load = AsyncMock()
        async with client_for(app_with_db(batch_db)) as c:
            data = await _register(c, folder)
            with patch.object(ModelService, "load_model_and_wait", load):
                r = await c.post("/v1/completions", json={"model": data["name"], "prompt": "hi",
                                                          "max_tokens": 4})
        assert r.status_code == 400, r.text
        assert r.json()["error"]["code"] == "model_not_generative"
        load.assert_not_awaited()


class TestGenerativeLocalModelsAreAllowed:
    @pytest.mark.parametrize("config", [LLAMA_CAUSAL, BART_GENERATOR])
    async def test_chat_reaches_the_load(self, batch_db, model_root, config):
        """Past the refusal and into the load, with this row's id, once. The stubbed load answers
        busy (503) so the test stops there without a real checkpoint."""
        folder = _folder(model_root, "gen", config)
        load = AsyncMock(side_effect=ModelBusyError("a load is in progress"))
        async with client_for(app_with_db(batch_db)) as c:
            data = await _register(c, folder)
            with patch.object(ModelService, "load_model_and_wait", load):
                r = await c.post("/v1/chat/completions", json=_chat(data["name"]))
        assert r.status_code == 503, r.text
        assert r.json()["error"]["code"] != "model_not_generative"
        assert load.await_count == 1
        assert load.await_args.args[0] == data["id"]


class TestRowsRegisteredBeforeTheFix:
    """A local row stored with NO tag (as every local registration was) is refused from its
    checkpoint's config at check time — no migration, no re-registration."""

    async def _untagged_row(self, factory, folder: Path, name: str) -> int:
        async with factory() as s:
            row = Model(name=name, source=ModelSource.LOCAL, repo_id=None, local_path=str(folder),
                        quantization=QuantizationType.FP16, cache_path=str(folder),
                        status=ModelStatus.READY, architecture=None, gguf_label="")
            s.add(row)
            await s.commit()
            return row.id

    async def test_an_existing_untagged_classifier_is_refused(self, batch_db, model_root):
        folder = _folder(model_root, "old-bart", BART_MNLI)
        model_id = await self._untagged_row(batch_db, folder, "old-bart")
        assert (await _row(batch_db, model_id)).architecture is None
        load = AsyncMock()
        async with client_for(app_with_db(batch_db)) as c:
            with patch.object(ModelService, "load_model_and_wait", load):
                r = await c.post("/v1/chat/completions", json=_chat("old-bart"))
        assert r.status_code == 400, r.text
        err = r.json()["error"]
        assert err["code"] == "model_not_generative"
        assert "text-classification" in err["message"]
        load.assert_not_awaited()

    async def test_an_existing_untagged_causal_lm_still_loads(self, batch_db, model_root):
        folder = _folder(model_root, "old-llama", LLAMA_CAUSAL)
        model_id = await self._untagged_row(batch_db, folder, "old-llama")
        load = AsyncMock(side_effect=ModelBusyError("a load is in progress"))
        async with client_for(app_with_db(batch_db)) as c:
            with patch.object(ModelService, "load_model_and_wait", load):
                r = await c.post("/v1/chat/completions", json=_chat("old-llama"))
        assert r.status_code == 503, r.text
        assert load.await_count == 1 and load.await_args.args[0] == model_id

    async def test_a_hub_default_tag_is_corrected_by_the_config(self, batch_db, model_root):
        """A Hub repo with no pipeline_tag is stored as the "text-generation" default; its own
        config.json still names the classification head and wins."""
        folder = _folder(model_root, "hub-clf", DEBERTA_NLI)
        async with batch_db() as s:
            row = Model(name="hub-clf", source=ModelSource.HUGGINGFACE, repo_id="o/hub-clf",
                        quantization=QuantizationType.FP16, cache_path=str(folder),
                        status=ModelStatus.READY, architecture="text-generation", gguf_label="")
            s.add(row)
            await s.commit()
        async with client_for(app_with_db(batch_db)) as c:
            with patch.object(ModelService, "load_model_and_wait", AsyncMock()) as load:
                r = await c.post("/v1/chat/completions", json=_chat("hub-clf"))
        assert r.status_code == 400 and r.json()["error"]["code"] == "model_not_generative"
        load.assert_not_awaited()


class TestOneTable:
    def test_every_task_head_the_loader_knows_is_refused(self):
        """The loader's table and the refusal set cannot drift: each head's tag is refused."""
        from millm.api.routes.openai.errors import is_embedding_only
        from millm.ml.model_loader import _TASK_HEAD_AUTO_CLASSES

        assert [(s, a) for s, a, _ in TASK_HEADS] == list(_TASK_HEAD_AUTO_CLASSES)
        for _suffix, _auto, tag in TASK_HEADS:
            assert is_embedding_only(tag), tag


class TestTheStoredTagStillDecides:
    async def test_a_hub_embedder_with_no_config_on_disk_is_refused_by_its_tag(self, batch_db):
        """The config is a second source, not a replacement: a row tagged non-generative is
        refused even when its checkpoint has no config.json to read (not downloaded, GGUF)."""
        async with batch_db() as s:
            s.add(Model(name="embedder", source=ModelSource.HUGGINGFACE, repo_id="o/embedder",
                        quantization=QuantizationType.FP16, cache_path="huggingface/o--embedder--FP16",
                        status=ModelStatus.READY, architecture="sentence-similarity", gguf_label=""))
            await s.commit()
        async with client_for(app_with_db(batch_db)) as c:
            with patch.object(ModelService, "load_model_and_wait", AsyncMock()) as load:
                r = await c.post("/v1/chat/completions", json=_chat("embedder"))
        assert r.status_code == 400 and r.json()["error"]["code"] == "model_not_generative"
        assert "sentence-similarity" in r.json()["error"]["message"]
        load.assert_not_awaited()
