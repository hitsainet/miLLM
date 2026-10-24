"""Two precisions of one repository, through the real `/v1` routes (2026-10-18, suite D56).

Live: `kushalpatil/jevify-gemma4-e4b` was registered twice, row 7 at Q8 (bitsandbytes) and row 13
at FP16, and a non-GGUF registration took the bare repo tail, so BOTH were `jevify-gemma4-e4b`:

* `system_fingerprint` named the OTHER row's precision: `_carrier_row` looked the resident up by
  name, took `.first()` with no ORDER BY, and never compared the row's id with the resident's;
* with nothing resident, a request naming it auto-loaded whichever row the database listed first;
* `/v1/models` showed one name for two models, and miDataworks could not pin either.

Real objects: the app (`create_app`), `ModelService` and `ModelRepository` over the SQLite test
session, `download_model`'s naming, and an `InferenceService` serving a tiny real Llama on the
CPU. Faked: the Hub (`get_model_info`), the background download worker, and
`load_model_and_wait` (a spy: what it is CALLED with is the auto-load's choice).

"Legacy" tests drop `uq_models_name` to rebuild the pre-027 table the live rows sat in.

MUTATION CONTROLS (each turns this file red; run 2026-10-18, see the commit):
  M1 `_carrier_row` back to the name lookup        -> fingerprint FP16/Q8 tests
  M2 `is_resident_row` compares names, not ids      -> fingerprint + listing tests
  M3 find_by_name exact `.first()` (no ambiguity)   -> legacy auto-load refusal test
  M4 drop the `:QUANT` tag in `_unique_served_name` -> download naming tests
  M5 drop the `:FP16` fallback                      -> legacy-holder download test
  M6 drop `uq_models_name` from the model           -> index test
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from millm.api.schemas.model import ModelDownloadRequest
from millm.core.errors import ModelNameTakenError
from millm.db.models.model import ModelSource, ModelStatus, QuantizationType
from millm.db.repositories.model_repository import ModelRepository
from millm.ml.model_loader import LoadedModel, LoadedModelState
from millm.services.model_service import ModelService
from tests.support.factories import make_model
from tests.unit.f25_fixtures import clear_loaded, word_model, word_tokenizer

REPO = "kushalpatil/jevify-gemma4-e4b"
NAME = "jevify-gemma4-e4b"
CHAT = {"messages": [{"role": "user", "content": "w1 w2 w3"}], "max_tokens": 2}


@pytest.fixture(autouse=True)
def _clean():
    clear_loaded()
    yield
    clear_loaded()


async def _drop_unique_name_index(engine) -> None:
    async with engine.begin() as conn:
        await conn.execute(sa.text("DROP INDEX uq_models_name"))


async def _legacy_rows(session) -> None:
    """Rows 7 (Q8) and 13 (FP16) under ONE name, inserted in the live order."""
    for row_id, quant in ((7, QuantizationType.Q8), (13, QuantizationType.FP16)):
        session.add(make_model(id=row_id, name=NAME, repo_id=REPO, quantization=quant,
                               cache_path=f"huggingface/x--{quant.value}",
                               status=ModelStatus.READY))
    await session.commit()


def _service(session) -> ModelService:
    downloader = MagicMock()
    downloader.get_model_info = MagicMock(return_value={"params": "7.9B",
                                                         "architecture": "text-generation"})
    service = ModelService(repository=ModelRepository(session), downloader=downloader)
    service._download_worker = lambda *a, **k: None  # no Hub, no thread writing the row
    service.load_model_and_wait = AsyncMock()
    return service


def _inference(resident_id: int | None, resident_name: str = NAME):
    from millm.services.inference_service import InferenceService

    if resident_id is not None:
        LoadedModelState().set(LoadedModel(
            model_id=resident_id, model_name=resident_name, model=word_model(seed=1),
            tokenizer=word_tokenizer(), loaded_at=datetime(2026, 10, 18), memory_used_mb=1,
            num_parameters=1, device="cpu", dtype="bfloat16",
        ))
    with patch("millm.services.inference_service.torch.cuda.is_available", return_value=False):
        inference = InferenceService(model_service=None)
    inference._device = "cpu"
    return inference


def _client(service, inference) -> httpx.AsyncClient:
    from millm.api.dependencies import get_inference_service, get_model_service
    from millm.main import create_app

    app = create_app()
    app.dependency_overrides[get_model_service] = lambda: service
    app.dependency_overrides[get_inference_service] = lambda: inference
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


async def _chat(client, model: str) -> httpx.Response:
    return await client.post("/v1/chat/completions", json={"model": model, **CHAT})


class TestTheFingerprintNamesTheResidentRow:
    """Legacy table, both rows named alike: the fingerprint must follow the RESIDENT's id."""

    @pytest.mark.parametrize("resident_id, precision", [(13, "/FP16:"), (7, "/Q8:")])
    async def test_each_resident_precision_is_the_one_reported(
        self, test_engine, test_session, resident_id, precision
    ):
        await _drop_unique_name_index(test_engine)
        await _legacy_rows(test_session)
        service = _service(test_session)

        async with _client(service, _inference(resident_id)) as client:
            response = await _chat(client, NAME)

        assert response.status_code == 200, response.text
        fingerprint = response.json()["system_fingerprint"]
        assert precision in fingerprint, fingerprint
        service.load_model_and_wait.assert_not_called()


class TestNothingResidentIsNotACoinToss:
    async def test_a_shared_name_is_refused_naming_both_rows(self, test_engine, test_session):
        """Before: whichever row the database listed first was auto-loaded."""
        await _drop_unique_name_index(test_engine)
        await _legacy_rows(test_session)
        service = _service(test_session)

        async with _client(service, _inference(None)) as client:
            response = await _chat(client, NAME)

        assert response.status_code == 400, response.text
        message = response.json()["error"]["message"]
        assert "id 7 (Q8)" in message and "id 13 (FP16)" in message
        service.load_model_and_wait.assert_not_called()


class TestTheListingKnowsTheResidentById:
    async def test_only_the_resident_row_takes_the_served_name(self, test_engine, test_session):
        """Legacy table, the resident (row 13) served under another name (an adapter's): the
        Q8 row 7 sharing its row name must keep its own entry, not be relabelled."""
        await _drop_unique_name_index(test_engine)
        await _legacy_rows(test_session)
        service = _service(test_session)
        inference = MagicMock()
        info = MagicMock(model_id=13, served_name="jevify-adapter", base_served_name=NAME)
        info.name = NAME
        inference.get_loaded_model_info = lambda: info

        async with _client(service, inference) as client:
            response = await client.get("/v1/models")

        ids = sorted(m["id"] for m in response.json()["data"])
        assert ids == sorted(["jevify-adapter", NAME]), ids


class TestRegistrationGivesEachPrecisionItsOwnName:
    async def test_fp16_keeps_the_bare_name_and_q8_is_tagged(self, test_session):
        service = _service(test_session)
        fp16 = await service.download_model(ModelDownloadRequest(
            source=ModelSource.HUGGINGFACE, repo_id=REPO, quantization=QuantizationType.FP16))
        q8 = await service.download_model(ModelDownloadRequest(
            source=ModelSource.HUGGINGFACE, repo_id=REPO, quantization=QuantizationType.Q8))

        assert (fp16.name, q8.name) == (NAME, f"{NAME}:Q8")

    async def test_both_are_listed_and_each_name_reaches_its_own_row(self, test_session):
        service = _service(test_session)
        fp16 = await service.download_model(ModelDownloadRequest(
            source=ModelSource.HUGGINGFACE, repo_id=REPO, quantization=QuantizationType.FP16))
        q8 = await service.download_model(ModelDownloadRequest(
            source=ModelSource.HUGGINGFACE, repo_id=REPO, quantization=QuantizationType.Q8))
        for row in (fp16, q8):
            await service.repository.update_status(row.id, ModelStatus.READY)

        async with _client(service, _inference(fp16.id, NAME)) as client:
            listed = sorted(m["id"] for m in (await client.get("/v1/models")).json()["data"])
            resident = await _chat(client, NAME)
            other = await _chat(client, f"{NAME}:Q8")

        assert listed == [NAME, f"{NAME}:Q8"]
        assert "/FP16:" in resident.json()["system_fingerprint"]
        # Naming the Q8 auto-loads THAT row (the spy cannot load it, so the route then answers
        # an error; the choice is what matters here).
        [call] = service.load_model_and_wait.await_args_list
        assert call.args[0] == q8.id, call
        assert other.status_code != 200

    async def test_a_legacy_untagged_holder_keeps_its_name(self, test_engine, test_session):
        """A Q8 row registered before the rule holds the bare name; the FP16 registered now is
        tagged instead, so the name a client already uses never moves."""
        test_session.add(make_model(id=7, name=NAME, repo_id=REPO,
                                    quantization=QuantizationType.Q8, cache_path="q8"))
        await test_session.commit()
        service = _service(test_session)

        fp16 = await service.download_model(ModelDownloadRequest(
            source=ModelSource.HUGGINGFACE, repo_id=REPO, quantization=QuantizationType.FP16))

        assert fp16.name == f"{NAME}:FP16"
        assert (await service.repository.get_by_id(7)).name == NAME

    async def test_a_typed_name_that_is_taken_is_refused(self, test_session):
        test_session.add(make_model(id=1, name="my-judge", repo_id="o/a", cache_path="a"))
        await test_session.commit()
        service = _service(test_session)

        with pytest.raises(ModelNameTakenError) as refused:
            await service.download_model(ModelDownloadRequest(
                source=ModelSource.HUGGINGFACE, repo_id="o/b",
                quantization=QuantizationType.FP16, custom_name="my-judge"))
        assert refused.value.details == {"name": "my-judge", "model_id": 1}


class TestTheDatabaseRefusesASecondRowWithOneName:
    async def test_uq_models_name(self, test_session):
        test_session.add(make_model(id=1, name=NAME, repo_id=REPO, cache_path="a"))
        await test_session.commit()
        test_session.add(make_model(id=2, name=NAME, repo_id=REPO,
                                    quantization=QuantizationType.Q8, cache_path="b"))
        with pytest.raises(IntegrityError):
            await test_session.commit()
