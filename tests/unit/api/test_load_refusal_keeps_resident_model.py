"""A refused load must not cost the operator the model they are serving.

The load runs in the background and unloads the resident model first, so a
placement refusal found there arrived as an ERROR status after the served model
was already gone. ModelService.load_model now checks placement BEFORE unloading,
against what each card will have once the resident model is gone: live free
memory plus that model's recorded usage on the card.

The node here: RTX 3080 Ti (index 0, 11 GB free) and RTX 3090 (index 1, 5 GB
free because the resident model holds 16 GB of it).

MUTATION CONTROLS (each must turn this file red):
  * drop the resident-usage add-back   -> "fits once the resident memory is counted back" fails
  * remove the pre-check call          -> both refusal tests see the model unloaded and a 202
Phase 2, 2026-09-14: M7 (Q4 skips the check again) -> test_a_quantized_model_no_split_holds_is_refused_before_the_unload
Review round 1, 2026-09-14 (mutate.py; restored and sha256-verified):
  R1-M3c the pre-check stops passing the checkpoint's is_pre_quantized
         -> test_a_pre_quantized_checkpoint_is_not_planned_with_bitsandbytes_factor,
            test_a_relative_cache_path_is_read_under_the_model_cache_dir,
            test_a_pre_quantized_q2_checkpoint_still_loads
  R1-M3d the pre-check reads the row's raw cache_path, not the resolved one
         -> test_a_relative_cache_path_is_read_under_the_model_cache_dir
  R1-M4  drop the Q2 refusal -> test_a_q2_transformers_checkpoint_is_refused_before_the_unload
  R1-M5  drop the GGUF_TENSOR_SPLIT pre-check
         -> test_a_list_shorter_than_the_cards_it_needs_is_refused_before_the_unload
  R1-M5c refuse any length that differs (`!=`), not only a shorter list
         -> test_a_list_longer_than_the_cards_it_found_is_left_to_the_loader
  (`<` -> `<=` SURVIVED and is an EQUIVALENT mutation: at the exact count the
  pre-check calls _gguf_tensor_split, which raises only when the lengths differ.
  R1-M5c is the variant that can change the answer.)
"""

from concurrent.futures import Future
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from millm.db.models.model import ModelStatus, QuantizationType
from millm.main import create_app
from millm.ml import model_loader
from millm.ml.model_loader import LoadedModel
from millm.services.model_service import ModelService
from tests.support.factories import make_model
from tests.support.fake_gpus import RTX_3090, TI_3080, fake_gpus

NODE = ((TI_3080, 11_000, 12_288), (RTX_3090, 5_000, 24_576))


class _RecordingExecutor:
    def __init__(self):
        self.calls = []

    def submit(self, fn, *args):
        self.calls.append((fn, args))
        future = Future()
        future.set_result(None)
        return future


def _service(model):
    repo = MagicMock()
    repo.get_by_id = AsyncMock(return_value=model)
    repo.update_status = AsyncMock(return_value=model)
    loader = MagicMock()
    loader.is_loaded = True
    loader.loaded_model_id = 9
    loader.state.current = LoadedModel(
        9, "resident", MagicMock(), MagicMock(), datetime.utcnow(),
        memory_used_mb=16_000, device="cuda:1", gpu_indices=[1],
        memory_by_device_mb={"cuda:1": 16_000},
    )
    svc = ModelService(repository=repo, downloader=MagicMock(), loader=loader, emitter=None)
    svc.unload_model = AsyncMock()
    svc._executor = _RecordingExecutor()
    return svc, repo


def _post(svc, body):
    from millm.api.dependencies import get_model_service

    app = create_app()
    app.dependency_overrides[get_model_service] = lambda: svc
    return TestClient(app).post("/api/models/3/load", json=body)


def _fp16(estimated_mb, **extra):
    return make_model(
        id=3, status=ModelStatus.READY, quantization=QuantizationType.FP16,
        estimated_memory_mb=estimated_mb, **extra,
    )


class TestARefusalKeepsTheServedModel:
    def test_an_explicit_card_too_small_is_refused_synchronously(self):
        svc, repo = _service(_fp16(18_000))
        with fake_gpus(*NODE):
            response = _post(svc, {"gpu": 0})

        assert response.status_code == 507, response.text
        assert "INSUFFICIENT_MEMORY" in response.text
        assert not svc.unload_model.called, "the served model was unloaded for a load that was refused"
        assert not repo.update_status.called
        assert svc._executor.calls == []

    def test_nothing_fits_even_summed_is_refused_synchronously(self):
        # 11 GB + (5 + 16) GB projected = 32 GB, for a 60 GB FP16 model.
        svc, _ = _service(_fp16(60_000))
        with fake_gpus(*NODE):
            response = _post(svc, {"gpu": "auto"})

        assert response.status_code == 507, response.text
        assert not svc.unload_model.called
        assert svc._executor.calls == []

    def test_the_refusal_carries_the_same_details_as_the_loader(self):
        svc, _ = _service(_fp16(18_000))
        with fake_gpus(*NODE):
            body = _post(svc, {"gpu": 0}).json()
        details = str(body)
        assert "18000" in details and "11000" in details

    def test_an_explicit_gguf_card_too_small_is_refused_synchronously(self, tmp_path):
        gguf = tmp_path / "big-Q4_K_M.gguf"
        with open(gguf, "wb") as handle:
            handle.truncate(12 * 1024 ** 3)  # sparse: 12 GB of weights, no disk used
        model = make_model(
            id=3, status=ModelStatus.READY, quantization=QuantizationType.Q4,
            gguf_label="Q4_K_M", gguf_files=["big-Q4_K_M.gguf"], cache_path=str(tmp_path),
        )
        svc, _ = _service(model)
        with fake_gpus(*NODE), patch.object(model_loader, "llama_supports_gpu_offload", lambda: True):
            response = _post(svc, {"gpu": 0})

        assert response.status_code == 507, response.text
        assert not svc.unload_model.called


class TestSplitsAreJudgedBeforeTheUnload:
    """Projected: 11 GB on card 0, 5 + 16 = 21 GB on card 1. A split's budgets
    are 9,976 + 19,976 = 29,952 MB."""

    def test_all_cards_that_cannot_hold_it_are_refused_synchronously(self):
        svc, _ = _service(_fp16(31_000))
        with fake_gpus(*NODE):
            response = _post(svc, {"gpu": "all"})

        assert response.status_code == 507, response.text
        assert not svc.unload_model.called
        assert svc._executor.calls == []

    def test_a_quantized_model_no_split_holds_is_refused_before_the_unload(self):
        """Q4 skipped the check on the grounds that it could spill to the CPU;
        with no spill, skipping it only moved the refusal past the unload."""
        model = make_model(
            id=3, status=ModelStatus.READY, quantization=QuantizationType.Q4,
            estimated_memory_mb=60_000,
        )
        svc, _ = _service(model)
        with fake_gpus(*NODE):
            response = _post(svc, {})

        assert response.status_code == 507, response.text
        assert not svc.unload_model.called

    def test_all_that_fits_is_accepted_and_forwarded(self):
        svc, _ = _service(_fp16(8_000))
        with fake_gpus(*NODE):
            response = _post(svc, {"gpu": "all"})

        assert response.status_code == 202, response.text
        [(_, args)] = svc._executor.calls
        assert args[-1] == "all"


class TestAValidSwitchStillHappens:
    def test_a_card_that_fits_once_the_resident_memory_is_counted_back_is_accepted(self):
        # Card 1 has 5 GB free now, 21 GB once the resident model is gone.
        svc, _ = _service(_fp16(18_000))
        with fake_gpus(*NODE):
            response = _post(svc, {"gpu": 1})

        assert response.status_code == 202, response.text
        svc.unload_model.assert_awaited_once_with(9)
        [(_, args)] = svc._executor.calls
        assert args[-1] == 1

    def test_auto_still_loads(self):
        svc, _ = _service(_fp16(8_000))
        with fake_gpus(*NODE):
            response = _post(svc, {})

        assert response.status_code == 202, response.text
        svc.unload_model.assert_awaited_once_with(9)
        assert svc._executor.calls[0][1][-1] is None

    def test_the_precheck_creates_no_cuda_context(self):
        svc, _ = _service(_fp16(8_000))
        with fake_gpus(*NODE) as fake:
            fake.forbid(0, 1)
            _post(svc, {"gpu": 1})
        assert fake.calls == []


def _checkpoint(directory, quantization_config):
    import json

    directory.mkdir(parents=True, exist_ok=True)
    config = {"model_type": "llama"}
    if quantization_config is not None:
        config["quantization_config"] = quantization_config
    (directory / "config.json").write_text(json.dumps(config))
    return directory


class TestThePrecheckReadsTheCheckpoint:
    """Review round 1, 2026-09-14: the pre-check planned every Q4/Q8 row with
    bitsandbytes' 0.9, including a GPTQ/AWQ checkpoint that gets no bitsandbytes.

    Projected budgets: 9,976 + 19,976 = 29,952 MB; with bitsandbytes' 0.9,
    8,978 + 17,978 = 26,956 MB. A 28,000 MB Q8 row fits the first, not the second.
    """

    @staticmethod
    def _q8(cache_path):
        return make_model(
            id=3, status=ModelStatus.READY, quantization=QuantizationType.Q8,
            estimated_memory_mb=28_000, cache_path=str(cache_path),
        )

    def test_a_pre_quantized_checkpoint_is_not_planned_with_bitsandbytes_factor(self, tmp_path):
        model = self._q8(_checkpoint(tmp_path, {"quant_method": "gptq", "bits": 8}))
        svc, _ = _service(model)
        with fake_gpus(*NODE):
            response = _post(svc, {})

        assert response.status_code == 202, response.text
        svc.unload_model.assert_awaited_once_with(9)

    def test_the_same_row_on_a_plain_checkpoint_is_refused_before_the_unload(self, tmp_path):
        svc, _ = _service(self._q8(_checkpoint(tmp_path, None)))
        with fake_gpus(*NODE):
            response = _post(svc, {})

        assert response.status_code == 507, response.text
        assert "26956" in response.text
        assert not svc.unload_model.called

    def test_a_relative_cache_path_is_read_under_the_model_cache_dir(self, tmp_path):
        """The row may store the path relative to MODEL_CACHE_DIR; the pre-check
        must open the checkpoint the background load will open."""
        from millm.core.config import settings

        _checkpoint(tmp_path / "org--gptq-model", {"quant_method": "gptq", "bits": 8})
        svc, _ = _service(self._q8("org--gptq-model"))
        with fake_gpus(*NODE), patch.object(settings, "MODEL_CACHE_DIR", str(tmp_path)):
            response = _post(svc, {})

        assert response.status_code == 202, response.text


class TestQ2IsNotLoadedAsSomethingElse:
    def test_a_q2_transformers_checkpoint_is_refused_before_the_unload(self, tmp_path):
        model = make_model(
            id=3, status=ModelStatus.READY, quantization=QuantizationType.Q2,
            estimated_memory_mb=3_600, cache_path=str(_checkpoint(tmp_path, None)),
        )
        svc, repo = _service(model)
        with fake_gpus(*NODE):
            response = _post(svc, {})

        assert response.status_code == 400, response.text
        assert "UNSUPPORTED_QUANTIZATION" in response.text
        assert "bfloat16" in response.text
        assert not svc.unload_model.called
        assert not repo.update_status.called
        assert svc._executor.calls == []

    def test_a_pre_quantized_q2_checkpoint_still_loads(self, tmp_path):
        model = make_model(
            id=3, status=ModelStatus.READY, quantization=QuantizationType.Q2,
            estimated_memory_mb=3_600,
            cache_path=str(_checkpoint(tmp_path, {"quant_method": "bitnet"})),
        )
        svc, _ = _service(model)
        with fake_gpus(*NODE):
            response = _post(svc, {})

        assert response.status_code == 202, response.text


class TestAGgufTensorSplitIsCheckedBeforeTheUnload:
    """Projected GGUF limits: 11,000 x 0.94 - 2,048 = 8,292 MB and
    21,000 x 0.94 - 2,048 = 17,692 MB. 22 GB of weights fit neither card alone
    ((22,528 + 2,048) / 0.94 = 26,145 MB > 21,000), so the split takes both."""

    @staticmethod
    def _gguf(tmp_path):
        gguf = tmp_path / "big-Q4_K_M.gguf"
        with open(gguf, "wb") as handle:
            handle.truncate(22 * 1024 ** 3)  # sparse
        return make_model(
            id=3, status=ModelStatus.READY, quantization=QuantizationType.Q4,
            gguf_label="Q4_K_M", gguf_files=["big-Q4_K_M.gguf"], cache_path=str(tmp_path),
        )

    def _post_with_split(self, tmp_path, tensor_split):
        from millm.core.config import settings

        svc, _ = _service(self._gguf(tmp_path))
        with fake_gpus(*NODE), \
                patch.object(model_loader, "llama_supports_gpu_offload", lambda: True), \
                patch.object(settings, "GGUF_TENSOR_SPLIT", tensor_split):
            return svc, _post(svc, {})

    def test_a_list_shorter_than_the_cards_it_needs_is_refused_before_the_unload(self, tmp_path):
        svc, response = self._post_with_split(tmp_path, "1")

        assert response.status_code == 500, response.text
        assert "GGUF_TENSOR_SPLIT has 1 value(s)" in response.text
        assert not svc.unload_model.called
        assert svc._executor.calls == []

    def test_one_value_per_card_is_accepted(self, tmp_path):
        svc, response = self._post_with_split(tmp_path, "1,3")

        assert response.status_code == 202, response.text
        svc.unload_model.assert_awaited_once_with(9)

    def test_a_list_longer_than_the_cards_it_found_is_left_to_the_loader(self, tmp_path):
        """The pre-check's cards are a LOWER bound: sized without the KV cache,
        the loader may take a third card. Here it finds two (limits 17,692 and
        16,752 MB cover 22,528) while the list names three — not a refusal
        here; only a list SHORTER than the cards found is."""
        from millm.core.config import settings

        svc, _ = _service(self._gguf(tmp_path))
        three_cards = NODE + (("RTX A5000", 20_000, 24_576),)
        with fake_gpus(*three_cards), \
                patch.object(model_loader, "llama_supports_gpu_offload", lambda: True), \
                patch.object(settings, "GGUF_TENSOR_SPLIT", "1,1,1"):
            response = _post(svc, {})

        assert response.status_code == 202, response.text
