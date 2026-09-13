"""Inference reads the model's own card(s), never an implicit GPU 0.

MUTATION CONTROLS (each must turn this file red):
  * draft device_map back to "auto"                -> "draft on the input device" fails
  * _kv_fits checks only gpu_indices[0]             -> "every card holds its share" fails
  * _get_input_device fallback back to self._device -> "falls back to its card" fails
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from millm.services.inference_service import InferenceService
from tests.support.fake_gpus import RTX_3090, TI_3080, fake_gpus


def _service(model, gpu_indices):
    svc = InferenceService.__new__(InferenceService)
    svc._device = "cpu"
    svc._speculative_model_id = "draft/model"
    svc._draft_model = None
    svc._model_state = SimpleNamespace(
        is_loaded=True,
        current=SimpleNamespace(model=model, gpu_indices=gpu_indices),
    )
    return svc


class TestTheDraftModel:
    def test_goes_whole_onto_the_main_models_input_device(self):
        svc = _service(SimpleNamespace(hf_device_map={"model.embed_tokens": 1}), [1])
        with patch("transformers.AutoModelForCausalLM") as factory:
            svc._get_draft_model()
        assert factory.from_pretrained.call_args.kwargs["device_map"] == {"": "cuda:1"}


class TestTheInputDevice:
    def test_falls_back_to_the_models_card_not_gpu0(self):
        broken = MagicMock()
        type(broken).hf_device_map = property(lambda self: (_ for _ in ()).throw(RuntimeError()))
        svc = _service(broken, [1])
        assert svc._get_input_device() == "cuda:1"

    def test_no_model_is_the_cpu_even_with_cards(self):
        with fake_gpus((TI_3080, 11_000, 12_288), (RTX_3090, 23_000, 24_576)):
            svc = InferenceService(model_service=None)
        assert svc._device == "cpu"


class TestKvCacheFit:
    def test_a_single_card_model_is_checked_on_its_card(self):
        with patch("millm.ml.memory_utils.verify_memory_available", return_value=(True, 9_000)) as verify:
            assert InferenceService._kv_fits(1_000, [1]) == (True, 9_000)
        assert [c.kwargs["device"] for c in verify.call_args_list] == [1]
        assert verify.call_args.args[0] == 1_000

    def test_every_card_holds_its_share_of_a_split_model(self):
        answers = {0: (False, 300), 1: (True, 20_000)}
        with patch(
            "millm.ml.memory_utils.verify_memory_available",
            side_effect=lambda mb, device: answers[device],
        ) as verify:
            fits, least_free = InferenceService._kv_fits(1_001, [0, 1])
        assert (fits, least_free) == (False, 300)
        assert sorted((c.args[0], c.kwargs["device"]) for c in verify.call_args_list) == [
            (501, 0),
            (501, 1),
        ]
