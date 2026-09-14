"""Inference reads the model's own card(s), never an implicit GPU 0.

MUTATION CONTROLS (each must turn this file red):
  * draft device_map back to "auto"                -> "draft on the input device" fails
  * _kv_fits checks only gpu_indices[0]             -> "every card holds its share" fails
  * _get_input_device fallback back to self._device -> "falls back to its card" fails
Phase 2, 2026-09-14 (mutate.py; restored and sha256-verified):
  M9  _release_draft_model no longer drops the draft
      -> test_a_model_change_releases_the_draft_so_the_next_lands_beside_the_new_model
  M9c _release_draft_model no longer re-arms a failed draft
      -> test_a_draft_that_failed_on_the_old_card_is_tried_again_after_a_reload
  M13 _kv_fits ignores the layer shares (even split)
      -> test_each_card_is_asked_for_its_share_of_the_layers, test_a_card_with_no_layers_is_not_asked,
         test_the_batch_sizing_reads_the_layer_split_from_the_model
  M13b the batch sizing passes no shares to _kv_fits
      -> test_the_batch_sizing_reads_the_layer_split_from_the_model
  M14 drop the embedding leaf-name match
      -> test_a_nested_multimodal_model_sends_inputs_to_its_embedding_card
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from millm.services.inference_service import InferenceService
from tests.support.fake_gpus import RTX_3090, TI_3080, fake_gpus

VERIFY = "millm.ml.memory_utils.verify_memory_available"


def _service(model, gpu_indices):
    svc = InferenceService.__new__(InferenceService)
    svc._device = "cpu"
    svc._speculative_model_id = "draft/model"
    svc._configured_speculative_model_id = "draft/model"
    svc._draft_model = None
    svc._cbm_backend = None
    svc._model_state = SimpleNamespace(
        is_loaded=True,
        current=SimpleNamespace(
            model=model, gpu_indices=gpu_indices, tokenizer=MagicMock(encode=lambda p: [1] * 10)
        ),
    )
    return svc


class TestTheDraftModel:
    def test_goes_whole_onto_the_main_models_input_device(self):
        svc = _service(SimpleNamespace(hf_device_map={"model.embed_tokens": 1}), [1])
        with patch("transformers.AutoModelForCausalLM") as factory:
            svc._get_draft_model()
        assert factory.from_pretrained.call_args.kwargs["device_map"] == {"": "cuda:1"}

    def test_a_model_change_releases_the_draft_so_the_next_lands_beside_the_new_model(self):
        """The draft was loaded once and never dropped: after the model moved to
        another card, the old card's draft kept proposing tokens to it."""
        svc = _service(SimpleNamespace(hf_device_map={"model.embed_tokens": 1}), [1])
        with patch("transformers.AutoModelForCausalLM") as factory:
            svc._get_draft_model()
            svc.on_model_unloading()
            svc._model_state.current = SimpleNamespace(
                model=SimpleNamespace(hf_device_map={"model.embed_tokens": 0}), gpu_indices=[0]
            )
            svc.on_model_loaded()
            svc._get_draft_model()
        assert [c.kwargs["device_map"] for c in factory.from_pretrained.call_args_list] == [
            {"": "cuda:1"},
            {"": "cuda:0"},
        ]

    def test_a_draft_that_failed_on_the_old_card_is_tried_again_after_a_reload(self):
        svc = _service(SimpleNamespace(hf_device_map={"model.embed_tokens": 1}), [1])
        with patch("transformers.AutoModelForCausalLM") as factory:
            factory.from_pretrained.side_effect = [RuntimeError("CUDA out of memory"), MagicMock()]
            assert svc._get_draft_model() is None
            assert svc._get_draft_model() is None, "a failed draft disables speculation"
            svc.on_model_loaded()
            assert svc._get_draft_model() is not None
        assert factory.from_pretrained.call_count == 2


class TestTheInputDevice:
    def test_falls_back_to_the_models_card_not_gpu0(self):
        broken = MagicMock()
        type(broken).hf_device_map = property(lambda self: (_ for _ in ()).throw(RuntimeError()))
        type(broken).get_input_embeddings = property(lambda self: (_ for _ in ()).throw(RuntimeError()))
        svc = _service(broken, [1])
        assert svc._get_input_device() == "cuda:1"

    def test_a_nested_multimodal_model_sends_inputs_to_its_embedding_card(self):
        model = SimpleNamespace(hf_device_map={
            "model.vision_tower": 0,
            "model.language_model.embed_tokens": 1,
            "model.language_model.layers.0": 1,
        })
        assert _service(model, [0, 1])._get_input_device() == "cuda:1"

    def test_no_model_is_the_cpu_even_with_cards(self):
        with fake_gpus((TI_3080, 11_000, 12_288), (RTX_3090, 23_000, 24_576)):
            svc = InferenceService(model_service=None)
        assert svc._device == "cpu"


class TestKvCacheFit:
    def test_a_single_card_model_is_checked_on_its_card(self):
        with patch(VERIFY, return_value=(True, 9_000)) as verify:
            assert InferenceService._kv_fits(1_000, [1]) == (True, 9_000)
        assert [c.kwargs["device"] for c in verify.call_args_list] == [1]
        assert verify.call_args.args[0] == 1_000

    def test_every_card_holds_its_share_of_a_split_model(self):
        answers = {0: (False, 300), 1: (True, 20_000)}
        with patch(VERIFY, side_effect=lambda mb, device: answers[device]) as verify:
            fits, least_free = InferenceService._kv_fits(1_001, [0, 1])
        assert (fits, least_free) == (False, 300)
        assert sorted((c.args[0], c.kwargs["device"]) for c in verify.call_args_list) == [
            (501, 0),
            (501, 1),
        ]

    def test_each_card_is_asked_for_its_share_of_the_layers(self):
        answers = {0: (True, 3_000), 1: (True, 20_000)}
        with patch(VERIFY, side_effect=lambda mb, device: answers[device]) as verify:
            assert InferenceService._kv_fits(1_000, [0, 1], {0: 0.25, 1: 0.75}) == (True, 3_000)
        assert sorted((c.args[0], c.kwargs["device"]) for c in verify.call_args_list) == [
            (250, 0),
            (750, 1),
        ]

    def test_a_card_with_no_layers_is_not_asked(self):
        with patch(VERIFY, return_value=(True, 20_000)) as verify:
            InferenceService._kv_fits(1_000, [0, 1], {0: 0.0, 1: 1.0})
        assert [(c.args[0], c.kwargs["device"]) for c in verify.call_args_list] == [(1_000, 1)]

    def test_the_batch_sizing_reads_the_layer_split_from_the_model(self):
        model = SimpleNamespace(hf_device_map={
            "model.embed_tokens": 0,
            "model.layers.0": 0,
            "model.layers.1": 1,
            "model.layers.2": 1,
            "model.layers.3": 1,
        })
        svc = _service(model, [0, 1])
        with patch.object(svc, "_project_kv_bytes", return_value=1_000 * 1024 * 1024), \
                patch(VERIFY, return_value=(True, 20_000)) as verify:
            svc._chunk_batch_for_memory(["a", "b"], 10)
        # 1,000 MB projected, +20% slack = 1,200 MB: a quarter on card 0.
        assert sorted((c.args[0], c.kwargs["device"]) for c in verify.call_args_list) == [
            (300, 0),
            (900, 1),
        ]
