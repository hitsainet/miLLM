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
Review round 1, 2026-09-14 (mutate.py; restored and sha256-verified):
  R1-M2 on_model_loaded starts continuous batching on a split model again (guard -> False)
      -> test_a_split_model_does_not_start_the_manager
Review round 2, 2026-09-14 (mutate.py; restored and sha256-verified):
  R2-M7  the draft always goes on the input device
      -> test_goes_on_the_split_card_with_the_most_free_memory, test_a_tie_goes_to_the_lower_index
  R2-M8  on_model_unloading does not suspend the draft
      -> test_no_draft_is_loaded_between_the_unload_and_the_next_load,
         test_a_draft_that_finishes_loading_after_the_unload_began_is_not_kept
  R2-M9  a draft that finishes loading after the unload began is kept
      -> test_a_draft_that_finishes_loading_after_the_unload_began_is_not_kept
  R2-M9b on_model_loaded does not lift the suspension
      -> both unload tests and test_a_model_change_releases_the_draft_so_the_next_lands_beside_the_new_model
  M9 re-run (_release_draft_model keeps the draft)
      -> test_a_model_change_releases_the_draft_..., test_no_draft_is_loaded_between_the_unload_...
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


class TestContinuousBatchingOnASplitModel:
    """transformers' PagedAttentionCache puts every layer's KV blocks on
    `model.device` (5.15.1 continuous_api.py:1001, cache.py:257). The layers of a
    split model on the other card would use a cache that is not on their card, so
    the manager is not started and requests take the serial path."""

    @staticmethod
    def _svc(gpu_indices):
        svc = _service(MagicMock(name="model"), gpu_indices)
        svc._model_state.current.supports_hooks = True
        svc._model_state.current.engine = "transformers"
        svc._cbm_backend = MagicMock(name="cbm")
        return svc

    def test_a_split_model_does_not_start_the_manager(self):
        svc = self._svc([0, 1])
        svc.on_model_loaded()
        assert svc._cbm_backend.start.call_count == 0

    def test_a_single_card_model_still_starts_it_with_that_model(self):
        svc = self._svc([1])
        svc.on_model_loaded()
        current = svc._model_state.current
        assert svc._cbm_backend.start.call_count == 1
        assert svc._cbm_backend.start.call_args.args == (current.model, current.tokenizer)


class TestTheDraftOnASplitModel:
    """Review round 2, 2026-09-14. A split fills its cards in index order, each
    but the last to its whole budget, and the lowest-index card holds the input
    embeddings: the input device is the FULLEST card of the split. The draft goes
    on whichever of the model's cards has the most free memory."""

    # The input embeddings on card 0, the fuller card here.
    SPLIT = {"model.embed_tokens": 0, "model.layers.0": 0, "model.layers.1": 1, "lm_head": 1}

    def test_goes_on_the_split_card_with_the_most_free_memory(self):
        svc = _service(SimpleNamespace(hf_device_map=self.SPLIT), [0, 1])
        assert svc._get_input_device() == "cuda:0", "the fixture must put the input device on the fuller card"
        with fake_gpus((TI_3080, 900, 12_288), (RTX_3090, 5_000, 24_576)), \
                patch("transformers.AutoModelForCausalLM") as factory:
            svc._get_draft_model()
        assert factory.from_pretrained.call_count == 1
        assert factory.from_pretrained.call_args.kwargs["device_map"] == {"": "cuda:1"}

    def test_a_tie_goes_to_the_lower_index(self):
        split = {"model.embed_tokens": 1, "model.layers.0": 1, "lm_head": 0}
        svc = _service(SimpleNamespace(hf_device_map=split), [0, 1])
        with fake_gpus((TI_3080, 5_000, 12_288), (RTX_3090, 5_000, 24_576)), \
                patch("transformers.AutoModelForCausalLM") as factory:
            svc._get_draft_model()
        assert factory.from_pretrained.call_args.kwargs["device_map"] == {"": "cuda:0"}

    def test_cards_that_cannot_be_read_leave_it_on_the_input_device(self):
        svc = _service(SimpleNamespace(hf_device_map=self.SPLIT), [0, 1])
        with fake_gpus((TI_3080, 900, 12_288), (RTX_3090, 5_000, 24_576)) as fake, \
                patch("transformers.AutoModelForCausalLM") as factory:
            fake.forbid(0, 1)
            svc._get_draft_model()
        assert factory.from_pretrained.call_args.kwargs["device_map"] == {"": "cuda:0"}


class TestTheDraftDuringAnUnload:
    """Review round 2, 2026-09-14. on_model_unloading released the draft and then
    ModelService drained pending requests for up to five seconds; each asked for
    the draft and loaded it again, beside the model being removed, holding its
    memory through the next load's placement."""

    def test_no_draft_is_loaded_between_the_unload_and_the_next_load(self):
        svc = _service(SimpleNamespace(hf_device_map={"model.embed_tokens": 1}), [1])
        with patch("transformers.AutoModelForCausalLM") as factory:
            assert svc._get_draft_model() is not None
            svc.on_model_unloading()
            assert svc._get_draft_model() is None, "a request drained during the unload loaded the draft again"
            assert factory.from_pretrained.call_count == 1
            svc.on_model_loaded()
            assert svc._get_draft_model() is not None
        assert factory.from_pretrained.call_count == 2
        assert svc._speculative_model_id == "draft/model", "suspension must not disable speculation"

    def test_a_draft_that_finishes_loading_after_the_unload_began_is_not_kept(self):
        svc = _service(SimpleNamespace(hf_device_map={"model.embed_tokens": 1}), [1])
        late = MagicMock(name="late draft")

        def _load_while_the_model_unloads(*args, **kwargs):
            svc.on_model_unloading()
            return late

        with patch("transformers.AutoModelForCausalLM") as factory:
            factory.from_pretrained.side_effect = _load_while_the_model_unloads
            assert svc._get_draft_model() is None
            assert svc._draft_model is None
            factory.from_pretrained.side_effect = None
            svc.on_model_loaded()
            assert svc._get_draft_model() is factory.from_pretrained.return_value
        assert not late.eval.called
