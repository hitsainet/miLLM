"""Encoder-decoder models must not have their output sliced away.

A decoder-only model returns [prompt..., generated...]; an encoder-decoder model
returns ONLY the decoder output. The generation path sliced off the prompt
length unconditionally, so every seq2seq request returned an empty string with
HTTP 200 and no error — a completely silent failure.

Reproduced on Falconsai/text_summarization (T5-small, 60.5M):
    prompt 30 tokens -> summary 23 tokens -> outputs[0][30:] == ''
The model itself was fine; it produced
    "FDA approved the drug after review. Regulators at the FDA cited safety..."
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from millm.services.inference_service import InferenceService


def _svc(is_encoder_decoder, config=None):
    """_model is a read-only property backed by the loaded-model state, so the
    model is injected through that state rather than assigned directly."""
    with patch("millm.services.inference_service.torch") as t:
        t.cuda.is_available.return_value = False
        svc = InferenceService(model_service=None)
    model = MagicMock()
    if config is not None:
        model.config = config
    else:
        model.config.is_encoder_decoder = is_encoder_decoder
    state = MagicMock()
    state.is_loaded = True
    state.current.model = model
    svc._model_state = state
    return svc


class TestSeq2SeqKeepsItsOutput:
    def test_encoder_decoder_output_is_not_sliced(self):
        """The decoder output IS the answer; there is no prompt to remove."""
        svc = _svc(True)
        # 23 generated tokens against a 30-token prompt — the real T5 case.
        out = torch.arange(23)
        got = svc._slice_generated(out, 30)
        assert len(got) == 23, (
            f"seq2seq output was truncated to {len(got)} tokens; the answer is gone"
        )
        assert torch.equal(got, out)

    def test_the_original_bug_produced_nothing(self):
        """Documents precisely what was wrong."""
        out = torch.arange(23)
        assert len(out[30:]) == 0

    def test_decoder_only_still_has_its_prompt_removed(self):
        """The fix must not leak the prompt back into decoder-only output."""
        svc = _svc(False)
        out = torch.arange(50)          # 30 prompt + 20 generated
        got = svc._slice_generated(out, 30)
        assert len(got) == 20
        assert got[0].item() == 30, "prompt tokens leaked into the completion"


class TestItFailsSafe:
    def test_a_model_without_the_flag_is_treated_as_decoder_only(self):
        """Absent config must not silently change existing behaviour."""
        svc = _svc(False, config=MagicMock(spec=[]))   # no attribute at all
        got = svc._slice_generated(torch.arange(50), 30)
        assert len(got) == 20

    def test_a_raising_config_does_not_break_generation(self):
        class _Boom:
            @property
            def is_encoder_decoder(self):
                raise RuntimeError("boom")

        svc = _svc(False, config=_Boom())
        got = svc._slice_generated(torch.arange(50), 30)
        assert len(got) == 20


class TestEveryGenerationPathUsesIt:
    """Three sites slice generated tokens; all must go through the helper."""

    def test_no_raw_slices_remain(self):
        src = open("millm/services/inference_service.py").read()
        assert "outputs[0][prompt_tokens:]" not in src, (
            "a raw prompt-length slice remains; seq2seq output would be lost there"
        )
        assert "outputs[row_idx][padded_width:]" not in src, (
            "the batched path still slices raw"
        )
        assert src.count("_slice_generated(") >= 4
