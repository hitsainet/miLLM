"""Unit tests for attach_set config/dtype resolution (Feature 12, task 4.4)."""

import torch

from millm.core.config import settings
from millm.services.sae_service import _resolve_attach_dtype


class TestAttachDtypeResolution:
    def test_float16_names(self):
        for name in ("float16", "fp16", "half", "FLOAT16", " Fp16 "):
            assert _resolve_attach_dtype(name) is torch.float16

    def test_bfloat16_names(self):
        assert _resolve_attach_dtype("bfloat16") is torch.bfloat16
        assert _resolve_attach_dtype("bf16") is torch.bfloat16

    def test_float32_names(self):
        assert _resolve_attach_dtype("float32") is torch.float32
        assert _resolve_attach_dtype("fp32") is torch.float32

    def test_unknown_falls_back_to_fp16_not_raise(self):
        assert _resolve_attach_dtype("nonsense") is torch.float16


class TestCircuitConfigDefaults:
    def test_vram_envelope_default_accommodates_the_contract_maximum(self):
        """The budget is ADVISORY — real capacity is gated against live free
        VRAM in attach_set. It was 200 MB: the two-SAE spike's close-out
        TARGET, not a capacity figure. A 5-SAE circuit on a 24 GB card sits at
        ~640 MB and tripped an "over the VRAM envelope" warning that read like
        a refusal, so a documentation number was masquerading as an
        operational limit.

        Assert the PROPERTY that matters rather than a literal: the budget must
        comfortably fit the 16-layer contract maximum at the measured 128 MB
        fp16 per SAE, or it will cry wolf on a legitimate circuit."""
        measured_mb_per_sae = 128
        assert (
            settings.MULTISAE_VRAM_ENVELOPE_MB
            >= settings.CIRCUIT_MAX_LAYERS * measured_mb_per_sae
        )

    def test_attach_dtype_default_is_fp16(self):
        assert _resolve_attach_dtype(settings.MULTISAE_ATTACH_DTYPE) is torch.float16

    def test_intensity_bounds_default(self):
        assert settings.CIRCUIT_INTENSITY_MIN == 0.0
        assert settings.CIRCUIT_INTENSITY_MAX == 2.0
