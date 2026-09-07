"""A GGUF model must be REFUSED by the interpretability paths, loudly.

SAE attachment, steering and sensing are `register_forward_hook` on a resolved
`nn.Module`. A llama.cpp model is a ctypes handle onto a C++ graph: no module
tree, no per-layer residual reachable from Python. These are not unimplemented
features, they are impossible ones.

Left unguarded, the failure is `_get_layer` exhausting six attribute paths and
raising "Could not find layer N. Model architecture may not be supported" — a
500 blaming the ARCHITECTURE for a limitation of the RUNTIME.

MUTATION CONTROLS (each must turn this file red):
  * remove the engine check in check_compatibility -> "refuses an SAE" fails
  * let on_model_loaded start CBM regardless       -> "CBM is not started" fails
  * drop the llamacpp branch in create_chat_completion -> "routes to llama.cpp" fails
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from millm.ml.model_loader import ENGINE_LLAMACPP, ENGINE_TRANSFORMERS, LoadedModel


def _loaded(engine: str) -> LoadedModel:
    return LoadedModel(
        model_id=1,
        model_name="m",
        model=MagicMock(),
        tokenizer=None if engine == ENGINE_LLAMACPP else MagicMock(),
        loaded_at=datetime.utcnow(),
        engine=engine,
    )


class TestSAEAttachmentIsRefused:
    @pytest.mark.asyncio
    async def test_compatibility_refuses_a_hookless_engine(self):
        """ONE guard: attach_sae, attach_set and the compatibility route all
        funnel through check_compatibility, so guarding each call site would be
        three chances to forget one."""
        from millm.services.sae_service import SAEService

        service = SAEService.__new__(SAEService)
        service.get_sae = AsyncMock(return_value=MagicMock(status=MagicMock(value="cached")))
        service._hooker = MagicMock()
        service._hooker.get_layer_count.return_value = 32

        state = MagicMock()
        state.is_loaded = True
        state.current = _loaded(ENGINE_LLAMACPP)

        with patch("millm.services.sae_service.LoadedModelState", return_value=state):
            result = await service.check_compatibility("sae-1", 12)

        joined = " ".join(result.errors)
        assert "llamacpp" in joined
        assert "forward hook" in joined, (
            "the message must name the RUNTIME limitation, not blame the "
            "model architecture the way _get_layer's ValueError does"
        )
        assert result.compatible is False

    @pytest.mark.asyncio
    async def test_a_transformers_model_is_not_refused_for_its_engine(self):
        """The guard must not fire on the ordinary case."""
        from millm.services.sae_service import SAEService

        service = SAEService.__new__(SAEService)
        sae = MagicMock()
        sae.status = MagicMock(value="cached")
        sae.d_in = 2304
        service.get_sae = AsyncMock(return_value=sae)
        service._hooker = MagicMock()
        service._hooker.get_layer_count.return_value = 32

        state = MagicMock()
        state.is_loaded = True
        current = _loaded(ENGINE_TRANSFORMERS)
        current.model.config.hidden_size = 2304
        state.current = current

        with patch("millm.services.sae_service.LoadedModelState", return_value=state):
            result = await service.check_compatibility("sae-1", 12)

        assert not any("forward hook" in e for e in result.errors)


class TestContinuousBatchingIsNotStarted:
    def test_cbm_is_not_started_for_a_gguf_model(self):
        """CBM is transformers' ContinuousBatchingManager. Handing it a ctypes
        handle and a None tokenizer fails deep inside generation instead of
        here, which is why this is the highest-value early guard."""
        from millm.services.inference_service import InferenceService

        svc = InferenceService.__new__(InferenceService)
        svc._cbm_backend = MagicMock()
        state = MagicMock()
        state.is_loaded = True
        state.current = _loaded(ENGINE_LLAMACPP)
        svc._model_state = state

        svc.on_model_loaded()

        assert not svc._cbm_backend.start.called

    def test_cbm_still_starts_for_a_transformers_model(self):
        from millm.services.inference_service import InferenceService

        svc = InferenceService.__new__(InferenceService)
        svc._cbm_backend = MagicMock()
        state = MagicMock()
        state.is_loaded = True
        state.current = _loaded(ENGINE_TRANSFORMERS)
        svc._model_state = state

        svc.on_model_loaded()

        assert svc._cbm_backend.start.called


class TestTheBackendIsNamedHonestly:
    def test_reports_llamacpp(self):
        from millm.services.inference_service import InferenceService

        svc = InferenceService.__new__(InferenceService)
        svc._cbm_backend = None
        state = MagicMock()
        state.current = _loaded(ENGINE_LLAMACPP)
        svc._model_state = state

        assert svc.backend_name == "llamacpp", (
            "'serial' would claim the transformers path is running"
        )

    def test_reports_serial_for_transformers(self):
        from millm.services.inference_service import InferenceService

        svc = InferenceService.__new__(InferenceService)
        svc._cbm_backend = None
        state = MagicMock()
        state.current = _loaded(ENGINE_TRANSFORMERS)
        svc._model_state = state

        assert svc.backend_name == "serial"
