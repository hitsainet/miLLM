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


class TestTheRefusalDoesNotItselfCrash:
    """The guard must RETURN, not merely record an error and fall through.

    Every check after it reaches into the model as a torch object, and
    `SAEHooker.get_layer_count` ends in `model.named_modules()` — an
    AttributeError on a llama.cpp handle, which `check_compatibility` only
    catches as ValueError. Falling through raised a 500 out of the very
    function that exists to answer "no, and here is why".

    The REAL hooker, deliberately: mocking `_hooker` (as the test above does,
    for its own purposes) makes this failure unreachable, which is why it went
    unnoticed.

    MUTATION CONTROL: replace the `return CompatibilityResult(...)` in the
    engine guard with `pass` -> this test raises AttributeError.
    """

    @pytest.mark.asyncio
    async def test_a_real_hooker_against_a_non_module_handle(self):
        from millm.ml.sae_hooker import SAEHooker
        from millm.services.sae_service import SAEService

        class FakeLlama:
            """No `config`, no `named_modules` — a ctypes handle's shape."""

            model_path = "/models/x-Q4_K_M.gguf"

        service = SAEService.__new__(SAEService)
        service.get_sae = AsyncMock(
            return_value=MagicMock(status=MagicMock(value="cached"))
        )
        service._hooker = SAEHooker()

        state = MagicMock()
        state.is_loaded = True
        state.current = _loaded(ENGINE_LLAMACPP)
        state.current.model = FakeLlama()

        with patch("millm.services.sae_service.LoadedModelState", return_value=state):
            result = await service.check_compatibility("sae-1", 12)

        assert result.compatible is False
        assert any("forward hook" in e for e in result.errors)
        assert not any("architecture" in e.lower() for e in result.errors), (
            "the runtime limitation must not be reported as an architecture "
            "problem — that is the exact misdiagnosis this guard replaces"
        )


class TestTextCompletionIsRefused:
    """/v1/completions is a public OpenAI route and reaches the transformers
    path directly. `self._tokenizer` is None on this engine, so the first line
    of its loop is `None(prompt, return_tensors="pt")` — a 500 deep inside
    generation instead of a 400 at the boundary.

    MUTATION CONTROL: drop the `_engine_is_llamacpp` guard at the top of
    `create_text_completion` -> this test fails with TypeError, not MiLLMError.
    """

    @pytest.mark.asyncio
    async def test_create_text_completion_refuses(self):
        from millm.core.errors import MiLLMError
        from millm.services.inference_service import InferenceService

        svc = InferenceService.__new__(InferenceService)
        svc._cbm_backend = None
        state = MagicMock()
        state.is_loaded = True
        state.current = _loaded(ENGINE_LLAMACPP)
        svc._model_state = state

        with pytest.raises(MiLLMError) as exc:
            await svc.create_text_completion(MagicMock())

        assert exc.value.code == "ENGINE_UNSUPPORTED"


class TestTheBackendInfoIsHonest:
    """/api/inference/status must not advertise streaming for an engine whose
    streaming path raises ENGINE_UNSUPPORTED.

    MUTATION CONTROL: remove the llamacpp branch from get_backend_info ->
    this test sees backend "serial" and streaming True.
    """

    def test_streaming_is_not_advertised(self):
        from millm.services.inference_service import InferenceService

        svc = InferenceService.__new__(InferenceService)
        svc._cbm_backend = None
        state = MagicMock()
        state.is_loaded = True
        state.current = _loaded(ENGINE_LLAMACPP)
        svc._model_state = state

        info = svc.get_backend_info()

        assert info["backend"] == "llamacpp"
        assert info["capabilities"]["streaming"] is False


class TestSamplingPenaltiesReachTheEngine:
    """The transformers path honours frequency_penalty/presence_penalty, so
    dropping them here would make the same request sample differently
    depending on which engine happens to hold the model.

    MUTATION CONTROL: delete the two penalty keys from `params` -> fails.
    """

    @pytest.mark.asyncio
    async def test_penalties_and_stops_are_forwarded(self):
        from millm.api.schemas.openai import ChatCompletionRequest
        from millm.services.inference_service import InferenceService

        svc = InferenceService.__new__(InferenceService)
        state = MagicMock()
        state.is_loaded = True
        current = _loaded(ENGINE_LLAMACPP)
        current.model.create_chat_completion.return_value = {
            "choices": [
                {"message": {"content": "hi"}, "finish_reason": "stop"}
            ],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 1,
                "total_tokens": 4,
            },
        }
        state.current = current
        svc._model_state = state

        class _NullQueue:
            def acquire(self):
                import contextlib

                @contextlib.asynccontextmanager
                async def _cm():
                    yield

                return _cm()

        svc._request_queue = _NullQueue()

        request = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hello"}],
            frequency_penalty=0.5,
            presence_penalty=-0.25,
            stop="END",
        )

        result = await svc._llamacpp_chat_completion(request)

        kwargs = current.model.create_chat_completion.call_args.kwargs
        assert kwargs["frequency_penalty"] == 0.5
        assert kwargs["presence_penalty"] == -0.25
        assert kwargs["stop"] == ["END"]
        assert result.choices[0].message.content == "hi"


class TestTheRefusalReachesTheClientAsARefusal:
    """`code=`/`status_code=` are CLASS attributes on MiLLMError, not
    constructor kwargs, so `MiLLMError(msg, code=..., status_code=...)` is a
    TypeError — every refusal raised that way surfaced as an unhandled 500
    instead of the 400 it meant to be. And an unmapped code falls back to
    a wrong error TYPE in the registered handler (millm_error_handler falls
    back to `(exc.status_code, "server_error")`), which an OpenAI client reads
    as a server fault to retry rather than a request to change.

    MUTATION CONTROLS:
      * raise MiLLMError(..., code=...) again        -> TypeError, test fails
      * drop ENGINE_UNSUPPORTED from ERROR_STATUS_MAP -> 400/server_error
    """

    def test_the_error_carries_its_code_and_status(self):
        from millm.core.errors import EngineUnsupportedError

        exc = EngineUnsupportedError("nope")
        assert exc.code == "ENGINE_UNSUPPORTED"
        assert exc.status_code == 400

    def test_the_openai_layer_maps_it_to_a_client_error(self):
        from millm.api.routes.openai.errors import ERROR_STATUS_MAP

        assert ERROR_STATUS_MAP["ENGINE_UNSUPPORTED"] == (
            400,
            "invalid_request_error",
        ), "server_error tells the caller to retry something that can never succeed"

    @pytest.mark.asyncio
    async def test_every_llamacpp_refusal_is_constructible(self):
        """Each raise site, actually raised — a refusal that throws TypeError
        on its way out is not a refusal."""
        from millm.core.errors import EngineUnsupportedError
        from millm.services.inference_service import InferenceService

        svc = InferenceService.__new__(InferenceService)
        state = MagicMock()
        state.is_loaded = True
        state.current = _loaded(ENGINE_LLAMACPP)
        svc._model_state = state

        from millm.api.schemas.openai import ChatCompletionRequest

        base = {"model": "m", "messages": [{"role": "user", "content": "x"}]}
        cases = [
            ChatCompletionRequest(**base, extra_messages=[[{"role": "user", "content": "y"}]]),
            ChatCompletionRequest(**base, profile="p"),
            ChatCompletionRequest(**base, chat_template_kwargs={"enable_thinking": False}),
        ]
        for request in cases:
            with pytest.raises(EngineUnsupportedError):
                await svc._llamacpp_chat_completion(request)

        with pytest.raises(EngineUnsupportedError):
            await svc._refuse_on_llamacpp("Streaming")


class TestNIsRefusedRatherThanSilentlyTruncated:
    """`n > 1` is honoured on the transformers path and impossible here.

    `_llamacpp_chat_completion` builds exactly ONE choice, so an unguarded
    n=3 returned a single-choice 200 with no error — the quiet degradation
    every other branch in this function refuses. A client demultiplexing on
    `index` silently loses two thirds of what it asked for.

    MUTATION CONTROL: delete the `n > 1` guard -> this test gets a
    ChatCompletionResponse with one choice instead of EngineUnsupportedError.
    """

    @pytest.mark.asyncio
    async def test_n_greater_than_one_refuses(self):
        from millm.api.schemas.openai import ChatCompletionRequest
        from millm.core.errors import EngineUnsupportedError
        from millm.services.inference_service import InferenceService

        svc = InferenceService.__new__(InferenceService)
        state = MagicMock()
        state.is_loaded = True
        state.current = _loaded(ENGINE_LLAMACPP)
        svc._model_state = state

        request = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "x"}],
            n=3,
        )

        with pytest.raises(EngineUnsupportedError) as exc:
            await svc._llamacpp_chat_completion(request)

        assert "n > 1" in str(exc.value)

    @pytest.mark.asyncio
    async def test_n_equals_one_still_generates(self):
        """The guard must bite on n>1 ONLY — the default must still serve."""
        from millm.api.schemas.openai import ChatCompletionRequest
        from millm.services.inference_service import InferenceService

        svc = InferenceService.__new__(InferenceService)
        state = MagicMock()
        state.is_loaded = True
        current = _loaded(ENGINE_LLAMACPP)
        current.model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        state.current = current
        svc._model_state = state

        class _NullQueue:
            def acquire(self):
                import contextlib

                @contextlib.asynccontextmanager
                async def _cm():
                    yield

                return _cm()

        svc._request_queue = _NullQueue()

        result = await svc._llamacpp_chat_completion(
            ChatCompletionRequest(
                model="m", messages=[{"role": "user", "content": "x"}]
            )
        )
        assert result.choices[0].message.content == "hi"

