"""Every MiLLMError code has an ERROR_STATUS_MAP row (Feature 25, 025_FTDD §5.5).

A code without a row reaches a /v1 client through the live handler's fallback,
`(exc.status_code, "server_error")` (`millm/api/exception_handlers.py`): the right status
wearing the wrong TYPE. That is how a scoring 400 — a token id outside the vocabulary —
reached OpenAI clients typed `server_error`, which their SDKs retry.

The set of codes is READ FROM THE CLASS HIERARCHY, never a hand-kept list: every module under
`millm` is imported (some error classes live beside their service, e.g.
`cluster_hub_service.HubUnavailableError`) and `MiLLMError.__subclasses__()` is walked
recursively. A new error class without a row fails here.

MUTATION CONTROL (M15): delete the `INVALID_SCORING_REQUEST` row -> red.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import millm
from millm.api.routes.openai.errors import ERROR_STATUS_MAP
from millm.core.errors import MiLLMError


def _import_everything() -> None:
    for module in pkgutil.walk_packages(millm.__path__, "millm."):
        if ".db.migrations" in module.name:
            continue
        try:
            importlib.import_module(module.name)
        except Exception:  # noqa: BLE001 - an optional engine (llama_cpp) may be absent
            continue


def _all_subclasses(cls: type) -> list[type]:
    found: list[type] = []
    for sub in cls.__subclasses__():
        found.append(sub)
        found.extend(_all_subclasses(sub))
    return found


_import_everything()
ERROR_CLASSES = sorted(
    {c for c in _all_subclasses(MiLLMError) if c.__module__.startswith("millm.")},
    key=lambda c: (c.__module__, c.__name__),
)


def test_the_walk_found_the_hierarchy():
    """Guard the guard: an import failure that found nothing would pass every row check."""
    codes = {c.code for c in ERROR_CLASSES}
    assert len(ERROR_CLASSES) > 40
    assert {"INVALID_SCORING_REQUEST", "FIELD_NOT_HONOURED", "HUB_UNAVAILABLE"} <= codes


@pytest.mark.parametrize("cls", ERROR_CLASSES, ids=lambda c: f"{c.__name__}:{c.code}")
def test_every_error_code_has_a_row(cls):
    assert cls.code in ERROR_STATUS_MAP, (
        f"{cls.__module__}.{cls.__name__} raises code {cls.code!r}, which has no "
        "ERROR_STATUS_MAP row; a /v1 client would receive it typed server_error"
    )


@pytest.mark.parametrize("cls", ERROR_CLASSES, ids=lambda c: f"{c.__name__}:{c.code}")
def test_a_client_error_is_never_typed_as_a_server_fault(cls):
    status, error_type = ERROR_STATUS_MAP[cls.code]
    if 400 <= status < 500:
        assert error_type != "server_error", cls.code


@pytest.mark.parametrize("code, status, error_type", [
    ("FIELD_NOT_HONOURED", 400, "invalid_request_error"),
    ("UNUSED_FIELDS_REFUSED", 400, "invalid_request_error"),
    ("RESPONSE_FORMAT_UNSUPPORTED", 400, "invalid_request_error"),
    ("NO_CHAT_TEMPLATE", 400, "invalid_request_error"),
    ("CONSTRAINED_OUTPUT_INVALID", 500, "server_error"),
    ("INVALID_SCORING_REQUEST", 400, "invalid_request_error"),
    ("NON_FINITE_LOGITS", 500, "server_error"),
])
def test_the_feature_25_rows(code, status, error_type):
    assert ERROR_STATUS_MAP[code] == (status, error_type)


def test_the_handler_carries_param_from_details():
    """Every Feature 25 refusal names its field; the envelope's `param` must carry it."""
    import asyncio
    import json
    from unittest.mock import MagicMock

    from millm.api.exception_handlers import millm_error_handler
    from millm.core.errors import FieldNotHonouredError

    request = MagicMock()
    request.url.path = "/v1/chat/completions"
    request.method = "POST"
    exc = FieldNotHonouredError("no", details={"param": "seed"})
    body = json.loads(asyncio.run(millm_error_handler(request, exc)).body)
    assert body["error"]["param"] == "seed"
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["code"] == "field_not_honoured"


def test_an_out_of_vocabulary_scoring_id_is_typed_invalid_request_over_http():
    """3.4: the latent defect itself, through the real route and the real handler."""
    from datetime import datetime
    from unittest.mock import AsyncMock, MagicMock, patch

    import torch
    from fastapi.testclient import TestClient
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    from millm.api.dependencies import get_inference_service, get_model_service
    from millm.main import create_app
    from millm.ml.model_loader import LoadedModel, LoadedModelState
    from millm.services.inference_service import InferenceService

    words = ["<s>", "[UNK]"] + [f"w{i}" for i in range(10)]
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>", unk_token="[UNK]")
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=len(words), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64)).eval()
    state = LoadedModelState()
    state.set(LoadedModel(model_id=1, model_name="tiny", model=model, tokenizer=fast,
                          loaded_at=datetime(2026, 10, 6), device="cpu", dtype="float32"))
    try:
        with patch("millm.services.inference_service.torch.cuda.is_available", return_value=False):
            inference = InferenceService(model_service=None)
        inference._device = "cpu"
        row = MagicMock()
        row.id, row.name, row.architecture, row.gguf_files = 1, "tiny", "text-generation", None
        svc = MagicMock()
        svc.find_model_by_name = AsyncMock(return_value=row)
        app = create_app()
        app.dependency_overrides[get_model_service] = lambda: svc
        app.dependency_overrides[get_inference_service] = lambda: inference
        response = TestClient(app).post("/v1/completions", json={
            "model": "tiny", "prompt": "w1 w2", "max_tokens": 1, "allowed_token_ids": [999]})
    finally:
        state._loaded = None
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error", error
    assert error["code"] == "invalid_scoring_request"
