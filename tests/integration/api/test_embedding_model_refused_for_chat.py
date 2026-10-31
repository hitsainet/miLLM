"""An embedding model must not be used for text generation.

Nemotron-3-Embed-8B-BF16 (architecture: sentence-similarity) was selected in a
chat client and answered with several hundred tokens of multilingual fragments:

    pronon� questionnaire)e contrari projekt отрима arrêté;"><shots 올 glo Lago

That is worse than an error, because it looks like output. The registry already
recorded the architecture; nothing checked it.

Reachable only since the OpenAI endpoints began loading whatever model a request
names — previously a fixed model was pinned and selecting another did nothing.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from millm.api.routes.openai.errors import is_embedding_only
from millm.main import create_app


def _model(name, architecture):
    m = MagicMock()
    m.id, m.name, m.architecture = 1, name, architecture
    return m


def _client(model):
    from millm.api.dependencies import get_inference_service, get_model_service

    svc = MagicMock()
    svc.find_model_by_name = AsyncMock(return_value=model)
    svc.get_locked_model = AsyncMock(return_value=None)
    svc.load_model_and_wait = AsyncMock()
    inference = MagicMock()
    inference.backend_name = "serial"
    inference.request_queue = MagicMock(pending_count=0, max_pending=5)
    inference.resolve_request_intensity = AsyncMock(return_value=None)
    inference.get_loaded_model_info = lambda: None

    app = create_app()
    app.dependency_overrides[get_model_service] = lambda: svc
    app.dependency_overrides[get_inference_service] = lambda: inference
    return TestClient(app), svc


BODY = {"model": "Nemotron-3-Embed-8B-BF16",
        "messages": [{"role": "user", "content": "hi"}]}


class TestTheClassifier:
    @pytest.mark.parametrize("arch", [
        "sentence-similarity", "feature-extraction", "sentence-transformers",
        "text-embedding", "Sentence-Similarity", "  sentence-similarity  ",
    ])
    def test_embedding_architectures_are_recognised(self, arch):
        assert is_embedding_only(arch)

    @pytest.mark.parametrize("arch", [
        "text-generation",          # LFM2.5, granite, gemma-2
        "any-to-any",               # gemma-4 — multimodal but DOES generate
        "image-text-to-text",       # gemma-3-4b — generates
        None, "",
    ])
    def test_generative_architectures_are_not_blocked(self, arch):
        assert not is_embedding_only(arch), (
            f"{arch!r} generates text and must not be refused"
        )


class TestChatRefusesEmbeddingModels:
    def test_it_returns_400_not_garbage(self):
        tc, _ = _client(_model("Nemotron-3-Embed-8B-BF16", "sentence-similarity"))
        with tc:
            r = tc.post("/v1/chat/completions", json=BODY)
        assert r.status_code == 400, r.status_code
        body = r.json()["error"]
        assert body["code"] == "model_not_generative"
        assert "embedding" in body["message"].lower()
        assert "/v1/embeddings" in body["message"], (
            "the refusal must say what the model IS for"
        )

    def test_it_refuses_BEFORE_loading_the_model(self):
        """Loading costs time and VRAM to reach a guaranteed-nonsense answer."""
        tc, svc = _client(_model("Nemotron-3-Embed-8B-BF16", "sentence-similarity"))
        with tc:
            tc.post("/v1/chat/completions", json=BODY)
        svc.load_model_and_wait.assert_not_awaited()

    def test_completions_refuses_too(self):
        tc, _ = _client(_model("Nemotron-3-Embed-8B-BF16", "sentence-similarity"))
        with tc:
            r = tc.post("/v1/completions",
                        json={"model": "Nemotron-3-Embed-8B-BF16", "prompt": "hi"})
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "model_not_generative"


class TestGenerativeModelsAreUnaffected:
    def test_a_multimodal_generative_model_still_proceeds(self):
        """gemma-4 is 'any-to-any' — multimodal, but it generates text."""
        tc, svc = _client(_model("gemma-4-12B-it", "any-to-any"))
        with tc:
            r = tc.post("/v1/chat/completions",
                        json={**BODY, "model": "gemma-4-12B-it"})
        assert r.status_code != 400 or \
            r.json().get("error", {}).get("code") != "model_not_generative"
        svc.load_model_and_wait.assert_awaited()
