"""POST /api/saes/preview through the REAL downloader (only the Hub client is mocked).

2026-10-18 hardware finding: previewing miStudio's private SAE repositories answered 500
INTERNAL_ERROR — the preview went to the Hub without the server's HF_TOKEN (download used it), and the
not-found surfaced as a bare ValueError."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from huggingface_hub.utils import RepositoryNotFoundError

from millm.api.dependencies import get_sae_service
from millm.api.exception_handlers import millm_error_handler
from millm.api.routes.management.saes import router
from millm.core.errors import MiLLMError
from millm.ml.sae_downloader import SAEDownloader


@pytest.fixture
def hub():
    return MagicMock()


@pytest.fixture
def client(hub):
    with patch.object(SAEDownloader, "__init__", lambda self, cache_dir: None):
        dl = SAEDownloader.__new__(SAEDownloader)
    dl._api = hub
    svc = MagicMock()
    svc.preview_repository = lambda repository_id, revision="main", token=None: dl.list_repository_files(
        repository_id, revision, token
    )
    app = FastAPI()
    app.include_router(router)
    app.add_exception_handler(MiLLMError, millm_error_handler)
    app.dependency_overrides[get_sae_service] = lambda: svc
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
async def test_private_repo_previews_with_the_server_token(client, hub, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_server")
    sib = MagicMock(rfilename="layer_12/params.safetensors", size=10)
    hub.repo_info.return_value = MagicMock(siblings=[sib])
    hub.list_repo_files.return_value = ["layer_12/params.safetensors"]
    async with client:
        r = await client.post("/api/saes/preview", json={"repository_id": "mistudio/private-sae"})
    assert r.status_code == 200, r.text
    assert hub.repo_info.call_count == 1
    assert hub.repo_info.call_args.kwargs["token"] == "hf_server"
    assert r.json()["data"]["repository_id"] == "mistudio/private-sae"


@pytest.mark.asyncio
async def test_unreadable_repo_is_a_404_not_a_500(client, hub, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    hub.repo_info.side_effect = RepositoryNotFoundError("401", response=MagicMock())
    async with client:
        r = await client.post("/api/saes/preview", json={"repository_id": "mistudio/private-sae"})
    assert r.status_code == 404, r.text
    assert r.json()["error"]["code"] == "REPO_NOT_FOUND"
