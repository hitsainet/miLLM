"""Feature 34 AC-2 / AC-3: EVERY Hub call in miLLM carries the credential from the one resolver.

Two halves, because each is blind where the other sees:

* BEHAVIOUR (AC-2): each call site is driven through its public entry point with only the Hub
  client mocked, a token SAVED in Settings and a DIFFERENT one in `HF_TOKEN`. The Hub call must
  receive the saved one (and the request's own token where the entry point takes one, and `False`
  when nothing is configured). Removing `hub_token(...)` from any site turns its test red.
* DISCOVERY (AC-3): an AST walk over `millm/` finds every call to a Hub entry point. The set found
  must EQUAL `INVENTORY` below: a new Hub call anywhere fails here until it is classified, and every
  HUB-class call must pass `token=`. Each HUB entry names the behavioural test that covers it.

All tokens are fake.
"""

from __future__ import annotations

import ast
import asyncio
import pathlib
from collections import Counter
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from millm.services import hf_token

SAVED = "hf_FAKEsaved00000000000000000000000sav2"
ENV = "hf_FAKEenv0000000000000000000000000env3"
REQ = "hf_FAKErequest000000000000000000000req1"


@pytest.fixture
def saved(monkeypatch):
    """A token saved in Settings, and a different one in the environment: the saved one must win."""
    monkeypatch.setenv("HF_TOKEN", ENV)
    hf_token._set_memory(token=SAVED)
    return SAVED


@pytest.fixture
def anonymous(monkeypatch):
    from millm.core.config import settings

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(settings, "HF_TOKEN", None)
    hf_token._reset_memory()


# ── Model downloader (Feature 1) ─────────────────────────────────────────────


@pytest.fixture
def model_dl(tmp_path):
    from millm.core.resilience import huggingface_circuit
    from millm.ml.model_downloader import ModelDownloader

    huggingface_circuit.reset()
    dl = ModelDownloader(cache_dir=str(tmp_path))
    dl.hf_api = MagicMock()
    yield dl
    huggingface_circuit.reset()


class TestModelDownloader:
    @pytest.mark.parametrize("explicit, expected", [(None, SAVED), (REQ, REQ), ("none", SAVED)])
    def test_download(self, saved, model_dl, explicit, expected):
        with patch("millm.ml.model_downloader.snapshot_download", return_value="/x") as snap:
            model_dl.download("org/model", "FP16", token=explicit)
        assert snap.call_count == 1
        assert snap.call_args.kwargs["token"] == expected

    def test_download_anonymous_sends_no_credential(self, anonymous, model_dl):
        with patch("millm.ml.model_downloader.snapshot_download", return_value="/x") as snap:
            model_dl.download("org/model", "FP16")
        assert snap.call_args.kwargs["token"] is False

    @pytest.mark.parametrize("explicit, expected", [(None, SAVED), (REQ, REQ)])
    def test_model_info(self, saved, model_dl, explicit, expected):
        model_dl.hf_api.model_info.return_value = MagicMock(siblings=[], safetensors=None, config={})
        try:
            model_dl.get_model_info("org/model", token=explicit)
        except Exception:  # noqa: BLE001 - only the call's credential is under test here
            pass
        assert model_dl.hf_api.model_info.call_count == 1
        assert model_dl.hf_api.model_info.call_args.kwargs["token"] == expected

    @pytest.mark.parametrize("explicit, expected", [(None, SAVED), (REQ, REQ)])
    def test_expected_download_size(self, saved, model_dl, explicit, expected):
        model_dl.hf_api.model_info.return_value = MagicMock(siblings=[])
        model_dl.get_expected_download_size("org/model", token=explicit)
        assert model_dl.hf_api.model_info.call_count == 1
        assert model_dl.hf_api.model_info.call_args.kwargs["token"] == expected


# ── SAE downloader (Feature 3) ───────────────────────────────────────────────


@pytest.fixture
def sae_dl(tmp_path):
    from millm.ml.sae_downloader import SAEDownloader

    with patch.object(SAEDownloader, "__init__", lambda self, cache_dir: None):
        dl = SAEDownloader.__new__(SAEDownloader)
    dl.cache_dir = tmp_path
    dl._api = MagicMock()
    return dl


class TestSAEDownloader:
    @pytest.mark.parametrize("explicit, expected", [(None, SAVED), (REQ, REQ)])
    async def test_download_and_its_repository_check(self, saved, sae_dl, explicit, expected):
        with patch("millm.ml.sae_downloader.snapshot_download", return_value="/x") as snap:
            await sae_dl.download("org/sae", token=explicit)
        assert sae_dl._api.repo_info.call_count == 1
        assert sae_dl._api.repo_info.call_args.kwargs["token"] == expected
        assert snap.call_count == 1 and snap.call_args.kwargs["token"] == expected

    @pytest.mark.parametrize("explicit, expected", [(None, SAVED), (REQ, REQ)])
    async def test_preview(self, saved, sae_dl, explicit, expected):
        sae_dl._api.repo_info.return_value = MagicMock(siblings=[])
        sae_dl._api.list_repo_files.return_value = []
        await sae_dl.list_repository_files("org/sae", token=explicit)
        assert sae_dl._api.repo_info.call_args.kwargs["token"] == expected
        assert sae_dl._api.list_repo_files.call_args.kwargs["token"] == expected

    async def test_preview_anonymous_sends_no_credential(self, anonymous, sae_dl):
        sae_dl._api.repo_info.return_value = MagicMock(siblings=[])
        sae_dl._api.list_repo_files.return_value = []
        await sae_dl.list_repository_files("org/sae")
        assert sae_dl._api.repo_info.call_args.kwargs["token"] is False


# ── Cluster and probe packs (Features 8, 24) ─────────────────────────────────


@pytest.fixture
def hub_api():
    from millm.services.cluster_hub_service import cluster_hub_circuit

    cluster_hub_circuit.reset()
    api = MagicMock()
    with patch("millm.services.cluster_hub_service.HfApi", return_value=api):
        yield api
    cluster_hub_circuit.reset()


@pytest.fixture(params=["cluster", "probe"])
def pack_service(request, tmp_path, monkeypatch):
    from millm.core.config import settings
    from millm.services.cluster_hub_service import ClusterHubService
    from millm.services.probe_hub_service import ProbeHubService

    monkeypatch.setattr(settings, "SAE_CACHE_DIR", str(tmp_path))
    return (ClusterHubService if request.param == "cluster" else ProbeHubService)(cache_ttl_s=60)


class TestPackHub:
    async def test_search(self, saved, hub_api, pack_service):
        hub_api.list_models.return_value = []
        await pack_service.search("q")
        assert hub_api.list_models.call_count == 1
        assert hub_api.list_models.call_args.kwargs["token"] == SAVED

    async def test_list_definitions(self, saved, hub_api, pack_service):
        hub_api.list_repo_files.return_value = []
        await pack_service.list_definitions("org/pack")
        assert hub_api.list_repo_files.call_count == 1
        assert hub_api.list_repo_files.call_args.kwargs["token"] == SAVED

    async def test_fetch_definition(self, saved, hub_api, pack_service, tmp_path):
        path = tmp_path / "def.json"
        path.write_text("{}")
        suffix = pack_service.DEFINITION_SUFFIX
        with patch("millm.services.cluster_hub_service.hf_hub_download", return_value=str(path)) as dl:
            with pytest.raises(Exception):  # an empty object fails the contract; the call is what matters
                await pack_service.fetch_definition("org/pack", f"a{suffix}")
        assert dl.call_count == 1 and dl.call_args.kwargs["token"] == SAVED

    async def test_anonymous_listing_sends_no_credential(self, anonymous, hub_api, pack_service):
        hub_api.list_models.return_value = []
        await pack_service.search("q")
        assert hub_api.list_models.call_args.kwargs["token"] is False

    async def test_a_listing_is_not_served_after_the_token_changes(self, saved, hub_api, pack_service):
        hub_api.list_models.return_value = []
        await pack_service.search("q")
        await pack_service.search("q")
        assert hub_api.list_models.call_count == 1, "the TTL cache must still serve a repeat"
        hf_token._reset_memory()
        await pack_service.search("q")
        assert hub_api.list_models.call_count == 2
        assert hub_api.list_models.call_args.kwargs["token"] == ENV


# ── Speculative-decoding draft model ─────────────────────────────────────────


class TestDraftModel:
    def test_draft_config(self, saved):
        from millm.services.inference_service import _draft_torch_dtype

        with patch("transformers.AutoConfig") as cfg:
            cfg.from_pretrained.return_value = SimpleNamespace(torch_dtype="bfloat16")
            _draft_torch_dtype("org/draft")
        assert cfg.from_pretrained.call_count == 1
        assert cfg.from_pretrained.call_args.kwargs["token"] == SAVED

    def test_draft_load(self, saved):
        from millm.services.inference_service import InferenceService

        svc = InferenceService.__new__(InferenceService)
        svc._device = "cpu"
        svc._speculative_model_id = "org/draft"
        svc._configured_speculative_model_id = "org/draft"
        svc._draft_model = None
        svc._cbm_backend = None
        svc._model_state = SimpleNamespace(
            is_loaded=True,
            current=SimpleNamespace(model=SimpleNamespace(hf_device_map={"x": 0}), gpu_indices=[0],
                                    tokenizer=MagicMock()),
        )
        with patch("transformers.AutoModelForCausalLM") as factory, \
             patch("millm.services.inference_service._draft_torch_dtype", return_value=None):
            svc._get_draft_model()
        assert factory.from_pretrained.call_count == 1
        assert factory.from_pretrained.call_args.kwargs["token"] == SAVED


# ── AC-3: discovery == inventory ─────────────────────────────────────────────

#: Names that reach the Hub (or construct a Hub client) when called.
HUB_NAMES = frozenset({
    "snapshot_download", "hf_hub_download", "model_info", "list_repo_files", "repo_info",
    "list_models", "whoami", "from_pretrained", "HfApi", "list_repo_tree", "get_paths_info",
    "file_exists", "repo_exists", "hf_hub_url", "HfFileSystem", "get_hf_file_metadata",
    "list_repo_refs", "dataset_info", "space_info",
})

HUB = "hub"          # reaches the Hub: must pass token=, and is covered by a behavioural test above
LOCAL = "local"      # from_pretrained on a local directory (the model cache): no Hub traffic
CLIENT = "client"    # HfApi() construction: the credential is passed per call, never here
NOT_HUB = "not_hub"  # a same-named method on a miLLM object (the model service's DB listing)

#: (file, enclosing function, callee) -> (class, count). Every entry was classified by reading it.
INVENTORY: dict[tuple[str, str, str], tuple[str, int]] = {
    ("millm/api/routes/management/models.py", "list_models", "list_models"): (NOT_HUB, 1),
    ("millm/main.py", "_auto_load_model", "list_models"): (NOT_HUB, 1),
    ("millm/ml/model_downloader.py", "__init__", "HfApi"): (CLIENT, 1),
    # token resolved by `download` -> TestModelDownloader.test_download
    ("millm/ml/model_downloader.py", "_snapshot_download_with_circuit", "snapshot_download"): (HUB, 1),
    # token resolved by `get_model_info` -> TestModelDownloader.test_model_info
    ("millm/ml/model_downloader.py", "_get_model_info_with_circuit", "model_info"): (HUB, 1),
    ("millm/ml/model_downloader.py", "get_expected_download_size", "model_info"): (HUB, 1),
    ("millm/ml/model_loader.py", "checkpoint_materialised_mb", "from_pretrained"): (LOCAL, 1),
    ("millm/ml/model_loader.py", "preflight_split", "from_pretrained"): (LOCAL, 1),
    ("millm/ml/model_loader.py", "load", "from_pretrained"): (LOCAL, 3),
    # The model weights: the request, then once more without an attention choice (2026-10-18).
    ("millm/ml/model_loader.py", "_from_pretrained_with_attn_fallback", "from_pretrained"): (LOCAL, 2),
    ("millm/ml/model_loader.py", "transformers_fit", "from_pretrained"): (LOCAL, 1),
    ("millm/ml/sae_downloader.py", "__init__", "HfApi"): (CLIENT, 1),
    # token resolved by `download` -> TestSAEDownloader.test_download_and_its_repository_check
    ("millm/ml/sae_downloader.py", "_download_sync", "snapshot_download"): (HUB, 1),
    ("millm/ml/sae_downloader.py", "_validate_repository", "repo_info"): (HUB, 1),
    # token resolved by `list_repository_files` -> TestSAEDownloader.test_preview
    ("millm/ml/sae_downloader.py", "_list_repository_files_sync", "repo_info"): (HUB, 1),
    ("millm/ml/sae_downloader.py", "_list_repository_files_sync", "list_repo_files"): (HUB, 1),
    ("millm/services/cluster_hub_service.py", "_list_models_sync", "HfApi"): (CLIENT, 1),
    ("millm/services/cluster_hub_service.py", "_list_models_sync", "list_models"): (HUB, 1),
    ("millm/services/cluster_hub_service.py", "_list_repo_files_sync", "HfApi"): (CLIENT, 1),
    ("millm/services/cluster_hub_service.py", "_list_repo_files_sync", "list_repo_files"): (HUB, 1),
    ("millm/services/cluster_hub_service.py", "_download_file_sync", "hf_hub_download"): (HUB, 1),
    ("millm/services/hf_token.py", "_whoami", "HfApi"): (CLIENT, 1),
    # -> test_hf_token.py::TestCheck::test_whoami_passes_the_token_to_the_hub
    ("millm/services/hf_token.py", "_whoami", "whoami"): (HUB, 1),
    ("millm/services/inference_service.py", "_draft_torch_dtype", "from_pretrained"): (HUB, 1),
    ("millm/services/inference_service.py", "_get_draft_model", "from_pretrained"): (HUB, 1),
}

_REPO = pathlib.Path(__file__).resolve().parents[3]


def _discover() -> list[tuple[tuple[str, str, str], ast.Call]]:
    found: list[tuple[tuple[str, str, str], ast.Call]] = []
    for path in sorted((_REPO / "millm").rglob("*.py")):
        rel = path.relative_to(_REPO).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        stack: list[str] = []

        class Visitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):  # noqa: N802
                stack.append(node.name)
                self.generic_visit(node)
                stack.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Call(self, node):  # noqa: N802
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
                if name in HUB_NAMES:
                    found.append(((rel, stack[-1] if stack else "<module>", name), node))
                self.generic_visit(node)

        Visitor().visit(tree)
    return found


def test_discovery_finds_calls_at_all():
    assert len(_discover()) >= 20, "the scanner found almost nothing, so it is not reading millm/"


def test_every_hub_call_is_classified():
    found = Counter(key for key, _ in _discover())
    declared = Counter({key: count for key, (_, count) in INVENTORY.items()})
    assert found == declared, (
        "Hub calls changed. A new call must carry `token=hub_token(...)` (millm/services/hf_token.py), "
        "get a behavioural test in this file, and an INVENTORY entry. "
        f"unexpected: {dict(found - declared)}; missing: {dict(declared - found)}"
    )


def test_every_hub_call_passes_a_token():
    for key, node in _discover():
        kind = INVENTORY.get(key, (None,))[0]
        keywords = {k.arg for k in node.keywords}
        if kind == HUB:
            assert "token" in keywords, f"{key} reaches the Hub without token="
        elif kind == LOCAL:
            assert node.args and ast.unparse(node.args[0]) == "cache_path", (
                f"{key} is classified LOCAL but does not load from cache_path"
            )
        elif kind == CLIENT:
            assert "token" not in keywords, f"{key}: pass the credential per call, not to the client"


def test_huggingface_hub_is_imported_only_where_inventoried():
    importers = set()
    for path in (_REPO / "millm").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "from huggingface_hub" in text or "import huggingface_hub" in text:
            importers.add(path.relative_to(_REPO).as_posix())
    assert importers == {
        "millm/ml/model_downloader.py",
        "millm/ml/sae_downloader.py",
        "millm/services/cluster_hub_service.py",
        "millm/services/hf_token.py",
    }


#: Names imported from huggingface_hub that never reach the network: exception types and the
#: no-op progress base. Anything else imported from the library must be in HUB_NAMES.
_HUB_IMPORTS_THAT_ARE_NOT_CALLS = frozenset({
    "EntryNotFoundError", "GatedRepoError", "HfHubHTTPError", "RepositoryNotFoundError",
    "RevisionNotFoundError",
})


def _names_imported_from_huggingface_hub() -> set[str]:
    names: set[str] = set()
    for path in (_REPO / "millm").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("huggingface_hub"):
                names.update(alias.name for alias in node.names)
    return names


def test_every_name_imported_from_the_hub_is_scanned_or_inert():
    """HUB_NAMES is hand-kept: a Hub entry point imported under a name it lacks would make the
    discovery scan above blind to every call of it. Each imported name is either scanned or a
    declared non-call (exceptions). Discovered for the huggingface_hub 2.x bump (2026-10-18)."""
    imported = _names_imported_from_huggingface_hub()
    assert imported, "found no `from huggingface_hub import` in millm/: the walk is not reading it"
    unscanned = imported - HUB_NAMES - _HUB_IMPORTS_THAT_ARE_NOT_CALLS
    assert not unscanned, f"imported from huggingface_hub but not in HUB_NAMES: {sorted(unscanned)}"


def test_hub_names_still_exist_in_the_installed_library():
    """A Hub major (1.x -> 2.x on 2026-10-18) can rename or remove an entry point. A name in
    HUB_NAMES that the INSTALLED library no longer has scans for a call nobody can make, while the
    replacement would go unscanned. `from_pretrained` is transformers', not the Hub's."""
    import huggingface_hub

    missing = sorted(
        n for n in HUB_NAMES - {"from_pretrained"}
        if not hasattr(huggingface_hub, n) and not hasattr(huggingface_hub.HfApi, n)
    )
    assert not missing, f"not in huggingface_hub {huggingface_hub.__version__}: {missing}"


def test_the_resolver_is_callable_from_worker_threads():
    """Most Hub calls run under `asyncio.to_thread`; `hub_token()` is sync and lock-protected."""
    async def run():
        hf_token._set_memory(token=SAVED)
        return await asyncio.gather(*(asyncio.to_thread(hf_token.hub_token) for _ in range(8)))

    assert asyncio.run(run()) == [SAVED] * 8


@pytest.mark.parametrize("first", ["millm.ml", "millm.ml.model_downloader", "millm.ml.sae_downloader",
                                   "millm.services", "millm.services.cluster_hub_service"])
def test_each_importer_imports_first_without_a_cycle(first):
    """The resolver first lived in `millm/services`; `millm.ml` importing it pulled in the services
    package, which imports `millm.ml` back — `import millm.ml` failed in a fresh process."""
    import subprocess
    import sys

    proc = subprocess.run([sys.executable, "-c", f"import {first}"], cwd=_REPO,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]


def test_the_resolver_imports_nothing_from_services_or_ml():
    tree = ast.parse((_REPO / "millm/core/hf_credentials.py").read_text(encoding="utf-8"))
    imported = [
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    ] + [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert imported, "the scanner read no imports"
    assert not [m for m in imported if m.startswith(("millm.services", "millm.ml"))]
