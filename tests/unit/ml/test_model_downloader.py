"""Unit tests for ModelDownloader."""

import shutil
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub.utils import (
    GatedRepoError,
    RepositoryNotFoundError,
)

from millm.core.errors import (
    DownloadFailedError,
    GatedModelError,
    RepoNotFoundError,
)
from millm.ml.model_downloader import (
    ModelDownloader,
    _DownloadSizePoller,
    _SilentTqdm,
)


@pytest.fixture(autouse=True)
def reset_hf_circuit():
    """Reset the shared HuggingFace circuit breaker before each test.

    CircuitBreaker._states is class-level, so failures in one test (e.g.
    test_download_cleans_up_on_failure) would otherwise open the circuit and
    cause unrelated get_model_info tests to fail with DownloadFailedError.
    """
    from millm.core.resilience import huggingface_circuit
    huggingface_circuit.reset()
    yield
    huggingface_circuit.reset()


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "model_cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def downloader(temp_cache_dir):
    """Create a ModelDownloader with a temporary cache directory."""
    return ModelDownloader(cache_dir=str(temp_cache_dir))


class TestModelDownloaderInit:
    """Tests for ModelDownloader initialization."""

    def test_creates_cache_directory(self, tmp_path):
        """Test that cache directory is created if it doesn't exist."""
        cache_dir = tmp_path / "new_cache"
        assert not cache_dir.exists()

        ModelDownloader(cache_dir=str(cache_dir))

        assert cache_dir.exists()

    def test_uses_provided_cache_dir(self, temp_cache_dir):
        """Test that provided cache directory is used."""
        downloader = ModelDownloader(cache_dir=str(temp_cache_dir))

        assert downloader.cache_dir == temp_cache_dir


class TestModelDownloaderLocalDir:
    """Tests for local directory path generation."""

    def test_get_local_dir_format(self, downloader):
        """Test that local directory path follows expected format."""
        local_dir = downloader._get_local_dir("google/gemma-2-2b", "Q4")

        assert "huggingface" in str(local_dir)
        assert "google--gemma-2-2b--Q4" in str(local_dir)

    def test_get_local_dir_handles_special_chars(self, downloader):
        """Test that repo IDs with special characters are handled."""
        local_dir = downloader._get_local_dir("meta-llama/Llama-3.1-8B", "FP16")

        assert "meta-llama--Llama-3.1-8B--FP16" in str(local_dir)


class TestModelDownloaderExists:
    """Tests for checking if model exists in cache."""

    def test_exists_returns_false_when_no_dir(self, downloader):
        """Test that exists returns False when directory doesn't exist."""
        assert not downloader.exists("nonexistent/model", "Q4")

    def test_exists_returns_false_when_empty_dir(self, downloader, temp_cache_dir):
        """Test that exists returns False when directory is empty."""
        local_dir = downloader._get_local_dir("google/gemma", "Q4")
        local_dir.mkdir(parents=True)

        assert not downloader.exists("google/gemma", "Q4")

    def test_exists_returns_true_with_config_json(self, downloader, temp_cache_dir):
        """Test that exists returns True when config.json is present."""
        local_dir = downloader._get_local_dir("google/gemma", "Q4")
        local_dir.mkdir(parents=True)
        (local_dir / "config.json").touch()

        assert downloader.exists("google/gemma", "Q4")

    def test_exists_returns_true_with_safetensors(self, downloader, temp_cache_dir):
        """Test that exists returns True when model.safetensors is present."""
        local_dir = downloader._get_local_dir("google/gemma", "Q4")
        local_dir.mkdir(parents=True)
        (local_dir / "model.safetensors").touch()

        assert downloader.exists("google/gemma", "Q4")


class TestModelDownloaderDownload:
    """Tests for the download method."""

    @patch("millm.ml.model_downloader.snapshot_download")
    def test_download_calls_snapshot_download(self, mock_snapshot, downloader, temp_cache_dir):
        """Test that download calls huggingface_hub's snapshot_download."""
        mock_snapshot.return_value = "/path/to/model"

        result = downloader.download("google/gemma-2-2b", "Q4")

        mock_snapshot.assert_called_once()
        call_kwargs = mock_snapshot.call_args[1]
        assert call_kwargs["repo_id"] == "google/gemma-2-2b"
        assert call_kwargs["local_dir_use_symlinks"] is False
        # resume_download is deprecated in huggingface_hub >= 1.x (resume is
        # automatic) and must no longer be passed.
        assert "resume_download" not in call_kwargs
        # Console progress bars are suppressed; progress comes from the poller.
        assert call_kwargs["tqdm_class"] is _SilentTqdm

    @patch("millm.ml.model_downloader.snapshot_download")
    def test_download_returns_local_path(self, mock_snapshot, downloader):
        """Test that download returns the local directory path."""
        result = downloader.download("google/gemma-2-2b", "Q4")

        expected_path = str(downloader._get_local_dir("google/gemma-2-2b", "Q4"))
        assert result == expected_path

    @patch("millm.ml.model_downloader.snapshot_download")
    def test_download_passes_token(self, mock_snapshot, downloader):
        """Test that token is passed to snapshot_download."""
        downloader.download("google/gemma-2-2b", "Q4", token="hf_test_token")

        call_kwargs = mock_snapshot.call_args[1]
        assert call_kwargs["token"] == "hf_test_token"

    @patch("millm.ml.model_downloader.snapshot_download")
    def test_download_raises_repo_not_found(self, mock_snapshot, downloader):
        """Test that RepoNotFoundError is raised for missing repos."""
        mock_snapshot.side_effect = RepositoryNotFoundError(
            "Not found", response=MagicMock()
        )

        with pytest.raises(RepoNotFoundError) as exc_info:
            downloader.download("nonexistent/model", "Q4")

        assert "nonexistent/model" in str(exc_info.value.message)
        assert exc_info.value.details["repo_id"] == "nonexistent/model"

    @patch("millm.ml.model_downloader.snapshot_download")
    def test_download_raises_gated_model_error(self, mock_snapshot, downloader):
        """Test that GatedModelError is raised for gated repos."""
        mock_snapshot.side_effect = GatedRepoError("Gated", response=MagicMock())

        with pytest.raises(GatedModelError) as exc_info:
            downloader.download("meta-llama/Llama-2-7b", "Q4")

        assert "meta-llama/Llama-2-7b" in str(exc_info.value.message)
        assert "gated" in str(exc_info.value.message).lower()

    @patch("millm.ml.model_downloader.snapshot_download")
    @patch("millm.ml.model_downloader.shutil.rmtree")
    def test_download_cleans_up_on_failure(self, mock_rmtree, mock_snapshot, downloader, temp_cache_dir):
        """Test that partial downloads are cleaned up on failure."""
        mock_snapshot.side_effect = Exception("Network error")

        # Create the directory to simulate partial download
        local_dir = downloader._get_local_dir("google/gemma", "Q4")
        local_dir.mkdir(parents=True)

        with pytest.raises(DownloadFailedError):
            downloader.download("google/gemma", "Q4")

        mock_rmtree.assert_called_once()

    @patch("millm.ml.model_downloader.snapshot_download")
    def test_download_reports_byte_progress_during_download(
        self, mock_snapshot, downloader
    ):
        """Progress must advance from on-disk bytes WHILE a large file is being
        written — not jump 0 -> 100 at the end.

        Regression for the tqdm_class bug: ``snapshot_download`` only applies the
        custom tqdm to the outer file-count bar, so the old tracker sat at 0%
        through a multi-GB shard. This proves the filesystem poller is wired into
        ``download()`` and reports a real mid-download percentage. Deleting the
        ``poller.start()`` line makes this test fail (reachability control).
        """
        local_dir = downloader._get_local_dir("google/gemma-2-2b", "Q4")
        updates: list = []

        def cb(pct, downloaded, total, speed):
            updates.append((pct, downloaded, total))

        def fake_snapshot(**kwargs):
            # Simulate a large shard landing on disk mid-download.
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / "model.safetensors").write_bytes(b"\0" * 5000)
            time.sleep(0.1)  # let the poller sample at least once

        mock_snapshot.side_effect = fake_snapshot

        with patch("millm.ml.model_downloader.POLL_INTERVAL_SECONDS", 0.02), patch.object(
            downloader, "get_expected_download_size", return_value=10_000
        ):
            downloader.download("google/gemma-2-2b", "Q4", progress_callback=cb)

        # A byte-based ~50% update (5000 / 10000) must have been reported while
        # the single file was on disk — not only a terminal 0 or 100.
        assert any(
            d == 5000 and t == 10_000 and 40.0 <= p <= 60.0 for p, d, t in updates
        ), f"expected a ~50% byte-based update mid-download, got {updates}"

    @patch("millm.ml.model_downloader.snapshot_download")
    def test_download_without_callback_skips_size_lookup(
        self, mock_snapshot, downloader
    ):
        """No progress_callback => no poller and no (network) size lookup."""
        with patch.object(
            downloader, "get_expected_download_size"
        ) as mock_size:
            downloader.download("google/gemma-2-2b", "Q4")

        mock_size.assert_not_called()


class TestDownloadSizePoller:
    """Tests for the filesystem-based download progress poller."""

    @staticmethod
    def _make_file(path: Path, size: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\0" * size)

    def test_reports_byte_based_percentage(self, tmp_path):
        calls: list = []
        poller = _DownloadSizePoller(
            local_dir=tmp_path,
            expected_total=1000,
            callback=lambda p, d, t, s: calls.append((p, d, t, s)),
        )
        self._make_file(tmp_path / "model.safetensors", 500)
        poller._emit()

        assert calls, "callback should have fired"
        pct, downloaded, total, _ = calls[-1]
        assert downloaded == 500
        assert total == 1000
        assert 49.0 <= pct <= 51.0

    def test_counts_incomplete_partial_files_in_subdir(self, tmp_path):
        """The in-progress ``*.incomplete`` file under ``.cache`` must count.

        This is the heart of the 0%-stuck fix: a shard is written to
        ``.cache/huggingface/download/<hash>.incomplete`` while downloading and
        its bytes must be visible before the file is finalized. Changing
        ``_dir_size`` from ``rglob`` to a top-level ``glob`` makes this fail.
        """
        calls: list = []
        poller = _DownloadSizePoller(
            local_dir=tmp_path,
            expected_total=1000,
            callback=lambda p, d, t, s: calls.append((p, d, t, s)),
        )
        self._make_file(tmp_path / "config.json", 250)  # finalized
        self._make_file(
            tmp_path / ".cache" / "huggingface" / "download" / "abc123.incomplete",
            250,  # still downloading
        )
        poller._emit()

        pct, downloaded, total, _ = calls[-1]
        assert downloaded == 500  # 250 finalized + 250 in-progress
        assert 49.0 <= pct <= 51.0

    def test_percentage_capped_at_99_until_complete(self, tmp_path):
        calls: list = []
        poller = _DownloadSizePoller(
            local_dir=tmp_path,
            expected_total=1000,
            callback=lambda p, d, t, s: calls.append((p, d, t, s)),
        )
        self._make_file(tmp_path / "model.bin", 1000)  # fully on disk
        poller._emit()

        # Never 100 until the caller confirms completion (avoids a premature
        # "done" while HF is still finalizing / verifying).
        assert calls[-1][0] == 99.0

    def test_unknown_total_still_reports_bytes(self, tmp_path):
        calls: list = []
        poller = _DownloadSizePoller(
            local_dir=tmp_path,
            expected_total=0,  # Hub metadata unavailable
            callback=lambda p, d, t, s: calls.append((p, d, t, s)),
        )
        self._make_file(tmp_path / "model.bin", 500)
        poller._emit()

        pct, downloaded, total, _ = calls[-1]
        assert pct == 0.0
        assert downloaded == 500  # bytes still surfaced so the UI shows movement
        assert total == 0

    def test_speed_computed_across_samples(self, tmp_path):
        calls: list = []
        poller = _DownloadSizePoller(
            local_dir=tmp_path,
            expected_total=10_000,
            callback=lambda p, d, t, s: calls.append((p, d, t, s)),
        )
        times = iter([100.0, 102.0])  # samples 2 seconds apart
        with patch(
            "millm.ml.model_downloader.time.monotonic", lambda: next(times)
        ):
            self._make_file(tmp_path / "part1", 1000)
            poller._emit()  # first sample: speed 0
            self._make_file(tmp_path / "part2", 3000)
            poller._emit()  # +3000 bytes over 2s => 1500 B/s

        speed = calls[-1][3]
        assert 1400.0 <= speed <= 1600.0

    def test_monotonic_progress_through_growing_file(self, tmp_path):
        calls: list = []
        poller = _DownloadSizePoller(
            local_dir=tmp_path,
            expected_total=1000,
            callback=lambda p, d, t, s: calls.append((p, d, t, s)),
        )
        target = tmp_path / "model.safetensors"
        for size in (100, 400, 800):
            self._make_file(target, size)
            poller._emit()

        pcts = [c[0] for c in calls]
        assert pcts == sorted(pcts)  # never goes backwards
        assert pcts[0] < pcts[-1]  # actually advances (not stuck at one value)

    def test_start_emits_total_immediately(self, tmp_path):
        """start() emits once up front so the UI gets the denominator early."""
        calls: list = []
        poller = _DownloadSizePoller(
            local_dir=tmp_path,
            expected_total=2000,
            callback=lambda p, d, t, s: calls.append((p, d, t, s)),
            interval=60.0,  # long, so only the immediate emit fires
        )
        poller.start()
        try:
            assert calls, "start() should emit immediately"
            assert calls[0][2] == 2000  # total present from the first event
        finally:
            poller.stop()


class TestModelDownloaderGetModelInfo:
    """Tests for get_model_info method."""

    @patch.object(ModelDownloader, "_extract_params", return_value="2B")
    @patch.object(ModelDownloader, "_check_trust_remote_code", return_value=False)
    def test_get_model_info_returns_dict(self, mock_trust, mock_params, downloader):
        """Test that get_model_info returns expected structure."""
        mock_info = MagicMock()
        mock_info.modelId = "google/gemma-2-2b"
        mock_info.gated = False
        mock_info.pipeline_tag = "text-generation"
        mock_info.library_name = "transformers"
        mock_info.downloads = 1000
        mock_info.likes = 50
        mock_info.siblings = []

        with patch.object(downloader.hf_api, "model_info", return_value=mock_info):
            result = downloader.get_model_info("google/gemma-2-2b")

        assert result["name"] == "gemma-2-2b"
        assert result["repo_id"] == "google/gemma-2-2b"
        assert result["params"] == "2B"
        assert result["architecture"] == "text-generation"
        assert result["is_gated"] is False
        assert result["requires_trust_remote_code"] is False

    def test_get_model_info_raises_repo_not_found(self, downloader):
        """Test that RepoNotFoundError is raised for missing repos."""
        with patch.object(downloader.hf_api, "model_info") as mock_info:
            mock_info.side_effect = RepositoryNotFoundError(
                "Not found", response=MagicMock()
            )

            with pytest.raises(RepoNotFoundError):
                downloader.get_model_info("nonexistent/model")

    def test_get_model_info_raises_gated_error(self, downloader):
        """Test that GatedModelError is raised for gated repos without token."""
        with patch.object(downloader.hf_api, "model_info") as mock_info:
            mock_info.side_effect = GatedRepoError("Gated", response=MagicMock())

            with pytest.raises(GatedModelError):
                downloader.get_model_info("meta-llama/Llama-2-7b")


class TestModelDownloaderExtractParams:
    """Tests for parameter extraction."""

    def test_extract_params_from_safetensors(self, downloader):
        """Test extraction from safetensors metadata."""
        mock_info = MagicMock()
        mock_safetensors = MagicMock()
        mock_safetensors.total = 2_000_000_000
        mock_info.safetensors = mock_safetensors
        mock_info.modelId = "test/model"

        result = downloader._extract_params(mock_info)

        assert result == "2.0B"

    def test_extract_params_trillions(self, downloader):
        """Test extraction for trillion-parameter models."""
        mock_info = MagicMock()
        mock_safetensors = MagicMock()
        mock_safetensors.total = 1_500_000_000_000
        mock_info.safetensors = mock_safetensors
        mock_info.modelId = "test/model"

        result = downloader._extract_params(mock_info)

        assert result == "1.5T"

    def test_extract_params_millions(self, downloader):
        """Test extraction for million-parameter models."""
        mock_info = MagicMock()
        mock_safetensors = MagicMock()
        mock_safetensors.total = 350_000_000
        mock_info.safetensors = mock_safetensors
        mock_info.modelId = "test/model"

        result = downloader._extract_params(mock_info)

        assert result == "350M"

    def test_extract_params_from_name(self, downloader):
        """Test extraction from model name when safetensors not available."""
        mock_info = MagicMock()
        mock_info.safetensors = None
        mock_info.modelId = "meta-llama/Llama-2-7b"

        result = downloader._extract_params(mock_info)

        assert result == "7B"

    def test_extract_params_unknown(self, downloader):
        """Test that unknown is returned when params can't be determined."""
        mock_info = MagicMock()
        mock_info.safetensors = None
        mock_info.modelId = "test/custom-model"

        result = downloader._extract_params(mock_info)

        assert result == "unknown"


class TestModelDownloaderTrustRemoteCode:
    """Tests for trust_remote_code detection."""

    def test_detects_custom_modeling_file(self, downloader):
        """Test that modeling_*.py files trigger trust_remote_code."""
        mock_info = MagicMock()
        mock_sibling = MagicMock()
        mock_sibling.rfilename = "modeling_custom.py"
        mock_info.siblings = [mock_sibling]

        result = downloader._check_trust_remote_code(mock_info)

        assert result is True

    def test_detects_configuration_file(self, downloader):
        """Test that configuration_*.py files trigger trust_remote_code."""
        mock_info = MagicMock()
        mock_sibling = MagicMock()
        mock_sibling.rfilename = "configuration_model.py"
        mock_info.siblings = [mock_sibling]

        result = downloader._check_trust_remote_code(mock_info)

        assert result is True

    def test_ignores_standard_files(self, downloader):
        """Test that standard files don't trigger trust_remote_code."""
        mock_info = MagicMock()
        mock_sibling1 = MagicMock()
        mock_sibling1.rfilename = "config.json"
        mock_sibling2 = MagicMock()
        mock_sibling2.rfilename = "model.safetensors"
        mock_info.siblings = [mock_sibling1, mock_sibling2]

        result = downloader._check_trust_remote_code(mock_info)

        assert result is False


class TestModelDownloaderDeleteCache:
    """Tests for delete_cached_model method."""

    def test_delete_returns_false_when_not_cached(self, downloader):
        """Test that delete returns False when model isn't cached."""
        result = downloader.delete_cached_model("nonexistent/model", "Q4")

        assert result is False

    def test_delete_removes_directory(self, downloader, temp_cache_dir):
        """Test that delete removes the cached model directory."""
        local_dir = downloader._get_local_dir("google/gemma", "Q4")
        local_dir.mkdir(parents=True)
        (local_dir / "model.safetensors").touch()

        result = downloader.delete_cached_model("google/gemma", "Q4")

        assert result is True
        assert not local_dir.exists()


class TestModelDownloaderCacheSize:
    """Tests for get_cache_size method."""

    def test_cache_size_returns_zero_when_not_cached(self, downloader):
        """Test that cache size returns 0 for non-existent models."""
        result = downloader.get_cache_size("nonexistent/model", "Q4")

        assert result == 0

    def test_cache_size_calculates_total(self, downloader, temp_cache_dir):
        """Test that cache size calculates total file size."""
        local_dir = downloader._get_local_dir("google/gemma", "Q4")
        local_dir.mkdir(parents=True)

        # Create files with known sizes
        file1 = local_dir / "config.json"
        file1.write_text("x" * 100)  # 100 bytes

        file2 = local_dir / "model.bin"
        file2.write_text("y" * 1000)  # 1000 bytes

        result = downloader.get_cache_size("google/gemma", "Q4")

        assert result == 1100
