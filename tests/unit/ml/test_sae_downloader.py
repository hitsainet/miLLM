"""Unit tests for SAEDownloader."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from huggingface_hub.utils import RepositoryNotFoundError

from millm.ml.sae_downloader import SAEDownloader


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "sae_cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def downloader(temp_cache_dir):
    """Create an SAEDownloader with a temporary cache directory."""
    with patch.object(SAEDownloader, "__init__", lambda self, cache_dir: None):
        dl = SAEDownloader.__new__(SAEDownloader)
        dl.cache_dir = temp_cache_dir
        dl._api = MagicMock()
        return dl


class TestSAEDownloaderDownload:
    """Tests for download method."""

    @pytest.mark.asyncio
    @patch("millm.ml.sae_downloader.snapshot_download")
    async def test_download_calls_snapshot_download(self, mock_snapshot, downloader):
        """Test that download calls snapshot_download with correct params."""
        mock_snapshot.return_value = "/path/to/sae"
        downloader._api.repo_info.return_value = MagicMock()

        result = await downloader.download("jbloom/gemma-2-2b-res-jb")

        mock_snapshot.assert_called_once()
        call_kwargs = mock_snapshot.call_args[1]
        assert call_kwargs["repo_id"] == "jbloom/gemma-2-2b-res-jb"
        assert call_kwargs["revision"] == "main"
        assert call_kwargs["resume_download"] is True

    @pytest.mark.asyncio
    @patch("millm.ml.sae_downloader.snapshot_download")
    async def test_download_passes_token(self, mock_snapshot, downloader):
        """Test that download passes token to snapshot_download."""
        mock_snapshot.return_value = "/path/to/sae"
        downloader._api.repo_info.return_value = MagicMock()

        await downloader.download(
            "jbloom/gemma-2-2b-res-jb",
            token="hf_test_token",
        )

        call_kwargs = mock_snapshot.call_args[1]
        assert call_kwargs["token"] == "hf_test_token"

    @pytest.mark.asyncio
    @patch("millm.ml.sae_downloader.snapshot_download")
    async def test_download_calls_progress_callback(self, mock_snapshot, downloader):
        """Test that download invokes the progress callback."""
        mock_snapshot.return_value = "/path/to/sae"
        downloader._api.repo_info.return_value = MagicMock()
        callback = MagicMock()

        await downloader.download(
            "jbloom/gemma-2-2b-res-jb",
            progress_callback=callback,
        )

        # Callback should be called at least twice: start and complete
        assert callback.call_count >= 2

        # First call should be downloading start
        first_call_args = callback.call_args_list[0][0][0]
        assert first_call_args["status"] == "downloading"
        assert first_call_args["percent"] == 0

        # Last call should be complete
        last_call_args = callback.call_args_list[-1][0][0]
        assert last_call_args["status"] == "complete"
        assert last_call_args["percent"] == 100

    @pytest.mark.asyncio
    async def test_download_handles_repository_not_found(self, downloader):
        """Test that download raises ValueError for missing repositories."""
        downloader._api.repo_info.side_effect = RepositoryNotFoundError("Not found")

        with pytest.raises(ValueError) as exc_info:
            await downloader.download("nonexistent/sae-repo")

        assert "not found" in str(exc_info.value).lower()


class TestSAEDownloaderListRepositoryFiles:
    """Tests for list_repository_files method."""

    @pytest.mark.asyncio
    async def test_returns_file_list(self, downloader):
        """Test that list_repository_files returns structured file data."""
        # Mock repo_info
        mock_sibling = MagicMock()
        mock_sibling.rfilename = "layer_12/width_16k/average_l0_50/params.npz"
        mock_sibling.size = 256_000_000

        mock_repo_info = MagicMock()
        mock_repo_info.siblings = [mock_sibling]
        downloader._api.repo_info.return_value = mock_repo_info

        # Mock list_repo_files
        downloader._api.list_repo_files.return_value = [
            "layer_12/width_16k/average_l0_50/params.npz",
        ]

        result = await downloader.list_repository_files("google/gemma-scope-2b-pt-res")

        assert result["repository_id"] == "google/gemma-scope-2b-pt-res"
        assert result["total_files"] >= 1
        assert len(result["files"]) >= 1

        first_file = result["files"][0]
        assert first_file["path"] == "layer_12/width_16k/average_l0_50/params.npz"
        assert first_file["layer"] == 12
        assert first_file["width"] == "16k"


class TestSAEDownloaderGenerateSaeId:
    """Tests for generate_sae_id method."""

    def test_generates_id_from_repo(self, downloader):
        """Test that generate_sae_id produces a filesystem-safe ID."""
        sae_id = downloader.generate_sae_id("jbloom/gemma-2-2b-res-jb")

        assert sae_id == "jbloom--gemma-2-2b-res-jb"
        assert "/" not in sae_id

    def test_includes_revision_when_not_main(self, downloader):
        """Test that non-main revisions are included in the ID."""
        sae_id = downloader.generate_sae_id(
            "jbloom/gemma-2-2b-res-jb",
            revision="v2.0",
        )

        assert "@v2.0" in sae_id

    def test_includes_file_path_directory(self, downloader):
        """Test that file path directory is included for unique IDs."""
        sae_id = downloader.generate_sae_id(
            "google/gemma-scope-2b-pt-res",
            file_path="layer_12/width_16k/average_l0_50/params.npz",
        )

        assert "layer_12--width_16k--average_l0_50" in sae_id

    def test_consistent_ids_for_same_input(self, downloader):
        """Test that the same inputs always produce the same ID."""
        id1 = downloader.generate_sae_id("jbloom/gemma-2-2b-res-jb")
        id2 = downloader.generate_sae_id("jbloom/gemma-2-2b-res-jb")

        assert id1 == id2


class TestSAEDownloaderExists:
    """Tests for is_cached method."""

    def test_returns_false_when_not_cached(self, downloader):
        """Test that is_cached returns False for non-existent SAEs."""
        assert downloader.is_cached("nonexistent/sae") is False

    def test_returns_true_when_snapshot_exists(self, downloader, temp_cache_dir):
        """Test that is_cached returns True when cache path exists."""
        # Create a mock cache path structure
        with patch.object(downloader, "get_cache_path", return_value=temp_cache_dir):
            assert downloader.is_cached("some/repo") is True
