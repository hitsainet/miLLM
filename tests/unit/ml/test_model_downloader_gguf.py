"""Choosing one quantization must download ONE quantization.

Before this, `snapshot_download` was called with no `allow_patterns` at all, so
selecting a 4.2 GB Q4 on a repo holding twenty-four quants pulled all of them —
about 250 GB — and reported success. The assertion that catches it is not "the
right files were requested" but "the OTHER files were not".

MUTATION CONTROLS (each must turn this file red):
  * drop `allow_patterns` from _snapshot_download_with_circuit -> "only the chosen
    quant" fails
  * pass `[]` instead of None when nothing is selected -> "whole repo" fails
  * drop `variant` from _get_local_dir -> "two Q4 variants" fails
  * read `size` instead of `lfs.size` -> "true size of an LFS file" fails
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from millm.ml.model_downloader import ModelDownloader, _safe_variant, _sibling_size


@pytest.fixture
def downloader(tmp_path):
    return ModelDownloader(cache_dir=str(tmp_path))


class TestOnlyTheChosenQuantIsFetched:
    def test_the_selection_becomes_the_allow_list(self, downloader):
        chosen = ["Qwen2.5-72B-Instruct-Q6_K/Qwen2.5-72B-Instruct-Q6_K-00001-of-00002.gguf",
                  "Qwen2.5-72B-Instruct-Q6_K/Qwen2.5-72B-Instruct-Q6_K-00002-of-00002.gguf"]

        with patch("millm.ml.model_downloader.snapshot_download") as snap:
            downloader.download(
                repo_id="bartowski/Qwen2.5-72B-Instruct-GGUF",
                quantization="Q4",
                file_paths=chosen,
                variant="Q6_K",
            )

        allow = snap.call_args.kwargs["allow_patterns"]
        assert allow == chosen, "every part of a split quant must be requested"

    def test_the_other_quantizations_are_NOT_fetched(self, downloader):
        """The assertion that fails loudly against a missing allow_patterns."""
        with patch("millm.ml.model_downloader.snapshot_download") as snap:
            downloader.download(
                repo_id="r/x-GGUF",
                quantization="Q4",
                file_paths=["x-Q4_K_M.gguf"],
                variant="Q4_K_M",
            )

        allow = snap.call_args.kwargs["allow_patterns"]
        assert allow is not None, "no allow list means the ENTIRE repo comes down"
        assert "x-Q8_0.gguf" not in allow
        assert not any(p.endswith("/*") or p == "*" for p in allow), (
            "a directory glob at the repo root matches every quantization"
        )

    def test_no_selection_still_downloads_the_whole_repo(self, downloader):
        """An ordinary safetensors model must be unaffected."""
        with patch("millm.ml.model_downloader.snapshot_download") as snap:
            downloader.download(repo_id="google/gemma-2-2b", quantization="FP16")

        assert snap.call_args.kwargs["allow_patterns"] is None, (
            "None means everything; an empty list would match NOTHING and "
            "report an empty directory as a successful download"
        )

    def test_the_revision_is_pinned(self, downloader):
        with patch("millm.ml.model_downloader.snapshot_download") as snap:
            downloader.download(repo_id="r/x", quantization="Q4", revision="abc123")

        assert snap.call_args.kwargs["revision"] == "abc123"


class TestVariantsDoNotCollide:
    def test_two_Q4_variants_get_two_directories(self, downloader):
        """Q4_K_M, Q4_K_S and Q4_0 are all "Q4" to the quantization enum."""
        a = downloader._get_local_dir("r/x", "Q4", "Q4_K_M")
        b = downloader._get_local_dir("r/x", "Q4", "Q4_K_S")

        assert a != b, "sharing a directory means deleting one destroys the other"

    def test_no_variant_keeps_the_original_path(self, downloader):
        """Existing rows must resolve where they already are."""
        assert downloader._get_local_dir("r/x", "Q4").name == "r--x--Q4"

    def test_a_label_cannot_escape_the_cache_directory(self, downloader):
        """The label arrives from a remote filename, so it is untrusted input."""
        escaped = downloader._get_local_dir("r/x", "Q4", "../../../etc")
        assert ".." not in str(escaped)
        assert str(escaped).startswith(str(downloader.cache_dir))

    def test_safe_variant_strips_separators(self):
        assert "/" not in _safe_variant("a/b")
        assert ".." not in _safe_variant("..")


class TestSizesAreTheTrueOnes:
    def test_prefers_the_lfs_size(self):
        """A multi-GB GGUF is LFS: `size` is the pointer, `lfs.size` is real."""
        sibling = SimpleNamespace(size=133, lfs=SimpleNamespace(size=4_218_473_152))
        assert _sibling_size(sibling) == 4_218_473_152

    def test_falls_back_when_there_is_no_lfs_record(self):
        assert _sibling_size(SimpleNamespace(size=512, lfs=None)) == 512

    def test_missing_sizes_read_as_zero_not_a_crash(self):
        assert _sibling_size(SimpleNamespace(size=None, lfs=None)) == 0


class TestTheProgressDenominator:
    def test_counts_only_the_selected_files(self, downloader):
        """Summing the whole repo pegs the bar near zero for the entire pull."""
        info = SimpleNamespace(
            siblings=[
                SimpleNamespace(rfilename="x-Q4_K_M.gguf", size=100, lfs=None),
                SimpleNamespace(rfilename="x-Q8_0.gguf", size=900, lfs=None),
            ]
        )
        with patch.object(downloader.hf_api, "model_info", return_value=info):
            total = downloader.get_expected_download_size(
                "r/x", file_paths=["x-Q4_K_M.gguf"]
            )

        assert total == 100, "the denominator must describe what is being fetched"
