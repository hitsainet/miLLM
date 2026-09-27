"""`sae.path` names a DIRECTORY, and resolving it is where a wrong assumption would land.

⚠ The contract's `path` field carries no description, so the PRODUCER is the authority:
miStudio writes `ExternalSAE.hf_filepath` into it — `layer_11`, or
`layer_12/width_16k/canonical` — the directory inside the repo holding `cfg.json` and
`sae_weights.safetensors`. The first version of `_resolve_sae_path` treated it as a file and
would have handed a directory to `SaeFeatureSlice.load`, which branches on
`.suffix == ".safetensors"` and falls through to `np.load`.

Caught by reading what miStudio writes, not by reading the field name. These tests keep it
caught, and pin the traversal guard beside it — `path` comes from an imported document.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from millm.core.errors import ProbeSaeMissingError
from millm.services.probe_arm_bridge import _resolve_sae_path, _weights_file

REPO = "mistudio/sae-lfm2p5-1p2b-instruct-l11-jumprelu"


class TestWeightsFileResolution:
    def test_a_directory_resolves_to_the_weights_inside_it(self, tmp_path):
        d = tmp_path / "layer_11"
        d.mkdir()
        (d / "cfg.json").write_text("{}")
        weights = d / "sae_weights.safetensors"
        weights.write_bytes(b"x")
        assert _weights_file(d) == str(weights)

    def test_a_file_path_still_works(self, tmp_path):
        """A future producer that names the file directly needs no change here."""
        f = tmp_path / "sae.safetensors"
        f.write_bytes(b"x")
        assert _weights_file(f) == str(f)

    def test_a_directory_with_no_weights_REFUSES_and_says_what_it_looked_for(self, tmp_path):
        """Naming the filenames tried is the difference between a fixable error and a mystery."""
        d = tmp_path / "layer_11"
        d.mkdir()
        (d / "cfg.json").write_text("{}")
        with pytest.raises(ProbeSaeMissingError) as exc:
            _weights_file(d)
        assert "looked_for" in exc.value.details
        assert "sae_weights.safetensors" in exc.value.details["looked_for"]

    def test_the_cfg_json_is_not_mistaken_for_weights(self, tmp_path):
        """Returning `cfg.json` would fail far away, inside the tensor loader."""
        d = tmp_path / "layer_11"
        d.mkdir()
        (d / "cfg.json").write_text("{}")
        with pytest.raises(ProbeSaeMissingError):
            _weights_file(d)


class TestTheDocumentCannotEscapeTheCache:
    @pytest.mark.asyncio
    async def test_a_traversing_path_is_refused(self, monkeypatch, tmp_path):
        """⚠ `path` comes from an IMPORTED DOCUMENT. `../../etc/...` must not read outside
        the SAE cache, and the refusal must not be a FileNotFoundError from deep in a loader.
        """
        from millm.core.config import settings

        monkeypatch.setattr(settings, "SAE_CACHE_DIR", str(tmp_path), raising=False)
        with pytest.raises(ProbeSaeMissingError) as exc:
            await _resolve_sae_path({"hf_repo": REPO, "path": "../../../etc/passwd"})
        assert "escapes" in str(exc.value).lower()

    @pytest.mark.asyncio
    async def test_a_missing_sae_names_the_repo_and_path_to_fetch(self, monkeypatch, tmp_path):
        """An operator needs to know WHAT to download, not that something was absent."""
        from millm.core.config import settings

        monkeypatch.setattr(settings, "SAE_CACHE_DIR", str(tmp_path), raising=False)
        with pytest.raises(ProbeSaeMissingError) as exc:
            await _resolve_sae_path({"hf_repo": REPO, "path": "layer_11"})
        assert exc.value.details["hf_repo"] == REPO
        assert exc.value.details["path"] == "layer_11"

    @pytest.mark.asyncio
    async def test_a_present_sae_directory_resolves_to_its_weights(self, monkeypatch, tmp_path):
        """The success path, through the real layout the cache uses."""
        from millm.core.config import settings

        monkeypatch.setattr(settings, "SAE_CACHE_DIR", str(tmp_path), raising=False)
        d = tmp_path / REPO.replace("/", "__") / "layer_11"
        d.mkdir(parents=True)
        weights = d / "sae_weights.safetensors"
        weights.write_bytes(b"x")
        resolved = await _resolve_sae_path({"hf_repo": REPO, "path": "layer_11"})
        assert resolved == str(weights)

    @pytest.mark.asyncio
    async def test_a_block_with_no_repo_or_path_refuses_rather_than_guessing(self, tmp_path, monkeypatch):
        from millm.core.config import settings

        monkeypatch.setattr(settings, "SAE_CACHE_DIR", str(tmp_path), raising=False)
        with pytest.raises(ProbeSaeMissingError):
            await _resolve_sae_path({"hf_repo": "", "path": ""})
