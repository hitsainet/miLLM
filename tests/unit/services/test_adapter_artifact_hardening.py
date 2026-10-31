"""Review round 1 (R1-B) hardening of the adapter artifact reader and manifest validator.

Each test here was written against a mutation that survived the existing suite, or a defect found
by reading, and was re-run as a negative control against the broken line:

- confinement is proved at ONE instant by `realpath`; the reads then re-walked the resolved string,
  so an intermediate directory swapped for a symlink in between was followed out of the root;
- `O_NOFOLLOW` on the final component was in the code and pinned by nothing;
- Python's `json` keeps the LAST of two equal keys (other readers keep the first) and accepts the
  non-JSON constants `NaN`/`Infinity`;
- the base hash comparison is case-insensitive on both sides, and nothing exercised the resident's;
- the vendored schema's bytes are checked against miForge only when miForge is checked out, so in
  CI (no checkout, addendum §5) a byte edit went unnoticed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from millm.core.errors import AdapterInvalidError, AdapterManifestInvalidError
from millm.ml.adapter_hooks import LoraHookSet
from millm.services import adapter_artifact
from millm.services.adapter_artifact import (
    ResidentBase,
    _read_regular,
    base_matches,
    prepare_adapter,
    resolve_adapter_dir,
)
from millm.services.adapter_manifest import VENDORED
from tests.unit.adapter_fixtures import BASE_NAME, tiny_llama, write_adapter

HUB = ResidentBase(manifest_sha256=None, names=frozenset({BASE_NAME}))

#: sha256 of the vendored `docs/schemas/adapter-manifest-v1.json` (md5 91cfe9f0…, identical to
#: miForge's at vendoring). Re-vendoring from miForge updates this line in the same commit.
VENDORED_SHA256 = "c120e08e4fcaaa3888138970f20f477b6b4f94213e5537147cfafd5897638779"


@pytest.fixture(scope="module")
def llama():
    return tiny_llama()


@pytest.fixture
def root(tmp_path) -> Path:
    r = tmp_path / "checkpoints"
    r.mkdir()
    return r


def roots(root: Path) -> list[tuple[str, str]]:
    return [("primary", str(root))]


class TestAnIntermediateDirectorySwappedAfterResolution:
    """`root/run/adapter` resolves inside the root; then `run` is replaced by a symlink to an
    attacker directory holding a complete, self-consistent adapter (its own manifest hash
    matches its own weights, so the hash check cannot catch it)."""

    def _swap_after_resolve(self, root, tmp_path, llama, monkeypatch):
        write_adapter(root / "run" / "adapter", llama, seed=1)
        write_adapter(tmp_path / "attacker" / "adapter", llama, seed=7)
        real_resolve = adapter_artifact.resolve_adapter_dir

        def resolve_then_swap(rel, roots_=None):
            location = real_resolve(rel, roots_)
            (root / "run").rename(root / "run.moved")
            (root / "run").symlink_to(tmp_path / "attacker")
            return location

        monkeypatch.setattr(adapter_artifact, "resolve_adapter_dir", resolve_then_swap)

    def test_the_swap_is_refused_and_nothing_is_built(self, root, tmp_path, llama, monkeypatch):
        self._swap_after_resolve(root, tmp_path, llama, monkeypatch)
        built = []
        real_build = LoraHookSet.build
        monkeypatch.setattr(
            LoraHookSet, "build", staticmethod(lambda *a, **k: built.append(1) or real_build(*a, **k))
        )
        with pytest.raises(AdapterInvalidError) as e:
            prepare_adapter("run/adapter", llama, HUB, roots=roots(root))
        assert e.value.details["reason"] == "unreadable"
        assert e.value.details["file"] == "adapter_manifest.json"
        assert built == []

    def test_the_fixture_separates_a_walk_from_a_plain_open(self, root, tmp_path, llama):
        """Without the descriptor walk, the resolved string reads the ATTACKER's file — so the
        test above fails against a plain `os.open(path)`, not merely by accident."""
        write_adapter(root / "run" / "adapter", llama, seed=1)
        write_adapter(tmp_path / "attacker" / "adapter", llama, seed=7)
        location = resolve_adapter_dir("run/adapter", roots(root))
        (root / "run").rename(root / "run.moved")
        (root / "run").symlink_to(tmp_path / "attacker")
        name = "adapter_model.safetensors"
        plain = _read_regular(location.files[name], max_bytes=1 << 30, label=name)
        assert plain == (tmp_path / "attacker" / "adapter" / name).read_bytes()
        with pytest.raises(AdapterInvalidError):
            _read_regular(location.files[name], max_bytes=1 << 30, label=name,
                          beneath=location.root)


class TestTheFinalComponentIsNotFollowed:
    def test_a_symlink_given_to_the_reader_is_refused(self, tmp_path):
        target = tmp_path / "real.json"
        target.write_text("{}")
        link = tmp_path / "link.json"
        link.symlink_to(target)
        with pytest.raises(AdapterInvalidError) as e:
            _read_regular(str(link), max_bytes=1024, label="link.json")
        assert e.value.details["reason"] == "unreadable"

    def test_a_symlink_is_refused_beneath_a_root_too(self, tmp_path):
        (tmp_path / "d").mkdir()
        (tmp_path / "d" / "real.json").write_text("{}")
        (tmp_path / "d" / "link.json").symlink_to(tmp_path / "d" / "real.json")
        with pytest.raises(AdapterInvalidError):
            _read_regular(str(tmp_path / "d" / "link.json"), max_bytes=1024, label="x",
                          beneath=str(tmp_path))
        assert _read_regular(str(tmp_path / "d" / "real.json"), max_bytes=1024, label="x",
                             beneath=str(tmp_path)) == b"{}"


class TestStrictJson:
    def _rewrite(self, path: Path, text: str) -> None:
        path.write_text(text)

    def test_a_duplicated_manifest_key_is_refused(self, root, llama):
        d = write_adapter(root / "a", llama)
        manifest = (d / "adapter_manifest.json").read_text()
        # A second, different weights_sha256: Python would silently keep this one.
        dup = manifest.replace('"weights_sha256"', '"weights_sha256": "' + "0" * 64 + '",\n  '
                               '"weights_sha256"', 1)
        assert json.loads(dup)["weights_sha256"] == json.loads(manifest)["weights_sha256"]
        self._rewrite(d / "adapter_manifest.json", dup)
        with pytest.raises(AdapterManifestInvalidError) as e:
            prepare_adapter("a", llama, HUB, roots=roots(root))
        assert "duplicate key 'weights_sha256'" in e.value.message

    def test_a_duplicated_config_key_is_refused(self, root, llama):
        d = write_adapter(root / "a", llama)
        config = (d / "adapter_config.json").read_text()
        self._rewrite(d / "adapter_config.json", config.replace('"r":', '"r": 4, "r":', 1))
        with pytest.raises(AdapterManifestInvalidError) as e:
            prepare_adapter("a", llama, HUB, roots=roots(root))
        assert "duplicate key 'r'" in e.value.message

    @pytest.mark.parametrize("constant", ["Infinity", "NaN", "-Infinity"])
    def test_a_non_json_constant_is_refused(self, root, llama, constant):
        d = write_adapter(root / "a", llama)
        manifest = json.loads((d / "adapter_manifest.json").read_text())
        text = json.dumps(manifest).replace('"alpha": 6.0', f'"alpha": {constant}')
        assert constant in text
        self._rewrite(d / "adapter_manifest.json", text)
        with pytest.raises(AdapterManifestInvalidError) as e:
            prepare_adapter("a", llama, HUB, roots=roots(root))
        assert "is not JSON" in e.value.message


class TestTheBaseHashIgnoresCase:
    H = "ab" * 32

    def test_an_upper_case_claim_matches_a_lower_case_resident(self):
        resident = ResidentBase(manifest_sha256=self.H, names=frozenset())
        assert base_matches({"name": "x", "manifest_sha256": self.H.upper()}, resident)

    def test_a_lower_case_claim_matches_an_upper_case_resident(self):
        resident = ResidentBase(manifest_sha256=self.H.upper(), names=frozenset())
        assert base_matches({"name": "x", "manifest_sha256": self.H}, resident)


def test_the_vendored_schema_bytes_are_pinned_without_a_miforge_checkout():
    """The cross-repo byte test skips when miForge is absent (CI). This pin is local, so a byte
    edit to the vendored copy goes red everywhere; re-vendoring updates the constant above."""
    assert hashlib.sha256(VENDORED.read_bytes()).hexdigest() == VENDORED_SHA256
    assert os.path.basename(VENDORED) == "adapter-manifest-v1.json"


def test_a_lone_surrogate_in_the_path_is_a_bad_path_not_a_500(root):
    """`os.path.realpath("a\\ud800")` raises UnicodeEncodeError, which no route maps."""
    with pytest.raises(AdapterInvalidError) as e:
        resolve_adapter_dir("a\ud800b", roots(root))
    assert e.value.details["reason"] == "bad_path"
