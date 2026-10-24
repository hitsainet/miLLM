"""The adapter artifact reader (`millm/services/adapter_artifact.py`) on tiny REAL models.

Every adapter is written to miForge's normative three-file layout and validated against the REAL
vendored schema. The rules pinned here, each with a mutation control:

- confinement: the directory AND each file resolve inside a checkpoint root; regular files only
  (a FIFO is refused without hanging);
- one read of the weights: bytes swapped on disk after the hash cannot reach the tensors;
- the digest is compared, and no tensor is materialised before it matches;
- `adapter_config.json` must agree with the manifest;
- the base matches by manifest hash when the resident has one, else by name or repo id (D-09);
- every `adapter_hooks` reason maps to its public code.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import typing
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from millm.core.errors import (
    AdapterBaseMismatchError,
    AdapterHashMismatchError,
    AdapterInstallFailedError,
    AdapterInvalidError,
    AdapterManifestInvalidError,
    AdapterSchemaNotVendoredError,
    AdapterShapeMismatchError,
)
from millm.ml import adapter_hooks
from millm.ml.adapter_hooks import AdapterHookError, AdapterPlanError, LoraHookSet
from millm.services import adapter_artifact, adapter_manifest
from millm.services.adapter_artifact import (
    PLAN_REASON_CODES,
    ResidentBase,
    adapter_roots,
    base_matches,
    local_manifest_sha256,
    map_hook_error,
    prepare_adapter,
    require_base_match,
    resident_base,
    resolve_adapter_dir,
)
from millm.services.adapter_manifest import vendored_validator
from tests.unit.adapter_fixtures import (
    ALPHA,
    BASE_NAME,
    LFM2_ENTRIES,
    LLAMA_ENTRIES,
    RANK,
    make_tensors,
    param_fingerprint,
    peft_key,
    target_paths,
    tiny_lfm2,
    tiny_llama,
    unset,
    write_adapter,
)

HUB = ResidentBase(manifest_sha256=None, names=frozenset({"Llama-3.1-8B-Instruct", BASE_NAME}))
H1 = "1" * 64
H2 = "2" * 64


@pytest.fixture(scope="module")
def llama():
    return tiny_llama()


@pytest.fixture
def root(tmp_path) -> Path:
    r = tmp_path / "checkpoints"
    r.mkdir()
    return r


def roots_of(*paths: Path) -> list[tuple[str, str]]:
    return [("primary", str(paths[0]))] + [(f"read:{i}", str(p)) for i, p in enumerate(paths[1:])]


def prepare(root: Path, rel: str, model, resident: ResidentBase = HUB, **kw):
    return prepare_adapter(rel, model, resident, roots=roots_of(root), **kw)


def ids() -> torch.Tensor:
    return torch.tensor([[1, 5, 9, 2, 33, 17, 4, 60]])


@pytest.fixture
def spies(monkeypatch):
    """Call counts of the two operations that materialise tensors."""
    calls = {"build": 0, "load": 0}
    real_build = LoraHookSet.build
    import safetensors.torch as st

    real_load = st.load

    def build(*a, **k):
        calls["build"] += 1
        return real_build(*a, **k)

    def load(*a, **k):
        calls["load"] += 1
        return real_load(*a, **k)

    monkeypatch.setattr(LoraHookSet, "build", staticmethod(build))
    monkeypatch.setattr(st, "load", load)
    return calls


# ---------------------------------------------------------------------------------------------
# The good path
# ---------------------------------------------------------------------------------------------


class TestAGoodAdapter:
    def test_it_prepares_every_target_in_fp32_on_cpu(self, root, llama):
        write_adapter(root / "r9" / "adapter-200", llama)
        art = prepare(root, "r9/adapter-200", llama)
        paths = target_paths(llama, LLAMA_ENTRIES)
        assert sorted(t.module_path for t in art.prepared.targets) == paths
        assert art.prepared.scaling == ALPHA / RANK
        for a, b in art.prepared.weights.values():
            assert a.dtype == b.dtype == torch.float32
            assert a.device.type == b.device.type == "cpu"
        assert art.root_relative == "r9/adapter-200"
        assert art.location.root_label == "primary"
        weights = (root / "r9/adapter-200/adapter_model.safetensors").read_bytes()
        manifest = (root / "r9/adapter-200/adapter_manifest.json").read_bytes()
        assert art.weights_sha256 == hashlib.sha256(weights).hexdigest()
        assert art.manifest_sha256 == hashlib.sha256(manifest).hexdigest()
        assert art.fields.name == f"{BASE_NAME}:r9:adapter-200"

    def test_the_prepared_tensors_are_the_files(self, root, llama):
        write_adapter(root / "a", llama, seed=7)
        art = prepare(root, "a", llama)
        expected = make_tensors(llama, target_paths(llama, LLAMA_ENTRIES), seed=7)
        for t in art.prepared.targets:
            a, b = art.prepared.weights[t.module_path]
            assert torch.equal(a, expected[t.a_key])
            assert torch.equal(b, expected[t.b_key])

    def test_preparing_touches_no_parameter_and_the_hooks_install(self, root, llama):
        write_adapter(root / "a", llama)
        before = param_fingerprint(llama)
        with torch.no_grad():
            base = llama(ids()).logits.clone()
        art = prepare(root, "a", llama)
        assert param_fingerprint(llama) == before
        handles = LoraHookSet.install(llama, art.prepared)
        try:
            with torch.no_grad():
                adapted = llama(ids()).logits
            assert not torch.allclose(adapted, base)
        finally:
            LoraHookSet.remove(handles)
        with torch.no_grad():
            assert torch.equal(llama(ids()).logits, base)

    def test_lfm2(self, root):
        model = tiny_lfm2()
        write_adapter(root / "a", model, entries=LFM2_ENTRIES)
        art = prepare(root, "a", model)
        assert {t.module_path for t in art.prepared.targets} == set(
            target_paths(model, LFM2_ENTRIES)
        )

    def test_bf16_model_keeps_fp32_adapter_tensors(self, root):
        model = tiny_llama(torch.bfloat16)
        write_adapter(root / "a", model)
        art = prepare(root, "a", model)
        assert all(a.dtype == torch.float32 for a, _ in art.prepared.weights.values())

    def test_absent_scaling_takes_the_default(self, root, llama):
        write_adapter(root / "a", llama, scaling=None)
        manifest = json.loads((root / "a/adapter_manifest.json").read_text())
        assert "scaling" not in manifest["adapter"]
        assert prepare(root, "a", llama).prepared.scaling == ALPHA / RANK

    def test_target_modules_in_another_order_agree(self, root, llama):
        write_adapter(root / "a", llama,
                      config_overrides={"target_modules": list(reversed(LLAMA_ENTRIES))})
        prepare(root, "a", llama)


class TestAManifestWrittenByTheToolLoads:
    """FTASKS 9.6: the published schema, a manifest written by `write_adapter_manifest`."""

    def test_the_tool_written_manifest_prepares(self, root, llama):
        from millm.tools.write_adapter_manifest import (
            build_manifest,
            validate_manifest,
            write_manifest,
        )

        d = write_adapter(root / "a", llama)
        (d / "adapter_manifest.json").unlink()
        m = build_manifest(d, base_name=BASE_NAME, base_manifest_sha256=None, run_id="r9",
                           step=200, name=None)
        validate_manifest(m)
        write_manifest(d, m, force=False)
        art = prepare(root, "a", llama, validator_factory=vendored_validator)
        assert art.fields.base_name == BASE_NAME

    def test_the_fixture_manifest_is_schema_valid(self, root, llama):
        from millm.tools.write_adapter_manifest import validate_manifest

        d = write_adapter(root / "a", llama)
        validate_manifest(json.loads((d / "adapter_manifest.json").read_text()))


# ---------------------------------------------------------------------------------------------
# Confinement and regular files (D-11)
# ---------------------------------------------------------------------------------------------


def reason_of(exc) -> str:
    return exc.details.get("reason")


class TestConfinement:
    @pytest.mark.parametrize(
        "rel", ["/etc", "../x", "a/../../x", "a/../a", "", ".", "a\x00b", "a\\b"]
    )
    def test_bad_paths_are_refused(self, root, rel, llama):
        with pytest.raises(AdapterInvalidError) as e:
            resolve_adapter_dir(rel, roots_of(root))
        assert reason_of(e.value) == "bad_path"

    def test_a_directory_symlink_out_of_the_root_is_refused(self, root, tmp_path, llama):
        outside = write_adapter(tmp_path / "outside", llama)
        (root / "link").symlink_to(outside)
        with pytest.raises(AdapterInvalidError) as e:
            prepare(root, "link", llama)
        assert reason_of(e.value) == "outside_root"

    def test_a_weights_symlink_out_of_the_directory_is_refused(self, root, tmp_path, llama):
        d = write_adapter(root / "a", llama)
        elsewhere = tmp_path / "elsewhere.safetensors"
        (d / "adapter_model.safetensors").rename(elsewhere)
        (d / "adapter_model.safetensors").symlink_to(elsewhere)
        with pytest.raises(AdapterInvalidError) as e:
            prepare(root, "a", llama)
        assert reason_of(e.value) == "outside_root"
        assert e.value.details["file"] == "adapter_model.safetensors"

    @pytest.mark.parametrize("name", ["adapter_manifest.json", "adapter_config.json",
                                      "adapter_model.safetensors"])
    def test_each_missing_file_is_refused(self, root, llama, name):
        d = write_adapter(root / "a", llama)
        (d / name).unlink()
        with pytest.raises(AdapterInvalidError) as e:
            prepare(root, "a", llama)
        assert reason_of(e.value) == "missing_file"
        assert e.value.details["file"] == name

    def test_a_missing_directory_is_refused(self, root, llama):
        with pytest.raises(AdapterInvalidError) as e:
            prepare(root, "nope", llama)
        assert reason_of(e.value) == "not_found"

    def test_a_fifo_is_refused_without_hanging(self, root, llama):
        d = write_adapter(root / "a", llama)
        (d / "adapter_model.safetensors").unlink()
        os.mkfifo(d / "adapter_model.safetensors")
        result: dict = {}

        def run():
            try:
                prepare(root, "a", llama)
            except Exception as e:  # noqa: BLE001 - captured for the assertion
                result["e"] = e

        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(timeout=10)
        assert not t.is_alive(), "reading a FIFO hung the worker"
        assert isinstance(result.get("e"), AdapterInvalidError)
        assert reason_of(result["e"]) == "not_regular_file"

    def test_a_directory_in_place_of_a_file_is_refused(self, root, llama):
        d = write_adapter(root / "a", llama)
        (d / "adapter_config.json").unlink()
        (d / "adapter_config.json").mkdir()
        with pytest.raises(AdapterInvalidError) as e:
            prepare(root, "a", llama)
        assert reason_of(e.value) == "not_regular_file"

    def test_error_bodies_name_only_root_relative_paths(self, root, tmp_path, llama):
        d = write_adapter(root / "r9" / "a", llama)
        (d / "adapter_model.safetensors").unlink()
        with pytest.raises(AdapterInvalidError) as e:
            prepare(root, "r9/a", llama)
        body = json.dumps(e.value.details) + e.value.message
        assert str(tmp_path) not in body
        assert "r9/a" in body

    def test_a_read_root_is_searched_and_labelled(self, root, tmp_path, llama):
        read = tmp_path / "miforge"
        write_adapter(read / "a", llama)
        art = prepare_adapter("a", llama, HUB, roots=roots_of(root, read))
        assert art.location.root_label == "read:0"

    def test_the_same_path_in_two_roots_is_ambiguous(self, root, tmp_path, llama):
        read = tmp_path / "miforge"
        write_adapter(root / "a", llama)
        write_adapter(read / "a", llama)
        with pytest.raises(AdapterInvalidError) as e:
            prepare_adapter("a", llama, HUB, roots=roots_of(root, read))
        assert reason_of(e.value) == "ambiguous"

    def test_the_roots_come_from_the_settings(self, monkeypatch, tmp_path):
        from millm.core.config import settings

        monkeypatch.setattr(settings, "WEIGHTS_CHECKPOINT_DIR", "/p")
        monkeypatch.setattr(settings, "WEIGHTS_CHECKPOINT_READ_ROOTS", "/r1, /r2")
        assert adapter_roots() == [("primary", "/p"), ("read:0", "/r1"), ("read:1", "/r2")]

    def test_the_default_roots_are_used_when_none_are_passed(self, monkeypatch, root, llama):
        from millm.core.config import settings

        write_adapter(root / "a", llama)
        monkeypatch.setattr(settings, "WEIGHTS_CHECKPOINT_DIR", str(root))
        monkeypatch.setattr(settings, "WEIGHTS_CHECKPOINT_READ_ROOTS", "")
        assert prepare_adapter("a", llama, HUB).root_relative == "a"


# ---------------------------------------------------------------------------------------------
# One read, the digest, and nothing materialised first (D-11, FTASKS 2.4)
# ---------------------------------------------------------------------------------------------


class TestTheHash:
    def test_a_one_byte_corruption_refuses_before_any_tensor_exists(self, root, llama, spies):
        write_adapter(root / "a", llama, corrupt_byte=True)
        with pytest.raises(AdapterHashMismatchError) as e:
            prepare(root, "a", llama)
        actual = hashlib.sha256((root / "a/adapter_model.safetensors").read_bytes()).hexdigest()
        assert e.value.details["actual"] == actual
        assert e.value.details["expected"] != actual
        assert spies == {"build": 0, "load": 0}

    def test_bytes_swapped_after_the_hash_cannot_reach_the_tensors(self, root, llama, monkeypatch):
        """The file is replaced on disk the moment its bytes are hashed. A loader that reopens
        the file would load the swapped tensors (whose hash never matched); one read cannot."""
        d = write_adapter(root / "a", llama, seed=1)
        weights = d / "adapter_model.safetensors"
        original = weights.read_bytes()
        swapped = make_tensors(llama, target_paths(llama, LLAMA_ENTRIES), seed=2)
        real = adapter_artifact._sha256_bytes

        def hash_then_swap(data: bytes) -> str:
            if data == original:
                save_file(swapped, str(weights))
            return real(data)

        monkeypatch.setattr(adapter_artifact, "_sha256_bytes", hash_then_swap)
        art = prepare(root, "a", llama)
        assert weights.read_bytes() != original
        expected = make_tensors(llama, target_paths(llama, LLAMA_ENTRIES), seed=1)
        for t in art.prepared.targets:
            assert torch.equal(art.prepared.weights[t.module_path][0], expected[t.a_key])
            assert torch.equal(art.prepared.weights[t.module_path][1], expected[t.b_key])

    def test_the_weights_file_is_opened_once(self, root, llama, monkeypatch):
        d = write_adapter(root / "a", llama)
        target = os.path.realpath(d / "adapter_model.safetensors")
        opened: list[str] = []
        real_open = os.open

        def counting_open(path, *a, **k):
            if os.path.realpath(path) == target:
                opened.append(path)
            return real_open(path, *a, **k)

        monkeypatch.setattr(os, "open", counting_open)
        prepare(root, "a", llama)
        assert len(opened) == 1

    def test_a_file_above_the_shape_bound_is_refused_before_reading(
        self, root, llama, spies, monkeypatch
    ):
        write_adapter(root / "a", llama)
        monkeypatch.setattr(adapter_artifact, "MAX_HEADER_BYTES", 0)
        monkeypatch.setattr(adapter_artifact, "MAX_ELEMENT_BYTES", 1)
        hashed = []
        monkeypatch.setattr(adapter_artifact, "_sha256_bytes", lambda d: hashed.append(d) or "x")
        with pytest.raises(AdapterInvalidError) as e:
            prepare(root, "a", llama)
        assert reason_of(e.value) == "too_large"
        assert e.value.details["file"] == "adapter_model.safetensors"
        assert hashed == [] and spies == {"build": 0, "load": 0}

    def test_the_size_cap_is_checked_from_fstat_before_any_byte_is_read(
        self, root, llama, monkeypatch
    ):
        d = write_adapter(root / "a", llama)
        size = (d / "adapter_model.safetensors").stat().st_size
        monkeypatch.setattr(adapter_artifact, "MAX_HEADER_BYTES", 0)
        monkeypatch.setattr(adapter_artifact, "MAX_ELEMENT_BYTES", 1)
        target = os.path.realpath(d / "adapter_model.safetensors")
        weights_fds: set[int] = set()
        reads: list[int] = []
        real_open, real_read = os.open, os.read

        def track_open(path, *a, **k):
            fd = real_open(path, *a, **k)
            if os.path.realpath(path) == target:
                weights_fds.add(fd)
            return fd

        def track_read(fd, n):
            if fd in weights_fds:
                reads.append(n)
            return real_read(fd, n)

        monkeypatch.setattr(os, "open", track_open)
        monkeypatch.setattr(os, "read", track_read)
        with pytest.raises(AdapterInvalidError) as e:
            prepare(root, "a", llama)
        assert e.value.details["size"] == size
        assert reads == []

    def test_a_file_that_grows_past_the_bound_while_read_is_refused(
        self, root, llama, monkeypatch
    ):
        """fstat said small; the read itself still stops at the bound."""
        d = write_adapter(root / "a", llama)
        monkeypatch.setattr(adapter_artifact, "MAX_HEADER_BYTES", 0)
        monkeypatch.setattr(adapter_artifact, "MAX_ELEMENT_BYTES", 1)
        target = os.path.realpath(d / "adapter_model.safetensors")
        weights_fds: set[int] = set()
        real_open, real_fstat = os.open, os.fstat

        def track_open(path, *a, **k):
            fd = real_open(path, *a, **k)
            if os.path.realpath(path) == target:
                weights_fds.add(fd)
            return fd

        def small_fstat(fd):
            st = real_fstat(fd)
            if fd in weights_fds:
                fields = list(st)
                fields[6] = 0  # st_size
                return os.stat_result(fields)
            return st

        monkeypatch.setattr(os, "open", track_open)
        monkeypatch.setattr(os, "fstat", small_fstat)
        with pytest.raises(AdapterInvalidError) as e:
            prepare(root, "a", llama)
        assert reason_of(e.value) == "too_large"
        assert "size" not in e.value.details

    def test_target_modules_matching_nothing_refuse_before_reading(self, root, llama):
        write_adapter(root / "a", llama, entries=("nothing_here",),
                      paths=target_paths(llama, ("q_proj",)))
        with pytest.raises(AdapterShapeMismatchError) as e:
            prepare(root, "a", llama)
        assert reason_of(e.value) == "target_unused"

    def test_a_truncated_safetensors_file_is_invalid(self, root, llama):
        d = write_adapter(root / "a", llama)
        weights = d / "adapter_model.safetensors"
        data = weights.read_bytes()[:-10]
        weights.write_bytes(data)
        m = json.loads((d / "adapter_manifest.json").read_text())
        m["weights_sha256"] = hashlib.sha256(data).hexdigest()
        (d / "adapter_manifest.json").write_text(json.dumps(m))
        with pytest.raises(AdapterInvalidError) as e:
            prepare(root, "a", llama)
        assert reason_of(e.value) == "unreadable"


# ---------------------------------------------------------------------------------------------
# Manifest, schema and config
# ---------------------------------------------------------------------------------------------


class TestManifestAndConfig:
    def test_an_absent_schema_refuses_through_the_default_factory(self, root, llama, monkeypatch,
                                                                  tmp_path):
        write_adapter(root / "a", llama)
        monkeypatch.setattr(adapter_manifest, "VENDORED", tmp_path / "absent.json")
        with pytest.raises(AdapterSchemaNotVendoredError):
            prepare(root, "a", llama)

    def test_an_unknown_key_is_refused(self, root, llama):
        write_adapter(root / "a", llama, manifest_overrides={"surprise": True})
        with pytest.raises(AdapterManifestInvalidError) as e:
            prepare(root, "a", llama)
        assert [p["path"] for p in e.value.details["errors"]] == ["$.surprise"]

    def test_an_unknown_scaling_is_refused(self, root, llama):
        write_adapter(root / "a", llama, scaling="rslora")
        with pytest.raises(AdapterManifestInvalidError) as e:
            prepare(root, "a", llama)
        assert "$.adapter.scaling" in [p["path"] for p in e.value.details["errors"]]

    def test_a_missing_required_field_is_refused(self, root, llama):
        write_adapter(root / "a", llama, manifest_overrides={"weights_sha256": unset()})
        with pytest.raises(AdapterManifestInvalidError):
            prepare(root, "a", llama)

    def test_a_manifest_that_is_not_json_is_refused(self, root, llama):
        d = write_adapter(root / "a", llama)
        (d / "adapter_manifest.json").write_text("{nope")
        with pytest.raises(AdapterManifestInvalidError):
            prepare(root, "a", llama)

    @pytest.mark.parametrize(
        "override,field",
        [
            ({"r": RANK * 2}, "r"),
            ({"lora_alpha": ALPHA + 1}, "lora_alpha"),
            ({"target_modules": ["q_proj", "v_proj"]}, "target_modules"),
        ],
    )
    def test_a_config_that_disagrees_with_the_manifest_is_refused(self, root, llama, override,
                                                                 field, spies):
        write_adapter(root / "a", llama, config_overrides=override)
        with pytest.raises(AdapterManifestInvalidError) as e:
            prepare(root, "a", llama)
        assert reason_of(e.value) == "config_disagrees"
        assert e.value.details["disagreements"][0]["field"] == field
        assert spies == {"build": 0, "load": 0}

    def test_an_unservable_config_is_refused(self, root, llama):
        write_adapter(root / "a", llama, config_overrides={"use_rslora": True})
        with pytest.raises(AdapterManifestInvalidError) as e:
            prepare(root, "a", llama)
        assert reason_of(e.value) == "config_unservable"

    def test_a_config_that_is_not_json_is_refused(self, root, llama):
        d = write_adapter(root / "a", llama)
        (d / "adapter_config.json").write_text("[")
        with pytest.raises(AdapterManifestInvalidError):
            prepare(root, "a", llama)


# ---------------------------------------------------------------------------------------------
# Shapes (FTASKS 2.5, acceptance 16)
# ---------------------------------------------------------------------------------------------


class TestShapes:
    def test_one_dimension_of_one_module_off_is_refused(self, root, llama, spies):
        bad = "model.layers.1.self_attn.v_proj"
        write_adapter(root / "a", llama, shape_off_by_one=bad)
        with pytest.raises(AdapterShapeMismatchError) as e:
            prepare(root, "a", llama)
        assert reason_of(e.value) == "shape"
        first = e.value.details["first"]
        assert first["module"] == bad
        assert "(24, 4)" in first["detail"] and "(25, 4)" in first["detail"]
        assert len(e.value.details["offenders"]) == 1
        assert spies["load"] == 0 and spies["build"] == 0

    def test_a_different_rank_on_one_module_is_refused(self, root, llama):
        """D-10: one alpha/rank scale, so another rank would get an unrecorded scale."""
        bad = "model.layers.0.mlp.down_proj"
        write_adapter(root / "a", llama, per_module_rank={bad: RANK + 2})
        with pytest.raises(AdapterShapeMismatchError) as e:
            prepare(root, "a", llama)
        assert reason_of(e.value) == "rank"
        assert e.value.details["first"]["module"] == bad

    def test_swapped_a_and_b_are_refused(self, root, llama):
        paths = target_paths(llama, LLAMA_ENTRIES)
        tensors = make_tensors(llama, paths)
        p = "model.layers.0.self_attn.v_proj"
        a, b = tensors[peft_key(p, "A")], tensors[peft_key(p, "B")]
        tensors[peft_key(p, "A")], tensors[peft_key(p, "B")] = b, a
        write_adapter(root / "a", llama, tensors=tensors)
        with pytest.raises(AdapterShapeMismatchError) as e:
            prepare(root, "a", llama)
        assert e.value.details["first"]["module"] == p

    def test_a_missing_module_is_refused(self, root, llama):
        paths = target_paths(llama, LLAMA_ENTRIES)
        tensors = make_tensors(llama, paths)
        tensors[peft_key("model.layers.7.self_attn.q_proj", "A")] = torch.randn(RANK, 48)
        tensors[peft_key("model.layers.7.self_attn.q_proj", "B")] = torch.randn(48, RANK)
        write_adapter(root / "a", llama, tensors=tensors)
        with pytest.raises(AdapterShapeMismatchError) as e:
            prepare(root, "a", llama)
        assert reason_of(e.value) == "missing_module"

    def test_a_non_linear_module_is_refused(self, root, llama):
        paths = target_paths(llama, LLAMA_ENTRIES)
        tensors = make_tensors(llama, paths)
        tensors[peft_key("model.embed_tokens", "A")] = torch.randn(RANK, 64)
        tensors[peft_key("model.embed_tokens", "B")] = torch.randn(48, RANK)
        write_adapter(root / "a", llama, tensors=tensors,
                      entries=LLAMA_ENTRIES + ("embed_tokens",))
        with pytest.raises(AdapterShapeMismatchError) as e:
            prepare(root, "a", llama)
        assert reason_of(e.value) == "unsupported_module"

    def test_an_incomplete_pair_is_refused(self, root, llama):
        paths = target_paths(llama, LLAMA_ENTRIES)
        tensors = make_tensors(llama, paths)
        del tensors[peft_key(paths[0], "B")]
        write_adapter(root / "a", llama, tensors=tensors)
        with pytest.raises(AdapterShapeMismatchError) as e:
            prepare(root, "a", llama)
        assert reason_of(e.value) == "incomplete_pair"

    def test_an_unparseable_key_is_refused_not_skipped(self, root, llama):
        paths = target_paths(llama, LLAMA_ENTRIES)
        tensors = make_tensors(llama, paths)
        tensors["something.else.weight"] = torch.zeros(2, 2)
        write_adapter(root / "a", llama, tensors=tensors)
        with pytest.raises(AdapterInvalidError) as e:
            prepare(root, "a", llama)
        assert reason_of(e.value) == "unsupported_tensor"

    def test_every_offender_is_listed(self, root, llama):
        write_adapter(root / "a", llama,
                      per_module_rank={"model.layers.0.mlp.down_proj": RANK + 1},
                      shape_off_by_one="model.layers.1.self_attn.v_proj")
        with pytest.raises(AdapterShapeMismatchError) as e:
            prepare(root, "a", llama)
        modules = {o.get("module") for o in e.value.details["offenders"]}
        assert {"model.layers.0.mlp.down_proj", "model.layers.1.self_attn.v_proj"} <= modules

    def test_a_non_finite_tensor_is_invalid(self, root, llama):
        paths = target_paths(llama, LLAMA_ENTRIES)
        tensors = make_tensors(llama, paths)
        tensors[peft_key(paths[0], "B")][0, 0] = float("nan")
        write_adapter(root / "a", llama, tensors=tensors)
        with pytest.raises(AdapterInvalidError) as e:
            prepare(root, "a", llama)
        assert reason_of(e.value) == "non_finite"


class TestReasonMapping:
    EXPECTED = {
        **{r: AdapterShapeMismatchError for r in (
            "rank", "shape", "missing_module", "unsupported_module", "fused_by_transformers",
            "incomplete_pair", "duplicate_target", "invalid_target_entry",
            "target_not_declared", "target_unused")},
        **{r: AdapterInvalidError for r in (
            "unsupported_tensor", "scaling", "dtype", "non_finite", "empty")},
        "install_failed": AdapterInstallFailedError,
    }

    def test_every_hook_reason_is_mapped_explicitly(self):
        assert set(PLAN_REASON_CODES) == set(typing.get_args(adapter_hooks.Reason))

    @pytest.mark.parametrize("reason", sorted(EXPECTED))
    def test_each_reason_maps_to_its_code(self, reason):
        err = map_hook_error(AdapterPlanError(reason, "m", offenders=({"reason": reason},)))
        assert type(err) is self.EXPECTED[reason]
        assert err.details["reason"] == reason
        assert err.details["offenders"] == [{"reason": reason}]

    def test_an_unknown_reason_is_invalid_not_dropped(self):
        err = map_hook_error(AdapterHookError("brand_new", "m"))
        assert isinstance(err, AdapterInvalidError)


# ---------------------------------------------------------------------------------------------
# Base identity (D-09, FTASKS 2.6)
# ---------------------------------------------------------------------------------------------


class TestBaseMatches:
    def test_a_hub_base_matches_by_repo_id(self):
        assert base_matches({"name": BASE_NAME, "manifest_sha256": None}, HUB)

    def test_a_hub_base_matches_by_row_name(self):
        assert base_matches({"name": "Llama-3.1-8B-Instruct"}, HUB)

    def test_another_name_is_refused(self):
        assert not base_matches({"name": "google/gemma-2-2b", "manifest_sha256": None}, HUB)

    def test_a_checkpoint_base_matches_by_hash(self):
        resident = ResidentBase(manifest_sha256=H1, names=frozenset({"x"}))
        assert base_matches({"name": "anything", "manifest_sha256": H1}, resident)
        assert base_matches({"name": "anything", "manifest_sha256": H1.upper()}, resident)

    def test_same_name_different_hash_is_refused(self):
        resident = ResidentBase(manifest_sha256=H1, names=frozenset({"ck"}))
        assert not base_matches({"name": "ck", "manifest_sha256": H2}, resident)

    def test_a_null_claim_does_not_match_a_hashed_resident(self):
        resident = ResidentBase(manifest_sha256=H1, names=frozenset({"ck"}))
        assert not base_matches({"name": "ck", "manifest_sha256": None}, resident)
        assert not base_matches({"name": "ck"}, resident)

    def test_a_hash_claim_does_not_match_a_hashless_resident(self):
        resident = ResidentBase(manifest_sha256=None, names=frozenset({"ck"}))
        assert not base_matches({"name": "ck", "manifest_sha256": H1}, resident)

    def test_none_and_none_match_by_name(self):
        resident = ResidentBase(manifest_sha256=None, names=frozenset({"ck"}))
        assert base_matches({"name": "ck", "manifest_sha256": None}, resident)

    def test_the_refusal_names_both_sides(self):
        resident = ResidentBase(manifest_sha256=H1, names=frozenset({"ck"}))
        with pytest.raises(AdapterBaseMismatchError) as e:
            require_base_match({"name": "ck", "manifest_sha256": H2}, resident)
        d = e.value.details
        assert d["rule"] == "manifest_sha256"
        assert d["adapter_base"] == {"name": "ck", "manifest_sha256": H2}
        assert d["resident"] == {"names": ["ck"], "manifest_sha256": H1}


class TestResidentBase:
    def test_a_hub_row(self):
        row = SimpleNamespace(name="Llama-3.1-8B-Instruct", repo_id=BASE_NAME,
                              source="huggingface", local_path=None)
        resident = resident_base(row)
        assert (resident.manifest_sha256, resident.names) == (HUB.manifest_sha256, HUB.names)
        assert resident.served_names == HUB.names  # no loaded model: the row's names only

    def test_a_loaded_identity_supplies_the_hash(self):
        row = SimpleNamespace(name="n", repo_id=None, source="huggingface", local_path=None)
        loaded = SimpleNamespace(weights_identity=SimpleNamespace(manifest_sha256=H1))
        assert resident_base(row, loaded).manifest_sha256 == H1

    def test_a_local_row_hashes_its_manifest(self, tmp_path):
        from millm.db.models.model import ModelSource

        (tmp_path / "manifest.json").write_bytes(b'{"name": "ck"}')
        row = SimpleNamespace(name="ck", repo_id=None, source=ModelSource.LOCAL,
                              local_path=str(tmp_path))
        rb = resident_base(row)
        assert rb.manifest_sha256 == hashlib.sha256(b'{"name": "ck"}').hexdigest()
        assert rb.names == frozenset({"ck"})

    def test_a_local_row_without_a_manifest_has_no_hash(self, tmp_path):
        row = SimpleNamespace(name="ck", repo_id=None, source="local", local_path=str(tmp_path))
        assert resident_base(row).manifest_sha256 is None

    def test_a_local_manifest_fifo_does_not_hang(self, tmp_path):
        os.mkfifo(tmp_path / "manifest.json")
        assert local_manifest_sha256(str(tmp_path)) is None

    def test_an_hf_row_never_reads_a_local_manifest(self, tmp_path):
        (tmp_path / "manifest.json").write_text("{}")
        row = SimpleNamespace(name="n", repo_id="o/n", source="huggingface",
                              local_path=str(tmp_path))
        assert resident_base(row).manifest_sha256 is None


class TestBaseInPrepare:
    def test_a_wrong_base_is_refused_before_any_tensor(self, root, llama, spies):
        write_adapter(root / "a", llama, base_name="google/gemma-2-2b")
        with pytest.raises(AdapterBaseMismatchError):
            prepare(root, "a", llama)
        assert spies == {"build": 0, "load": 0}

    def test_a_checkpoint_base_by_hash(self, root, llama):
        write_adapter(root / "a", llama, base_name="ck:r7:step-500", base_hash=H1)
        resident = ResidentBase(manifest_sha256=H1, names=frozenset({"ck"}))
        prepare(root, "a", llama, resident)

    def test_a_hashed_claim_against_a_hub_resident_is_refused(self, root, llama):
        write_adapter(root / "a", llama, base_hash=H1)
        with pytest.raises(AdapterBaseMismatchError):
            prepare(root, "a", llama)
