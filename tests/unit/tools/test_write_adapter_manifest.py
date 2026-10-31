"""`python -m millm.tools.write_adapter_manifest` (addendum D-08).

The manifest it writes must validate against the VENDORED schema (not a test copy), must pin the
hash of the file actually on disk, and must never be written for a PEFT configuration whose served
behaviour schema v1 cannot express.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from millm.tools import write_adapter_manifest as tool
from millm.tools.write_adapter_manifest import ManifestToolError, build_manifest, main

RANK = 4
PATHS = ["model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.v_proj",
         "model.layers.1.self_attn.q_proj"]


def peft_config(**overrides) -> dict:
    config = {
        "peft_type": "LORA",
        "base_model_name_or_path": "meta-llama/Llama-3.1-8B-Instruct",
        "r": RANK,
        "lora_alpha": 8,
        "lora_dropout": 0.05,
        "target_modules": ["v_proj", "q_proj"],
        "bias": "none",
        "use_rslora": False,
        "use_dora": False,
        "fan_in_fan_out": False,
        "rank_pattern": {},
        "alpha_pattern": {},
        "modules_to_save": None,
    }
    config.update(overrides)
    return config


def write_peft_dir(directory: Path, *, config: dict | None = None, paths=PATHS, rank: int = RANK) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text(json.dumps(config or peft_config()))
    gen = torch.Generator().manual_seed(0)
    tensors = {}
    for path in paths:
        tensors[f"base_model.model.{path}.lora_A.weight"] = torch.randn(rank, 48, generator=gen)
        tensors[f"base_model.model.{path}.lora_B.weight"] = torch.randn(48, rank, generator=gen)
    save_file(tensors, str(directory / "adapter_model.safetensors"))
    return directory


def run(directory: Path, *extra: str) -> int:
    return main([str(directory), "--base-name", "meta-llama/Llama-3.1-8B-Instruct", *extra])


class TestItWritesASchemaValidManifest:
    def test_the_written_manifest_validates_against_the_vendored_schema(self, tmp_path):
        import jsonschema

        d = write_peft_dir(tmp_path / "run-a")
        assert run(d) == 0
        manifest = json.loads((d / "adapter_manifest.json").read_text())
        schema = json.loads(tool.VENDORED_SCHEMA.read_text())
        jsonschema.Draft202012Validator(schema).validate(manifest)
        assert manifest["kind"] == "miforge.adapter-manifest"
        assert manifest["schema_version"] == "1"

    def test_the_vendored_schema_is_the_one_in_docs(self):
        assert tool.VENDORED_SCHEMA == (
            Path(__file__).resolve().parents[3] / "docs" / "schemas" / "adapter-manifest-v1.json"
        )
        assert tool.VENDORED_SCHEMA.is_file()

    def test_the_fields_come_from_the_peft_config(self, tmp_path):
        d = write_peft_dir(tmp_path / "run-a")
        assert run(d, "--run-id", "r1", "--step", "500") == 0
        m = json.loads((d / "adapter_manifest.json").read_text())
        assert m["adapter"] == {"rank": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"],
                                "dropout": 0.05, "scaling": "alpha_over_r"}
        assert m["base"] == {"name": "meta-llama/Llama-3.1-8B-Instruct", "manifest_sha256": None}
        assert m["name"] == "meta-llama/Llama-3.1-8B-Instruct:r1:adapter-500"
        assert (m["run_id"], m["step"]) == ("r1", 500)

    def test_the_hash_is_the_files(self, tmp_path):
        d = write_peft_dir(tmp_path / "run-a")
        assert run(d) == 0
        m = json.loads((d / "adapter_manifest.json").read_text())
        want = hashlib.sha256((d / "adapter_model.safetensors").read_bytes()).hexdigest()
        assert m["weights_sha256"] == want

    def test_created_at_is_rfc3339_utc(self, tmp_path):
        d = write_peft_dir(tmp_path / "run-a")
        assert run(d) == 0
        created = json.loads((d / "adapter_manifest.json").read_text())["created_at"]
        assert created.endswith("Z") and tool.is_rfc3339(created)

    def test_base_manifest_sha256_and_name_are_passed_through(self, tmp_path):
        d = write_peft_dir(tmp_path / "run-a")
        h = "ab" * 32
        assert run(d, "--base-manifest-sha256", h, "--name", "custom:adapter-1") == 0
        m = json.loads((d / "adapter_manifest.json").read_text())
        assert m["base"]["manifest_sha256"] == h
        assert m["name"] == "custom:adapter-1"

    def test_run_id_defaults_to_the_directory_name(self, tmp_path):
        d = write_peft_dir(tmp_path / "my run!")
        assert run(d) == 0
        assert json.loads((d / "adapter_manifest.json").read_text())["run_id"] == "my-run"


class TestOverwrite:
    def test_an_existing_manifest_is_not_replaced(self, tmp_path, capsys):
        d = write_peft_dir(tmp_path / "run-a")
        (d / "adapter_manifest.json").write_text("{\"keep\": true}")
        assert run(d) == 1
        assert json.loads((d / "adapter_manifest.json").read_text()) == {"keep": True}
        assert "--force" in capsys.readouterr().err

    def test_force_replaces_it(self, tmp_path):
        d = write_peft_dir(tmp_path / "run-a")
        (d / "adapter_manifest.json").write_text("{\"keep\": true}")
        assert run(d, "--force") == 0
        assert "weights_sha256" in json.loads((d / "adapter_manifest.json").read_text())

    def test_a_refusal_writes_nothing(self, tmp_path):
        d = write_peft_dir(tmp_path / "run-a", config=peft_config(use_dora=True))
        assert run(d) == 1
        assert sorted(p.name for p in d.iterdir()) == ["adapter_config.json",
                                                       "adapter_model.safetensors"]


class TestRefusals:
    @pytest.mark.parametrize("overrides,needle", [
        ({"use_rslora": True}, "use_rslora"),
        ({"use_dora": True}, "use_dora"),
        ({"bias": "lora_only"}, "bias"),
        ({"bias": "all"}, "bias"),
        ({"lora_bias": True}, "lora_bias"),
        ({"rank_pattern": {"q_proj": 8}}, "rank_pattern"),
        ({"alpha_pattern": {"q_proj": 16}}, "alpha_pattern"),
        ({"fan_in_fan_out": True}, "fan_in_fan_out"),
        ({"modules_to_save": ["lm_head"]}, "modules_to_save"),
        ({"target_modules": ".*(q|v)_proj"}, "is the string .*a regex or 'all-linear'"),
        ({"target_modules": "all-linear"}, "is the string 'all-linear'"),
        ({"target_modules": []}, "target_modules"),
        ({"peft_type": "IA3"}, "peft_type"),
        ({"r": 0}, "r must"),
        ({"lora_alpha": 0}, "lora_alpha"),
    ])
    def test_unservable_configs_are_refused(self, tmp_path, overrides, needle):
        d = write_peft_dir(tmp_path / "run", config=peft_config(**overrides))
        with pytest.raises(ManifestToolError, match=needle):
            build_manifest(d, base_name="b", base_manifest_sha256=None, run_id=None, step=0, name=None)

    def test_a_rank_that_disagrees_with_r_is_refused(self, tmp_path):
        d = write_peft_dir(tmp_path / "run", rank=8)
        with pytest.raises(ManifestToolError, match="not rank 4"):
            build_manifest(d, base_name="b", base_manifest_sha256=None, run_id=None, step=0, name=None)

    def test_target_modules_that_disagree_with_the_keys_are_refused(self, tmp_path):
        d = write_peft_dir(tmp_path / "run", config=peft_config(target_modules=["q_proj", "k_proj"]))
        with pytest.raises(ManifestToolError) as e:
            build_manifest(d, base_name="b", base_manifest_sha256=None, run_id=None, step=0, name=None)
        assert "target_not_declared" in str(e.value) and "target_unused" in str(e.value)

    def test_a_non_peft_key_is_refused(self, tmp_path):
        d = write_peft_dir(tmp_path / "run")
        save_file({"model.layers.0.q_proj.weight": torch.zeros(2, 2)},
                  str(d / "adapter_model.safetensors"))
        with pytest.raises(ManifestToolError, match="not a PEFT LoRA key"):
            build_manifest(d, base_name="b", base_manifest_sha256=None, run_id=None, step=0, name=None)

    def test_missing_files_are_refused(self, tmp_path):
        d = write_peft_dir(tmp_path / "run")
        (d / "adapter_model.safetensors").unlink()
        with pytest.raises(ManifestToolError, match="not found"):
            build_manifest(d, base_name="b", base_manifest_sha256=None, run_id=None, step=0, name=None)

    def test_a_weights_symlink_out_of_the_directory_is_refused(self, tmp_path):
        d = write_peft_dir(tmp_path / "run")
        outside = tmp_path / "elsewhere.safetensors"
        (d / "adapter_model.safetensors").rename(outside)
        os.symlink(outside, d / "adapter_model.safetensors")
        with pytest.raises(ManifestToolError, match="outside"):
            build_manifest(d, base_name="b", base_manifest_sha256=None, run_id=None, step=0, name=None)

    def test_a_fifo_is_refused_rather_than_read(self, tmp_path):
        d = write_peft_dir(tmp_path / "run")
        (d / "adapter_model.safetensors").unlink()
        os.mkfifo(d / "adapter_model.safetensors")
        with pytest.raises(ManifestToolError, match="regular file"):
            build_manifest(d, base_name="b", base_manifest_sha256=None, run_id=None, step=0, name=None)

    @pytest.mark.parametrize("h", ["AB" * 32, "ab" * 31, "zz" * 32])
    def test_a_malformed_base_hash_is_refused(self, tmp_path, h):
        d = write_peft_dir(tmp_path / "run")
        with pytest.raises(ManifestToolError, match="64 lowercase hex"):
            build_manifest(d, base_name="b", base_manifest_sha256=h, run_id=None, step=0, name=None)

    def test_a_negative_step_is_refused(self, tmp_path):
        d = write_peft_dir(tmp_path / "run")
        with pytest.raises(ManifestToolError, match="--step"):
            build_manifest(d, base_name="b", base_manifest_sha256=None, run_id=None, step=-1, name=None)

    def test_a_missing_schema_is_refused_not_skipped(self, tmp_path):
        d = write_peft_dir(tmp_path / "run")
        assert run(d, "--schema", str(tmp_path / "nope.json")) == 1
        assert not (d / "adapter_manifest.json").exists()


class TestValidation:
    def _manifest(self, tmp_path) -> dict:
        d = write_peft_dir(tmp_path / "run")
        return build_manifest(d, base_name="b", base_manifest_sha256=None, run_id=None, step=0,
                              name=None)

    def test_a_valid_manifest_passes(self, tmp_path):
        tool.validate_manifest(self._manifest(tmp_path))

    @pytest.mark.parametrize("bad", ["not-a-date", "2026-13-01T00:00:00Z", "2026-10-14 05:00:00",
                                     "2026-10-14T05:00:00"])
    def test_date_time_is_actually_checked(self, tmp_path, bad):
        """jsonschema skips `format: date-time` without rfc3339-validator (addendum D-29)."""
        m = self._manifest(tmp_path)
        m["created_at"] = bad
        with pytest.raises(ManifestToolError, match="created_at"):
            tool.validate_manifest(m)

    def test_an_unknown_field_fails(self, tmp_path):
        m = self._manifest(tmp_path)
        m["extra"] = 1
        with pytest.raises(ManifestToolError):
            tool.validate_manifest(m)

    @pytest.mark.parametrize("value", ["2026-10-14T05:00:00Z", "2026-10-14T05:00:00.123+02:00",
                                       "2026-10-14t05:00:00z"])
    def test_rfc3339_accepts(self, value):
        assert tool.is_rfc3339(value)


def test_it_runs_as_a_module(tmp_path):
    import subprocess
    import sys

    d = write_peft_dir(tmp_path / "run")
    proc = subprocess.run(
        [sys.executable, "-m", "millm.tools.write_adapter_manifest", str(d), "--base-name", "b"],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parents[3], timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert (d / "adapter_manifest.json").is_file()
