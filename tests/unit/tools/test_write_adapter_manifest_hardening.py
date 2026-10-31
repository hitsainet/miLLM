"""Review round 1 (R1-B) additions for `millm/tools/write_adapter_manifest.py`.

- PEFT LoRA *variants* whose forward is not `x + s·B·A·x` (QALoRA, activated LoRA, Arrow, …) were
  written a manifest and then served as plain LoRA, because only DoRA and rsLoRA were refused.
  `read_peft_config` is also the loader's config check, so these refusals hold at load too.
- `lora_alpha: Infinity` (Python's json accepts it) was written into a manifest as the non-JSON
  token `Infinity`.
- Without `--force`, an existence check followed by `os.replace` overwrote a manifest written in
  between. Publication is now `os.link`, which fails if the name exists.
- The tool and the loader share one RFC 3339 rule (`millm.services.rfc3339`).
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from millm.core.errors import AdapterManifestInvalidError
from millm.services import rfc3339
from millm.services.adapter_artifact import check_config_agrees
from millm.services.adapter_manifest import extract_fields
from millm.tools import write_adapter_manifest as tool
from millm.tools.write_adapter_manifest import ManifestToolError, read_peft_config
from tests.unit.tools.test_write_adapter_manifest import peft_config, run, write_peft_dir

VARIANTS = {
    "use_qalora": True,
    "use_bdlora": {"block": 2},
    "alora_invocation_tokens": [1, 2, 3],
    "arrow_config": {"top_k": 2},
    "velora_config": {"x": 1},
    "monteclora_config": {"x": 1},
    "kasa_config": {"x": 1},
    "layer_replication": [[0, 2]],
    "target_parameters": ["experts.gate_up_proj"],
    "trainable_token_indices": [5, 6],
}


@pytest.mark.parametrize("key,value", list(VARIANTS.items()))
def test_a_lora_variant_is_refused_by_the_tool(tmp_path, capsys, key, value):
    d = write_peft_dir(tmp_path / "a", config=peft_config(**{key: value}))
    assert run(d) == 1
    assert key in capsys.readouterr().err
    assert not (d / "adapter_manifest.json").exists()


@pytest.mark.parametrize("key,value", list(VARIANTS.items()))
def test_a_lora_variant_is_refused_by_the_loader_config_check(key, value):
    fields = extract_fields({
        "name": "n", "run_id": "r", "step": 0, "weights_sha256": "0" * 64,
        "base": {"name": "b"},
        "adapter": {"rank": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
    })
    with pytest.raises(AdapterManifestInvalidError) as e:
        check_config_agrees(peft_config(**{key: value}), fields)
    assert e.value.details["reason"] == "config_unservable"


@pytest.mark.parametrize("falsy", [None, False, [], {}])
def test_an_unset_variant_field_is_accepted(falsy):
    """A current PEFT writes every field, unset ones as null/false/empty."""
    config = peft_config(**{k: falsy for k in VARIANTS})
    assert read_peft_config(config)[0] == 4


@pytest.mark.parametrize("alpha", [float("inf"), float("nan")])
def test_a_non_finite_alpha_is_refused(tmp_path, alpha):
    with pytest.raises(ManifestToolError):
        read_peft_config(peft_config(lora_alpha=alpha))


def test_a_manifest_with_infinity_is_never_written(tmp_path):
    (tmp_path / "d").mkdir()
    with pytest.raises(ManifestToolError):
        tool.write_manifest(tmp_path / "d", {"adapter": {"alpha": float("inf")}}, force=False)
    assert os.listdir(tmp_path / "d") == []


def test_a_manifest_appearing_after_the_check_is_not_overwritten(tmp_path, monkeypatch):
    d = write_peft_dir(tmp_path / "a")
    target = d / "adapter_manifest.json"
    real_mkstemp = tempfile.mkstemp

    def mkstemp_then_race(*args, **kwargs):
        result = real_mkstemp(*args, **kwargs)
        target.write_text('{"written": "by someone else"}')
        return result

    monkeypatch.setattr(tool.tempfile, "mkstemp", mkstemp_then_race)
    with pytest.raises(ManifestToolError, match="already exists"):
        tool.write_manifest(d, {"k": 1}, force=False)
    assert json.loads(target.read_text()) == {"written": "by someone else"}
    assert sorted(p.name for p in d.iterdir() if p.name.startswith(".adapter_manifest")) == []


def _no_hard_links(monkeypatch):
    def refuse(*_a, **_k):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(tool.os, "link", refuse)


def test_a_volume_without_hard_links_still_gets_a_manifest(tmp_path, monkeypatch):
    """R2-C: `os.link` raises EPERM/ENOTSUP on volumes without hard links, and the tool died with
    a traceback, writing nothing. Control: drop the `except OSError` fallback -> PermissionError."""
    d = write_peft_dir(tmp_path / "a")
    _no_hard_links(monkeypatch)
    tool.write_manifest(d, {"k": 3}, force=False)
    assert json.loads((d / "adapter_manifest.json").read_text()) == {"k": 3}
    assert [p.name for p in d.iterdir() if p.name.startswith(".adapter_manifest")] == []


def test_without_hard_links_a_manifest_appearing_meanwhile_is_still_not_overwritten(
    tmp_path, monkeypatch
):
    d = write_peft_dir(tmp_path / "a")
    target = d / "adapter_manifest.json"
    real_mkstemp = tempfile.mkstemp

    def mkstemp_then_race(*args, **kwargs):
        result = real_mkstemp(*args, **kwargs)
        target.write_text('{"written": "by someone else"}')
        return result

    monkeypatch.setattr(tool.tempfile, "mkstemp", mkstemp_then_race)
    _no_hard_links(monkeypatch)
    with pytest.raises(ManifestToolError, match="already exists"):
        tool.write_manifest(d, {"k": 1}, force=False)
    assert json.loads(target.read_text()) == {"written": "by someone else"}
    assert [p.name for p in d.iterdir() if p.name.startswith(".adapter_manifest")] == []


def test_force_still_replaces_atomically(tmp_path):
    d = write_peft_dir(tmp_path / "a")
    (d / "adapter_manifest.json").write_text("{}")
    tool.write_manifest(d, {"k": 2}, force=True)
    assert json.loads((d / "adapter_manifest.json").read_text()) == {"k": 2}


def test_the_tool_uses_the_shared_rfc3339_rule():
    assert tool.is_rfc3339 is rfc3339.is_rfc3339
    assert tool.utc_now_rfc3339 is rfc3339.utc_now_rfc3339
    from millm.services import adapter_manifest
    assert not hasattr(adapter_manifest, "is_rfc3339") or (
        adapter_manifest.is_rfc3339 is rfc3339.is_rfc3339
    )
    assert not tool.is_rfc3339("2026-10-13T12:00:00")  # no offset
