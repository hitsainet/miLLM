"""Tiny real models and adapter directories for Feature 33 tests.

`write_adapter` produces miForge's normative THREE-file layout (schema `adapter-manifest-v1.json`):
`adapter_model.safetensors`, `adapter_config.json` (PEFT's own) and `adapter_manifest.json`.

The fixtures disagree with the defects they guard against:
- projections are NON-SQUARE (hidden 48, intermediate 80, two KV heads so k/v project to 24), so a
  swapped A and B is a shape error, not a silent transpose;
- lora_B is NON-ZERO on every target, so the delta changes the output;
- `alpha != rank`, so a dropped or inverted scale is visible;
- the bad-shape and per-module-rank options change ONE dimension of ONE module only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import torch
from safetensors.torch import save_file
from torch import nn

RANK = 4
ALPHA = 6.0
BASE_NAME = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA_ENTRIES = ("q_proj", "v_proj", "down_proj")
LFM2_ENTRIES = ("q_proj", "w2")
CREATED_AT = "2026-10-14T12:00:00Z"
_UNSET: Any = object()


def tiny_llama(dtype: torch.dtype = torch.float32) -> nn.Module:
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=48,
        intermediate_size=80,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=12,
        max_position_embeddings=64,
    )
    return LlamaForCausalLM(config).to(dtype).eval()


def tiny_lfm2(dtype: torch.dtype = torch.float32) -> nn.Module:
    from transformers import Lfm2Config, Lfm2ForCausalLM

    torch.manual_seed(0)
    config = Lfm2Config(
        vocab_size=64,
        hidden_size=48,
        intermediate_size=80,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        layer_types=["conv", "full_attention"],
        block_auto_adjust_ff_dim=False,
        max_position_embeddings=64,
    )
    return Lfm2ForCausalLM(config).to(dtype).eval()


def peft_key(path: str, which: str) -> str:
    return f"base_model.model.{path}.lora_{which}.weight"


def target_paths(model: nn.Module, entries: Iterable[str]) -> list[str]:
    """Every linear module path ending in one of `entries` on a `.` boundary."""
    entries = tuple(entries)
    return sorted(
        n for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and any(n == e or n.endswith("." + e) for e in entries)
    )


def make_tensors(
    model: nn.Module,
    paths: Sequence[str],
    *,
    rank: int = RANK,
    seed: int = 1,
    shape_off_by_one: Optional[str] = None,
    per_module_rank: Optional[dict[str, int]] = None,
) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    tensors: dict[str, torch.Tensor] = {}
    for path in paths:
        module = model.get_submodule(path)
        r = (per_module_rank or {}).get(path, rank)
        out_f = module.out_features + (1 if path == shape_off_by_one else 0)
        tensors[peft_key(path, "A")] = torch.randn(r, module.in_features, generator=gen) * 0.2
        tensors[peft_key(path, "B")] = torch.randn(out_f, r, generator=gen) * 0.2
    return tensors


def peft_config(*, rank: int = RANK, alpha: float = ALPHA, entries: Sequence[str] = LLAMA_ENTRIES,
                **overrides: Any) -> dict:
    config = {
        "peft_type": "LORA",
        "base_model_name_or_path": BASE_NAME,
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "target_modules": list(entries),
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


def write_adapter(
    directory: Path,
    model: nn.Module,
    *,
    entries: Sequence[str] = LLAMA_ENTRIES,
    paths: Optional[Sequence[str]] = None,
    rank: int = RANK,
    alpha: float = ALPHA,
    seed: int = 1,
    base_name: str = BASE_NAME,
    base_hash: Optional[str] = None,
    scaling: Any = "alpha_over_r",
    corrupt_byte: bool = False,
    shape_off_by_one: Optional[str] = None,
    per_module_rank: Optional[dict[str, int]] = None,
    tensors: Optional[dict[str, torch.Tensor]] = None,
    manifest_overrides: Optional[dict] = None,
    config_overrides: Optional[dict] = None,
) -> Path:
    """Write the three files into `directory` and return it.

    `scaling=None` omits `adapter.scaling` (the schema default applies). `corrupt_byte` flips one
    byte of the tensor data AFTER the manifest's hash was taken. `manifest_overrides` is merged
    one level deep (`{"adapter": {...}}` updates the adapter object); a value of `_UNSET` removes
    the key.
    """
    directory.mkdir(parents=True, exist_ok=True)
    if paths is None:
        paths = target_paths(model, entries)
    if tensors is None:
        tensors = make_tensors(
            model, paths, rank=rank, seed=seed, shape_off_by_one=shape_off_by_one,
            per_module_rank=per_module_rank,
        )
    weights = directory / "adapter_model.safetensors"
    save_file(tensors, str(weights))
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    if corrupt_byte:
        data = bytearray(weights.read_bytes())
        data[-3] ^= 0x01
        weights.write_bytes(bytes(data))

    adapter: dict[str, Any] = {
        "rank": rank,
        "alpha": alpha,
        "target_modules": sorted(set(entries)),
        "dropout": 0.0,
    }
    if scaling is not None:
        adapter["scaling"] = scaling
    manifest: dict[str, Any] = {
        "kind": "miforge.adapter-manifest",
        "schema_version": "1",
        "name": f"{base_name}:r9:adapter-200",
        "run_id": "r9",
        "step": 200,
        "base": {"name": base_name, "manifest_sha256": base_hash},
        "adapter": adapter,
        "weights_sha256": digest,
        "created_at": CREATED_AT,
    }
    for key, value in (manifest_overrides or {}).items():
        if value is _UNSET:
            manifest.pop(key, None)
        elif isinstance(value, dict) and isinstance(manifest.get(key), dict):
            for k, v in value.items():
                if v is _UNSET:
                    manifest[key].pop(k, None)
                else:
                    manifest[key][k] = v
        else:
            manifest[key] = value
    (directory / "adapter_manifest.json").write_text(json.dumps(manifest, indent=2))

    config = peft_config(rank=rank, alpha=alpha, entries=entries)
    config.update(config_overrides or {})
    (directory / "adapter_config.json").write_text(json.dumps(config, indent=2))
    return directory


def unset() -> Any:
    """The sentinel `write_adapter` reads as 'remove this key'."""
    return _UNSET


def param_fingerprint(model: nn.Module) -> dict[str, tuple[int, str]]:
    """{name: (data_ptr, sha256 of the bytes)} for every parameter."""
    return {
        name: (
            p.data_ptr(),
            hashlib.sha256(p.detach().cpu().float().numpy().tobytes()).hexdigest(),
        )
        for name, p in model.named_parameters()
    }
