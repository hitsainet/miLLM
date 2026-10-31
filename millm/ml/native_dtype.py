"""The dtype a model row loads at: the checkpoint's own precision, by the rule miStudio shares.

⚠ WHY THIS EXISTS. This server loaded every non-GGUF row in bfloat16 — FP16- and FP32-labelled rows
alike — while miStudio (which trains the probes, SAEs and circuits served here) cast every 16-bit
row to float16. A probe fitted on float16 activations then failed parity here, combined Δ 0.251
against a 0.10 tolerance; and the 0.10 itself had been raised to absorb exactly that gap. Phase 0
of the 2026-10-03 fix reproduced this server's parity numbers to four decimals by re-scoring at
bfloat16, one request at a time.

THE RULE — `docs/schemas/native-dtype-cases.json`, byte-identical in both repos and tested against
by both resolvers:

    FP32 row                 -> float32. The row asked for 4 bytes a parameter; loading it at 16
                                bits was a mislabel.
    FP16 / Q8 / Q4 / Q2 row  -> the checkpoint's own 16-bit dtype (bfloat16 or float16). A
                                checkpoint recording float32, or nothing, gets bfloat16 — float16's
                                65,504 ceiling is the NaN risk the loader's old comment cited.

For a bitsandbytes row the resolved dtype is both `torch_dtype` and the 4-bit compute dtype.
GGUF rows are outside this rule: llama.cpp owns their precision.

A config that names no dtype gives `None` here, never transformers' own default — presenting a
default as "the checkpoint's dtype" would be a fabricated fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import torch

#: Every dtype the rule produces. Pinned against the contract's `model.load_dtype` enum.
LOAD_DTYPES: Tuple[str, ...] = ("float16", "bfloat16", "float32")

_TORCH = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
_ALIASES = {
    "float16": "float16", "half": "float16", "fp16": "float16", "f16": "float16",
    "bfloat16": "bfloat16", "bf16": "bfloat16",
    "float32": "float32", "float": "float32", "fp32": "float32", "f32": "float32",
}
DEFAULT_16BIT = "bfloat16"


def normalise_dtype_name(value: Any) -> Optional[str]:
    """`torch.bfloat16`, `"torch.bfloat16"`, `"bf16"` -> `"bfloat16"`; unknown -> None."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if text.startswith("torch."):
        text = text[len("torch."):]
    return _ALIASES.get(text)


def _field(source: Any, name: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)


def checkpoint_dtype_of(config: Any) -> Tuple[Optional[str], str]:
    """`(name, source)` — what the checkpoint RECORDS: `dtype`, `torch_dtype`, then `text_config`.

    For a transformers config object only values the checkpoint wrote are trusted: `to_diff_dict`
    keeps what differs from the class defaults, so a transformers-supplied default does not pass
    for a recorded one.
    """
    if config is None:
        return None, "default"
    raw = config
    if not isinstance(config, Mapping) and hasattr(config, "to_diff_dict"):
        try:
            raw = config.to_diff_dict()
        except Exception:  # noqa: BLE001 - fall back to attribute reads
            raw = config
    for name in ("dtype", "torch_dtype"):
        found = normalise_dtype_name(_field(raw, name))
        if found:
            return found, "config"
    text = _field(raw, "text_config")
    if text is not None:
        for name in ("dtype", "torch_dtype"):
            found = normalise_dtype_name(_field(text, name))
            if found:
                return found, "text_config"
    return None, "default"


@dataclass(frozen=True)
class ResolvedDtype:
    """What a row loads at and why. ONE object for the plan, the preflight and the load."""

    torch_dtype: torch.dtype
    name: str
    checkpoint_dtype: Optional[str]
    source: str
    quantization: str

    @property
    def storage_name(self) -> str:
        return "float32" if self.name == "float32" else "float16"


def resolve_load_dtype(
    quantization: Any, checkpoint_dtype: Optional[str], source: str = "config",
    pre_quantized: bool = False,
) -> ResolvedDtype:
    """THE rule. `quantization` is a QuantizationType or its value ("FP16", "Q4", ...).
    A pre-quantized checkpoint keeps the 16-bit rule whatever its label (shared table v2)."""
    quant = str(getattr(quantization, "value", quantization)).upper()
    if pre_quantized and quant == "FP32":
        quant = "FP16"
    recorded = normalise_dtype_name(checkpoint_dtype)
    if recorded is None:
        source = "default"
    if quant == "FP32":
        name = "float32"
    elif quant in ("FP16", "Q8", "Q4", "Q2", "GPTQ", "AWQ", "BITNET"):
        # Pre-quantized formats keep their non-quantized modules at the 16-bit rule too.
        name = recorded if recorded in ("bfloat16", "float16") else DEFAULT_16BIT
    else:
        raise ValueError(f"unknown quantization {quantization!r}")
    return ResolvedDtype(_TORCH[name], name, recorded, source, quant)


def is_pre_quantized(config: Any) -> bool:
    """Whether a checkpoint ships already quantized (its config carries `quantization_config`)."""
    return config is not None and _field(config, "quantization_config") is not None


def resolve_for_config(quantization: Any, config: Any) -> ResolvedDtype:
    recorded, source = checkpoint_dtype_of(config)
    return resolve_load_dtype(quantization, recorded, source, pre_quantized=is_pre_quantized(config))


def rule_quantization(quantization: Any, is_pre_quantized: bool) -> str:
    """The row label the rule is applied with. A PRE-QUANTIZED checkpoint (GPTQ, AWQ, FP8...) keeps
    its unquantized and dequantized modules at the 16-bit rule whatever the row is labelled: an
    FP32 label on such a checkpoint would otherwise load those modules at float32 while the size
    plan (`checkpoint_materialised_mb`) assumed 16 bits — and GPTQ/AWQ kernels at float32 are not
    a supported combination anyway."""
    return "FP16" if is_pre_quantized else str(getattr(quantization, "value", quantization))
