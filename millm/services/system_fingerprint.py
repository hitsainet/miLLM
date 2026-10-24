"""`system_fingerprint` for chat and text completion responses (Feature 25, FR-25.13.8-13.10).

THE GUARANTEE: the fingerprint names the model, its revision, its precision and the engine, and
a part miLLM does not know is written `unrecorded` — never guessed. A labelling job records it
against every row, so a guessed part would be a false provenance claim, not a cosmetic one.

Format: `millm:<name>@<revision>:<dtype>/<quantization>:<engine>`, e.g.
`millm:LFM2.5-1.2B-Instruct@0f604ada:bfloat16/FP16:transformers`.

* `revision` is the row's (nullable, `millm/db/models/model.py`).
* `dtype` is the RESOLVED load precision on `LoadedModel` (native-dtype rule); the loader's
  sentinel `"unknown"` maps to `unrecorded`.
* `quantization` is the exact GGUF label when there is one, else the row's quantization.
* `engine` is the resident engine; with nothing resident, the row's.

Stable while the loaded configuration is unchanged and different when any part changes. Pure: no
I/O. Feature 26's batch output lines reuse it.
"""

from __future__ import annotations

from typing import Any

UNRECORDED = "unrecorded"


def _part(value: Any) -> str:
    if isinstance(value, str) and value.strip() and value.strip().lower() != "unknown":
        return value.strip()
    return UNRECORDED


def build_system_fingerprint(row: Any, loaded: Any) -> str:
    name = _part(getattr(row, "name", None))
    revision = _part(getattr(row, "revision", None))
    dtype = _part(getattr(loaded, "dtype", None)) if loaded is not None else UNRECORDED
    quantization = getattr(row, "quantization", None)
    quantization = getattr(quantization, "value", quantization)
    label = getattr(row, "gguf_label", None)
    quant = _part(label) if _part(label) != UNRECORDED else _part(quantization)
    engine = _part(getattr(loaded, "engine", None)) if loaded is not None else UNRECORDED
    if engine == UNRECORDED:
        engine = "llamacpp" if getattr(row, "gguf_files", None) else "transformers"
    return f"millm:{name}@{revision}:{dtype}/{quant}:{engine}"
