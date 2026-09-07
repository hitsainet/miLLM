"""Real objects for tests, instead of doubles that invent their own fields.

A `Model` row is 22 fields and no behaviour. Mocking it buys nothing and costs a
specific, repeated bug: `MagicMock` answers EVERY attribute with another mock,
so a column added to the ORM silently starts coming back as a mock object rather
than a value. That has now happened three times in one day, and once it reached
`delete_cached_model` as a directory-name suffix.

`MagicMock(spec=Model)` does not help. `spec` rejects attributes that do not
exist on the class — it says nothing about the ones that do. `gguf_label` exists,
so the spec'd double happily returned a mock for it.

Constructing a real `Model` fixes it at the root: SQLAlchemy column defaults are
applied at INSERT, not at construction, so every field is spelled out below. Add
a column and you give it a default HERE, once, or the omission shows up in one
place instead of silently in twenty.

Mock collaborators — repositories, downloaders, loaders, the torch model — they
do I/O or have behaviour worth faking. Do not mock rows.
"""

from datetime import datetime
from typing import Any

from millm.db.models.model import Model, ModelSource, ModelStatus, QuantizationType

#: An ordinary, fully-downloaded HuggingFace model. Deliberately NOT a GGUF one:
#: the common case should be the default, and a GGUF row is one override away.
_DEFAULTS: dict[str, Any] = {
    "id": 1,
    "name": "gemma-2-2b",
    "source": ModelSource.HUGGINGFACE,
    "repo_id": "google/gemma-2-2b",
    "local_path": None,
    "params": "2B",
    "architecture": "text-generation",
    "quantization": QuantizationType.Q4,
    # '' not None — this is what the column stores for "no GGUF selection", and
    # it is part of the uniqueness constraint, where NULLs compare as distinct.
    "gguf_label": "",
    "gguf_files": None,
    "revision": None,
    "disk_size_mb": 1500,
    "estimated_memory_mb": 2000,
    "cache_path": "huggingface/google--gemma-2-2b--Q4",
    "config_json": None,
    "trust_remote_code": False,
    "status": ModelStatus.READY,
    "error_message": None,
    "locked": False,
    "created_at": datetime(2026, 1, 1),
    "updated_at": datetime(2026, 1, 1),
    "loaded_at": None,
}


def make_model(**overrides: Any) -> Model:
    """Build a real `Model` row, overriding whichever fields the test cares about.

    >>> make_model(gguf_label="Q5_K_M", quantization=QuantizationType.Q8).gguf_label
    'Q5_K_M'
    """
    unknown = set(overrides) - set(_DEFAULTS)
    if unknown:
        # A typo'd override would otherwise be silently ignored, and the test
        # would pass against a default it never meant to use.
        raise TypeError(f"make_model() got unexpected field(s): {sorted(unknown)}")
    return Model(**{**_DEFAULTS, **overrides})


def make_gguf_model(**overrides: Any) -> Model:
    """A model downloaded as a single GGUF quantization."""
    gguf: dict[str, Any] = {
        "name": "zora-v1.13-gguf",
        "repo_id": "sovasoft/zora-v1.13-gguf",
        "params": "8.2B",
        "quantization": QuantizationType.Q8,
        "gguf_label": "Q5_K_M",
        "gguf_files": ["zora-v1.13-Q5_K_M.gguf"],
        "revision": "141f6348f24ca1a2932a9976c2b40c23419f4a36",
        "cache_path": "huggingface/sovasoft--zora-v1.13-gguf--Q8--Q5_K_M",
    }
    return make_model(**{**gguf, **overrides})
