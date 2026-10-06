"""Shared fixtures for Feature 27's stateless-scoring tests: a TINY REAL Llama, real probe rows in a
real SQLite repository, and a loaded identity that matches them.

Mocks stop at the model's identity lookup (which needs a model row and a download directory); the
scoring service, the probe construction, the forward and `_verdict_for` always run for real.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any

from millm.services.probe_identity import LoadedIdentity
from tests.unit.f25_fixtures import CHAT_TEMPLATE
from tests.unit.probe_fixtures import probe_definition

HF_ID = "tiny/llama"
REVISION = "0" * 40
D = 16
N_LAYERS = 2


def tiny_definition(name: str = "tiny-probe", *, layer: int = 1, threshold: Any = 1.0,
                    hf_id: str = HF_ID, weights: list[float] | None = None) -> dict:
    doc = probe_definition()
    doc["name"] = name
    doc["model"] = {
        "hf_id": hf_id, "revision": REVISION, "d_model": D, "n_layers": N_LAYERS,
        "architecture": "LlamaForCausalLM",
        "chat_template_sha256": hashlib.sha256(CHAT_TEMPLATE.encode()).hexdigest(),
    }
    doc["read"] = {"layer": layer, "hook_point": "resid_post"}
    doc["head"] = {
        "weights": weights or [0.25] * D, "bias": 0.0,
        "norm_mean": [0.0] * D, "norm_std": [1.0] * D, "attention_query": None,
    }
    doc["decision"] = copy.deepcopy(doc["decision"])
    doc["decision"]["threshold"] = threshold
    doc["test_vectors"]["vectors"][0]["token_ids"] = [2, 6, 7, 8]
    return doc


def tiny_identity(**over: Any) -> LoadedIdentity:
    base = dict(hf_id=HF_ID, d_model=D, n_layers=N_LAYERS, chat_template=CHAT_TEMPLATE,
                revision=REVISION, revision_source="download_dir")
    base.update(over)
    return LoadedIdentity(**base)
