"""A valid `mistudio.probe-definition/v1` document, and knobs to make it invalid on purpose.

Shared across phases 2–7. It is SYNTHETIC: small widths, round numbers, and scores that are
obviously made up, so nothing here can be mistaken for measured evidence. The real
`tests/fixtures/lfm2_probe_definition.json` arrives from miStudio 033 acceptance at task 10.1 and
is what proves the two systems agree on real data; this one proves the code paths.

⚠ The default is rung 2 with no acknowledgement. Rung 0 and 1 documents MUST carry one — the
contract refuses them otherwise — so a test that wants a low-rung probe has to supply it, which is
the behaviour under test rather than an inconvenience.
"""

from __future__ import annotations

import copy
from typing import Any

D_MODEL = 8
N_LAYERS = 16


def probe_definition(**overrides: Any) -> dict:
    """A valid dense probe definition. Top-level keys can be overridden or replaced."""
    doc: dict[str, Any] = {
        "kind": "mistudio.probe-definition/v1",
        "name": "high-stakes",
        "description": "synthetic fixture — not measured evidence",
        "concept": "high-stakes",
        "model": {
            "hf_id": "LiquidAI/LFM2.5-1.2B-Instruct",
            "revision": "0f604ada3f766f9f257460c4c9f0b5d6f69d431b",
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "architecture": "Lfm2ForCausalLM",
            "chat_template_sha256": "a" * 64,
        },
        "read": {"layer": 11, "hook_point": "resid_post"},
        "scope": "all",
        "basis": "residual",
        "head": {
            "weights": [0.5, -0.25, 1.0, 0.0, 0.75, -1.5, 0.125, 2.0],
            "bias": 0.25,
            "norm_mean": [0.0] * D_MODEL,
            "norm_std": [1.0] * D_MODEL,
            "attention_query": None,
        },
        "aggregation": {"rule": "mean", "params": {}, "streamable": True},
        "decision": {
            "threshold": 1.0,
            "target_fpr": 0.01,
            "realised_fpr": 0.0097,
            "threshold_source": "validation_negatives",
        },
        "evidence": {
            "rung": 2,
            "rung_language": "detects on unseen tasks",
            "acknowledgement": None,
            "evaluations": [{"auroc": 0.88, "distribution": "out_of_distribution"}],
        },
        "provenance": {"mistudio_probe_id": "pm_fixture01", "built_at": "2026-09-27T00:00:00Z"},
        "test_vectors": {
            "tolerance": 0.001,
            "authoritative_input": "token_ids",
            "messages_reproduce_token_ids": False,
            "vectors": [
                {
                    "messages": [{"role": "user", "content": "escalate this now"}],
                    "token_ids": [1, 2, 3, 4],
                    "token_scores": [0.5, 1.5, 2.0, 1.0],
                    "score": 1.25,
                    "verdict": True,
                }
            ],
        },
    }
    doc.update(copy.deepcopy(overrides))
    return doc


def sae_probe_definition(k: int = 4, **overrides: Any) -> dict:
    """A k-sparse SAE probe: the head is one weight per selected feature, not per d_model."""
    doc = probe_definition()
    doc["basis"] = "sae_features"
    doc["head"]["weights"] = [1.0, -0.5, 0.25, 2.0][:k]
    doc["head"]["norm_mean"] = [0.0] * k
    doc["head"]["norm_std"] = [1.0] * k
    doc["sae"] = {
        "hf_repo": "mistudio/sae-lfm2p5-1p2b-instruct-l11-jumprelu",
        "path": "layer_11",
        "revision": "53943b26" + "0" * 32,
        "weights_sha256": "b" * 64,
        "architecture": "jumprelu",
        "d_model": D_MODEL,
        "n_features": 64,
        "normalization": {"mode": "constant_norm_rescale", "source": "sae_row"},
        "feature_indices": sorted([3, 11, 29, 47][:k]),
    }
    doc.update(copy.deepcopy(overrides))
    return doc


def acknowledged(reason: str = "exploratory monitor; gates nothing") -> dict:
    return {"by": "operator", "at": "2026-09-27T00:00:00Z", "reason": reason}
