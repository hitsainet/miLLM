"""Per-run behaviour fixtures for 039 (FTASKS 0.6, TD39-36, FR-39.32).

* `random_key()` mints a valid behaviour key (`^[a-z0-9][a-z0-9-]{0,62}[a-z0-9]$`) that is
  different on every run, so no test can come to depend on a particular behaviour's name and the
  generality guard (038 FR-38.1) never sees one.
* `definition(...)` builds a `mistudio.probe-definition/v1` document for the tiny real Llama of
  `tests/unit/f25_fixtures.py` whose `test_vectors` are COMPUTED on that model, per run, through
  the serving code (`ProbeRequestContext` over `build_probe_forward`). Parity therefore passes for
  real — no stubbed parity engine (TD39-36) — and a test that wants a failing probe changes exactly
  the field under test (`break_parity=True` shifts the recorded score).

Multi-probe fixtures should differ in score, layer, threshold and behaviour (FPRD §3): pass a
different `layer`, `threshold` and `scale` per probe.
"""

from __future__ import annotations

import copy
import secrets
from types import SimpleNamespace
from typing import Any, Optional

import torch

#: The token ids every computed vector scores (in the word tokenizer's vocabulary).
VECTOR_IDS = [2, 6, 7, 8]


def random_key(prefix: str = "b") -> str:
    """A fresh valid behaviour key, e.g. `b-3fa91c2e`."""
    return f"{prefix}-{secrets.token_hex(4)}"


def _head(scale: float, d: int = 16) -> dict[str, Any]:
    gen = torch.Generator().manual_seed(int(abs(scale) * 1000) % 2**31)
    weights = (torch.randn(d, generator=gen) * scale).tolist()
    return {"weights": weights, "bias": 0.0, "norm_mean": [0.0] * d, "norm_std": [1.0] * d,
            "attention_query": None}


def definition(
    name: str,
    *,
    key: Optional[str],
    version: Optional[int] = 1,
    layer: int = 1,
    threshold: Optional[float] = 0.0,
    scale: float = 0.25,
    rule: str = "mean",
    model: Any = None,
    hf_id: Optional[str] = None,
    break_parity: bool = False,
    rung: int = 2,
) -> dict[str, Any]:
    """A definition for the tiny model whose test vector reproduces on it (unless broken)."""
    from tests.unit.f25_fixtures import word_model
    from tests.unit.score_fixtures import tiny_definition

    doc = tiny_definition(name, layer=layer, threshold=threshold,
                          **({"hf_id": hf_id} if hf_id else {}))
    doc["head"] = _head(scale)
    doc["aggregation"] = {"rule": rule, "params": {}, "streamable": True}
    doc["evidence"] = copy.deepcopy(doc["evidence"])
    doc["evidence"]["rung"] = rung
    if rung < 2:
        doc["evidence"]["acknowledgement"] = {
            "by": "test", "at": "2026-10-24T00:00:00Z", "reason": "fixture"}
    if key is not None:
        doc["behaviour"] = {"key": key, "name": key.replace("-", " "), "version": version}
    else:
        doc.pop("behaviour", None)
    token_scores, score = compute_vector(doc, model or word_model())
    if break_parity:
        score += 5.0
        token_scores = [t + 5.0 for t in token_scores]
    doc["test_vectors"] = {
        "tolerance": 1e-4, "authoritative_input": "token_ids",
        "messages_reproduce_token_ids": False,
        "vectors": [{"messages": [{"role": "user", "content": "w1 w2"}],
                     "token_ids": list(VECTOR_IDS), "token_scores": token_scores,
                     "score": score,
                     "verdict": (score >= threshold) if threshold is not None else None}],
    }
    return doc


def compute_vector(doc: dict[str, Any], model: Any) -> tuple[list[float], float]:
    """`(token_scores, score)` of `doc`'s head on `VECTOR_IDS`, through the serving code."""
    from millm.services.probe_arm_bridge import build_probe_forward
    from millm.services.probe_arming import armed_probe_from_row
    from millm.services.probe_runtime import ProbeRequestContext

    row = SimpleNamespace(
        id="pr_factory000", name=doc["name"], definition=doc, rule=doc["aggregation"]["rule"],
        scope=doc.get("scope", "all"), layer=doc["read"]["layer"],
        rung=doc["evidence"]["rung"], threshold=doc["decision"]["threshold"],
        threshold_revision=1,
    )
    armed = armed_probe_from_row(row, windows=[])
    context = ProbeRequestContext("factory", [armed])
    build_probe_forward(model, [armed.layer])(torch.tensor([VECTOR_IDS]), context)
    verdicts = context.finish()
    verdict = next(v for v in verdicts if v.window == armed.scope)
    return [float(x) for x in context.token_scores_for(armed.probe_id)], float(verdict.score)


async def import_probe(session: Any, doc: dict[str, Any]) -> Any:
    """Import `doc` through the real `ProbeService` (the projections, digests and member map)."""
    from millm.db.repositories.probe_repository import ProbeRepository
    from millm.services.probe_service import ProbeService

    return await ProbeService(ProbeRepository(session)).import_definition(
        doc, raw_bytes=1, on_conflict="rename", origin="file")
