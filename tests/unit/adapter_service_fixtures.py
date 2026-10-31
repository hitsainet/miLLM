"""A real AdapterService over a tiny real Llama, the REAL barrier and a real InferenceService.

Only the database row lookup is a stand-in (it returns the row the resident was loaded from).
Everything an adapter load touches — the artifact reader, the vendored schema, the hooks, the
`WeightBarrier.serving_pause` that holds the admission slot, the base epoch, the served-name
overlay, the drop callback — is the production object, so a test cannot pass by a fake agreeing
with the code.

`AdapterEnv` cleans up after itself: the process adapter slot is emptied (hooks removed), the
resident is cleared, the barrier and the lease registry are reset.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import AsyncMock

import torch

from millm.ml.model_loader import LoadedModelState
from millm.services.adapter_service import AdapterService, get_adapter_state
from millm.services.model_lease import LeaseRegistry, set_lease_registry
from millm.services.weight_barrier import WeightBarrier, set_weight_barrier
from tests.unit.adapter_fixtures import BASE_NAME, tiny_llama, write_adapter
from tests.unit.f25_fixtures import clear_loaded, make_service, word_tokenizer

ADAPTER_NAME = f"{BASE_NAME}:r9:adapter-200"
OPERATOR_TOKEN = "operator-plaintext-token"
TRAINER_TOKEN = "trainer-plaintext-token"
PROMPT_IDS = torch.tensor([[0, 6, 7, 8, 9]])


def weights_tokens() -> str:
    """A `WEIGHTS_TOKENS` value with one operator and one trainer token."""
    return json.dumps([
        {"label": "ops", "role": "operator",
         "sha256": hashlib.sha256(OPERATOR_TOKEN.encode()).hexdigest()},
        {"label": "forge", "role": "trainer",
         "sha256": hashlib.sha256(TRAINER_TOKEN.encode()).hexdigest()},
    ])


@dataclass
class AdapterEnv:
    root: Path
    model: Any
    inference: Any
    barrier: WeightBarrier
    leases: LeaseRegistry
    row: Any
    row_lookup: AsyncMock
    service: AdapterService
    #: Every registered row name, as the database would report them (D-32).
    names_lookup: AsyncMock

    def write(self, rel: str = "r9/adapter-step-000200", **kwargs: Any) -> str:
        write_adapter(self.root / rel, self.model, **kwargs)
        return rel

    def logits(self) -> torch.Tensor:
        with torch.no_grad():
            return self.model(PROMPT_IDS).logits.clone()

    def close(self) -> None:
        get_adapter_state().take()
        get_adapter_state().release_claim()
        clear_loaded()
        set_weight_barrier(None)
        set_lease_registry(None)


def build_env(tmp_path: Path, *, dtype: torch.dtype = torch.float32,
              quantization: str = "none") -> AdapterEnv:
    get_adapter_state().take()
    model = tiny_llama(dtype)
    inference = make_service(model, word_tokenizer(), name="tiny")
    LoadedModelState().current.quantization_method = quantization
    barrier = WeightBarrier(inference=lambda: inference)
    set_weight_barrier(barrier)
    leases = LeaseRegistry()
    set_lease_registry(leases)
    root = tmp_path / "checkpoints"
    root.mkdir()
    row = SimpleNamespace(name="tiny", repo_id=BASE_NAME, source=None, local_path=None)
    row_lookup = AsyncMock(return_value=row)
    names_lookup = AsyncMock(return_value=["tiny", BASE_NAME])
    service = AdapterService(
        inference=lambda: inference,
        barrier=lambda: barrier,
        leases=lambda: leases,
        row_lookup=row_lookup,
        names_lookup=names_lookup,
        roots=lambda: [("primary", str(root))],
    )
    return AdapterEnv(
        root, model, inference, barrier, leases, row, row_lookup, service, names_lookup
    )


def identity(role: str = "operator", label: Optional[str] = None) -> Any:
    from millm.core.weights_auth import WeightsIdentity

    return WeightsIdentity(label=label or ("ops" if role == "operator" else "forge"), role=role)
