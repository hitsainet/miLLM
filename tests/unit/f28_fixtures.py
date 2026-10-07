"""Shared fixtures for Feature 28 tests: a TINY REAL Llama, REAL `LoadedSAE`s hooked through the
real `SAEHooker`, registered in the REAL `AttachedSAEState` singleton, and a real (SQLite)
database behind `millm.db.base.async_session_factory` for profile and circuit reads.

Two SAEs at two layers (0 and 1), from different seeds, so a test that selects or disables "the
other entry" cannot pass because both entries are one object (fixtures must not agree by
construction, 028 FTASKS Notes).
"""

from __future__ import annotations

import hashlib
import struct
from contextlib import contextmanager
from typing import Any, Optional
from unittest.mock import patch

import pytest
import pytest_asyncio
import torch
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from millm.api.schemas.openai import ChatCompletionRequest, ChatMessage, TextCompletionRequest
from millm.db.base import Base
from millm.ml.sae_config import SAEConfig
from millm.ml.sae_hooker import SAEHooker
from millm.ml.sae_wrapper import LoadedSAE
from millm.services.sae_service import AttachedSAEState
from tests.unit.f25_fixtures import clear_loaded, make_service, word_model, word_tokenizer

D_SAE = 8
SAE_A = "sae_alpha"
SAE_B = "sae_beta"


def make_sae(seed: int, d_sae: int = D_SAE) -> LoadedSAE:
    torch.manual_seed(seed)
    return LoadedSAE(
        W_enc=torch.randn(16, d_sae), b_enc=torch.zeros(d_sae),
        W_dec=torch.randn(d_sae, 16) * 3.0, b_dec=torch.zeros(16),
        config=SAEConfig(d_in=16, d_sae=d_sae, model_name="t", hook_name="t", hook_layer=0),
    )


class Served:
    """A service, its model, and the attached SAEs (sae_id → LoadedSAE)."""

    def __init__(self, svc: Any, model: Any, saes: dict[tuple[str, int], LoadedSAE],
                 handles: list) -> None:
        self.svc = svc
        self.model = model
        self.saes = saes
        self.handles = handles

    def sae(self, sae_id: str, layer: int) -> LoadedSAE:
        return self.saes[(sae_id, layer)]

    def state(self) -> list[tuple[str, int, dict, bool]]:
        """Every entry's (id, layer, values, enabled) — the exact restore target."""
        return [(e.sae_id, e.layer, e.sae.get_steering_values(), e.sae.is_steering_enabled)
                for e in AttachedSAEState().entries()]


def attach(model: Any, entries: list[tuple[str, int, int]]) -> Served:
    """Attach `(sae_id, layer, seed)` entries for real: hooked and registered."""
    state = AttachedSAEState()
    saes: dict[tuple[str, int], LoadedSAE] = {}
    handles = []
    for sae_id, layer, seed in entries:
        sae = make_sae(seed)
        handle = SAEHooker().install(model, layer, sae)
        state.set(sae, sae_id, layer, handle)
        saes[(sae_id, layer)] = sae
        handles.append(handle)
    return Served(None, model, saes, handles)


@pytest.fixture
def clean_state():
    AttachedSAEState().reset_for_tests()
    yield
    AttachedSAEState().reset_for_tests()
    clear_loaded()


def build(entries: list[tuple[str, int, int]]) -> Served:
    model, tokenizer = word_model(), word_tokenizer()
    svc = make_service(model, tokenizer)
    served = attach(model, entries)
    served.svc = svc
    return served


@pytest_asyncio.fixture
async def db(monkeypatch):
    """A real SQLite database behind `async_session_factory` (profiles, circuits, claims)."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)

    @event.listens_for(engine.sync_engine, "connect")
    def _fk(dbapi_connection, _record):  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr("millm.db.base.async_session_factory", factory)
    yield factory
    await engine.dispose()


async def add_profile(factory: Any, name: str, steering: dict, *, active: bool = False,
                      intensity: float = 1.0, sae_id: Optional[str] = None,
                      layer: Optional[int] = None) -> None:
    from millm.db.models.profile import Profile

    async with factory() as session:
        session.add(Profile(
            id=f"prof_{name}", name=name, steering={str(k): v for k, v in steering.items()},
            is_active=active, intensity=intensity, sae_id=sae_id, layer=layer,
        ))
        await session.commit()


def chat(**over) -> ChatCompletionRequest:
    base = dict(model="tiny", messages=[ChatMessage(role="user", content="w1 w2 w3")],
                max_tokens=6, temperature=0.0)
    base.update(over)
    return ChatCompletionRequest(**base)


def text(**over) -> TextCompletionRequest:
    base = dict(model="tiny", prompt="w1 w2 w3", max_tokens=6, temperature=0.0)
    base.update(over)
    return TextCompletionRequest(**base)


@contextmanager
def no_cuda():
    with patch("millm.services.inference_service.torch.cuda.is_available", return_value=False):
        yield


def independent_hash(sae_id: str, applied: dict[int, float]) -> str:
    """The FTDD §5.3 hash computed here, NOT through `millm.core.steering_state`, so a test
    comparing the two cannot pass because both read one function."""
    body = f"millm.steering-set/v1\nsae={sae_id}\n" + "".join(
        f"{i}:{struct.pack('>d', applied[i]).hex()}\n" for i in sorted(applied) if applied[i] != 0
    )
    return "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()
