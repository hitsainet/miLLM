"""Flow (e): the three status surfaces and the Admin UI agree on names and values (review RX).

`GET /api/weights/status` (031, FR-31.16.1), `GET /api/weights/adapter` (033, D-28) and the
`weights` block of `GET /api/health/detailed` (FR-31.16.3) describe one barrier and one
attachment. Driven through the REAL app across five states (idle; attached; quiesced; adapter
applied; torn), every state is read from all three and compared field by field, and the live
JSON's key sets are compared with the TypeScript interfaces the Admin UI's Weights card reads
(`admin-ui/src/types/api.ts`), parsed from the file, never a hand-kept copy.

Harness and fakes: `tests/unit/weights_e2e_fixtures.py`.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest
import torch

from millm.api.schemas.weights import WeightsHealthSummary, WeightsStatus
from millm.api.schemas.weights_adapters import AdapterStatus
from millm.core.config import settings
from millm.services.adapter_service import AdapterService, addressable_names, get_adapter_service
from millm.services.model_lease import get_lease_registry
from tests.unit.adapter_fixtures import write_adapter
from tests.unit.weights_e2e_fixtures import (  # noqa: F401 - flow is a fixture
    OPS,
    ROW_REPO,
    attach,
    flow,
    holder,
)

TYPES = Path(__file__).resolve().parents[3] / "admin-ui" / "src" / "types" / "api.ts"


def _ts_keys(interface: str) -> set[str]:
    """The top-level member names of `export interface <interface> { ... }` in api.ts."""
    text = TYPES.read_text(encoding="utf-8")
    match = re.search(r"export interface " + interface + r" \{(.*?)\n\}", text, re.S)
    assert match, f"{interface} is not in {TYPES}"
    body = re.sub(r"/\*.*?\*/", "", match.group(1), flags=re.S)
    keys, depth = set(), 0
    for line in body.splitlines():
        stripped = line.strip()
        if depth == 0:
            m = re.match(r"([a-z][a-z0-9_]*)\??:", stripped)
            if m:
                keys.add(m.group(1))
        depth += stripped.count("{") - stripped.count("}")
    return keys


class TestNamesAgree:
    def test_status_schema_matches_the_ui_interface(self):
        assert _ts_keys("WeightsStatus") == set(WeightsStatus.model_fields)

    def test_adapter_schema_matches_the_ui_interface(self):
        assert _ts_keys("AdapterStatus") == set(AdapterStatus.model_fields)

    def test_attachment_and_commit_blocks_match(self):
        from millm.api.schemas.weights import AttachmentStatus, CommitStatus

        assert _ts_keys("WeightsAttachment") == set(AttachmentStatus.model_fields)
        assert _ts_keys("WeightsCommitStatus") == set(CommitStatus.model_fields)

    def test_health_block_names_are_status_names(self):
        """Every health field is a status field of the same meaning (attachment_state is the
        status's attachment.state), so a reader of both never translates."""
        health = set(WeightsHealthSummary.model_fields) - {"error", "attachment_state"}
        assert health <= set(WeightsStatus.model_fields)


@pytest.fixture
def adapters(flow):
    repo = flow.models.repository

    async def names():
        return sorted(addressable_names(await repo.get_all()))

    service = AdapterService(
        inference=lambda: flow.inference, barrier=lambda: flow.barrier,
        leases=get_lease_registry, row_lookup=repo.get_by_id, names_lookup=names,
        roots=lambda: [("primary", str(flow.root))],
    )
    flow.app.dependency_overrides[get_adapter_service] = lambda: service
    return service


async def _read(client) -> tuple[dict, dict, dict]:
    status = (await client.get("/api/weights/status")).json()["data"]
    adapter = (await client.get("/api/weights/adapter")).json()["data"]
    health = (await client.get("/api/health/detailed")).json()["weights"]
    return status, adapter, health


def _agree(status: dict, adapter: dict, health: dict) -> None:
    assert set(status) == set(WeightsStatus.model_fields)
    assert set(adapter) == set(AdapterStatus.model_fields)
    assert status["adapter"] == adapter
    assert health["error"] is None
    assert health["weights_state"] == status["weights_state"]
    assert health["barrier"] == status["barrier"]
    assert health["attached"] == status["attached"]
    expected_state = status["attachment"]["state"] if status["attachment"] else None
    assert health["attachment_state"] == expected_state


async def test_every_state_reads_the_same_on_all_three(flow, adapters, monkeypatch):
    seen = []
    async with flow.client() as client:
        status, adapter, health = await _read(client)
        _agree(status, adapter, health)
        seen.append((status["weights_state"], status["barrier"], status["attached"]))

        attachment_id = await attach(client)
        status, adapter, health = await _read(client)
        _agree(status, adapter, health)
        assert health["attachment_state"] == "active"

        # Stale (R-01.4, criterion 3): no heartbeat within the window. Health must say so too.
        timeout = settings.WEIGHTS_HEARTBEAT_TIMEOUT_S
        monkeypatch.setattr(settings, "WEIGHTS_HEARTBEAT_TIMEOUT_S", 0.0)
        status, adapter, health = await _read(client)
        _agree(status, adapter, health)
        assert status["attachment"]["state"] == "stale" and status["attachment"]["stale_since"]
        assert health["attachment_state"] == "stale"
        monkeypatch.setattr(settings, "WEIGHTS_HEARTBEAT_TIMEOUT_S", timeout)
        assert (await client.post("/api/weights/heartbeat", headers=holder(attachment_id))
                ).status_code == 200
        assert (await _read(client))[2]["attachment_state"] == "active"

        h = holder(attachment_id)
        assert (await client.post("/api/weights/quiesce", headers=h)).status_code == 200
        status, adapter, health = await _read(client)
        _agree(status, adapter, health)
        assert status["barrier"] == "quiesced" and status["held_requests"] == 0
        assert (await client.post("/api/weights/resume", headers=h)).status_code == 200

        write_adapter(flow.root / "adapters" / "a1", flow.model, base_name=ROW_REPO,
                      manifest_overrides={"name": "org/tiny:r:adapter-1"})
        r = await client.post("/api/weights/adapter/load", json={"path": "adapters/a1"},
                              headers=OPS)
        assert r.status_code == 200, r.text
        status, adapter, health = await _read(client)
        _agree(status, adapter, health)
        assert adapter["applied"] is True
        # D-41: the resident name is the BASE's (the attached process sees base tensors); the
        # adapter is reported in the adapter block.
        assert adapter["name"] == "org/tiny:r:adapter-1"
        assert status["resident_model"]["name"] == "tiny"  # the row's served name
        assert status["adapter"]["name"] == "org/tiny:r:adapter-1"
        assert adapter["base_steps_since_load"] == 0

        monkeypatch.setattr(settings, "WEIGHTS_COMMIT_WATCHDOG_MS", 30)
        assert (await client.post("/api/weights/quiesce", headers=h)).status_code == 200
        assert (await client.post("/api/weights/commit-begin", headers=h)).status_code == 200
        status, adapter, health = await _read(client)
        _agree(status, adapter, health)
        assert status["weights_state"] == "writing" and status["commit"]["began_at"]
        with torch.no_grad():
            next(flow.model.parameters()).add_(1.0)
        for _ in range(300):
            if flow.barrier.snapshot().weights_state == "torn":
                break
            await asyncio.sleep(0.01)
        status, adapter, health = await _read(client)
        _agree(status, adapter, health)
        assert status["weights_state"] == "torn"
