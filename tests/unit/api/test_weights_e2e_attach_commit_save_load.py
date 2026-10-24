"""Flow (a), end to end through the REAL app (review RX; BRD-01 §5 criteria 2, 4, 11 on the CPU).

attach → quiesce → commit-begin → in-place write (what a trainer does through its handles) →
commit-end → resume → `/v1` answers from the new weights with `X-miLLM-Model` `commits=1` →
save (holder) → more commits → save again (parent check) → in-place load of the EARLIER
checkpoint (addresses kept, so the attachment's handles stay valid) → served name and identity
→ lineage. Harness and fakes: `tests/unit/weights_e2e_fixtures.py`.
"""

from __future__ import annotations

import torch

from millm.services.resident_identity import identity_of, served_name
from tests.unit.weights_e2e_fixtures import (  # noqa: F401 - flow is a fixture
    ROW_NAME,
    ROW_REPO,
    addresses,
    attach,
    chat,
    flow,
    holder,
    manifest,
    params,
)


def _text(response) -> str:
    assert response.status_code == 200, response.text
    return response.json()["choices"][0]["message"]["content"]


async def _commit(client, attachment_id, write) -> dict:
    """One bracketed in-place write, the trainer's step (R-01.6, R-01.21, R-01.7)."""
    h = holder(attachment_id)
    q = await client.post("/api/weights/quiesce", headers=h)
    assert q.status_code == 200, q.text
    assert q.json()["data"]["weights_state"] == "clean"
    b = await client.post("/api/weights/commit-begin", headers=h)
    assert b.status_code == 200, b.text
    assert b.json()["data"]["weights_state"] == "writing"
    write()
    e = await client.post("/api/weights/commit-end", headers=h)
    assert e.status_code == 200, e.text
    r = await client.post("/api/weights/resume", headers=h)
    assert r.status_code == 200, r.text
    assert r.json()["data"] == {"barrier": "serving", "weights_state": "clean"}
    return e.json()["data"]


def _nudge(model, seed: int):
    def write():
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn(p.shape, generator=g).to(p.dtype) * 0.5)
    return write


async def test_attach_commit_save_load_lineage(flow):
    model = flow.model
    fixed = addresses(model)
    async with flow.client() as client:
        r0 = await chat(client)
        t0 = _text(r0)
        assert r0.headers["X-miLLM-Model"] == f'"{ROW_NAME}";origin=model;commits=0'

        attachment_id = await attach(client)
        status = (await client.get("/api/weights/status")).json()["data"]
        assert status["attached"] is True and status["attachment"]["label"] == "forge"
        assert status["resident_model"] == {"model_id": 1, "name": ROW_NAME,
                                            "manifest_hash": None}

        # --- one commit on the plain row (criterion 2) --------------------------------------
        end = await _commit(client, attachment_id, _nudge(model, 1))
        assert end["steps_since_load"] == 1 and end["overran"] is False
        values_a = params(model)
        r1 = await chat(client)
        t1 = _text(r1)
        assert t1 != t0, "the completion after the commit must come from the new weights"
        assert r1.headers["X-miLLM-Model"] == f'"{ROW_NAME}";origin=model;commits=1'
        assert r1.json()["model"] == ROW_NAME

        # --- save (holder only, R-01.31) -------------------------------------------------------
        refused = await client.post("/api/weights/save", json={"manifest": manifest(step=1)},
                                    headers={"Authorization": holder(attachment_id)[
                                        "Authorization"]})
        assert refused.status_code == 403
        assert refused.json()["error"]["code"] == "NOT_ATTACHMENT_HOLDER"
        s1 = await client.post("/api/weights/save", json={"manifest": manifest(step=1)},
                               headers=holder(attachment_id))
        assert s1.status_code == 201, s1.text
        step1 = s1.json()["data"]
        name1 = f"{ROW_REPO}:run:step-1"
        assert step1["name"] == name1 and step1["path"] == "run/step-000001"
        assert step1["parent_check"] == "match" and step1["identity_set"] is True
        # criterion 4, first half: the manifest's hash matches a fresh hash of the file.
        import hashlib

        weights_file = flow.root / "run" / "step-000001" / "model.safetensors"
        assert hashlib.sha256(weights_file.read_bytes()).hexdigest() == step1["weights_sha256"]
        assert served_name() == name1
        status = (await client.get("/api/weights/status")).json()["data"]
        assert status["last_save"]["checkpoint_sha256"] == step1["weights_sha256"]
        assert status["last_save"]["path"] == "run/step-000001"
        assert status["resident_model"]["name"] == name1
        assert status["resident_model"]["manifest_hash"] == step1["manifest_sha256"]

        # The served name moved to the checkpoint's; `commits=` counts from the identity.
        r_saved = await chat(client, name1)
        assert _text(r_saved) == t1
        assert r_saved.headers["X-miLLM-Model"] == f'"{name1}";origin=checkpoint;commits=0'

        # --- an unsaved commit: name unchanged, commits=1 (D-18) ------------------------------
        await _commit(client, attachment_id, _nudge(model, 2))
        r2 = await chat(client, name1)
        t2 = _text(r2)
        assert t2 != t1
        assert r2.headers["X-miLLM-Model"] == f'"{name1}";origin=checkpoint;commits=1'
        lineage = (await client.get("/api/weights/lineage")).json()["data"]
        assert lineage["resident"] == {"name": name1, "origin": "checkpoint",
                                       "manifest_sha256": step1["manifest_sha256"],
                                       "commits_since_identity": 1}

        # --- save step 2 with step 1 as its parent ------------------------------------------
        s2 = await client.post("/api/weights/save",
                               json={"manifest": manifest(step=2, parent=name1)},
                               headers=holder(attachment_id))
        assert s2.status_code == 201, s2.text
        assert s2.json()["data"]["parent_check"] == "match"
        name2 = f"{ROW_REPO}:run:step-2"
        lineage = (await client.get("/api/weights/lineage")).json()["data"]
        assert [link["name"] for link in lineage["chain"]] == [name2, name1]
        assert lineage["complete"] is True and lineage["break"] is None
        assert lineage["base_model"] == ROW_REPO and lineage["weights_verified"] is False

        # --- in-place load of the EARLIER checkpoint by the holder (criterion 11, CPU half) -----
        load = await client.post("/api/weights/load", json={"name": name1},
                                 headers=holder(attachment_id))
        assert load.status_code == 200, load.text
        body = load.json()["data"]
        assert body["mode"] == "in_place" and body["name"] == name1
        assert body["weights_state"] == "clean" and body["steps_since_load"] == 0
        assert body["weights_sha256"] == step1["weights_sha256"]
        # Addresses kept: the attachment's handles still point at the served tensors.
        assert addresses(model) == fixed
        assert flow.weights.verify_addresses("test") == []
        assert all(torch.equal(p, values_a[n]) for n, p in model.named_parameters())
        r3 = await chat(client, name1)
        assert _text(r3) == t1
        assert r3.headers["X-miLLM-Model"] == f'"{name1}";origin=checkpoint;commits=0'
        assert identity_of(flow.resident).name == name1

        lineage = (await client.get("/api/weights/lineage")).json()["data"]
        assert lineage["resident"]["name"] == name1
        assert lineage["resident"]["commits_since_identity"] == 0
        assert [link["name"] for link in lineage["chain"]] == [name1]
        models = [m["id"] for m in (await client.get("/v1/models")).json()["data"]]
        assert name1 in models

        # The attachment survived the in-place load and still holds.
        status = (await client.get("/api/weights/status")).json()["data"]
        assert status["attached"] is True and status["weights_state"] == "clean"
        assert status["steps_since_load"] == 0
        d = await client.post("/api/weights/detach", headers=holder(attachment_id))
        assert d.status_code == 200, d.text
