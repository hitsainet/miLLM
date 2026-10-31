"""D-38 (review RX F1), end to end through the REAL app: a request naming the original row after a
save moved the resident identity off it is refused, never answered by reloading the row.

Before D-38 the same request was answered with an implicit `reload=True`: while attached it got
`409 already_attached`, and once detached it silently reloaded the row from disk, discarding the
trained weights and any unsaved commits. Now it is `404 model_not_found` (the unknown-model shape)
naming the resident's served name and `POST /api/weights/load {"model_id": N}`, and the resident is
untouched. The deliberate load still works.

Harness and fakes: `tests/unit/weights_e2e_fixtures.py`.

MUTATION CONTROL (FX-B): rule 3's plain-row branch restored to `Resolution(row=row, reload=True)`
-> every test here but the deliberate-load one goes red.
"""

from __future__ import annotations

import torch

from millm.services.resident_identity import identity_of, served_name
from tests.unit.weights_e2e_fixtures import (  # noqa: F401 - flow is a fixture
    OPS,
    ROW_ID,
    ROW_NAME,
    ROW_REPO,
    attach,
    chat,
    flow,
    holder,
    manifest,
    params,
)

SAVED = f"{ROW_REPO}:run:step-1"


def _nudge(model) -> None:
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.25)


async def _commit(client, attachment_id, model) -> None:
    h = holder(attachment_id)
    for path in ("/api/weights/quiesce", "/api/weights/commit-begin"):
        assert (await client.post(path, headers=h)).status_code == 200
    _nudge(model)
    for path in ("/api/weights/commit-end", "/api/weights/resume"):
        assert (await client.post(path, headers=h)).status_code == 200


async def _train_and_save(flow, client) -> str:
    """attach → one commit → save. Returns the attachment id (still attached)."""
    attachment_id = await attach(client)
    await _commit(client, attachment_id, flow.model)
    saved = await client.post("/api/weights/save", json={"manifest": manifest(step=1)},
                              headers=holder(attachment_id))
    assert saved.status_code == 201, saved.text
    assert served_name() == SAVED
    return attachment_id


def _assert_refused(response) -> None:
    assert response.status_code == 404, response.text
    error = response.json()["error"]
    assert error["code"] == "model_not_found" and error["param"] == "model"
    assert f"'{SAVED}'" in error["message"]
    assert f'POST /api/weights/load {{"model_id": {ROW_ID}}}' in error["message"]


def _assert_untouched(flow, before: dict) -> None:
    assert flow.loader.loads == [], "the row was reloaded from disk"
    assert flow.resident is not None and flow.resident.model is flow.model
    assert all(torch.equal(p, before[n]) for n, p in flow.model.named_parameters())
    assert identity_of(flow.resident).name == SAVED


async def test_attached_a_chat_naming_the_row_is_refused_and_the_resident_kept(flow):
    async with flow.client() as client:
        attachment_id = await _train_and_save(flow, client)
        before = params(flow.model)

        _assert_refused(await chat(client, ROW_NAME))
        _assert_untouched(flow, before)

        # The attachment is undisturbed and the saved name still answers.
        status = (await client.get("/api/weights/status")).json()["data"]
        assert status["attached"] is True and status["weights_state"] == "clean"
        assert (await chat(client, SAVED)).status_code == 200
        assert (await client.post("/api/weights/detach",
                                  headers=holder(attachment_id))).status_code == 200


async def test_detached_a_chat_naming_the_row_is_refused_and_the_resident_kept(flow):
    """The silent-loss case: nothing holds the weights, and an implicit reload would discard
    the trained values and the unsaved commit made after the save."""
    async with flow.client() as client:
        attachment_id = await _train_and_save(flow, client)
        await _commit(client, attachment_id, flow.model)  # an unsaved commit
        assert (await client.post("/api/weights/detach",
                                  headers=holder(attachment_id))).status_code == 200
        before = params(flow.model)

        _assert_refused(await chat(client, ROW_NAME))
        r = await client.post("/v1/completions",
                              json={"model": ROW_NAME, "prompt": "w1 w2", "max_tokens": 2})
        _assert_refused(r)
        _assert_untouched(flow, before)
        assert (await chat(client, SAVED)).status_code == 200


async def test_the_deliberate_load_still_restores_the_row(flow):
    async with flow.client() as client:
        attachment_id = await _train_and_save(flow, client)
        assert (await client.post("/api/weights/detach",
                                  headers=holder(attachment_id))).status_code == 200
        _assert_refused(await chat(client, ROW_NAME))

        loaded = await client.post("/api/weights/load", json={"model_id": ROW_ID}, headers=OPS)
        assert loaded.status_code == 200, loaded.text
        assert loaded.json()["data"]["mode"] == "full_reload"
        assert [load["model_id"] for load in flow.loader.loads] == [ROW_ID]
        r = await chat(client, ROW_NAME)
        assert r.status_code == 200, r.text
        assert r.headers["X-miLLM-Model"] == f'"{ROW_NAME}";origin=model;commits=0'
