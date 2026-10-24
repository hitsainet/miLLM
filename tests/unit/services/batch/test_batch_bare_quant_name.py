"""D-46 (review R2-A F-B): a batch line naming a bare quantization repo name is refused at
VALIDATION, as the synchronous route refuses it, not at run time.

`ModelRepository.find_by_name` maps `org/q-GGUF` onto its one tagged row `org/q-GGUF:Q4_K_M`.
Before D-46 the batch validator accepted such a line (the row resolved), the batch went
`in_progress`, and every line then failed the D-31 run-time re-check with 404 — validation and
run disagreeing. Now the shared resolver refuses it up front with the same 404 `model_not_found`.

Real app, real BatchRunner, real ModelService over the REAL repository on SQLite, a tiny real
model served under the tagged name (tests/unit/batch_fixtures.py).

Mutation control FB-1 (progress log): the resolver's bare-name refusal removed -> both cases red.
"""

from __future__ import annotations

import pytest_asyncio

from tests.unit.batch_fixtures import (  # noqa: F401
    BatchHarness,
    batch_db,
    batch_dir,
    client_for,
    completion_line,
)

BARE = "org/q-GGUF"
TAGGED = "org/q-GGUF:Q4_K_M"


@pytest_asyncio.fixture
async def tagged(batch_db, batch_dir):  # noqa: F811
    from tests.unit.f25_fixtures import clear_loaded

    h = BatchHarness(batch_db, model_name=TAGGED)
    await h.seed_model(name=TAGGED)
    yield h
    await h.runner.stop()
    clear_loaded()


async def _create(h, lines):
    async with client_for(h.app()) as client:
        created = await h.create(client, lines)
        assert created.status_code == 200, created.text
        await h.settle_validation()
        batch = await h.batch(created.json()["id"])
        got = (await client.get(f"/v1/batches/{batch.id}")).json()
        errors = await h.lines(client, batch.error_file_id)
    return batch, got, errors


async def test_a_bare_name_line_is_refused_at_validation(tagged):
    batch, got, errors = await _create(tagged, [
        completion_line("bare-1", model=BARE),
        completion_line("bare-2", model=BARE),
    ])
    assert batch.status == "failed", got
    assert got["request_counts"] == {"total": 2, "completed": 0, "failed": 2}
    data = got["errors"]["data"]
    assert [(e["line"], e["code"], e["param"]) for e in data] == [
        (1, "model_not_found", "body.model"), (2, "model_not_found", "body.model")]
    assert data[0]["message"] == (
        f"Line 1: The model '{BARE}' does not exist. Available: {TAGGED}")
    # The error FILE has one validation refusal per line, and nothing ran.
    assert [(e["custom_id"], e["error"]["code"], e["response"]) for e in errors] == [
        ("bare-1", "model_not_found", None), ("bare-2", "model_not_found", None)]
    assert got["in_progress_at"] is None


async def test_the_exact_tagged_name_validates_and_runs(tagged):
    batch, got, _ = await _create(tagged, [completion_line("ok", model=TAGGED)])
    assert batch.status == "in_progress", got
    assert got["request_counts"]["failed"] == 0
