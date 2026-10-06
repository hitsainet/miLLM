"""Feature 26 task 3.7: the files surface, through the REAL app (`create_app()`).

Reachability: every route is asserted present in the live OpenAPI document AND called; the
payload is asserted, not just the status. Removing the `files_router` include turns every test
here red (mutation control M12).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from millm.core.config import settings
from tests.unit.batch_fixtures import (  # noqa: F401  (fixtures)
    app_with_db,
    batch_db,
    batch_dir,
    client_for,
    completion_line,
    jsonl,
    upload,
)

FILE_FIELDS = {
    "id", "object", "bytes", "created_at", "expires_at", "filename", "purpose", "status",
    "status_details",
}


def _stored_files(root) -> list:
    return [p for p in root.rglob("*") if p.is_file()] if root.exists() else []


def test_every_files_route_is_in_the_live_openapi_document():
    from millm.main import create_app

    paths = create_app().openapi()["paths"]
    assert {"post", "get"} <= set(paths["/v1/files"])
    assert "get" in paths["/v1/files/{file_id}"] and "delete" in paths["/v1/files/{file_id}"]
    assert "get" in paths["/v1/files/{file_id}/content"]
    assert "multipart/form-data" in paths["/v1/files"]["post"]["requestBody"]["content"]


async def test_upload_returns_openais_file_object(batch_db, batch_dir):
    data = jsonl([completion_line("a"), completion_line("b")])
    async with client_for(app_with_db(batch_db)) as client:
        response = await upload(
            client, data, headers={"X-miLLM-Lease": "x", "X-miLLM-Load-Policy": "refuse"}
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == FILE_FIELDS
    assert body["object"] == "file" and body["purpose"] == "batch"
    assert body["bytes"] == len(data) and body["filename"] == "in.jsonl"
    assert body["status"] == "processed" and body["id"].startswith("file-")
    assert body["expires_at"] - body["created_at"] == 30 * 86400  # T-68
    stored = _stored_files(batch_dir)
    assert len(stored) == 1 and stored[0].read_bytes() == data
    assert "in.jsonl" not in str(stored[0]), "the client filename reached the filesystem"


async def test_a_bad_purpose_is_refused_naming_it(batch_db, batch_dir):
    async with client_for(app_with_db(batch_db)) as client:
        response = await upload(client, b"{}\n", purpose="fine-tune")
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["param"] == "purpose" and error["type"] == "invalid_request_error"
    assert _stored_files(batch_dir) == []


async def test_an_upload_over_the_row_limit_is_refused_and_nothing_stored(
    batch_db, batch_dir, monkeypatch
):
    monkeypatch.setattr(settings, "BATCH_MAX_ROWS", 3)
    data = jsonl([completion_line(str(i)) for i in range(5)])
    async with client_for(app_with_db(batch_db)) as client:
        response = await upload(client, data)
        listed = await client.get("/v1/files")
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "batch_file_limit"
    assert "5 lines" in error["message"] and "3" in error["message"]
    assert _stored_files(batch_dir) == []
    assert listed.json()["data"] == []


async def test_an_upload_over_the_byte_limit_is_refused_while_copying(
    batch_db, batch_dir, monkeypatch
):
    monkeypatch.setattr(settings, "BATCH_MAX_FILE_BYTES", 100)
    data = jsonl([completion_line(str(i)) for i in range(10)])
    async with client_for(app_with_db(batch_db)) as client:
        response = await upload(client, data)
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "batch_file_limit"
    assert f"{len(data)} bytes" in error["message"], "the MEASURED size must be named"
    assert _stored_files(batch_dir) == []


async def test_a_declared_oversize_upload_is_refused_before_the_body_is_read(
    batch_db, batch_dir, monkeypatch
):
    monkeypatch.setattr(settings, "BATCH_MAX_FILE_BYTES", 10)
    from millm.api.routes.openai.files import MULTIPART_ALLOWANCE

    async with client_for(app_with_db(batch_db)) as client:
        response = await client.post(
            "/v1/files",
            content=b"x",
            headers={
                "Content-Length": str(10 + MULTIPART_ALLOWANCE + 1),
                "Content-Type": "multipart/form-data; boundary=zz",
            },
        )
    assert response.status_code == 400
    assert "declares" in response.json()["error"]["message"]


async def test_content_streams_with_the_stated_media_type(batch_db, batch_dir):
    data = jsonl([completion_line("a")])
    async with client_for(app_with_db(batch_db)) as client:
        file_id = (await upload(client, data)).json()["id"]
        got = await client.get(f"/v1/files/{file_id}")
        content = await client.get(f"/v1/files/{file_id}/content")
    assert got.json()["id"] == file_id
    assert content.status_code == 200
    assert content.headers["content-type"] == "application/jsonl"
    assert content.content == data


async def test_unknown_ids_are_404(batch_db, batch_dir):
    async with client_for(app_with_db(batch_db)) as client:
        a = await client.get("/v1/files/file-nope")
        b = await client.get("/v1/files/file-nope/content")
        c = await client.delete("/v1/files/file-nope")
    assert [r.status_code for r in (a, b, c)] == [404, 404, 404]
    assert a.json()["error"]["code"] == "file_not_found"


async def test_list_is_newest_first_filterable_and_paged(batch_db, batch_dir):
    async with client_for(app_with_db(batch_db)) as client:
        ids = [(await upload(client, jsonl([completion_line(str(i))]))).json()["id"]
               for i in range(3)]
        everything = (await client.get("/v1/files")).json()
        page = (await client.get("/v1/files", params={"limit": 2})).json()
        rest = (await client.get("/v1/files", params={"limit": 2, "after": page["last_id"]})).json()
        other = (await client.get("/v1/files", params={"purpose": "batch_output"})).json()
    assert everything["object"] == "list"
    assert [f["id"] for f in everything["data"]] == list(reversed(ids))
    assert page["has_more"] is True and len(page["data"]) == 2
    assert page["first_id"] == ids[2] and page["last_id"] == ids[1]
    assert [f["id"] for f in rest["data"]] == [ids[0]] and rest["has_more"] is False
    assert other["data"] == []


async def test_delete_is_refused_while_a_live_batch_references_it_and_allowed_after(
    batch_db, batch_dir
):
    from millm.db.models.batch import Batch

    async with client_for(app_with_db(batch_db)) as client:
        file_id = (await upload(client, jsonl([completion_line("a")]))).json()["id"]
        now = datetime.now(timezone.utc)
        async with batch_db() as session:
            session.add(Batch(
                id="batch_live", endpoint="/v1/completions", completion_window="24h",
                status="in_progress", input_file_id=file_id, pack=True,
                output_expires_after_s=2_592_000, created_at=now,
                expires_at=now + timedelta(hours=24),
            ))
            await session.commit()
        refused = await client.delete(f"/v1/files/{file_id}")
        async with batch_db() as session:
            batch = await session.get(Batch, "batch_live")
            batch.status = "completed"
            await session.commit()
        allowed = await client.delete(f"/v1/files/{file_id}")
        content = await client.get(f"/v1/files/{file_id}/content")
        again = await client.get(f"/v1/files/{file_id}")
    assert refused.status_code == 409
    assert refused.json()["error"]["code"] == "file_in_use"
    assert "batch_live" in refused.json()["error"]["message"]
    assert allowed.status_code == 200
    assert allowed.json() == {"id": file_id, "object": "file", "deleted": True}
    assert _stored_files(batch_dir) == []
    assert content.status_code == 404 and content.json()["error"]["code"] == "file_deleted"
    assert again.json()["status"] == "deleted"
