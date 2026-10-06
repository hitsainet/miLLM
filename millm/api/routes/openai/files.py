"""OpenAI-shaped files routes for the Batch API (Feature 26, FR-26.1.1, FR-26.6.5, FR-26.6.9-10).

POST   /v1/files                 upload a JSONL file (`purpose: "batch"`), caps enforced while copying
GET    /v1/files                 list, newest first, filterable by `purpose`
GET    /v1/files/{id}            the file object
GET    /v1/files/{id}/content    the bytes, `Content-Type: application/jsonl`
DELETE /v1/files/{id}            delete the bytes; 409 while a non-terminal batch references it

⚠ The upload route reads `Content-Length` BEFORE it parses the multipart body, so an upload that
declares more than the cap is refused without reading it (FR-26.8.3). That is why the body is
parsed by hand (`request.form()`) rather than declared as `UploadFile` parameters: FastAPI parses
declared form parameters before the handler runs. The OpenAPI document still describes it
(`openapi_extra`). Every miLLM route accepts `X-miLLM-Load-Policy` and `X-miLLM-Lease` and is not
broken by them (FR-26.1.7); these routes ignore both.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Annotated, Any, Literal, Optional

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse

from millm.api.dependencies import DbSession
from millm.core.batch_values import FilePurpose, FileStatus
from millm.core.config import settings
from millm.core.errors import (
    BatchFileDeletedError,
    BatchFileExpiredError,
    BatchFileInUseError,
    BatchFileLimitError,
    BatchFileNotFoundError,
    InvalidBatchRequestError,
)
from millm.core.logging import get_logger
from millm.db.models.batch import BatchFile
from millm.db.repositories.batch_repository import BatchRepository
from millm.services.batch.files import get_file_store, new_file_id
from millm.services.batch.state import file_object, openai_list, utcnow

router = APIRouter()
logger = get_logger(__name__)

#: The results media type (FTDD TD6; miStudio 034 FR-22 reads it). Stated once.
JSONL_MEDIA_TYPE = "application/jsonl"
#: Room for multipart boundaries and the `purpose` field over the file cap.
MULTIPART_ALLOWANCE = 64 * 1024

_UPLOAD_SCHEMA = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["file", "purpose"],
                    "properties": {
                        "file": {"type": "string", "format": "binary"},
                        "purpose": {"type": "string", "enum": ["batch"]},
                    },
                }
            }
        },
    }
}


async def _get(repo: BatchRepository, file_id: str) -> BatchFile:
    row = await repo.get_file(file_id)
    if row is None:
        raise BatchFileNotFoundError(f"No file {file_id!r}.", details={"param": "file_id"})
    return row


@router.post("/files", openapi_extra=_UPLOAD_SCHEMA)
async def upload_file(request: Request, session: DbSession) -> dict[str, Any]:
    """Upload a JSONL batch input file (FR-26.1.1). Over a cap → 400 naming it; nothing stored."""
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit():
        if int(declared) > settings.BATCH_MAX_FILE_BYTES + MULTIPART_ALLOWANCE:
            raise BatchFileLimitError(
                f"The upload declares {int(declared)} bytes; the file limit is "
                f"{settings.BATCH_MAX_FILE_BYTES} bytes (BATCH_MAX_FILE_BYTES). Nothing was read "
                "or stored; split the file.",
                details={"param": "file", "limit": "bytes", "max": settings.BATCH_MAX_FILE_BYTES,
                         "measured": int(declared)},
            )
    form = await request.form(max_files=1, max_fields=8)
    try:
        purpose = form.get("purpose")
        if purpose != FilePurpose.BATCH.value:
            raise InvalidBatchRequestError(
                f"purpose must be 'batch'; got {purpose!r}.", details={"param": "purpose"}
            )
        upload = form.get("file")
        if upload is None or isinstance(upload, str) or not hasattr(upload, "file"):
            raise InvalidBatchRequestError(
                "A multipart field 'file' holding the JSONL file is required.",
                details={"param": "file"},
            )
        now = utcnow()
        file_id = new_file_id()
        store = get_file_store()
        path = store.relative_path(file_id, now)
        stored = await asyncio.to_thread(
            store.write_upload, upload.file, path,
            max_bytes=settings.BATCH_MAX_FILE_BYTES, max_rows=settings.BATCH_MAX_ROWS,
        )
        filename = (getattr(upload, "filename", None) or "upload.jsonl")[:255]
    finally:
        await form.close()
    row = BatchFile(
        id=file_id, purpose=FilePurpose.BATCH.value, filename=filename, bytes=stored.bytes,
        line_count=stored.line_count, storage_path=stored.storage_path, sha256=stored.sha256,
        status=FileStatus.PROCESSED.value, created_at=now,
        expires_at=now + timedelta(days=settings.BATCH_FILE_RETENTION_DAYS),
    )
    try:
        await BatchRepository(session).create_file(row)
    except BaseException:
        await asyncio.to_thread(store.delete, stored.storage_path)
        raise
    logger.info("batch_file_uploaded", file_id=file_id, bytes=stored.bytes, lines=stored.line_count)
    return file_object(row)


@router.get("/files")
async def list_files(
    session: DbSession,
    purpose: Annotated[Optional[str], Query()] = None,
    limit: Annotated[int, Query(ge=1, le=10000)] = 100,
    after: Annotated[Optional[str], Query()] = None,
    order: Annotated[Literal["asc", "desc"], Query()] = "desc",
) -> dict[str, Any]:
    """Files newest first, in OpenAI's list shape (FR-26.6.9, T-69)."""
    rows, more = await BatchRepository(session).list_files(
        purpose=purpose, limit=limit, after=after, order=order
    )
    return openai_list([file_object(r) for r in rows], more)


@router.get("/files/{file_id}")
async def get_file(file_id: str, session: DbSession) -> dict[str, Any]:
    return file_object(await _get(BatchRepository(session), file_id))


@router.get("/files/{file_id}/content")
async def get_file_content(file_id: str, session: DbSession) -> StreamingResponse:
    """The bytes, streamed, with one stated media type (FR-26.6.5)."""
    row = await _get(BatchRepository(session), file_id)
    if row.status == FileStatus.EXPIRED.value:
        raise BatchFileExpiredError(
            f"File {file_id} expired at {row.expires_at.isoformat()}; its content was removed "
            "by retention. The record remains.",
            details={"param": "file_id"},
        )
    store = get_file_store()
    if row.status == FileStatus.DELETED.value or not store.exists(row.storage_path):
        raise BatchFileDeletedError(
            f"File {file_id} was deleted; its content is gone.", details={"param": "file_id"}
        )
    return StreamingResponse(
        store.iter_bytes(row.storage_path),
        media_type=JSONL_MEDIA_TYPE,
        headers={"Content-Length": str(row.bytes)},
    )


@router.delete("/files/{file_id}")
async def delete_file(file_id: str, session: DbSession) -> dict[str, Any]:
    """Delete the bytes and mark the record (FR-26.6.10). Refused while a live batch needs it."""
    repo = BatchRepository(session)
    row = await _get(repo, file_id)
    if row.status in (FileStatus.DELETED.value, FileStatus.EXPIRED.value):
        raise BatchFileDeletedError(
            f"File {file_id} is already {row.status}.", details={"param": "file_id"}
        )
    referencing = await repo.batch_referencing(file_id)
    if referencing is not None:
        raise BatchFileInUseError(
            f"File {file_id} is in use by batch {referencing}, which has not finished; cancel "
            "it or wait for it to end.",
            details={"param": "file_id", "batch_id": referencing},
        )
    await repo.mark_file(file_id, FileStatus.DELETED, utcnow())
    await asyncio.to_thread(get_file_store().delete, row.storage_path)
    return {"id": file_id, "object": "file", "deleted": True}
