"""Request and response schemas for the checkpoint routes (Feature 32, FTDD §5.1–§5.4).

Three routes: `POST /api/weights/save`, `POST /api/weights/load`, `GET /api/weights/lineage`.
Instantiate (FR-32.8) is DEFERRED (addendum scope), so it has no schema here.

The save body's `manifest` is typed `Any` on purpose: the vendored schema is the only authority on
its fields, and the service validates it so a refusal is `400 INVALID_MANIFEST` naming every field
by JSON path, not a bare 422 from a second, hand-kept copy of the field list.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SaveRequest(BaseModel):
    """`POST /api/weights/save`. The caller's manifest fields; miLLM writes the rest."""

    model_config = ConfigDict(extra="forbid")

    manifest: Any = Field(
        ...,
        description=(
            "The trainer-written manifest fields (run_id, step, domain, signal_source, parent, "
            "optimizer, and the optional ones). base_model, name, created_at and weights_sha256 "
            "are written by miLLM and refused here."
        ),
    )


class SaveResponse(BaseModel):
    """`201` body (FPRD FR-32.1.10)."""

    name: str
    path: str = Field(..., description="The step directory, relative to the checkpoint root")
    weights_sha256: str
    manifest_sha256: str
    bytes: int = Field(..., description="Total bytes written into the step directory")
    snapshot_ms: float = Field(..., description="How long serving was paused")
    write_ms: float = Field(..., description="Write, fsync, hash and publish, outside the pause")
    parent_check: Literal["match", "mismatch"]
    latest: str = Field(..., description="What `latest` now points at")
    identity_set: bool = Field(
        ...,
        description=(
            "False when a load replaced the resident weights while this checkpoint was being "
            "written: the checkpoint is published, but the resident model is not renamed to it."
        ),
    )


class LoadRequest(BaseModel):
    """`POST /api/weights/load`: exactly one of `name`, `path` or `model_id`."""

    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = Field(None, description="A fully qualified checkpoint name")
    path: Optional[str] = Field(
        None, description="A step directory, relative to a checkpoint root or absolute under one"
    )
    model_id: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "A plain model row to reload inside the barrier: the recovery path from `torn` when "
            "no checkpoint exists (addendum D-01). Never a checkpoint carrier row."
        ),
    )
    gpu: int | str | None = Field(None, description="Full reload only; as POST /api/models/{id}/load")
    trust_remote_code: bool = Field(
        False,
        description=(
            "Full reload only. A checkpoint whose config.json names `auto_map` (remote code) is "
            "refused unless this is true AND the token is an operator's (addendum D-23)."
        ),
    )

    @field_validator("gpu", mode="before")
    @classmethod
    def _gpu(cls, value: Any) -> int | str | None:
        from millm.api.schemas.model import ModelLoadRequest

        return ModelLoadRequest.validate_gpu(value)

    @model_validator(mode="after")
    def _exactly_one_target(self) -> "LoadRequest":
        given = [k for k in ("name", "path", "model_id") if getattr(self, k) is not None]
        if len(given) != 1:
            raise ValueError("give exactly one of name, path or model_id")
        return self


class LoadResponse(BaseModel):
    """`200` body (FPRD FR-32.5.13). Hashes are null for a `model_id` row reload."""

    name: str
    mode: Literal["in_place", "full_reload"]
    weights_sha256: Optional[str]
    manifest_sha256: Optional[str]
    verify_ms: float
    write_ms: float
    weights_state: str
    steps_since_load: int


class LineageResident(BaseModel):
    name: str
    origin: Literal["model", "checkpoint", "instantiated"]
    manifest_sha256: Optional[str]
    commits_since_identity: int


class LineageLink(BaseModel):
    name: str
    run_id: str
    step: int
    path: str = Field(..., description="The step directory, relative to its checkpoint root")
    root: str = Field(..., description="The checkpoint root the link was found under")
    manifest_sha256: str
    weights_sha256: str
    parent: Optional[str]
    merged_adapter: Optional[dict[str, Any]]


class LineageBreak(BaseModel):
    at: str
    reason: Literal[
        "manifest_missing", "manifest_invalid", "name_mismatch", "cycle", "depth_limit",
        "ambiguous",
    ]


class LineageResponse(BaseModel):
    """`GET /api/weights/lineage` (FTDD §5.4). Read from manifests; weights are never hashed."""

    model_config = ConfigDict(populate_by_name=True)

    resident: LineageResident
    base_model: str
    chain: list[LineageLink]
    complete: bool
    break_: Optional[LineageBreak] = Field(None, alias="break")
    weights_verified: Literal[False] = False
