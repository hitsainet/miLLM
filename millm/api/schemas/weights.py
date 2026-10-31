"""Response models for the weight-attachment routes (Feature 31; 031 FTDD §5.2-§5.3, FPRD
FR-31.16).

Every route answers in the management envelope (`ApiResponse[...]`). Field names are the
contract the Admin UI's Weights card reads (`admin-ui/src/types/api.ts`, `WeightsStatus`) and the
manual documents (`manual/docs/api/weights.md`); rename nothing without both.

⚠ NO MODEL HERE CARRIES THE ATTACHMENT ID except `AttachResponse` (returned once, FR-31.1.6),
and none carries a token (FR-31.14.9). `WeightsStatus` forbids extra fields, so a status dict
that grew an `attachment_id` key fails validation instead of reaching the unauthenticated
status route (FR-31.16.2).

`null` means "not recorded" or "not applicable", never zero: `steps_since_load` is null when
nothing is resident, and `last_save` stays null until Feature 32 records a save.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from millm.api.schemas.weights_adapters import AdapterStatus

WeightsStateLiteral = Literal["clean", "writing", "torn"]
BarrierLiteral = Literal["serving", "quiesced"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- attach (FTDD §5.3) ------------------------------------------------------------------------


class AttachedModel(_Strict):
    model_id: int
    name: Optional[str]
    manifest_hash: Optional[str] = Field(
        None, description="The loaded checkpoint's manifest hash; null when not loaded from one."
    )


class HeartbeatTerms(_Strict):
    timeout_s: float
    interval_hint_s: float


class ProducerVersions(_Strict):
    torch: Optional[str] = None
    cuda: Optional[str] = None
    driver: Optional[str] = None


class StorageHandle(BaseModel):
    """One CUDA storage, with the fields PyTorch's reducer emits (FR-31.1.3)."""

    model_config = ConfigDict(extra="allow")

    index: int


class ParameterHandle(BaseModel):
    """One parameter: its module path and where it lives in a storage (FR-31.1.1)."""

    model_config = ConfigDict(extra="allow")

    name: str
    storage: int


class ReleaseTerms(_Strict):
    """The release contract (addendum D-47): what an attachment pins until it is released."""

    pinned_bytes: int = Field(
        ..., description="Bytes of GPU memory the exported storages hold; they stay reserved "
        "until every handle is rebuilt once and released, or miLLM restarts.",
    )
    contract: str = Field(..., description="What the consumer must do to release them.")


class AttachResponse(_Strict):
    attachment_id: str = Field(
        ..., description="Returned exactly once; send it as X-miLLM-Attachment on holder calls."
    )
    model: AttachedModel
    weights_state: WeightsStateLiteral
    steps_since_load: int
    heartbeat: HeartbeatTerms
    producer: ProducerVersions
    storages: list[StorageHandle]
    parameters: list[ParameterHandle]
    release: ReleaseTerms
    config: Optional[dict[str, Any]] = Field(
        None,
        description="The served model's architecture config exactly as loaded (its own JSON "
        "form, every attribute stated), so a trainer rebuilds the graph around the handles "
        "from the config this process instantiated. Null when the model has no Hugging Face "
        "config. Additive (suite decision D39, 2026-10-17).",
    )


# --- holder routes (FTDD §5.2) -----------------------------------------------------------------


class DetachResponse(_Strict):
    weights_state: WeightsStateLiteral
    barrier: BarrierLiteral


class HeartbeatResponse(_Strict):
    attachment_state: Literal["active"]
    timeout_s: float
    interval_hint_s: float = Field(
        description="The suggested heartbeat interval (WEIGHTS_HEARTBEAT_INTERVAL_HINT_S)."
    )
    weights_state: WeightsStateLiteral


class QuiesceResponse(_Strict):
    held_requests: int
    held_batch_chunks: int
    weights_state: WeightsStateLiteral
    quiesced_at: str
    hold_deadline: str


class ResumeResponse(_Strict):
    barrier: BarrierLiteral
    weights_state: WeightsStateLiteral


class CommitBeginResponse(_Strict):
    weights_state: WeightsStateLiteral
    began_at: str
    watchdog_ms: int


class CommitEndResponse(_Strict):
    weights_state: WeightsStateLiteral
    steps_since_load: int
    commit_ms: float
    overran: bool


class ClearedAttachment(_Strict):
    role: str
    label: str
    stale_since: Optional[str]


class ClearStaleResponse(_Strict):
    cleared: ClearedAttachment
    weights_state: WeightsStateLiteral
    barrier: BarrierLiteral


# --- status (FPRD FR-31.16.1) ------------------------------------------------------------------


class AttachmentStatus(_Strict):
    state: Literal["active", "stale"]
    role: str
    label: str
    attached_at: str
    last_heartbeat_at: Optional[str]
    stale_since: Optional[str]


class ResidentModelStatus(_Strict):
    model_id: int
    name: Optional[str]
    manifest_hash: Optional[str]


class CudaMemory(_Strict):
    """One card, as this process's CUDA caching allocator reports it."""

    device_index: int
    allocated_mb: int
    reserved_mb: int


class LastSave(_Strict):
    checkpoint_sha256: str
    at: str
    path: str


class CommitStatus(_Strict):
    began_at: Optional[str]
    open_ms: Optional[float]
    watchdog_ms: int
    overran: bool


class LastOverrun(_Strict):
    at: str
    commit_ms: float


class WeightsStatus(_Strict):
    """`GET /api/weights/status` (FR-31.16.1). No attachment ID, no token (FR-31.16.2)."""

    attached: bool
    attachment: Optional[AttachmentStatus]
    barrier: BarrierLiteral
    quiesced_at: Optional[str]
    held_requests: Optional[int] = Field(
        None, description="Interactive requests waiting while quiesced; null while serving."
    )
    weights_state: WeightsStateLiteral
    resident_model: Optional[ResidentModelStatus]
    steps_since_load: Optional[int] = Field(
        None, description="Completed commits since the parameter set became resident; null "
        "when nothing is resident (not recorded), never 0 for that case.",
    )
    last_save: Optional[LastSave]
    commit: Optional[CommitStatus]
    commit_overran: bool
    last_hold_timeout_at: Optional[str]
    last_overrun: Optional[LastOverrun]
    pause_unavailable: Optional[Literal["cbm", "concurrency"]] = Field(
        None,
        description="Why the barrier cannot pause serving right now — continuous batching is "
        "enabled (`cbm`) or the request queue has more than one slot (`concurrency`) — so "
        "quiesce, a weights load and an adapter change would be refused with "
        "BARRIER_UNAVAILABLE; null when a pause is possible.",
    )
    adapter: Optional[AdapterStatus] = Field(
        None,
        description="The adapter block, the same model GET /api/weights/adapter returns "
        "(addendum D-28); null when it could not be read — the status never fails over it.",
    )
    cuda_memory: list[CudaMemory] | None = Field(
        None,
        description="Allocated and reserved MiB per card in this process (addendum D-47). With "
        "nothing resident, allocated memory well above zero was never released by a consumer "
        "(handles never rebuilt, or a trainer that died or exited holding them); only a restart "
        "returns it. Null when CUDA is not available.",
    )
    exports_since_start: int = Field(
        0, description="Handle exports (attaches) this process has made since it started.",
    )


class WeightsHealthSummary(BaseModel):
    """The `weights` block of `/api/health/detailed` (FR-31.16.3). `error` replaces the facts
    when the barrier or the attachment could not be read; the endpoint itself never fails."""

    weights_state: Optional[WeightsStateLiteral] = None
    barrier: Optional[BarrierLiteral] = None
    attached: Optional[bool] = None
    attachment_state: Optional[Literal["active", "stale"]] = None
    error: Optional[str] = None

