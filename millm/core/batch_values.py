"""The Batch API's value sets (Feature 26) — pure enums, importable from the ORM layer.

They live in `core` so `millm.db.models.batch` can build its CHECK constraints from them without
importing the `millm.services` package (whose `__init__` pulls in the model service). The
transition table and serialisers that use them are in `millm.services.batch.state`.
"""

from __future__ import annotations

from enum import StrEnum


class BatchStatus(StrEnum):
    """OpenAI's batch statuses (FR-26.3.1). THE list."""

    VALIDATING = "validating"
    IN_PROGRESS = "in_progress"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


TERMINAL: frozenset[BatchStatus] = frozenset(
    {BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED, BatchStatus.EXPIRED}
)


class RowState(StrEnum):
    """One input line's state."""

    PENDING = "pending"
    INVALID = "invalid"
    DONE = "done"
    FAILED = "failed"


class RowKind(StrEnum):
    """What a valid line does when it runs (FTID §7 step 9)."""

    SCORING = "scoring"
    GENERATION = "generation"
    EMBEDDING = "embedding"
    PROBE = "probe"


class FileStatus(StrEnum):
    """OpenAI's `processed`/`error`, plus the two retention states."""

    PROCESSED = "processed"
    ERROR = "error"
    DELETED = "deleted"
    EXPIRED = "expired"


class FilePurpose(StrEnum):
    BATCH = "batch"
    BATCH_OUTPUT = "batch_output"


class WaitingReason(StrEnum):
    QUEUED = "queued"
    MODEL_NOT_RESIDENT = "model_not_resident"
    LEASE_UNAVAILABLE = "lease_unavailable"
