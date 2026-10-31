"""Request schemas for the Batch API (Feature 26, FR-26.1.2).

Responses are built by `millm.services.batch.state` (`batch_object`, `file_object`,
`openai_list`) so the route, the socket event and the tests read ONE serialiser.

`BatchCreateRequest` is `extra="allow"`: an undeclared field is REPORTED in
`X-miLLM-Ignored-Fields`, or refused under `X-miLLM-Strict: true` — Feature 25's rule (FR-26.1.6),
applied by `request_policy.evaluate_control`. It is never silently dropped.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

#: OpenAI's bounds for `output_expires_after.seconds` (FR-26.9.5).
OUTPUT_EXPIRES_MIN_S = 3_600
OUTPUT_EXPIRES_MAX_S = 2_592_000


class OutputExpiresAfter(BaseModel):
    """OpenAI's `{anchor: "created_at", seconds: N}`. Bounds are checked by the route, which names
    them in the refusal (a pydantic range error would not say "3600 to 2592000")."""

    model_config = ConfigDict(extra="forbid")

    anchor: Literal["created_at"] = "created_at"
    seconds: int


class BatchCreateRequest(BaseModel):
    """`POST /v1/batches`."""

    model_config = ConfigDict(extra="allow")

    input_file_id: str = Field(min_length=1, max_length=64)
    endpoint: str = Field(min_length=1, max_length=64)
    completion_window: str = Field(min_length=1, max_length=16)
    #: OpenAI's: up to 16 string pairs.
    metadata: Optional[dict[str, str]] = Field(default=None, max_length=16)
    output_expires_after: Optional[OutputExpiresAfter] = None
    #: miLLM extension (FR-26.5.1, T-63): omitted → BATCH_PACK_DEFAULT.
    pack: Optional[bool] = None
