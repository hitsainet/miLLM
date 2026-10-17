"""`POST /api/probes/score` — stateless probe scoring (Feature 27, FR-27.4).

Every model is `extra="forbid"`: a misspelt field is a 422, never a silent drop (FTDD §4).
Shape refusals — zero or too many inputs, too many probes, two kinds in one input,
`prompt_tokens` past the input, the `text` verification gate — are `INVALID_PROBE_SCORE_REQUEST`
(400) from `probe_scoring.check_shape`, per the FTDD §5.1 refusal table, and read the caps from
settings at request time rather than freezing them into the schema at import.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from millm.services.probe_scope import WINDOWS


class ScoreMessage(BaseModel):
    """One chat turn of a `messages` input."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class ProbeScoreInput(BaseModel):
    """Exactly one of `token_ids`, `messages`, `text` (FR-27.4a).

    `token_ids` is authoritative when given and may carry `prompt_tokens`, the prompt length
    (T-72). `prompt_tokens` with any other kind is refused: for `messages` it is DERIVED, and a
    caller-supplied one could disagree with the render.
    """

    model_config = ConfigDict(extra="forbid")

    token_ids: Optional[list[int]] = Field(default=None, min_length=1)
    prompt_tokens: Optional[int] = Field(default=None, ge=0)
    messages: Optional[list[ScoreMessage]] = Field(default=None, min_length=1)
    text: Optional[str] = Field(default=None, min_length=1)

    def kinds(self) -> list[str]:
        """The input kinds this input carries. Exactly one is valid (`check_shape` refuses the
        rest with `INVALID_PROBE_SCORE_REQUEST`, a 400, per the FTDD's refusal table)."""
        return [k for k in ("token_ids", "messages", "text") if getattr(self, k) is not None]


class ProbeScoreRequest(BaseModel):
    """`{probe_ids?, inputs, windows?, return_token_ids?}`.

    `probe_ids` omitted means every imported probe that matches the loaded model; the rest are
    listed as skipped with their reason (T-75). Given, any mismatch refuses the request.
    `windows` omitted gives the arm-time default (`resolve_windows`).
    """

    model_config = ConfigDict(extra="forbid")

    probe_ids: Optional[list[str]] = Field(default=None, min_length=1)
    inputs: list[ProbeScoreInput]
    windows: Optional[list[str]] = None
    return_token_ids: bool = False

    @field_validator("windows")
    @classmethod
    def _known_windows(cls, value: Optional[list[str]]) -> Optional[list[str]]:
        for name in value or ():
            if name not in WINDOWS:
                raise ValueError(f"unknown window {name!r}; known: {', '.join(WINDOWS)}")
        return value
