"""The `millm` extension object on OpenAI-shaped responses, and the request fields that fill it.

Owned by Feature 27; Features 25, 26 and 28 add their own fields to `MillmExtension` here, so each
feature extends one model without touching the others (FTDD §12). The object is OMITTED from a
response that did not ask for it, so a client sending no new field receives exactly the body it
received before (FR-27.1e).
"""

from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Said on every activations block (FR-27.2e, T-77). Neither read point is a counterfactual:
#: steering at earlier layers, and tokens generated under steering, still shape the residual at
#: this layer.
READ_POINT_NOTE = (
    "{read_point} is what the model computed at this layer for this request; it is not an "
    "unsteered counterfactual — steering at earlier layers, and tokens generated under steering, "
    "still shape the residual here"
)


class PositionRange(BaseModel):
    """`{start, end}`: absolute positions over the processed sequence, half-open."""

    model_config = ConfigDict(extra="forbid")

    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> "PositionRange":
        if self.end <= self.start:
            raise ValueError("positions.end must be greater than positions.start")
        return self


class ReturnSaeActivations(BaseModel):
    """`return_sae_activations` on `/v1/chat/completions` and `/v1/completions` (FR-27.1).

    `positions` is absolute over the processed sequence (prompt, then generated tokens). The final
    sampled token is never fed back, so it has no activation (FR-27.1c). `features` restricts the
    candidates; `top_k` picks the largest among them, as the monitoring configuration does.
    """

    model_config = ConfigDict(extra="forbid")

    sae_id: Optional[str] = None
    features: Optional[list[int]] = Field(default=None, min_length=1)
    top_k: int = Field(ge=1)
    positions: Union[Literal["last", "prompt", "completion", "all"], PositionRange]
    read_point: Literal["post_steering", "pre_steering"] = "post_steering"

    @model_validator(mode="after")
    def _features_non_negative(self) -> "ReturnSaeActivations":
        if self.features is not None and any(f < 0 for f in self.features):
            raise ValueError("features must be non-negative SAE feature indices")
        return self


class SaeActivationFeature(BaseModel):
    index: int
    value: float


class SaeActivationPosition(BaseModel):
    position: int
    token_id: Optional[int] = None
    features: list[SaeActivationFeature]


class SaeActivationsBlock(BaseModel):
    sae_id: str
    layer: int
    read_point: Literal["post_steering", "pre_steering", "unsteered"]
    positions: list[SaeActivationPosition]
    note: str


class MillmExtension(BaseModel):
    """The `millm` object. Every field is optional and omitted when absent."""

    sae_activations: Optional[SaeActivationsBlock] = Field(
        default=None, exclude_if=lambda v: v is None
    )
