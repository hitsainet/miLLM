"""The `X-miLLM-*` provenance a response carries — built in ONE place (Feature 26, FR-26.10.1).

The synchronous routes set these as response headers; the Batch API puts the SAME dict under each
output line's `response.millm.headers`. Before this module the header construction lived inline
in each route body, and a batch line would have needed a second copy — which is how a header gets
lost in one of the two (FTDD §12 risk "A route header lost in batch lines").

Two halves, because two of the values must be resolved BEFORE generation (a streaming response
commits its headers before the first byte) and the rest only exist AFTER it (the seed scope the
path that ran recorded, the constraint it applied, the probe verdicts, whether the circuit dial
actually applied):

* `pre_generation(request, inference, chat=...)` — resets the request-scoped context, resolves
  the steering-intensity echo and the circuit-rung phrase.
* `post_generation(...)` — the header dict for a finished non-streaming response.
* `finish_body(result, model_row, inference)` — the body-level `system_fingerprint` (FR-25.13.8).

⚠ Feature 28's `X-miLLM-Steering` value (FR-28.3.10) is NOT in this tree; today the header is set
only on a scoring response carrying activations (X-09). It joins `post_generation` when 028 lands,
and every batch line then carries it with no batch-side change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from millm.api.request_policy import IGNORED_FIELDS_HEADER
from millm.services.inference_service import (
    circuit_apply_failed,
    get_probe_verdicts,
    get_request_outcome,
    reset_steering_memo,
)
from millm.services.system_fingerprint import build_system_fingerprint


def seed_header(seed: int, scope: str) -> str:
    """`X-miLLM-Seed: 7;scope="request"` (FR-25.13.6, FR-25.14). Sent only when the request
    carried a seed — miLLM never chooses one of its own (T-60)."""
    return f'{seed};scope="{scope}"'


@dataclass(frozen=True)
class PreGeneration:
    """What must be known before generation starts."""

    echo_intensity: Optional[str] = None
    echo_circuit_rung: Optional[str] = None


async def pre_generation(request: Any, inference: Any, *, chat: bool) -> PreGeneration:
    """Reset the request-scoped context and resolve the two pre-generation echoes.

    ⚠ The reset is not optional for a batch: every row of a chunk runs in ONE task, so without
    it row N would report row N-1's probe verdicts and seed scope.
    """
    reset_steering_memo()
    if not chat:
        return PreGeneration()
    echo_circuit_rung = None
    try:
        rung_info = await inference.active_circuit_rung()
        if rung_info is not None:
            # Structured (RFC 8941): the rung stays trivially parseable as an int and the phrase
            # is a quoted-string, so punctuation in the ladder vocabulary can never break a
            # naive parser.
            echo_circuit_rung = f'{rung_info[0]}; language="{rung_info[1]}"'
    except Exception:  # noqa: BLE001 - observability must never fail a chat request
        echo_circuit_rung = None
    echo_intensity = None
    if getattr(request, "steering_intensity", None) is not None:
        # For streaming, the echo resolution doubles as the pre-commit 404 check for a named
        # profile (one profile read, not two).
        effective = await inference.resolve_request_intensity(
            request, ensure_named_profile=bool(getattr(request, "stream", False))
        )
        if effective is not None:
            echo_intensity = f"{effective:g}"
    return PreGeneration(echo_intensity=echo_intensity, echo_circuit_rung=echo_circuit_rung)


def post_generation(
    request: Any,
    inference: Any,
    *,
    endpoint: str,
    pre: PreGeneration,
    ignored_header: Optional[str],
) -> dict[str, str]:
    """Every `X-miLLM-*` header a finished non-streaming response carries, from one function.

    `endpoint` is `chat`, `completions` or `embeddings`.
    """
    from millm.api.routes.openai.chat import build_probe_verdicts_header

    headers: dict[str, str] = {"X-miLLM-Backend": inference.backend_name}
    if endpoint == "embeddings":
        if ignored_header:
            headers[IGNORED_FIELDS_HEADER] = ignored_header
        return headers
    if endpoint == "chat":
        # X-miLLM-Batch advertises the batched-generation extension. Both request schemas
        # accept `extra_messages` as an extra field on an older server and return a single
        # choice; this header is the capability probe that makes the difference observable.
        extra = getattr(request, "extra_messages", None)
        headers["X-miLLM-Batch"] = str(len(extra) + 1 if extra else 1)
        if pre.echo_intensity is not None:
            headers["X-miLLM-Steering-Intensity"] = pre.echo_intensity
    if ignored_header:
        headers[IGNORED_FIELDS_HEADER] = ignored_header
    outcome = get_request_outcome()
    if getattr(request, "seed", None) is not None:
        # The scope the path that RAN recorded; the up-front rule only if it recorded none.
        headers["X-miLLM-Seed"] = seed_header(
            request.seed, outcome.get("seed_scope") or inference.seed_scope_for(request)
        )
    if endpoint == "chat" and outcome.get("constrained"):
        headers["X-miLLM-Constrained"] = outcome["constrained"]
    # F18 R3-01: the rung header is decided AFTER generation — the dial can fail to apply, and a
    # header claiming causal evidence for an intervention that did not run would be false.
    if pre.echo_circuit_rung is not None and not circuit_apply_failed():
        headers["X-miLLM-Circuit-Rung"] = pre.echo_circuit_rung
    # Probe verdicts (FR-24.7, FR-27.8g), read after generation: they do not exist before.
    probe_header = build_probe_verdicts_header(get_probe_verdicts())
    if probe_header:
        headers["X-miLLM-Probe-Verdicts"] = probe_header
    if request.wants_scores() and getattr(request, "return_sae_activations", None) is not None:
        # X-09: scoring is always unsteered, and a scoring response carrying activations says so.
        headers["X-miLLM-Steering"] = "none"
    return headers


def finish_body(result: Any, model_row: Any, inference: Any) -> Any:
    """Set the body-level provenance the routes set (FR-25.13.8)."""
    if hasattr(result, "system_fingerprint"):
        result.system_fingerprint = build_system_fingerprint(model_row, inference.loaded_model())
    return result
