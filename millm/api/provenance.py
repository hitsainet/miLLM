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

Feature 28 (FR-28.3.10): `X-miLLM-Steering` is set here, from the report the generation path
PUBLISHED (`get_steering_report()`, computed from a snapshot of what the hooks applied — never
from the request). The synchronous routes and every Batch API line read this one function, so a
header and a batch line cannot disagree; the streaming chunk serialises the same report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from millm.api.request_policy import IGNORED_FIELDS_HEADER
from millm.core.logging import get_logger
from millm.services.inference_service import (
    circuit_apply_failed,
    current_served_model,
    get_probe_verdicts,
    get_request_outcome,
    get_steering_report,
    reset_steering_memo,
    served_model_header_value,
)

logger = get_logger(__name__)

#: 032 FR-32.6.7: the served model, as captured inside the request's admission slot.
MODEL_HEADER = "X-miLLM-Model"

#: The header when a generation path published no report — a defect (every path publishes one,
#: `test_steering_report_every_path.py`), reported honestly rather than guessed or omitted.
STEERING_UNREPORTED = "unknown;reason=read_failed"


def _wants_scores(request: Any) -> bool:
    wants = getattr(request, "wants_scores", None)
    return bool(callable(wants) and wants())


def steering_header_value(request: Any) -> str:
    """`X-miLLM-Steering` for a finished request (FR-28.3.8, FR-28.3.10).

    Scoring is always unsteered (X-09) and its report is the constant `none` whatever path served
    it, including the Batch API's packed scorer, which builds its body without the service's
    scoring entry point.
    """
    wants = getattr(request, "wants_scores", None)
    if callable(wants) and wants():
        return "none"
    report = get_steering_report()
    if report is None:
        logger.warning("steering_report_missing",
                       detail="a generation path published no steering report")
        return STEERING_UNREPORTED
    return str(report.header)
from millm.services.system_fingerprint import build_system_fingerprint


def model_header_value() -> Optional[str]:
    """`X-miLLM-Model` for a finished request (032 FR-32.6.7, FR-32.6.9), or None.

    Read from the capture `_admit` took with the slot held, never from the loader: by the time
    the headers are built a weights load or an adapter flip may have changed what is resident.
    None (the header is omitted) when no capture exists, which no path that took a slot produces;
    a header guessed from the loader would be the very mislabel the capture exists to prevent.
    """
    served = current_served_model()
    if served is None:
        return None
    return served_model_header_value(served)


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
        # ⚠ A REQUEST THAT CANNOT BE STEERED BY THE CIRCUIT GETS NO RUNG (hardware 2026-10-16).
        # Inline `steering` replaces every attached SAE's values for this request (`features: []`
        # disables them all), so the active circuit steers nothing in it; echoing its rung put an
        # evidence claim on a response the circuit did not touch — `X-miLLM-Steering: none`
        # beside `X-miLLM-Circuit-Rung: 0` on the deployed image.
        rung_info = (
            await inference.active_circuit_rung()
            if getattr(request, "steering", None) is None else None
        )
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
            # λ resolved to 0 (`"off"`, `0`) runs the request unsteered — the same rule as above:
            # no circuit-attributable steering, so no rung. Decided here because a stream commits
            # its headers before generation.
            if effective == 0:
                echo_circuit_rung = None
    return PreGeneration(echo_intensity=echo_intensity, echo_circuit_rung=echo_circuit_rung)


def circuit_steered(report: Any) -> bool:
    """Whether the published steering report says a circuit steered this response.

    `None` (no report published — a defect `steering_header_value` reports separately) keeps the
    pre-generation decision rather than guessing; a report that could not be read (`unknown`) or
    that names no circuit item withholds the rung, because the rung header is an evidence claim
    and must never describe a response the circuit did not steer.
    """
    if report is None:
        return True
    return any(getattr(item, "kind", None) == "circuit" for item in report.items)


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
    model_header = model_header_value()
    if model_header is not None:
        headers[MODEL_HEADER] = model_header
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
    # And it is decided against the report of what RAN: a scoring request (always unsteered), an
    # inline set, or an operator write that replaced the circuit's values mid-request all leave
    # no circuit item, and then there is no rung to state.
    if (pre.echo_circuit_rung is not None and not circuit_apply_failed()
            and not _wants_scores(request) and circuit_steered(get_steering_report())):
        headers["X-miLLM-Circuit-Rung"] = pre.echo_circuit_rung
    # Probe verdicts (FR-24.7, FR-27.8g), read after generation: they do not exist before.
    probe_header = build_probe_verdicts_header(get_probe_verdicts())
    if probe_header:
        headers["X-miLLM-Probe-Verdicts"] = probe_header
    # Feature 28: every generation response states its steering, read after generation from the
    # snapshot the path took before restoring (FR-28.3.1). Scoring answers `none` (X-09).
    headers["X-miLLM-Steering"] = steering_header_value(request)
    return headers


def finish_body(result: Any, model_row: Any, inference: Any) -> Any:
    """Set the body-level provenance the routes set (FR-25.13.8)."""
    if hasattr(result, "system_fingerprint"):
        result.system_fingerprint = build_system_fingerprint(model_row, inference.loaded_model())
    return result
