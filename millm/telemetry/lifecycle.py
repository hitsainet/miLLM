"""`mi.probe.lifecycle` records (FR-38.20; FTDD §6.5): import, arm, refused arm, disarm, parity,
recalibrate and delete, emitted from the SERVICE methods so a route, the arm bridge and the
model-unload disarm all emit.

⚠ THE OPERATOR'S `reason` IS NEVER READ. The arm acknowledgement's reason, the recalibration's
history reason and the disarm's `paused_reason` are free text; `record_for` reads only the named
identity fields of the probe row (and an error's CODE), so no argument can carry text here. The
allowlist is the backstop (`mi.lifecycle.code` admits only `CODE_LIKE` or `lower_snake`).

No OpenTelemetry import (D38.17).
"""

from __future__ import annotations

from typing import Any

SUBJECT_KINDS = ("probe", "probe_set", "rule_set")


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def record_for(
    probe: Any,
    action: str,
    *,
    outcome: str = "ok",
    code: str | None = None,
    previous_threshold_revision: int | None = None,
    previous_decision_digest: str | None = None,
) -> dict[str, Any]:
    """The lifecycle attributes for one probe row. Reads NAMED FIELDS ONLY (module doc)."""
    behaviour = getattr(probe, "behaviour_key", None)
    if behaviour is None:
        definition = getattr(probe, "definition", None)
        block = definition.get("behaviour") if isinstance(definition, dict) else None
        behaviour = block.get("key") if isinstance(block, dict) else None
    return {
        "mi.lifecycle.subject_kind": "probe",
        "mi.lifecycle.action": action,
        "mi.lifecycle.outcome": outcome,
        "mi.lifecycle.code": code,
        "mi.lifecycle.probe_id": getattr(probe, "id", None),
        "mi.lifecycle.behaviour": behaviour,
        "mi.lifecycle.detector_digest": getattr(probe, "detector_digest", None),
        "mi.lifecycle.threshold_revision": _int(getattr(probe, "threshold_revision", None)),
        "mi.lifecycle.decision_revision": _int(getattr(probe, "decision_revision", None)),
        "mi.lifecycle.decision_digest": getattr(probe, "decision_digest", None),
        "mi.lifecycle.previous_threshold_revision": _int(previous_threshold_revision),
        "mi.lifecycle.previous_decision_digest": previous_decision_digest,
    }


def error_code(exc: BaseException) -> str:
    """A refusal's CODE (`PROBE_PARITY_FAILED`), else its class name. Never its message."""
    code = getattr(exc, "code", None)
    code = getattr(code, "value", code)
    # A refusal's code is the PUBLIC one the route answers with (upper case), whichever case the
    # raising helper used (`ProbeRecalibrationRefused` carries lower case).
    return code.upper() if isinstance(code, str) and code else type(exc).__name__
