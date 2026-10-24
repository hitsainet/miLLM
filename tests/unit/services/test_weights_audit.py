"""`weights_audit`: one line per gated call, typed fields only (031 FR-31.15; addendum §2.1)."""

from __future__ import annotations

import inspect
import json
from datetime import datetime
from typing import Any

import pytest

import millm.services.weights_audit as audit_module
from millm.core.weights_auth import WeightsIdentity
from millm.services.weights_audit import weights_audit

SECRET = "attachment-id-or-token-that-must-never-be-logged"


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def __getattr__(self, level: str):
        def record(event: str, *args: Any, **kwargs: Any) -> None:
            self.calls.append((level, event, kwargs))

        return record


@pytest.fixture
def rec(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(audit_module, "logger", recorder)
    return recorder


def test_the_signature_is_closed_typed_keywords_only():
    """No `**kwargs`: a free-form field is how a token reaches a log line by accident."""
    params = inspect.signature(weights_audit).parameters
    assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    assert not any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params.values())
    assert list(params) == [
        "operation", "outcome", "identity", "code", "reason", "attachment_ref",
        "weights_sha256", "checkpoint_sha256", "adapter", "mode", "commit_ms", "steps", "path",
        "state_before", "elapsed_ms",
    ]
    assert all(params[n].kind is inspect.Parameter.KEYWORD_ONLY for n in list(params)[3:])
    with pytest.raises(TypeError):
        weights_audit("weights_save", "accepted", None, token=SECRET)  # type: ignore[call-arg]


def test_an_accepted_call_is_info_with_identity_model_state_and_time(rec):
    weights_audit("weights_save", "accepted", WeightsIdentity("miforge", "trainer"),
                  attachment_ref="abcd1234", checkpoint_sha256="f" * 64, path="run/step-000005")
    [(level, event, fields)] = rec.calls
    assert (level, event) == ("info", "weights_audit")
    assert fields["operation"] == "weights_save" and fields["outcome"] == "accepted"
    assert (fields["role"], fields["label"]) == ("trainer", "miforge")
    assert fields["attachment_ref"] == "abcd1234"
    assert fields["checkpoint_sha256"] == "f" * 64 and fields["path"] == "run/step-000005"
    assert fields["weights_state"] == "clean"
    assert "resident_model" in fields and fields["code"] is None
    assert datetime.fromisoformat(fields["timestamp"]).utcoffset().total_seconds() == 0
    # Operation-specific fields not passed are absent, not invented.
    assert "commit_ms" not in fields and "adapter" not in fields
    assert "weights_state_before" not in fields and "elapsed_ms" not in fields


def test_state_before_and_elapsed_ms_are_logged_under_fr_31_15_2_names(rec):
    weights_audit("commit_end", "accepted", WeightsIdentity("miforge", "trainer"),
                  state_before="torn", elapsed_ms=2104.5, commit_ms=2104.5, steps=8)
    [(_, _, fields)] = rec.calls
    assert fields["weights_state_before"] == "torn" and fields["elapsed_ms"] == 2104.5
    assert fields["weights_state"] == "clean"  # the state after


def test_a_refusal_is_a_warning_and_unauthenticated_is_null_identity(rec):
    weights_audit("attach", "refused", None, code="WEIGHTS_UNAUTHORIZED", reason="missing")
    [(level, _, fields)] = rec.calls
    assert level == "warning"
    assert fields["role"] is None and fields["label"] is None
    assert fields["code"] == "WEIGHTS_UNAUTHORIZED" and fields["reason"] == "missing"


def test_outcome_is_strictly_accepted_or_refused(rec):
    with pytest.raises(ValueError):
        weights_audit("x", "ok", None)  # type: ignore[arg-type]
    assert rec.calls == []


def test_the_resident_model_is_the_served_name(rec, monkeypatch):
    import millm.services.resident_identity as ri

    monkeypatch.setattr(ri, "served_name", lambda loaded=None: "base:step-000007")
    weights_audit("weights_load", "accepted", None)
    assert rec.calls[0][2]["resident_model"] == "base:step-000007"


def test_a_failed_context_read_never_fails_the_audit(rec, monkeypatch):
    import millm.services.resident_identity as ri

    def boom(loaded=None):
        raise RuntimeError("no state")

    monkeypatch.setattr(ri, "served_name", boom)
    weights_audit("weights_load", "accepted", None)
    events = [c[1] for c in rec.calls]
    assert events == ["weights_audit_context_unavailable", "weights_audit"]
    assert rec.calls[1][2]["resident_model"] is None


def test_no_secret_in_any_line(rec):
    """Every field a caller can pass, filled; none can carry an ID or a token by name."""
    weights_audit("commit-end", "accepted", WeightsIdentity("miforge", "trainer"),
                  attachment_ref=SECRET[:8], commit_ms=12.5, steps=3, mode="in_place",
                  adapter="run:adapter-200", weights_sha256="a" * 64)
    assert SECRET not in json.dumps(rec.calls, default=str)
