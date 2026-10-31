"""The one builder (FTASKS 2.5 – 2.8): descriptor, request context, ids, no database.

* the arm-time `ProbeDescriptor` carries every detector field, and building a result never reads
  the probe row again (D38.18);
* `ProbeRequestContext.finish(interrupted=…)` gives `partial` when passes were observed and the
  window resolves, `unavailable` otherwise, and `finish()` with no argument is unchanged;
* the builder succeeds with the repositories patched to raise;
* 10,000 ids are unique and monotonic.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import jsonschema
import torch

from millm.api.schemas.probe_result import strict_schema, wire
from millm.ml.probe_head import ProbeHead
from millm.services.probe_arming import armed_probe_from_row
from millm.services.probe_digest import decision_digest, detector_digest
from millm.services.probe_result import build_probe_results, uuid7
from millm.services.probe_runtime import ArmedProbe, ExecutionFacts, ProbeRequestContext
from tests.unit.score_fixtures import tiny_definition

STRICT = jsonschema.Draft202012Validator(strict_schema())


class _Row:
    """A stored probe row that COUNTS every attribute read, and can be poisoned."""

    def __init__(self, definition: dict) -> None:
        self.__dict__["_reads"] = 0
        self.__dict__["_poisoned"] = False
        model = definition["model"]
        values = dict(
            id="pr_0123456789ab", name="probe one", definition=definition,
            rule=definition["aggregation"]["rule"], scope=definition["scope"],
            layer=definition["read"]["layer"], rung=2, threshold=definition["decision"]["threshold"],
            threshold_revision=3, threshold_calibration_id="cal_7", basis=definition["basis"],
            revision=model["revision"], template_sha256=model["chat_template_sha256"],
            parity={"passed": True, "model": {"hf_id": model["hf_id"]}},
            detector_digest=None, decision_digest=None, detector_digest_source=None,
            identity_algorithm=None, decision_revision=4,
        )
        self.__dict__.update(values)

    def __getattribute__(self, name):
        if not name.startswith("_") and object.__getattribute__(self, "_poisoned"):
            raise AssertionError(f"the probe row was read at request time ({name})")
        return object.__getattribute__(self, name)


def _definition() -> dict:
    doc = tiny_definition("probe one", layer=0, threshold=0.0)
    doc["behaviour"] = {"key": "b-placeholder", "name": "placeholder", "version": 2}
    doc["model"]["load_dtype"] = "float32"
    return doc


def test_the_descriptor_carries_every_field_and_the_row_is_never_read_again():
    definition = _definition()
    row = _Row(definition)
    armed = armed_probe_from_row(row, windows=["all"])
    d = armed.descriptor
    assert d.behaviour_key == "b-placeholder" and d.behaviour_version == 2
    assert d.detector_digest == detector_digest(definition)
    assert d.detector_digest_source == "computed"
    assert d.identity_algorithm == "mistudio.probe-identity/1"
    assert d.hook == "resid_post" and d.basis == "residual" and d.target == "positive_class"
    assert d.template_sha256 == definition["model"]["chat_template_sha256"]
    assert d.model_revision == definition["model"]["revision"] and d.precision == "float32"
    assert d.parity_status == "passed"
    assert d.decision.decision_digest == decision_digest(definition["decision"])
    assert d.decision.decision_revision == 4 and d.decision.calibration_id == "cal_7"

    row.__dict__["_poisoned"] = True  # from here on, any read of the row raises
    from millm.services.probe_runtime import Verdict

    verdict = Verdict(probe_id=armed.probe_id, name=armed.name, rung=2,
                      rung_language="detects on unseen tasks", scored=True, score=0.5,
                      threshold=0.0, fires=True, n_scored_tokens=3, threshold_revision=3)
    context = SimpleNamespace(request_id="chatcmpl-1", probes=[armed], session=None, trace=None,
                              execution=ExecutionFacts("transformers", "serial"))
    data = wire(build_probe_results([verdict], context)[0])
    STRICT.validate(data)
    assert data["detector_digest"] == d.detector_digest
    assert data["decision_digest"] == d.decision.decision_digest
    assert data["decision_revision"] == 4 and data["threshold_revision"] == 3
    assert data["behaviour"] == {"key": "b-placeholder", "version": 2}
    assert data["layer"] == 0 and data["rule"] == "mean" and data["definition_scope"] == "all"
    assert data["parity_status"] == "passed" and data["precision"] == "float32"


def test_the_builder_reads_no_database():
    """With every repository patched to raise, the build still succeeds."""
    from millm.services.probe_runtime import Verdict

    boom = AssertionError("the builder touched a repository")
    with patch("millm.db.repositories.probe_repository.ProbeRepository.get", side_effect=boom), \
            patch("millm.db.repositories.probe_repository.ProbeEventRepository.list_events",
                  side_effect=boom), \
            patch("millm.db.base.async_session_factory", side_effect=boom):
        results = build_probe_results([Verdict(
            probe_id="pr_0123456789ab", name="p", rung=2, rung_language="detects on unseen tasks",
            scored=True, score=1.0, threshold=0.5, fires=True, n_scored_tokens=1,
        )], SimpleNamespace(request_id="r1", probes=(), session=None, trace=None))
    assert len(results) == 1


def _probe(**over) -> ArmedProbe:
    base = dict(probe_id="pr_0123456789ab", name="probe", head=ProbeHead(weight=torch.ones(8)),
                rule="mean", scope="all", layer=0, rung=2,
                rung_language="detects on unseen tasks", threshold=0.0)
    base.update(over)
    return ArmedProbe(**base)


def test_finish_with_no_argument_is_unchanged():
    context = ProbeRequestContext("req", [_probe()])
    context.observe(0, torch.randn(1, 4, 8))
    verdict = context.finish()[0]
    assert verdict.scored and context.interrupted is None
    data = wire(build_probe_results([verdict], context)[0])
    assert data["status"] == "final" and data["outcome"] == "completed"


def test_an_interrupted_request_with_observed_passes_is_partial():
    context = ProbeRequestContext("req", [_probe()])
    context.observe(0, torch.randn(1, 4, 8))
    verdicts = context.finish(interrupted="client_disconnected")
    data = wire(build_probe_results(verdicts, context)[0])
    STRICT.validate(data)
    assert data["outcome"] == "interrupted" and data["status"] == "partial"
    assert data["reason"] == "client_disconnected"
    assert data["score"] is not None and data["n_scored_tokens"] == 4
    assert data["verdict"] is None, "a partial result is never a verdict"


def test_an_interrupted_request_before_any_pass_is_unavailable():
    context = ProbeRequestContext("req", [_probe()])
    verdicts = context.finish(interrupted="cancelled")
    data = wire(build_probe_results(verdicts, context)[0])
    STRICT.validate(data)
    assert data["outcome"] == "interrupted" and data["status"] == "unavailable"
    assert data["score"] is None and data["reason"] == "cancelled"


def test_ten_thousand_ids_are_unique_and_monotonic():
    ids = [uuid7() for _ in range(10_000)]
    assert len({str(i) for i in ids}) == 10_000
    assert [i.int for i in ids] == sorted(i.int for i in ids), "minted ids must sort in time"
    assert all(i.version == 7 and i.variant == "specified in RFC 4122" for i in ids)


# ── review R1: lines no test caught ─────────────────────────────────────────────────


def _verdict(**over):
    from millm.services.probe_runtime import Verdict

    base = dict(probe_id="pr_0123456789ab", name="probe", rung=2,
                rung_language="detects on unseen tasks", scored=True, score=0.5, threshold=0.0,
                fires=True, n_scored_tokens=3)
    base.update(over)
    return Verdict(**base)


def _ctx(probes=()):
    return SimpleNamespace(request_id="chatcmpl-r1", probes=list(probes), session=None, trace=None)


def test_no_steering_report_is_unknown_and_an_empty_one_is_none():
    """FR-38.6: a result never claims "unsteered" without a report saying so."""
    from millm.services.inference_service import set_steering_report
    from millm.services.steering_report import SteeringReport

    set_steering_report(None)
    assert wire(build_probe_results([_verdict()], _ctx())[0])["steering_ref"]["kind"] == "unknown"
    set_steering_report(SteeringReport.none())
    try:
        assert wire(build_probe_results([_verdict()], _ctx())[0])["steering_ref"]["kind"] == "none"
    finally:
        set_steering_report(None)


def test_a_batch_row_never_belongs_to_a_session():
    """BR-001: the stored `session_id` column is now a projection of the result, so the builder's
    reset is the only thing keeping a batch row out of a conversation."""
    from millm.services.batch.state import BATCH_ROW

    context = _ctx()
    context.session = SimpleNamespace(id="ps_0123456789", turn_index=4)
    token = BATCH_ROW.set(("batch_0123456789abcdef", 3))
    try:
        data = wire(build_probe_results([_verdict()], context)[0])
    finally:
        BATCH_ROW.reset(token)
    assert data["origin"] == "batch" and data["batch_line"] == 3
    assert data["session_id"] is None and data["turn_index"] is None


def test_a_result_with_no_score_counts_no_tokens():
    data = wire(build_probe_results([_verdict(score=float("nan"), n_scored_tokens=7)], _ctx())[0])
    STRICT.validate(data)
    assert data["outcome"] == "failed" and data["n_scored_tokens"] == 0


def test_the_decisions_target_fpr_travels():
    row = _Row(_definition())
    armed = armed_probe_from_row(row, windows=["all"])
    data = wire(build_probe_results([_verdict(probe_id=armed.probe_id)], _ctx([armed]))[0])
    assert data["decision"]["target_fpr"] == _definition()["decision"]["target_fpr"]
    assert data["decision"]["target_fpr"] is not None


def test_strict_mode_raises_on_a_broken_rule(monkeypatch):
    """`MILLM_STRICT_RESULTS=1` is what makes a builder defect loud in every other test; if
    `_enforce` stopped raising, those tests would keep passing over broken results."""
    import pytest

    from millm.services.probe_result import ProbeResultRuleViolation, _enforce

    good = build_probe_results([_verdict()], _ctx())[0]
    broken = good.model_copy(update={"fires": not good.verdict})
    monkeypatch.setenv("MILLM_STRICT_RESULTS", "1")
    with pytest.raises(ProbeResultRuleViolation):
        _enforce(broken)
    monkeypatch.setenv("MILLM_STRICT_RESULTS", "0")
    _enforce(broken)  # logged, never raised, in production


def test_an_unlisted_descriptor_value_reads_unknown_and_never_raises():
    """A rule, basis, hook, scope, parity status or digest source the contract does not list is
    `unknown`, not a ValidationError that loses the request's events (review R1)."""
    from millm.services.probe_runtime import ProbeDescriptor

    armed = _probe(rule="ema", scope="paragraph", layer=-1, descriptor=ProbeDescriptor(
        hook="resid_pre", basis="pca", target="negative_class", parity_status="flaky",
        detector_digest="a" * 64, detector_digest_source="guessed", identity_algorithm="other/9"))
    data = wire(build_probe_results([_verdict()], _ctx([armed]))[0])
    STRICT.validate(data)
    assert (data["rule"], data["definition_scope"], data["hook"], data["basis"],
            data["target"], data["parity_status"], data["detector_digest_source"]) == (
        "unknown",) * 7
    assert data["layer"] is None and data["identity_algorithm"] is None
