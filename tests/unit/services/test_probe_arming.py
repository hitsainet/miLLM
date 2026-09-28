"""The four arming gates, their order, and what must be true afterwards.

The ordering tests are the ones worth keeping: a probe that is both wrong-model and low-rung should
be told "wrong model", because asking someone to acknowledge a probe that cannot run is asking them
to consent to nothing.

And the atomicity tests: a probe that is half-armed — a row saying armed with no hook, or a hook
with no row — is worse than one that refused, because status would describe a monitor that is not
monitoring.
"""

from __future__ import annotations

import hashlib

import pytest
import torch
from torch import nn

from millm.core.errors import (
    ProbeLimitError,
    ProbeModelMismatchError,
    ProbeParityFailedError,
    UnvalidatedProbeError,
)
from millm.db.repositories.probe_repository import ProbeRepository
from millm.services.probe_arming import ProbeArmingService, head_from_definition
from millm.services.probe_identity import LoadedIdentity
from millm.services.probe_runtime import ProbeRuntimeState
from millm.services.probe_service import ProbeService
from tests.unit.probe_fixtures import acknowledged, probe_definition

pytestmark = pytest.mark.asyncio

TEMPLATE = "{{ messages }}"
TEMPLATE_SHA = hashlib.sha256(TEMPLATE.encode()).hexdigest()


class TinyLayer(nn.Module):
    def forward(self, x):
        return (x, None)


class TinyModel(nn.Module):
    def __init__(self, n=16):
        super().__init__()
        inner = nn.Module()
        inner.layers = nn.ModuleList([TinyLayer() for _ in range(n)])
        self.model = inner


def loaded(**over) -> LoadedIdentity:
    base = dict(
        hf_id="LiquidAI/LFM2.5-1.2B-Instruct",
        d_model=8,
        n_layers=16,
        chat_template=TEMPLATE,
        revision="0f604ada3f766f9f257460c4c9f0b5d6f69d431b",
        revision_source="download_metadata",
        supports_hooks=True,
    )
    base.update(over)
    return LoadedIdentity(**base)


def definition(**over) -> dict:
    doc = probe_definition(**over)
    doc["model"]["chat_template_sha256"] = TEMPLATE_SHA
    # A head of ones over d=8 with activations of 0.5 gives 4.0 per token; the fixture's vector
    # records exactly that so parity passes by construction.
    doc["head"]["weights"] = [1.0] * 8
    doc["head"]["bias"] = 0.0
    doc["head"]["norm_mean"] = [0.0] * 8
    doc["head"]["norm_std"] = [1.0] * 8
    doc["test_vectors"]["vectors"] = [
        {
            "messages": [{"role": "user", "content": "x"}],
            "token_ids": [1, 2, 3],
            "token_scores": [4.0, 4.0, 4.0],
            "score": 4.0,
            "verdict": True,
        }
    ]
    return doc


def forward_at(value: float):
    def _forward(input_ids, context):
        n = input_ids.shape[1]
        context.observe(11, torch.full((1, n, 8), value))

    return _forward


@pytest.fixture(autouse=True)
def clean_state():
    ProbeRuntimeState.reset_for_tests()
    yield
    ProbeRuntimeState.reset_for_tests()


@pytest.fixture
async def ctx(test_session):
    repo = ProbeRepository(test_session)
    service = ProbeService(repo)
    arming = ProbeArmingService(repo, ProbeRuntimeState())
    return repo, service, arming


class TestHappyPath:
    async def test_all_four_gates_pass_and_the_probe_is_armed(self, ctx):
        repo, service, arming = ctx
        probe = await service.import_definition(definition())
        armed = await arming.arm(
            probe, model=TinyModel(), loaded=loaded(), forward=forward_at(0.5)
        )
        assert armed.probe_id == probe.id
        assert (await repo.get(probe.id)).armed is True
        assert ProbeRuntimeState().has_armed() is True

    async def test_the_parity_report_is_stored_even_on_success(self, ctx):
        repo, service, arming = ctx
        probe = await service.import_definition(definition())
        await arming.arm(probe, model=TinyModel(), loaded=loaded(), forward=forward_at(0.5))
        stored = (await repo.get(probe.id)).parity
        assert stored["passed"] is True
        assert stored["max_abs_diff"] == pytest.approx(0.0)

    async def test_a_rung_two_probe_needs_no_acknowledgement(self, ctx):
        _repo, service, arming = ctx
        probe = await service.import_definition(definition())
        await arming.arm(probe, model=TinyModel(), loaded=loaded(), forward=forward_at(0.5))
        assert probe.arm_acknowledgement is None


class TestGateOrder:
    async def test_a_wrong_model_is_reported_before_the_rung_gate(self, ctx):
        """⚠ Asking someone to acknowledge a probe that cannot run is asking them to consent to
        nothing. The useful message is "wrong model"."""
        _repo, service, arming = ctx
        doc = definition()
        doc["evidence"] = {"rung": 1, "rung_language": "detects on held-out data",
                           "acknowledgement": acknowledged(), "evaluations": []}
        probe = await service.import_definition(doc)
        with pytest.raises(ProbeModelMismatchError):
            await arming.arm(
                probe,
                model=TinyModel(),
                loaded=loaded(hf_id="Qwen/Qwen2.5-7B-Instruct"),
                forward=forward_at(0.5),
            )

    async def test_parity_is_not_run_when_an_earlier_gate_fails(self, ctx):
        """Parity is the only gate that costs a forward pass."""
        _repo, service, arming = ctx
        probe = await service.import_definition(definition())
        calls = []

        def counting_forward(input_ids, context):
            calls.append(1)
            forward_at(0.5)(input_ids, context)

        with pytest.raises(ProbeModelMismatchError):
            await arming.arm(
                probe, model=TinyModel(), loaded=loaded(d_model=4096), forward=counting_forward
            )
        assert calls == [], "the model was run for a probe that had already failed a free gate"


class TestTheRungGate:
    async def test_below_rung_two_is_refused_without_an_acknowledgement(self, ctx):
        _repo, service, arming = ctx
        doc = definition()
        doc["evidence"] = {"rung": 1, "rung_language": "detects on held-out data",
                           "acknowledgement": acknowledged(), "evaluations": []}
        probe = await service.import_definition(doc)
        with pytest.raises(UnvalidatedProbeError) as exc:
            await arming.arm(probe, model=TinyModel(), loaded=loaded(), forward=forward_at(0.5))
        assert exc.value.details["rung"] == 1
        assert exc.value.details["next_step"]

    async def test_acknowledging_it_arms_and_STORES_the_consent(self, ctx):
        """⚠ A second consent, not a duplicate of the definition's.

        The person who exported a weak probe and the person arming it against live traffic are not
        necessarily the same person, and only the second is choosing to monitor with it.
        """
        repo, service, arming = ctx
        doc = definition()
        doc["evidence"] = {"rung": 0, "rung_language": "trained",
                           "acknowledgement": acknowledged("exported for review"),
                           "evaluations": []}
        probe = await service.import_definition(doc)
        await arming.arm(
            probe,
            model=TinyModel(),
            loaded=loaded(),
            forward=forward_at(0.5),
            acknowledge_below_rung2=True,
            reason="watching this one by hand",
        )
        stored = await repo.get(probe.id)
        assert stored.armed is True
        assert stored.arm_acknowledgement["reason"] == "watching this one by hand"
        assert stored.arm_acknowledgement["rung"] == 0
        # the definition's own acknowledgement is untouched and still distinct
        assert stored.definition_acknowledgement["reason"] == "exported for review"


class TestParityGate:
    async def test_a_probe_that_scores_differently_is_refused(self, ctx):
        repo, service, arming = ctx
        probe = await service.import_definition(definition())
        with pytest.raises(ProbeParityFailedError) as exc:
            # 0.9 * 8 = 7.2 per token, against a recorded 4.0
            await arming.arm(probe, model=TinyModel(), loaded=loaded(), forward=forward_at(0.9))
        assert exc.value.details["max_abs_diff"] == pytest.approx(3.2)
        assert exc.value.details["vector_index"] == 0

    async def test_a_failed_arm_leaves_NOTHING_armed(self, ctx):
        """⚠ Half-armed is worse than refused: status would describe a monitor that is not
        monitoring."""
        repo, service, arming = ctx
        probe = await service.import_definition(definition())
        with pytest.raises(ProbeParityFailedError):
            await arming.arm(probe, model=TinyModel(), loaded=loaded(), forward=forward_at(0.9))
        assert (await repo.get(probe.id)).armed is False
        assert ProbeRuntimeState().has_armed() is False

    async def test_the_failed_parity_report_is_still_stored(self, ctx):
        """The operator needs to see WHY, and a refusal that leaves no record makes them re-run
        the gate to find out."""
        repo, service, arming = ctx
        probe = await service.import_definition(definition())
        with pytest.raises(ProbeParityFailedError):
            await arming.arm(probe, model=TinyModel(), loaded=loaded(), forward=forward_at(0.9))
        assert (await repo.get(probe.id)).parity["passed"] is False


class TestTheLimit:
    async def test_arming_beyond_the_cap_is_refused(self, ctx, monkeypatch):
        from millm.core.config import settings

        monkeypatch.setattr(settings, "PROBE_MAX_ARMED", 2)
        repo, service, arming = ctx
        probes = [
            await service.import_definition(definition(name=f"p{i}")) for i in range(3)
        ]
        for probe in probes[:2]:
            await arming.arm(probe, model=TinyModel(), loaded=loaded(), forward=forward_at(0.5))
        with pytest.raises(ProbeLimitError) as exc:
            await arming.arm(probes[2], model=TinyModel(), loaded=loaded(), forward=forward_at(0.5))
        assert exc.value.details["max_armed"] == 2


class TestDisarm:
    async def test_disarm_clears_both_the_runtime_and_the_row(self, ctx):
        repo, service, arming = ctx
        probe = await service.import_definition(definition())
        await arming.arm(probe, model=TinyModel(), loaded=loaded(), forward=forward_at(0.5))
        await arming.disarm(probe, reason="operator")
        assert (await repo.get(probe.id)).armed is False
        assert ProbeRuntimeState().has_armed() is False

    async def test_disarm_all_clears_both_sides(self, ctx):
        """⚠ Clearing only the runtime leaves rows claiming to be armed after a restart; clearing
        only the rows leaves hooks on a model nobody is tracking."""
        repo, service, arming = ctx
        for i in range(2):
            probe = await service.import_definition(definition(name=f"p{i}"))
            await arming.arm(probe, model=TinyModel(), loaded=loaded(), forward=forward_at(0.5))
        assert await arming.disarm_all("model_changed") == 2
        assert ProbeRuntimeState().has_armed() is False
        assert await repo.count_armed() == 0
        assert all(p.paused_reason == "model_changed" for p in await repo.list())


class TestHeadConstruction:
    def test_the_head_is_float32_whatever_the_model_is(self):
        """Scoring in half precision would introduce a difference from miStudio's float32 fit
        that parity would then have to absorb — spending the tolerance on a saving nobody needs."""
        head = head_from_definition(definition())
        assert head.weight.dtype == torch.float32
        assert head.std.dtype == torch.float32

    def test_the_attention_query_travels_when_present(self):
        doc = definition()
        doc["aggregation"] = {"rule": "attention", "params": {}, "streamable": True}
        doc["head"]["attention_query"] = [0.5] * 8
        head = head_from_definition(doc)
        assert head.attention_query is not None and head.attention_query.shape == (8,)


class TestAnUnverifiableScopeSaysSo:
    """⚠ "Nothing could be compared" is not "the numbers disagree".

    Only `scope: all` is exactly reproducible. A `prompt` or `response` probe has no
    reproducible positions — the contract records the token ids but not which ones miStudio's
    role mask selected — so every vector comes back incomparable and NOTHING is scored.

    That surfaced as `PROBE_PARITY_FAILED`, *"this build does not reproduce the scores miStudio
    recorded"*, over a report with **0 of 16 comparable and 0 tokens scored**. An operator
    reading that debugs their build, their model and their precision, none of which is involved.
    Reported 2026-09-28 against an L6 probe exported with `scope: prompt`.

    This estate already has the rule, from the mirror-image defect: a parity check that reports
    "incorrect" against a correct implementation is worse than none, because it is believed the
    first time.
    """

    @staticmethod
    def _probe(scope: str):
        from unittest.mock import MagicMock

        row = MagicMock()
        row.id = "pr_s"
        row.name = "scoped"
        row.layer = 6
        row.rule = "mean"
        row.scope = scope
        row.basis = "residual"
        row.threshold = 1.0
        row.rung = 2
        row.armed = False
        row.definition = {
            "read": {"layer": 6, "hook_point": "resid_post"},
            "aggregation": {"rule": "mean"},
            "scope": scope,
            "basis": "residual",
            "head": {
                "weights": [0.1] * 8,
                "bias": 0.0,
                "norm_mean": [0.0] * 8,
                "norm_std": [1.0] * 8,
            },
            "model": {"hf_id": "LiquidAI/LFM2.5-1.2B-Instruct", "d_model": 8, "n_layers": 16},
            "decision": {"threshold": 1.0},
            "evidence": {"rung": 2},
            "test_vectors": {
                "tolerance": 0.05,
                "vectors": [
                    {"token_ids": [1, 2, 3], "token_scores": [0.1, 0.2, 0.3], "score": 0.2}
                ],
            },
        }
        return row

    @pytest.mark.asyncio
    @pytest.mark.parametrize("scope", ["prompt", "response"])
    async def test_it_refuses_with_its_own_code_not_parity_failed(self, scope):
        from unittest.mock import AsyncMock, MagicMock

        from millm.core.errors import ProbeParityFailedError, ProbeScopeUnverifiableError
        from millm.services.probe_arming import ProbeArmingService
        from millm.services.probe_identity import LoadedIdentity

        repo = MagicMock()
        repo.count_armed = AsyncMock(return_value=0)
        repo.update = AsyncMock()
        identity = LoadedIdentity(
            hf_id="LiquidAI/LFM2.5-1.2B-Instruct", d_model=8, n_layers=16
        )

        with pytest.raises(ProbeScopeUnverifiableError) as exc:
            await ProbeArmingService(repo).arm(
                self._probe(scope),
                model=MagicMock(),
                loaded=identity,
                forward=lambda *a, **k: None,
            )
        assert not isinstance(exc.value, ProbeParityFailedError)
        assert exc.value.code == "PROBE_SCOPE_UNVERIFIABLE"

    @pytest.mark.asyncio
    async def test_the_message_names_the_scope_and_the_remedy(self):
        """The operator must be able to act on it: the fix is on the PRODUCER side."""
        from unittest.mock import AsyncMock, MagicMock

        from millm.core.errors import ProbeScopeUnverifiableError
        from millm.services.probe_arming import ProbeArmingService
        from millm.services.probe_identity import LoadedIdentity

        repo = MagicMock()
        repo.count_armed = AsyncMock(return_value=0)
        repo.update = AsyncMock()

        with pytest.raises(ProbeScopeUnverifiableError) as exc:
            await ProbeArmingService(repo).arm(
                self._probe("prompt"),
                model=MagicMock(),
                loaded=LoadedIdentity(
                    hf_id="LiquidAI/LFM2.5-1.2B-Instruct", d_model=8, n_layers=16
                ),
                forward=lambda *a, **k: None,
            )
        text = str(exc.value)
        assert "'prompt'" in text
        assert "scope 'all'" in text
        assert "not a disagreement about the numbers" in text
        assert exc.value.details["scope"] == "prompt"

    @pytest.mark.asyncio
    async def test_a_REAL_parity_failure_still_reports_as_one(self):
        """⚠ Specificity. If every refusal became 'unverifiable scope', the gate would stop
        reporting the thing it exists for."""
        from unittest.mock import AsyncMock, MagicMock

        import torch

        from millm.core.errors import ProbeParityFailedError
        from millm.services.probe_arming import ProbeArmingService
        from millm.services.probe_identity import LoadedIdentity
        from millm.services.probe_runtime import ProbeRequestContext

        repo = MagicMock()
        repo.count_armed = AsyncMock(return_value=0)
        repo.update = AsyncMock()

        def forward(ids, context: ProbeRequestContext):
            context.observe(6, torch.zeros(1, ids.shape[1], 8))

        with pytest.raises(ProbeParityFailedError):
            await ProbeArmingService(repo).arm(
                self._probe("all"),          # reproducible scope, wrong numbers
                model=MagicMock(),
                loaded=LoadedIdentity(
                    hf_id="LiquidAI/LFM2.5-1.2B-Instruct", d_model=8, n_layers=16
                ),
                forward=forward,
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "scope,expect_all",
        [("prompt", True), ("response", True), ("all", False)],
    )
    async def test_the_scope_reason_is_all_or_nothing(self, scope, expect_all):
        """⚠ THIS INVARIANT IS WHAT MAKES THE ARMING BRANCH SAFE, so it is pinned here.

        `arm()` refuses with PROBE_SCOPE_UNVERIFIABLE only when **every** vector is
        unverifiable, and reports a parity failure otherwise. A mutation relaxing that to
        "any vector" survives the suite — measured, not argued: `NOT_COMPARABLE_SCOPE` is
        set from `scope_is_reproducible(probe.scope)`, which is constant across vectors, so
        the set is either empty or the whole list and the two spellings agree today.

        If a later contract carries scope per vector, that stops being true and a single
        unverifiable vector would hide a real disagreement on the other fifteen. This test
        goes red at that moment, which is the point of it.
        """
        from unittest.mock import MagicMock

        import torch

        from millm.services.probe_parity import NOT_COMPARABLE_SCOPE, ProbeParityEngine
        from millm.services.probe_runtime import ArmedProbe, ProbeRequestContext

        row = self._probe(scope)
        definition = dict(row.definition)
        definition["test_vectors"] = {
            "tolerance": 0.05,
            "vectors": [
                {"token_ids": [1, 2, 3], "token_scores": [0.1, 0.2, 0.3], "score": 0.2},
                {"token_ids": [4, 5], "token_scores": [0.4, 0.5], "score": 0.45},
                {"token_ids": [6], "token_scores": [0.6], "score": 0.6},
            ],
        }

        def forward(ids, context: ProbeRequestContext):
            context.observe(6, torch.zeros(1, ids.shape[1], 8))

        probe = ArmedProbe(
            probe_id="pr_s",
            name="scoped",
            head=head_from_definition(definition),
            rule="mean",
            scope=scope,
            layer=6,
            rung=2,
            rung_language="held-out",
            threshold=1.0,
        )
        report = ProbeParityEngine(forward).run(
            probe=probe, definition=definition, tolerance=0.05
        )
        expected_n = len(definition["test_vectors"]["vectors"])

        flagged = [v for v in report.vectors if v.reason == NOT_COMPARABLE_SCOPE]
        assert len(report.vectors) == expected_n
        if expect_all:
            assert len(flagged) == len(report.vectors), (
                "a partially-flagged report would make 'any' and 'all' disagree in arm()"
            )
        else:
            assert flagged == []
