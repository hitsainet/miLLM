"""End-to-end probe-monitor workflow (Feature 24, task 10.2).

Drives the REAL path with the REAL artifacts miStudio's 033 acceptance produced —
`tests/fixtures/lfm2_probe_definition*.json` are the actual published files, copied byte
for byte off the node, not hand-written approximations:

  import → parity (from `token_ids`) → the four arming gates → scoring a request →
  the verdict header and the streaming chunk → an event with no prompt text in it.

⚠ **REAL DEFINITIONS MATTER MORE HERE THAN ANYWHERE ELSE IN THIS FEATURE.** Every serious
defect in the 033 export arc validated against the contract while being wrong, and a
hand-written fixture is written by whoever also wrote the code — so it agrees with the
code by construction, which is the single most common reason a suite stays green over a
critical bug in this estate. These three carry 2048-wide and 128-wide heads, sixteen test
vectors each, rungs 3, 2 and 0, and a real SAE reference.

The model is a tiny randomly-initialised Llama, so the SCORES here are meaningless. That
is deliberate: what is under test is the plumbing, and a test that needed the real 1.2B
model would run on the node and nowhere else. Parity against miStudio's recorded scores
is task 10.3, on hardware.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from millm.api.schemas.probe import ProbeDefinitionV1
from millm.core.errors import (
    ProbeLimitError,
    ProbeModelMismatchError,
    UnvalidatedProbeError,
)
from millm.services.probe_arming import ProbeArmingService, armed_probe_from_row
from millm.services.probe_identity import LoadedIdentity
from millm.services.probe_runtime import ProbeRequestContext, ProbeRuntimeState

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
DENSE = FIXTURES / "lfm2_probe_definition.json"
SAE = FIXTURES / "lfm2_probe_definition_sae.json"
RUNG0 = FIXTURES / "lfm2_probe_definition_rung0.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def _row(definition: dict, probe_id: str = "pr_1", armed: bool = False):
    """A probe row as the repository returns one, from a REAL definition.

    Every field is read OUT of the document rather than asserted beside it: a row whose
    layer disagreed with its definition's layer is a defect the fixture should be able to
    expose, and hardcoding `layer=11` here would hide it.
    """
    read = definition["read"]
    row = MagicMock()
    row.id = probe_id
    row.name = definition["name"]
    row.hf_id = definition["model"]["hf_id"]
    row.layer = read["layer"]
    row.rule = definition["aggregation"]["rule"]
    row.scope = definition["scope"]
    row.basis = definition["basis"]
    row.threshold = (definition.get("decision") or {}).get("threshold")
    row.rung = definition["evidence"]["rung"]
    row.armed = armed
    row.paused_reason = None
    row.parity = None
    row.definition = definition
    return row


def _repo(count_armed: int = 0):
    repo = MagicMock()
    repo.count_armed = AsyncMock(return_value=count_armed)
    repo.update = AsyncMock()
    return repo


def _identity(definition: dict, **over) -> LoadedIdentity:
    """The loaded model, as the definition describes it — so identity PASSES by default.

    Built from the document so a test for a MISMATCH has to state the difference
    explicitly. A fixture that differed by accident would make the mismatch tests pass
    for the wrong reason and the success tests fail mysteriously.
    """
    model = definition["model"]
    fields = dict(
        hf_id=model["hf_id"],
        d_model=model["d_model"],
        n_layers=model["n_layers"],
        chat_template=None,
        revision=model.get("revision"),
        revision_source="download_metadata",
    )
    # The definition records a template HASH, not the template. Supply a string that
    # hashes to it where one is recorded, so the gate compares equal.
    recorded = model.get("chat_template_sha256")
    fields.update(over)
    if recorded and fields.get("chat_template") is None:
        fields["chat_template"] = _MATCHES_RECORDED
        identity = _IdentityWithRecordedTemplate(**fields)
        object.__setattr__(identity, "_recorded", recorded)
        return identity
    return LoadedIdentity(**fields)


class _IdentityWithRecordedTemplate(LoadedIdentity):
    """A loaded identity that reports a GIVEN template hash.

    ⚠ The gate compares `chat_template_sha256`, which the real class computes from the
    template text. The fixture records a hash of LFM2.5's real template and sha256 cannot
    be inverted, so a test that wants the template gate to PASS has to supply the hash.

    This overrides only what the test cannot otherwise produce. The production comparison
    in `check_identity` is untouched, and the mismatch tests below go through the ordinary
    class with a real differing template — so the gate is exercised in both directions by
    the path that ships, not by this shim.
    """

    _recorded: str | None = None

    @property
    def chat_template_sha256(self):  # type: ignore[override]
        if self.chat_template is None:
            return None
        if self._recorded is not None and self.chat_template == _MATCHES_RECORDED:
            return self._recorded
        return super().chat_template_sha256


#: The sentinel a test passes as `chat_template` to mean "whatever the document recorded".
_MATCHES_RECORDED = "<the template this probe was fitted under>"


def _row_at_model_width(definition: dict, model, *, probe_id: str = "pr_1", layer: int = 1):
    """A row from a real definition, re-dimensioned to the tiny model.

    ⚠ The DOCUMENT is real; only the widths are the test model's. A 2048-wide head cannot
    be dotted with 32-wide hidden states, so a test that runs a real forward pass has to
    narrow it — and narrowing `model.d_model` to match means the identity gate passes
    trivially in these three tests. That is fine and deliberate: the identity gate is
    exercised field by field in `test_identity_refuses_on_each_field_and_NAMES_it`, against
    the real widths. What these tests check is that a real document completes all four
    gates and ends up armed.
    """
    width = model.config.hidden_size
    row = _row(definition, probe_id=probe_id)
    row.definition = dict(definition)
    row.definition["head"] = _head_at_width(definition["head"], width)
    row.definition["model"] = dict(definition["model"], d_model=width)
    row.definition["read"] = dict(definition["read"], layer=layer)
    row.layer = layer
    return row


def _rescore_vectors_for(row, model) -> None:
    """Re-record the document's test-vector scores against THIS model, in place.

    ⚠ **READ THIS BEFORE CALLING IT A WEAKENED TEST.** A randomly-initialised 32-wide Llama
    cannot reproduce scores measured on the real LFM2.5-1.2B, so a real document against
    this build is REFUSED at the parity gate — correctly, and
    `test_parity_REFUSES_the_wrong_build` asserts exactly that, with the real recorded
    numbers untouched.

    What these three tests need is the other half: that a document whose scores this build
    DOES reproduce completes all four gates and ends up armed, with its acknowledgement
    stored. That needs scores this build produces, so they are computed here by running the
    real forward through the real accumulator — the same `combine()` the live path calls,
    not a shortcut.

    The numbers are therefore meaningless as detection; the plumbing they exercise is not.
    Parity against miStudio's OWN recorded scores is task 10.3, on the node, with the real
    model. A unit test cannot do it and should not pretend to.
    """
    armed = armed_probe_from_row(row)
    forward = _forward_from(model)
    vectors = []
    for vector in row.definition["test_vectors"]["vectors"]:
        token_ids = list(vector["token_ids"])
        context = ProbeRequestContext("rescore", [armed])
        forward(torch.tensor([token_ids], dtype=torch.long), context)
        verdict = context.finish()[0]
        assert verdict.scored, f"rescoring failed: {verdict.not_scored_reason}"
        vectors.append(
            dict(
                vector,
                score=verdict.score,
                token_scores=context.token_scores_for(armed.probe_id),
            )
        )
    row.definition["test_vectors"] = dict(
        row.definition["test_vectors"], vectors=vectors
    )


def _head_at_width(head: dict, width: int) -> dict:
    """The fixture's head, re-shaped to a width the tiny model actually has.

    ⚠ EVERY vector, not just `weights`. `ProbeHead` refuses a head whose normalisation
    does not match its weights — deliberately, because a mismatched `norm_std` is a silent
    scale error — and my first version replaced `weights` alone, so the guard caught the
    test rather than the test catching anything.
    """
    out = dict(head)
    out["weights"] = [0.01] * width
    for name in ("norm_mean", "norm_std", "attention_query"):
        if head.get(name) is not None:
            out[name] = [1.0] * width if name == "norm_std" else [0.0] * width
    return out


def _forward_from(model) -> object:
    """The parity `forward`, against a real (tiny) torch model, via the real hooker."""
    from millm.services.probe_arm_bridge import build_parity_forward

    return build_parity_forward(model, 1)


@pytest.fixture(scope="module")
def tiny_llama():
    """A tiny REAL Llama — a randomly-initialised one, on CPU.

    Real because the hook target, the output shape and the device are exactly what a
    hand-built stub gets wrong: three capture loops in miStudio's 032 arc built their mask
    on the model's device while the hook returns activations on the CPU, and every test
    passed because the tiny model ran on CPU where the two agree.
    """
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        # ⚠ Wide enough for the REAL token ids in the fixtures. LFM2.5's vocabulary reaches
        # 64402, and a 512-token tiny model raises IndexError inside the embedding the
        # moment parity replays a recorded vector — which is the fixtures being real
        # working as intended, not a problem with them.
        vocab_size=65536,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(config).eval()
    return model


@pytest.fixture(autouse=True)
def clean_runtime():
    ProbeRuntimeState.reset_for_tests()
    yield
    ProbeRuntimeState.reset_for_tests()


# ── the documents ─────────────────────────────────────────────────────────────


class TestTheRealDefinitionsValidate:
    """The vendored contract accepts the files miStudio actually published.

    ⚠ This is the byte-level half of the co-release. `docs/schemas/probe-definition-v1.json`
    is asserted byte-identical elsewhere; here the pydantic mirror is asserted to accept
    real documents, which is a different claim — a mirror can match the schema file and
    still reject a valid document through a validator the schema does not express.
    """

    @pytest.mark.parametrize("path", [DENSE, SAE, RUNG0], ids=["dense", "sae", "rung0"])
    def test_it_validates(self, path):
        model = ProbeDefinitionV1.model_validate(load(path))
        # ⚠ The kind DOES carry the `/v1` suffix here. Circuits and clusters do not — I
        # asserted the circuit convention first and it failed, which is the useful kind of
        # failure: a real document settles it and a hand-written fixture would have been
        # written to whichever convention I believed.
        assert model.kind == "mistudio.probe-definition/v1"
        assert model.read.hook_point == "resid_post"

    def test_the_fixtures_cover_the_cases_that_matter(self):
        """A fixture set that happened to be three copies of one probe would pass every
        test below it while covering one path."""
        docs = [load(p) for p in (DENSE, SAE, RUNG0)]
        assert {d["basis"] for d in docs} == {"residual", "sae_features"}
        assert {d["evidence"]["rung"] for d in docs} == {0, 2, 3}
        assert {len(d["head"]["weights"]) for d in docs} == {2048, 128}
        # The k-sparse head is the SAE's k, NOT d_model — the thing miStudio's exporter
        # refuses to infer.
        sae_doc = next(d for d in docs if d["basis"] == "sae_features")
        assert len(sae_doc["head"]["weights"]) == len(sae_doc["sae"]["feature_indices"])
        assert len(sae_doc["head"]["weights"]) != sae_doc["model"]["d_model"]

    @pytest.mark.parametrize("path", [DENSE, SAE, RUNG0], ids=["dense", "sae", "rung0"])
    def test_every_vector_carries_token_ids(self, path):
        """Parity scores from `token_ids`, so a document without them cannot be checked.

        miStudio measured re-rendered `messages` missing by up to 1.153 while the recorded
        ids reproduced at 0.000e+00 on all sixteen.
        """
        vectors = load(path)["test_vectors"]["vectors"]
        assert len(vectors) == 16
        assert all(v.get("token_ids") for v in vectors)


# ── the gates, in order ───────────────────────────────────────────────────────


class TestTheArmingGatesOnRealDocuments:
    @pytest.mark.asyncio
    async def test_a_rung3_probe_arms(self, tiny_llama):
        definition = load(DENSE)
        row = _row_at_model_width(definition, tiny_llama)
        _rescore_vectors_for(row, tiny_llama)
        service = ProbeArmingService(_repo())
        armed = await service.arm(
            row,
            model=tiny_llama,
            loaded=_identity(row.definition),
            forward=_forward_from(tiny_llama),
        )
        assert armed.probe_id == "pr_1"
        assert ProbeRuntimeState().has_armed()

    @pytest.mark.asyncio
    async def test_a_rung0_probe_is_REFUSED_without_an_acknowledgement(self, tiny_llama):
        definition = load(RUNG0)
        service = ProbeArmingService(_repo())
        with pytest.raises(UnvalidatedProbeError) as exc:
            await service.arm(
                _row(definition),
                model=tiny_llama,
                loaded=_identity(definition),
                forward=_forward_from(tiny_llama),
            )
        # The refusal has to be actionable: the rung, its words, and what would raise it.
        assert exc.value.details["rung"] == 0
        assert exc.value.details["rung_language"]
        assert exc.value.details["next_step"]
        assert not ProbeRuntimeState().has_armed(), (
            "a refused arm left the probe in the runtime — it would score with no row "
            "claiming to be armed, so nothing could disarm it"
        )

    @pytest.mark.asyncio
    async def test_the_same_rung0_probe_arms_WITH_one(self, tiny_llama):
        definition = load(RUNG0)
        repo = _repo()
        row = _row_at_model_width(definition, tiny_llama)
        _rescore_vectors_for(row, tiny_llama)
        await ProbeArmingService(repo).arm(
            row,
            model=tiny_llama,
            loaded=_identity(row.definition),
            forward=_forward_from(tiny_llama),
            acknowledge_below_rung2=True,
            reason="integration test",
            by="tests",
        )
        # The acknowledgement is RECORDED, not merely accepted. An arm that took the flag
        # and stored nothing would leave no trace of who chose to monitor with a rung-0
        # detector.
        stored = [c for c in repo.update.await_args_list if "arm_acknowledgement" in c.kwargs]
        assert stored, "arming below rung 2 stored no acknowledgement"
        ack = stored[-1].kwargs["arm_acknowledgement"]
        assert ack["by"] == "tests"
        assert ack["reason"] == "integration test"
        assert ack["rung"] == 0

    @pytest.mark.asyncio
    async def test_a_rung3_probe_needs_no_acknowledgement(self, tiny_llama):
        """Specificity: if the gate fired for everything, the test above proves nothing."""
        definition = load(DENSE)
        repo = _repo()
        row = _row_at_model_width(definition, tiny_llama)
        _rescore_vectors_for(row, tiny_llama)
        await ProbeArmingService(repo).arm(
            row,
            model=tiny_llama,
            loaded=_identity(row.definition),
            forward=_forward_from(tiny_llama),
        )
        acks = [
            c.kwargs["arm_acknowledgement"]
            for c in repo.update.await_args_list
            if "arm_acknowledgement" in c.kwargs
        ]
        assert acks and acks[-1] is None, (
            "a rung-3 probe recorded an acknowledgement nobody was asked for"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "field",
        ["hf_id", "d_model", "n_layers", "chat_template"],
    )
    async def test_identity_refuses_on_each_field_and_NAMES_it(self, tiny_llama, field):
        """Every mismatch is named, not the first.

        Stopping at the first difference makes an operator fix one field, retry, and meet
        the next — never learning whether they loaded the wrong model or imported the
        wrong probe.
        """
        definition = load(DENSE)
        wrong = {
            "hf_id": "Qwen/Qwen2.5-7B-Instruct",
            "d_model": 3584,
            "n_layers": 28,
            "chat_template": "a template this probe was never fitted under",
        }[field]
        with pytest.raises(ProbeModelMismatchError) as exc:
            await ProbeArmingService(_repo()).arm(
                _row(definition),
                model=tiny_llama,
                loaded=_identity(definition, **{field: wrong}),
                forward=_forward_from(tiny_llama),
            )
        named = str(exc.value.details)
        key = "chat_template" if field == "chat_template" else field
        assert key in named, f"the refusal does not name {key}: {named}"

    @pytest.mark.asyncio
    async def test_the_limit_gate_runs_before_the_model_is_touched(self, tiny_llama):
        """Parity is the only gate that costs a forward pass, so it is last.

        The forward callable here RAISES: if anything ran it, this test fails loudly
        instead of passing a little slower.
        """
        from millm.core.config import settings

        def exploding_forward(*_a, **_k):
            raise AssertionError("the model was run before a free gate had refused")

        with pytest.raises(ProbeLimitError):
            await ProbeArmingService(_repo(count_armed=settings.PROBE_MAX_ARMED)).arm(
                _row(load(DENSE)),
                model=tiny_llama,
                loaded=_identity(load(DENSE)),
                forward=exploding_forward,
            )


    @pytest.mark.asyncio
    async def test_parity_REFUSES_the_wrong_build(self, tiny_llama):
        """⚠ THE HEADLINE GUARANTEE, and it fell out of the test rather than being aimed at.

        The real document's sixteen vectors carry the scores miStudio measured on
        LFM2.5-1.2B. This build is a randomly-initialised 32-wide Llama, so it reproduces
        none of them — and arming is refused. Without this gate the probe would report
        under the authority of an AUROC measured on something else, which is the failure the
        whole export contract exists to prevent.

        Note what is NOT relaxed here: the document's recorded numbers are untouched. Only
        the head width is narrowed, so the refusal is about the SCORES and not a shape.
        """
        from millm.core.errors import ProbeParityFailedError

        definition = load(DENSE)
        row = _row_at_model_width(definition, tiny_llama)  # widths only; scores intact
        repo = _repo()
        with pytest.raises(ProbeParityFailedError) as exc:
            await ProbeArmingService(repo).arm(
                row,
                model=tiny_llama,
                loaded=_identity(row.definition),
                forward=_forward_from(tiny_llama),
            )
        assert not ProbeRuntimeState().has_armed()
        # The failed report is STORED, so an operator can see why rather than only that.
        stored = [c for c in repo.update.await_args_list if "parity" in c.kwargs]
        assert stored, "a refused arm stored no parity report — nothing says why"
        report = stored[-1].kwargs["parity"]
        assert report["passed"] is False
        assert report["max_abs_diff"] is not None, (
            "a null max_abs_diff means nothing was comparable, which is a different "
            "refusal from 'the numbers disagree' and must not be reported as this one"
        )
        assert exc.value.details["max_abs_diff"] == report["max_abs_diff"]


# ── scoring a request ─────────────────────────────────────────────────────────


class TestScoringARequest:
    @pytest.mark.asyncio
    async def test_an_armed_probe_scores_a_forward_pass(self, tiny_llama):
        definition = load(DENSE)
        # d_model must match the model for a real pass; the fixture's head is 2048-wide
        # and the tiny model is 32-wide, so score THIS model's width with the real
        # accumulator rather than pretending the shapes agree.
        row = _row_at_model_width(definition, tiny_llama)

        armed = armed_probe_from_row(row)
        context = ProbeRequestContext("req_1", [armed])
        forward = _forward_from(tiny_llama)
        forward(torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long), context)

        verdicts = context.finish()
        assert len(verdicts) == 1
        verdict = verdicts[0]
        assert verdict.scored, f"not scored: {verdict.not_scored_reason}"
        assert verdict.score is not None
        # Five tokens in, five per-token scores out. A rule that silently scored one
        # position would still produce a plausible number.
        assert len(context.token_scores_for(armed.probe_id)) == 5

    @pytest.mark.asyncio
    async def test_a_batched_pass_is_NOT_scored_and_says_why(self, tiny_llama):
        """Two rows interleave positions the request-scoped context cannot attribute.

        Scoring row 0 and labelling it as the request's verdict would be a confident wrong
        answer; `not_scored` with a reason is the honest one.
        """
        definition = load(DENSE)
        row = _row_at_model_width(definition, tiny_llama)

        armed = armed_probe_from_row(row)
        context = ProbeRequestContext("req_2", [armed])
        forward = _forward_from(tiny_llama)
        forward(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long), context)

        verdict = context.finish()[0]
        assert not verdict.scored
        assert verdict.not_scored_reason, "a quiet probe with no reason reads as 'nothing detected'"
