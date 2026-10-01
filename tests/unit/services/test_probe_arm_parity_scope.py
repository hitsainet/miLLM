"""Parity runs on the probe's own scope, and says what actually went wrong.

⚠ REPORTED FROM THE RUNNING SYSTEM 2026-10-01. Five arm attempts across two probes were refused
with *"This build does not reproduce the scores miStudio recorded for this probe"* — while every
one of the 16 vectors came back:

    "comparable": false, "reason": "prompt_boundary_unknown", "n_tokens": 0
    "max_abs_diff": null, "scored_tokens": 0

**Nothing had been compared.** `max_abs_diff: null` is not "the numbers differ", it is "there are
no numbers". The message named a disagreement that did not exist, and an operator reasonably read
it as "my build is wrong" and went looking for a model mismatch.

TWO DEFECTS, and they compound:

1. **Parity was handed the operator's windows.** A test vector is a bare `token_ids` sequence with
   no prompt/response split, so the parity context never calls `set_prompt_length`. Arm on
   `['all','prompt','response']` and two of the three verdicts are `prompt_boundary_unknown` with
   zero scored tokens, permanently and by construction. `POST /probes/{id}/parity` already passed
   `windows=[]`; the ARM path did not — so the same check reached two different conclusions
   depending on which door you came through.

2. **The refusal described the wrong failure.** "Could not score anything" and "scored something
   and it disagreed" need different words, because they send you to different places.
"""

from __future__ import annotations

import torch

from millm.ml.probe_head import ProbeHead
from millm.services.probe_runtime import ArmedProbe, ProbeRequestContext

D = 8


def _probe(windows):
    return ArmedProbe(
        probe_id="pr_x", name="x", head=ProbeHead(weight=torch.ones(D), bias=0.0, layer=11),
        rule="mean", scope="all", layer=11, rung=2, rung_language="r",
        threshold=1.0, windows=windows,
    )


class TestTheParityContextIsTheReasonExtraWindowsAreNoise:
    """The mechanism, pinned. These verdicts can never be scored in a parity run."""

    def test_extra_windows_are_unscoreable_without_a_prompt_boundary(self):
        ctx = ProbeRequestContext("parity:0", [_probe(("all", "prompt", "response"))])
        ctx.observe(11, torch.full((1, 12, D), 0.5))   # no set_prompt_length — as parity runs
        by = {v.window: v for v in ctx.finish()}
        assert by["all"].scored is True, "the scope's own window must still score"
        assert by["prompt"].scored is False
        assert by["prompt"].not_scored_reason == "prompt_boundary_unknown"
        assert by["response"].scored is False

    def test_the_scopes_window_alone_produces_exactly_one_verdict(self):
        """What the arm path now hands the engine. One verdict, scored, nothing to select among."""
        ctx = ProbeRequestContext("parity:0", [_probe(())])
        ctx.observe(11, torch.full((1, 12, D), 0.5))
        verdicts = ctx.finish()
        assert len(verdicts) == 1
        assert verdicts[0].window == "all" and verdicts[0].scored is True


class TestArmingRunsParityOnTheScopeNotTheWindows:
    """⚠ ASSERTED ON THE CALL, because the behaviour is one argument deep and a green arm proves
    nothing about which object the engine received."""

    @staticmethod
    def _arm_source():
        import inspect

        from millm.services.probe_arming import ProbeArmingService

        return inspect.getsource(ProbeArmingService.arm)

    def test_a_separate_probe_is_built_for_parity(self):
        src = self._arm_source()
        assert "windows=[]" in src, (
            "arming no longer builds a parity-only probe; the engine is being handed the "
            "operator's windows again, which generates unscoreable verdicts"
        )

    def test_the_engine_is_given_that_one(self):
        import ast

        tree = ast.parse(self._arm_source().strip())
        runs = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "run"
        ]
        assert runs, "no `.run(...)` call found in arm — this guard has stopped looking"
        first = runs[0].args[0]
        assert isinstance(first, ast.Name) and first.id == "for_parity", (
            f"the parity engine is given {ast.dump(first)[:60]}, not the scope-only probe"
        )


class TestTheRefusalSaysWhichFailureItWas:
    """⚠ THESE TEST BEHAVIOUR BECAUSE A SOURCE SCRAPE DID NOT.

    The first version of this asserted the message text appeared in `arm`'s source. Replacing the
    branch's `if` with `if False:` left the suite GREEN — the string is still in the file, the
    branch is simply dead. A guard satisfied by the wrong occurrence, which is the recurring
    failure here; the recorded remedy is to extract the decision and call it.
    """

    @staticmethod
    def _vectors(*diffs, reason=None):
        return {"vectors": [
            {"max_abs_diff": d, "reason": (reason if d is None else None)} for d in diffs
        ]}

    def test_all_vectors_unscored_is_detected(self):
        from millm.services.probe_arming import parity_scored_nothing

        assert parity_scored_nothing(self._vectors(None, None, None)) is True

    def test_one_scored_vector_means_something_WAS_compared(self):
        """Specificity: a single number makes it a real disagreement, not an empty run."""
        from millm.services.probe_arming import parity_scored_nothing

        assert parity_scored_nothing(self._vectors(None, 0.4, None)) is False

    def test_no_vectors_at_all_counts_as_nothing_compared(self):
        from millm.services.probe_arming import parity_scored_nothing

        assert parity_scored_nothing({"vectors": []}) is True
        assert parity_scored_nothing({}) is True
        assert parity_scored_nothing(None) is True

    def test_the_message_names_the_empty_run_and_its_reason(self):
        from millm.services.probe_arming import parity_refusal_message

        msg = parity_refusal_message(
            self._vectors(None, None, reason="prompt_boundary_unknown")
        )
        assert "could not score any test vector" in msg
        assert "not a disagreement about the numbers" in msg
        assert "prompt_boundary_unknown" in msg, (
            "the reason every vector gave is the one actionable fact and it is missing"
        )

    def test_a_real_disagreement_keeps_the_old_wording(self):
        from millm.services.probe_arming import parity_refusal_message

        msg = parity_refusal_message(self._vectors(0.9, 0.2))
        assert "does not reproduce the scores miStudio recorded" in msg
        assert "could not score" not in msg

    def test_arm_CALLS_the_decision_rather_than_inlining_it_again(self):
        import ast
        import inspect

        from millm.services.probe_arming import ProbeArmingService

        tree = ast.parse(inspect.getsource(ProbeArmingService.arm).strip())
        called = {
            n.func.id for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        assert "parity_refusal_message" in called, (
            f"arm no longer calls the refusal decision; calls: {sorted(called)}"
        )
