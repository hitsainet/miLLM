"""A probe's scope actually restricts which positions it scores.

⚠ IT DID NOT, FOR THE WHOLE LIFE OF THE FEATURE. `scored_mask` was complete and correct and had
no production caller; `ProbeRequestContext` was built with `mask=None`, so `window(None, ...)`
returned all-True and every armed probe scored every position whatever its scope said. A
`prompt`-scoped probe would have scored the model's own output with weights that never saw one —
which `probe_scope`'s own opening paragraph calls a different detector.

That was invisible because the only scope anyone had ever armed was `all`, for which all-True is
correct. An arming gate added 2026-09-28 refused the other two rather than serving them wrongly;
this wires them and the gate widens.

Two things here are tested behaviourally rather than by reading the wiring:

* the SCORED POSITIONS, by giving each position a distinctive activation and checking which ones
  reach the score. A test that only asserted "a mask was passed" would pass against a mask that
  was built and then ignored.
* the BOUNDARY, which is stated by the caller and never inferred. Taking the first pass's length
  is right for an ordinary generation and silently wrong under chunked prefill, and the failure
  mode is scoring the model's output as the user's words.
"""

from __future__ import annotations

import pytest
import torch

from millm.services.probe_runtime import ArmedProbe, ProbeRequestContext
from millm.ml.probe_head import ProbeHead

D = 4


def _probe(probe_id: str, scope: str, rule: str = "mean") -> ArmedProbe:
    """A head that reads position magnitude straight through: score == the activation value."""
    return ArmedProbe(
        probe_id=probe_id,
        name=probe_id,
        head=ProbeHead(weight=torch.tensor([1.0, 0.0, 0.0, 0.0]), bias=0.0),
        rule=rule,
        scope=scope,
        layer=1,
        rung=2,
        rung_language="held-out",
        threshold=None,
    )


def _acts(values):
    """(1, T, D) where channel 0 of position i is values[i]."""
    row = torch.zeros(1, len(values), D)
    for i, v in enumerate(values):
        row[0, i, 0] = float(v)
    return row


class TestTheScopeRestrictsWhatIsScored:
    """Prompt positions carry 10, generated positions carry 100, so the mean names the set."""

    @pytest.mark.parametrize(
        "scope,expected_mean,expected_n",
        [
            ("all", (10 * 3 + 100 * 2) / 5, 5),
            ("prompt", 10.0, 3),
            ("response", 100.0, 2),
        ],
    )
    def test_one_pass_covering_prompt_and_response(self, scope, expected_mean, expected_n):
        probe = _probe("pr_1", scope)
        ctx = ProbeRequestContext("req", [probe])
        ctx.set_prompt_length(3)
        ctx.observe(1, _acts([10, 10, 10, 100, 100]))

        verdict = ctx.finish()[0]
        assert verdict.scored, verdict.not_scored_reason
        assert verdict.n_scored_tokens == expected_n
        assert verdict.score == pytest.approx(expected_mean)

    def test_across_a_prefill_and_several_decode_steps(self):
        """The realistic shape: one prefill pass then one pass per generated token. The window
        has to track the ABSOLUTE position, not the position within a pass."""
        for scope, expected_mean, expected_n in (
            ("all", (10 * 3 + 100 * 2) / 5, 5),
            ("prompt", 10.0, 3),
            ("response", 100.0, 2),
        ):
            probe = _probe("pr_1", scope)
            ctx = ProbeRequestContext("req", [probe])
            ctx.set_prompt_length(3)
            ctx.observe(1, _acts([10, 10, 10]))   # prefill
            ctx.observe(1, _acts([100]))          # decode 1
            ctx.observe(1, _acts([100]))          # decode 2

            verdict = ctx.finish()[0]
            assert verdict.scored, f"{scope}: {verdict.not_scored_reason}"
            assert verdict.n_scored_tokens == expected_n, scope
            assert verdict.score == pytest.approx(expected_mean), scope

    def test_two_probes_on_one_layer_with_DIFFERENT_scopes(self):
        """⚠ The window used to be built ONCE PER PASS and shared by every probe on the layer,
        which is correct only while they all agree — and scope is a per-probe field. Both probes
        would have read whichever window was built first."""
        prompt_probe = _probe("pr_prompt", "prompt")
        response_probe = _probe("pr_response", "response")
        ctx = ProbeRequestContext("req", [prompt_probe, response_probe])
        ctx.set_prompt_length(3)
        ctx.observe(1, _acts([10, 10, 10, 100, 100]))

        by_id = {v.probe_id: v for v in ctx.finish()}
        assert by_id["pr_prompt"].score == pytest.approx(10.0)
        assert by_id["pr_response"].score == pytest.approx(100.0)
        assert by_id["pr_prompt"].n_scored_tokens == 3
        assert by_id["pr_response"].n_scored_tokens == 2

    def test_a_response_probe_on_a_prompt_only_request_scores_nothing(self):
        """Not a zero — the probe never looked. `no_scored_tokens` is the honest verdict."""
        ctx = ProbeRequestContext("req", [_probe("pr_1", "response")])
        ctx.set_prompt_length(3)
        ctx.observe(1, _acts([10, 10, 10]))

        verdict = ctx.finish()[0]
        assert verdict.scored is False
        assert verdict.not_scored_reason == "no_scored_tokens"


class TestTheBoundaryIsStatedNotGuessed:
    def test_without_a_boundary_a_scoped_probe_refuses(self):
        """⚠ The alternative is guessing, and a wrong guess scores the model's own output under
        the name of the user's prompt."""
        ctx = ProbeRequestContext("req", [_probe("pr_1", "prompt")])
        ctx.observe(1, _acts([10, 10, 10, 100, 100]))

        verdict = ctx.finish()[0]
        assert verdict.scored is False
        assert verdict.not_scored_reason == "prompt_boundary_unknown"

    def test_without_a_boundary_an_all_probe_is_unaffected(self):
        """⚠ Specificity. `all` needs no boundary, and requiring one would break every probe
        actually in service."""
        ctx = ProbeRequestContext("req", [_probe("pr_1", "all")])
        ctx.observe(1, _acts([10, 10, 10, 100, 100]))

        verdict = ctx.finish()[0]
        assert verdict.scored is True
        assert verdict.n_scored_tokens == 5

    def test_the_same_boundary_twice_is_fine(self):
        """Several call sites may report it; agreeing is not a conflict."""
        ctx = ProbeRequestContext("req", [_probe("pr_1", "prompt")])
        ctx.set_prompt_length(3)
        ctx.set_prompt_length(3)
        ctx.observe(1, _acts([10, 10, 10, 100, 100]))
        assert ctx.finish()[0].n_scored_tokens == 3

    def test_two_DIFFERENT_boundaries_refuse_rather_than_pick_one(self):
        """Two callers disagreeing means scoring under either is a guess."""
        ctx = ProbeRequestContext("req", [_probe("pr_1", "prompt")])
        ctx.set_prompt_length(3)
        ctx.set_prompt_length(4)
        ctx.observe(1, _acts([10, 10, 10, 100, 100]))

        verdict = ctx.finish()[0]
        assert verdict.scored is False
        assert "prompt_boundary_conflict" in verdict.not_scored_reason

    def test_a_negative_boundary_refuses(self):
        ctx = ProbeRequestContext("req", [_probe("pr_1", "prompt")])
        ctx.set_prompt_length(-1)
        ctx.observe(1, _acts([10, 10]))
        assert "prompt_boundary_invalid" in ctx.finish()[0].not_scored_reason


class TestEveryGenerationPathStatesIt:
    """Wiring, by AST. Four paths open a probe context; one that forgets the boundary would
    report `prompt_boundary_unknown` on every scoped probe — honest, but broken on that path
    only, and nothing else would say so."""

    @staticmethod
    def _counts():
        import ast
        import inspect

        from millm.services import inference_service

        tree = ast.parse(inspect.getsource(inference_service))
        begins = notes = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "_probe_begin":
                    begins += 1
                elif node.func.attr == "_probe_note_prompt_length":
                    notes += 1
        return begins, notes

    def test_the_scan_sees_the_begin_sites(self):
        """Three REGISTERED begins: the serial chat, streaming chat and text paths, which score.

        Was four until Feature 27: the CBM streaming path moved to `_probe_begin_detached`
        (FR-27.8h), which marks the request `continuous_batching` and needs no boundary — a
        detached context never scores. Every generation path's wiring is proved behaviourally by
        `test_probe_paths_discovered.py`; this keeps the boundary-report check honest.
        """
        begins, _ = self._counts()
        assert begins >= 3, f"expected at least 3 _probe_begin call sites, found {begins}"

    def test_every_begin_is_matched_by_a_boundary_report(self):
        begins, notes = self._counts()
        assert notes >= begins, (
            f"{begins} paths open a probe context but only {notes} report the prompt boundary; "
            f"a scoped probe on the unreported path would never score"
        )
