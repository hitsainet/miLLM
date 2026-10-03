"""A window's own bar and the length bands, together — the combination nothing tested.

⚠ FOUND ON A REAL CHAT, 2026-10-03, AFTER BOTH FEATURES HAD SHIPPED AND BEEN REVIEWED. Per-window
bars (2026-09-30) and per-length bands (2026-10-02) were each tested alone; every test that
exercised one left the other empty. In the verdict path the length lookup ran AFTER the window's
bar was chosen and replaced it, so with both present every window was judged against bands cut
from the WHOLE-REQUEST negatives:

    L16 response   own bar 24.20   judged at 12.35   (half the bar: false positives on short replies)
    L11 prompt     own bar 14.69   judged at 37.75   (silenced a verdict that should have fired)

The bands are quantiles of the probe's own-scope distribution — miStudio cuts them in the same
pass as the global bar — so they apply to that window and no other.

Also here: the `response` window's weights were never trained on model replies. A bar cut from
that window's negatives retired its `provisional` marker, so its verdicts read as measured, while
the UI's own copy says they are reported as provisional. Both reasons now mark it.
"""

from __future__ import annotations

import torch

from millm.ml.probe_head import ProbeHead
from millm.services.probe_runtime import ArmedProbe, ProbeRequestContext
from millm.services.probe_scope import window_weights_trained

D = 4

# The production shape: a short band and an open-ended one, cut on the `all` window.
BANDS = [
    {"min_tokens": 0, "max_tokens": 203, "threshold": 12.35, "threshold_source": "band"},
    {"min_tokens": 204, "max_tokens": None, "threshold": 30.0, "threshold_source": "band"},
]


def armed(*, windows=("all", "prompt", "response"), window_thresholds=None, bands=BANDS,
          threshold=17.24, scope="all"):
    return ArmedProbe(
        probe_id="pr_l16",
        name="pr_l16",
        head=ProbeHead(weight=torch.ones(D), bias=0.0, layer=1),
        rule="mean",
        scope=scope,
        layer=1,
        rung=2,
        rung_language="detects on unseen tasks",
        threshold=threshold,
        windows=windows,
        window_thresholds=(
            {"all": 17.24, "prompt": 14.69, "response": 24.20}
            if window_thresholds is None else window_thresholds
        ),
        length_bands=list(bands),
    )


def verdicts(probe, *, prompt_tokens=86, generated=40, prompt_value=20.0, reply_value=20.0):
    """Every token scores `value` (a `mean` of D ones over value/D), so the score is exact."""
    ctx = ProbeRequestContext("req", [probe])
    ctx.set_prompt_length(prompt_tokens)
    ctx.observe(1, torch.full((1, prompt_tokens, D), prompt_value / D))
    for _ in range(generated):
        ctx.observe(1, torch.full((1, 1, D), reply_value / D))
    return {v.window: v for v in ctx.finish()}


class TestEachWindowIsJudgedAgainstItsOwnBar:
    def test_the_response_window_keeps_its_own_bar(self):
        got = verdicts(armed())["response"]
        assert got.threshold == 24.20, (
            f"response judged at {got.threshold}: the length band cut on the whole-request "
            f"negatives replaced the bar cut on this window's own"
        )
        assert got.fires is False, "a 20.0 reply against its own 24.20 must not fire"

    def test_the_prompt_window_keeps_its_own_bar(self):
        got = verdicts(armed(), prompt_tokens=300)["prompt"]
        assert got.threshold == 14.69
        assert got.fires is True, "a 20.0 prompt against its own 14.69 must fire"

    def test_the_probes_own_window_still_uses_the_bands(self):
        """Specificity: the bands must still apply where they were cut, or this fix removed them."""
        short = verdicts(armed(), prompt_tokens=86, generated=40)["all"]
        assert short.n_scored_tokens == 126
        assert short.threshold == 12.35
        long = verdicts(armed(), prompt_tokens=300, generated=40)["all"]
        assert long.threshold == 30.0

    def test_a_window_without_its_own_bar_falls_back_to_the_global_not_a_band(self):
        got = verdicts(armed(window_thresholds={}), generated=40)["response"]
        assert got.threshold == 17.24

    def test_without_bands_nothing_changes(self):
        got = verdicts(armed(bands=[]))
        assert got["all"].threshold == 17.24
        assert got["prompt"].threshold == 14.69
        assert got["response"].threshold == 24.20


class TestTheResponseWindowIsProvisionalUntilItsWeightsAreTrained:
    def test_a_calibrated_response_bar_does_not_make_it_measured(self):
        got = verdicts(armed())
        assert got["response"].provisional is True, (
            "the response window has its own bar, but its weights never saw a model reply"
        )

    def test_the_trained_windows_are_not_flagged(self):
        got = verdicts(armed())
        assert got["prompt"].provisional is False
        assert got["all"].provisional is False

    def test_it_still_fires_by_operator_decision(self):
        got = verdicts(armed(window_thresholds={"response": 10.0}))["response"]
        assert got.provisional is True and got.fires is True

    def test_the_rule(self):
        assert window_weights_trained("all", "prompt") is True
        assert window_weights_trained("all", "all") is True
        assert window_weights_trained("all", "response") is False
        assert window_weights_trained("response", "response") is True


class TestStreamedRequestsCaptureTheGeneratedIds:
    """⚠ The response events of a streamed chat stored no context text: the id capture that feeds
    the recorder was installed only when SAE sensing was active, so with probes alone the recorder
    got the PROMPT ids and every response position fell past their end."""

    def test_the_capture_is_installed_when_probes_are_armed(self):
        import ast
        import inspect

        from millm.services import inference_service

        tree = ast.parse(inspect.getsource(inference_service))
        guards = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            body_calls = {
                getattr(n.func, "id", None)
                for stmt in node.body for n in ast.walk(stmt) if isinstance(n, ast.Call)
            }
            if "_make_id_capture_criteria" in body_calls:
                guards.append({n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)})
        assert guards, "the scan found no guarded id capture — it is looking at the wrong shape"
        for names in guards:
            assert "_probe_ctx" in names, (
                f"the id capture is guarded by {sorted(names)} only, so a streamed request with "
                f"probes armed records response events with no context"
            )
