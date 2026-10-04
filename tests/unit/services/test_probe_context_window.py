"""The decoded prompt window on a probe event — and the fact that nothing produced it.

⚠ EVERY OTHER PART OF THIS FEATURE SHIPPED. The `context_text` / `context_token_ids` columns
(migration 016), the `GET /api/probes/events/{id}` route documented as the only one that serves
them, the admin UI modal that renders them, `record(contexts=...)`, the broadcast stripping, the
privacy tests that assert the stripping — and `PROBE_EVENT_CONTEXT_TOKENS: int = 24` **had no reader
anywhere in the repo**. `_probe_record` never passed `contexts`, so `context_text` was NULL on every
event ever written and the modal opened empty.

Reported 2026-09-28: *"the window to see the associated prompt still does not open."* Verified on the
node first — the route returned HTTP 200 with `context_text: None` — so this is a producer defect,
not a route or UI one.

The sharpest detail: the comment directly above the offending call documents the SAME omission being
fixed for the argument beside it (`overhead_ms` "had NO production caller"). One was wired, its
neighbour was not, in the same review round. So the guard here is not "does a window function exist"
but "does every call site pass the ids, and does a real payload carry the text".
"""

from __future__ import annotations

import ast
import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from millm.services.probe_context import context_window, contexts_for


class FakeTokenizer:
    """Decodes to a readable marker per id, so a window's CONTENT is assertable."""

    def decode(self, ids, skip_special_tokens=True):  # noqa: ANN001
        return " ".join(f"t{int(i)}" for i in ids)


def _verdict(probe_id="pr_1", top=None, window="all"):
    v = MagicMock()
    v.probe_id = probe_id
    v.top_positions = [] if top is None else list(top)
    # ⚠ SET EXPLICITLY. A MagicMock answers every attribute, so a `getattr(v, "window", "all")`
    # in production would read a Mock here and the test would pass while the real key was
    # garbage. Stating it is what makes this fixture able to fail.
    v.window = window
    return v


class TestTheWindow:
    def test_it_centres_on_the_position(self):
        text, window = context_window(torch.arange(100), 50, 2, FakeTokenizer())
        assert window == [48, 49, 50, 51, 52]
        assert text == "t48 t49 t50 t51 t52"

    def test_it_clamps_at_the_start_without_wrapping(self):
        _text, window = context_window(torch.arange(100), 1, 5, FakeTokenizer())
        assert window == [0, 1, 2, 3, 4, 5, 6], "a negative lower bound must clamp, not wrap"

    def test_it_clamps_at_the_end(self):
        _text, window = context_window(torch.arange(10), 9, 5, FakeTokenizer())
        assert window == [4, 5, 6, 7, 8, 9]

    def test_a_2d_batch_tensor_uses_row_0(self):
        ids = torch.arange(20).unsqueeze(0)
        _text, window = context_window(ids, 3, 1, FakeTokenizer())
        assert window == [2, 3, 4]

    def test_a_plain_list_works_because_the_cbm_path_holds_one(self):
        _text, window = context_window(list(range(20)), 3, 1, FakeTokenizer())
        assert window == [2, 3, 4]

    @pytest.mark.parametrize(
        "ids,pos,k,tok",
        [
            (torch.arange(5), 9, 2, FakeTokenizer()),   # position beyond the ids
            (torch.arange(5), -1, 2, FakeTokenizer()),  # negative position
            (torch.arange(5), 1, 0, FakeTokenizer()),   # context disabled
            (None, 1, 2, FakeTokenizer()),              # no ids captured
            (torch.arange(5), 1, 2, None),              # model not loaded
        ],
    )
    def test_it_returns_nothing_rather_than_a_wrong_window(self, ids, pos, k, tok):
        """⚠ `(None, None)` is a deliberate answer, not a failure.

        A decode-phase position against prompt-only ids is legitimate — the streaming paths fall
        back to `inputs["input_ids"]`. Returning a window from the wrong end would tell the
        operator the verdict was about text it never scored, which is the one thing this modal
        exists to get right.
        """
        assert context_window(ids, pos, k, tok) == (None, None)

    def test_a_raising_tokenizer_does_not_cost_the_verdict(self):
        class Boom:
            def decode(self, *a, **k):
                raise RuntimeError("tokenizer exploded")

        assert context_window(torch.arange(10), 5, 2, Boom()) == (None, None)


class TestContextsFor:
    def test_it_keys_by_probe_id_AND_window(self):
        out = contexts_for(
            [_verdict("pr_a", [10]), _verdict("pr_b", [20])],
            torch.arange(100),
            1,
            FakeTokenizer(),
        )
        assert set(out) == {("pr_a", "all"), ("pr_b", "all")}
        assert out[("pr_a", "all")]["context_token_ids"] == [9, 10, 11]
        assert out[("pr_b", "all")]["context_token_ids"] == [19, 20, 21]

    def test_a_verdict_with_no_top_position_contributes_nothing(self):
        """⚠ No entry, not an empty one: `context_text IS NULL` must keep meaning "no window",
        not "a window we could not fill". An unscored verdict is the common case."""
        out = contexts_for([_verdict("pr_a", None)], torch.arange(100), 2, FakeTokenizer())
        assert out == {}

    def test_two_windows_of_ONE_probe_keep_separate_contexts(self):
        """⚠ THE REASON THE KEY IS A PAIR. One probe reports several windows and each one's top
        position is somewhere different — the prompt window peaks in the person's words, the
        response window in the model's. Keyed by probe id alone the second silently overwrote the
        first and both events opened on the same text while showing different scores. Nothing
        raised; the reader was simply shown the wrong evidence for one of them."""
        out = contexts_for(
            [_verdict("pr_a", [10], window="prompt"), _verdict("pr_a", [90], window="response")],
            torch.arange(100), 1, FakeTokenizer(),
        )
        assert set(out) == {("pr_a", "prompt"), ("pr_a", "response")}
        assert out[("pr_a", "prompt")]["context_token_ids"] == [9, 10, 11]
        assert out[("pr_a", "response")]["context_token_ids"] == [89, 90, 91]

    def test_it_uses_the_FIRST_top_position(self):
        """`top_positions` is sorted descending by score, so [0] is the top firing position —
        which is what the setting's own comment describes."""
        out = contexts_for([_verdict("pr_a", [40, 10, 90])], torch.arange(100), 0 + 1, FakeTokenizer())
        assert out[("pr_a", "all")]["context_token_ids"] == [39, 40, 41]


class TestItActuallyReachesTheEvent:
    """The half that was missing. A window function nobody calls is the defect, not the fix."""

    @pytest.mark.asyncio
    async def test_probe_record_forwards_a_populated_contexts(self):
        from millm.services.inference_service import InferenceService
        from millm.services.probe_runtime import ProbeRequestContext

        probe = MagicMock()
        probe.probe_id = "pr_1"
        probe.name = "p"
        probe.layer = 1
        probe.rule = "mean"
        probe.scope = "all"
        probe.rung = 2
        probe.rung_language = "held-out"
        probe.threshold = 0.0
        probe.rule_params = {}
        probe.encoder = None
        probe.head = MagicMock()

        context = ProbeRequestContext("req", [probe])

        verdicts = [_verdict("pr_1", [5])]

        service = MagicMock()
        service.record = AsyncMock()

        me = MagicMock()
        me._tokenizer = FakeTokenizer()
        me.is_model_loaded = MagicMock(return_value=True)

        import millm.api.dependencies as deps

        original = getattr(deps, "_probe_event_service", None)
        deps._probe_event_service = service
        try:
            await InferenceService._probe_record(
                me, context, verdicts, full_ids=torch.arange(50)
            )
        finally:
            deps._probe_event_service = original

        service.record.assert_awaited_once()
        kwargs = service.record.await_args.kwargs
        assert "contexts" in kwargs, (
            "record() was called WITHOUT contexts — context_text stays NULL on every event and "
            "the UI's prompt window opens empty"
        )
        got = kwargs["contexts"]
        assert got and ("pr_1", "all") in got, f"contexts was {got!r}"
        assert got[("pr_1", "all")]["context_text"], "the window carries no decoded text"
        assert got[("pr_1", "all")]["context_token_ids"]

    @pytest.mark.asyncio
    async def test_it_sends_None_rather_than_an_empty_dict_when_there_is_no_window(self):
        """So a row's NULL context means "nothing to show", not "an empty map was stored"."""
        from millm.services.inference_service import InferenceService
        from millm.services.probe_runtime import ProbeRequestContext

        service = MagicMock()
        service.record = AsyncMock()
        me = MagicMock()
        me._tokenizer = FakeTokenizer()
        me.is_model_loaded = MagicMock(return_value=True)

        import millm.api.dependencies as deps

        original = getattr(deps, "_probe_event_service", None)
        deps._probe_event_service = service
        try:
            await InferenceService._probe_record(
                me,
                ProbeRequestContext("req", []),
                [_verdict("pr_1", None)],
                full_ids=torch.arange(50),
            )
        finally:
            deps._probe_event_service = original

        assert service.record.await_args.kwargs["contexts"] is None

    @pytest.mark.asyncio
    async def test_the_configured_window_size_is_actually_read(self, monkeypatch):
        """⚠ `PROBE_EVENT_CONTEXT_TOKENS` had ZERO readers in the whole repo while its own comment
        described this feature. A setting nothing reads is a lie in the config file.

        Asserted BEHAVIOURALLY — the window's width changes with the setting — rather than by
        scraping the source for the name. And patched as an attribute on the settings singleton,
        not by rebinding a module name: `_probe_record` imports `settings` inside its own body, so
        a module-level patch would silently do nothing.
        """
        from millm.core.config import settings
        from millm.services.inference_service import InferenceService
        from millm.services.probe_runtime import ProbeRequestContext

        widths = {}
        for k in (1, 7):
            monkeypatch.setattr(settings, "PROBE_EVENT_CONTEXT_TOKENS", k)
            service = MagicMock()
            service.record = AsyncMock()
            me = MagicMock()
            me._tokenizer = FakeTokenizer()
            me.is_model_loaded = MagicMock(return_value=True)

            import millm.api.dependencies as deps

            original = getattr(deps, "_probe_event_service", None)
            deps._probe_event_service = service
            try:
                await InferenceService._probe_record(
                    me,
                    ProbeRequestContext("req", []),
                    [_verdict("pr_1", [25])],
                    full_ids=torch.arange(100),
                )
            finally:
                deps._probe_event_service = original
            widths[k] = len(
                service.record.await_args.kwargs["contexts"][("pr_1", "all")]["context_token_ids"]
            )

        assert widths == {1: 3, 7: 15}, (
            f"the window did not track PROBE_EVENT_CONTEXT_TOKENS: {widths} "
            f"(expected 2k+1 tokens for each k)"
        )


class TestEveryCallSitePassesTheIds:
    """⚠ The durable guard, on the AST, for a KEYWORD on each CALL.

    There are four generation paths and each calls `_probe_record` separately. A fifth added without
    `full_ids` would lose the prompt window on that path only — silently, because the event is still
    written and `context_text` is nullable. That is the shape of the original defect.
    """

    @staticmethod
    def _call_sites():
        from millm.services import inference_service

        tree = ast.parse(inspect.getsource(inference_service))
        return [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_probe_record"
        ]

    def test_the_scan_finds_the_call_sites_at_all(self):
        """A source scan that matches nothing asserts nothing — this repo has shipped two that
        failed open."""
        sites = self._call_sites()
        assert len(sites) >= 4, f"expected at least 4 _probe_record call sites, found {len(sites)}"

    def test_every_one_passes_full_ids(self):
        missing = [
            node.lineno
            for node in self._call_sites()
            if "full_ids" not in {kw.arg for kw in node.keywords}
        ]
        assert not missing, (
            f"_probe_record called without full_ids at line(s) {missing} — that path records "
            f"events with no prompt window, and nothing else will report it"
        )


class TestTheContextStaysInsideItsWindow:
    """⚠ FOUND ON A REAL CHAT, 2026-10-04: a prompt verdict peaking near the end of the prompt
    displayed the start of the model's REPLY, and a response verdict the end of the prompt. The
    context is evidence for one window's verdict, so it shows only that window's tokens."""

    IDS = list(range(100, 120))  # 20 tokens: prompt is the first 12, the reply the last 8
    N_PROMPT = 12

    def test_a_prompt_peak_at_the_end_of_the_prompt_shows_no_reply(self):
        out = contexts_for(
            [_verdict(top=[11], window="prompt")], self.IDS, 5, FakeTokenizer(), prompt_length=self.N_PROMPT,
        )
        ids = out[("pr_1", "prompt")]["context_token_ids"]
        assert ids == list(range(106, 112)), "the context ran past the prompt into the reply"

    def test_a_response_peak_at_the_start_of_the_reply_shows_no_prompt(self):
        out = contexts_for(
            [_verdict(top=[12], window="response")], self.IDS, 5, FakeTokenizer(), prompt_length=self.N_PROMPT,
        )
        ids = out[("pr_1", "response")]["context_token_ids"]
        assert ids == list(range(112, 118)), "the context reached back into the prompt"

    def test_the_all_window_is_unclipped(self):
        out = contexts_for(
            [_verdict(top=[11], window="all")], self.IDS, 5, FakeTokenizer(), prompt_length=self.N_PROMPT,
        )
        assert out[("pr_1", "all")]["context_token_ids"] == list(range(106, 117))

    def test_an_unknown_boundary_is_not_guessed(self):
        out = contexts_for([_verdict(top=[11], window="prompt")], self.IDS, 5, FakeTokenizer())
        assert out[("pr_1", "prompt")]["context_token_ids"] == list(range(106, 117))

    def test_the_recorder_passes_the_boundary(self):
        """Wiring: the clip is inert unless the recorder hands over the prompt length."""
        from millm.services import inference_service

        import textwrap

        tree = ast.parse(textwrap.dedent(inspect.getsource(inference_service.InferenceService._probe_record)))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "contexts_for"]
        assert len(calls) == 1
        passed = {k.arg: ast.unparse(k.value) for k in calls[0].keywords}
        assert passed.get("prompt_length") == "getattr(context, 'prompt_length', None)"

    def test_the_request_context_exposes_its_prompt_length(self):
        from millm.services.probe_runtime import ProbeRequestContext

        ctx = ProbeRequestContext("r", [])
        assert ctx.prompt_length is None
        ctx.set_prompt_length(12)
        assert ctx.prompt_length == 12
