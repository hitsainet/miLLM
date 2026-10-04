"""The `last_user` window and per-window length bands (operator, 2026-10-04).

`prompt` is everything before the reply, so on a client that resends the conversation (Open
WebUI, LibreChat, most agent frameworks) a high-stakes earlier turn kept firing on every later one,
and a long system prompt or a retrieved document diluted or triggered it. `last_user` reads the
newest user message alone, over the SAME span miStudio calibrated it on: the message's header,
content and end-of-turn, found by rendering the conversation's prefixes with the same template.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from millm.ml.probe_head import ProbeHead
from millm.services.probe_arming import (
    BASE_DEFAULT_WINDOWS,
    resolve_windows,
    window_length_bands_from_definition,
)
from millm.services.probe_context import contexts_for
from millm.services.probe_runtime import ArmedProbe, ProbeRequestContext
from millm.services.probe_scope import WINDOWS, window_weights_trained
from millm.services.probe_turns import (
    NO_CHAT_TEMPLATE,
    NO_USER_TURN,
    SPAN_UNRESOLVED,
    last_user_token_span,
)

WORDS = ["<s>", "<|user|>", "<|assistant|>", "<|system|>", "<|end|>", "be", "brief", "first",
         "question", "an", "answer", "second", "virus", "spreads", "fast", "a", "b", "x", "y", "[UNK]"]


@pytest.fixture(scope="module")
def tokenizer():
    """A REAL fast tokenizer with a REAL chat template — the span is computed by rendering, so a
    stub would agree with whatever the code expects."""
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>", unk_token="[UNK]")
    fast.chat_template = (
        "<s> {% for m in messages %}<|{{ m.role }}|> {{ m.content }} <|end|> {% endfor %}"
        "{% if add_generation_prompt %}<|assistant|> {% endif %}"
    )
    return fast


CHAT = [
    {"role": "system", "content": "be brief"},
    {"role": "user", "content": "first question"},
    {"role": "assistant", "content": "an answer"},
    {"role": "user", "content": "virus spreads fast"},
]


def _served(tokenizer, messages, *, extra_bos=False):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return ([0] + ids) if extra_bos else ids


class TestTheSpan:
    def test_it_is_the_newest_user_turn_with_its_header_and_end(self, tokenizer):
        served = _served(tokenizer, CHAT)
        span, reason = last_user_token_span(tokenizer, CHAT, served)
        assert reason is None
        assert tokenizer.convert_ids_to_tokens(served[span[0]: span[1]]) == [
            "<|user|>", "virus", "spreads", "fast", "<|end|>",
        ]

    def test_a_bos_the_tokenizer_prepended_shifts_it(self, tokenizer):
        """Serving may add a BOS the template render does not carry; positions must follow."""
        plain, _ = last_user_token_span(tokenizer, CHAT, _served(tokenizer, CHAT))
        shifted, _ = last_user_token_span(tokenizer, CHAT, _served(tokenizer, CHAT, extra_bos=True))
        assert shifted == (plain[0] + 1, plain[1] + 1)

    def test_a_first_and_only_user_turn_starts_at_its_own_header(self, tokenizer):
        """Review round 1 (H1): message 0 also holds the BOS and any template preamble; the window
        starts at the message's own role header, as miStudio calibrates it."""
        messages = [{"role": "user", "content": "virus spreads fast"}]
        served = _served(tokenizer, messages)
        span, _ = last_user_token_span(tokenizer, messages, served)
        assert tokenizer.convert_ids_to_tokens(served[span[0]: span[1]]) == [
            "<|user|>", "virus", "spreads", "fast", "<|end|>",
        ]

    def test_no_user_turn_is_a_reason(self, tokenizer):
        messages = [{"role": "system", "content": "be brief"}]
        assert last_user_token_span(tokenizer, messages, _served(tokenizer, messages)) == (None, NO_USER_TURN)

    def test_ids_that_are_not_this_render_are_refused_not_guessed(self, tokenizer):
        assert last_user_token_span(tokenizer, CHAT, [1, 2, 3]) == (None, SPAN_UNRESOLVED)

    def test_no_chat_template_is_a_reason(self, tokenizer):
        bare = PreTrainedTokenizerFast(tokenizer_object=tokenizer.backend_tokenizer, unk_token="[UNK]")
        assert last_user_token_span(bare, CHAT, [0]) == (None, NO_CHAT_TEMPLATE)


D = 4


def _armed(**over):
    base = dict(
        probe_id="pr_1", name="p", head=ProbeHead(weight=torch.ones(D), bias=0.0, layer=1),
        rule="mean", scope="all", layer=1, rung=2, rung_language="x", threshold=10.0,
        windows=("last_user",), window_thresholds={"last_user": 2.0},
    )
    base.update(over)
    return ArmedProbe(**base)


def _run(probe, *, span, reason=None, n_prompt=10, generated=3):
    ctx = ProbeRequestContext("r", [probe])
    ctx.set_prompt_length(n_prompt)
    ctx.set_last_user_span(span, reason)
    values = torch.zeros((1, n_prompt, D))
    if span:
        values[0, span[0]: span[1]] = 3.0 / D  # the newest message scores 3, history scores 0
    ctx.observe(1, values)
    for _ in range(generated):
        ctx.observe(1, torch.zeros((1, 1, D)))
    return {v.window: v for v in ctx.finish()}


class TestTheWindowAtServeTime:
    def test_it_scores_only_the_span(self):
        got = _run(_armed(), span=(6, 10))["last_user"]
        assert got.scored and got.n_scored_tokens == 4
        assert got.score == pytest.approx(3.0), "history leaked into the newest message's mean"
        assert got.threshold == 2.0 and got.fires is True

    def test_without_a_span_it_says_why(self):
        got = _run(_armed(), span=None, reason=NO_USER_TURN)["last_user"]
        assert got.scored is False and got.not_scored_reason == NO_USER_TURN

    def test_it_is_not_provisional_for_an_all_probe(self):
        assert _run(_armed(), span=(6, 10))["last_user"].provisional is False
        assert window_weights_trained("all", "last_user") is True

    def test_a_windows_own_bands_refine_its_own_bar(self):
        bands = [
            {"min_tokens": 0, "max_tokens": 5, "threshold": 4.0},
            {"min_tokens": 6, "max_tokens": None, "threshold": 1.0},
        ]
        got = _run(_armed(window_length_bands={"last_user": bands}), span=(6, 10))["last_user"]
        assert got.n_scored_tokens == 4 and got.threshold == 4.0, (
            "a 4-token turn must be judged against this window's own short band"
        )

    def test_it_is_a_known_window_on_by_default_only_with_its_own_bar(self):
        """Review round 1 (M4): an older probe has no `last_user` bar, so defaulting the window on
        would fire provisionally against the global bar on traffic that was quiet before."""
        assert "last_user" in WINDOWS and "last_user" not in BASE_DEFAULT_WINDOWS
        assert "last_user" not in resolve_windows(None, probe_scope="all")
        assert "last_user" in resolve_windows(None, probe_scope="all", calibrated={"last_user": 2.0})
        assert resolve_windows(["last_user"], probe_scope="all") == ("last_user",)


class TestReadingAWindowsBandsFromTheDefinition:
    BANDS = [{"min_tokens": 0, "max_tokens": None, "threshold": 2.5}]

    def test_bands_need_the_windows_own_bar(self):
        definition = {"decision": {"windows": {
            "last_user": {"threshold": 2.0, "length_bands": self.BANDS},
            "prompt": {"length_bands": self.BANDS},  # no bar: the bands refine nothing
        }}}
        assert window_length_bands_from_definition(definition) == {"last_user": self.BANDS}

    def test_a_torn_table_is_dropped_whole(self):
        torn = [{"min_tokens": 0, "max_tokens": 5, "threshold": 2.0}]  # closed final band
        definition = {"decision": {"windows": {"last_user": {"threshold": 2.0, "length_bands": torn}}}}
        assert window_length_bands_from_definition(definition) == {}


class TestRecalibrationRefusesWhatWouldNeverApply:
    def test_an_unknown_window_is_refused(self):
        from millm.services.probe_recalibration import (
            ProbeRecalibrationRefused,
            every_submitted_bar_survives_parsing,
        )

        with pytest.raises(ProbeRecalibrationRefused, match="not windows this server can report"):
            every_submitted_bar_survives_parsing({"windows": {"newest": {"threshold": 1.0}}})

    def test_a_torn_window_band_table_is_refused(self):
        from millm.services.probe_recalibration import (
            ProbeRecalibrationRefused,
            every_submitted_bar_survives_parsing,
        )

        torn = [{"min_tokens": 0, "max_tokens": 5, "threshold": 2.0}]
        with pytest.raises(ProbeRecalibrationRefused, match="per-length tables for window"):
            every_submitted_bar_survives_parsing(
                {"windows": {"last_user": {"threshold": 1.0, "length_bands": torn}}}
            )


class TestTheContextAndTheWiring:
    def test_the_context_stays_inside_the_span(self):
        from unittest.mock import MagicMock

        verdict = MagicMock()
        verdict.probe_id, verdict.window, verdict.top_positions = "pr_1", "last_user", [7]
        tok = MagicMock()
        tok.decode = lambda ids, skip_special_tokens=True: " ".join(map(str, ids))
        out = contexts_for([verdict], list(range(100, 120)), 5, tok, prompt_length=12,
                           last_user_span=(6, 10))
        assert out[("pr_1", "last_user")]["context_token_ids"] == list(range(106, 110))

    def test_both_chat_paths_record_the_span_and_text_completion_states_why_not(self):
        from millm.services import inference_service

        source = textwrap.dedent(inspect.getsource(inference_service.InferenceService))
        tree = ast.parse(source)
        calls = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "_probe_note_last_user_span"
        ]
        assert len(calls) == 2, "the non-streamed and streamed chat paths must both record the span"
        for call in calls:
            assert ast.unparse(call.args[1]) == "request.messages"
        reasons = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "set_last_user_span"
            and any(isinstance(a, ast.Constant) and a.value == "text_completion_has_no_user_turn" for a in n.args)
        ]
        assert reasons, "a raw-text completion must say why it has no last_user window"


class TestTheWindowBandsAreWiredEndToEnd:
    """⚠ WRITTEN BECAUSE THREE MUTATIONS SURVIVED: arming not reading a window's bands,
    recalibration not refreshing them, and the span helper skipping its prefix check."""

    BANDS = [
        {"min_tokens": 0, "max_tokens": 5, "threshold": 4.0},
        {"min_tokens": 6, "max_tokens": None, "threshold": 1.0},
    ]

    def test_arming_reads_each_windows_own_bands(self):
        from tests.unit.services.test_probe_recalibration import _Row, decision, definition

        from millm.services.probe_arming import armed_probe_from_row

        row = _Row(definition=definition(decision=decision(
            windows={"last_user": {"threshold": 2.0, "length_bands": self.BANDS}}
        )))
        probe = armed_probe_from_row(row, windows=["last_user"])
        assert probe.window_length_bands == {"last_user": self.BANDS}

    @pytest.mark.asyncio
    async def test_recalibration_refreshes_the_live_windows_bands(self):
        from tests.unit.services.test_probe_recalibration import (
            MISTUDIO_ID, _Repo, _Row, armed, decision,
        )

        from millm.services.probe_recalibration import ProbeRecalibrationService
        from millm.services.probe_runtime import ProbeRuntimeState
        from tests.unit.services.test_probe_recalibration import arm_into

        state = ProbeRuntimeState()
        arm_into(state, armed(windows=("last_user",), window_thresholds={"last_user": 2.0}))
        new = [dict(band, threshold=band["threshold"] + 10.0) for band in self.BANDS]
        outcome = await ProbeRecalibrationService(_Repo(), state).recalibrate(
            _Row(),
            decision=decision(windows={"last_user": {"threshold": 12.0, "length_bands": new}}),
            mistudio_probe_id=MISTUDIO_ID,
            reason="window bands moved",
        )
        assert outcome["registry_updated"] is True
        live = state.get(armed().probe_id)
        assert live.window_length_bands == {"last_user": new}, (
            "the live probe kept its old window bands, so verdicts are judged as before the re-cut"
        )

    def test_a_template_that_rewrites_earlier_turns_is_refused(self):
        """The prefix property, checked: a template whose render of a turn depends on what
        follows would place the span over the wrong tokens."""
        tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="[UNK]"))
        tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>", unk_token="[UNK]")
        # The LAST message is rendered under a different header, so a turn reads differently once
        # another follows it: the prefix render is not a token-prefix of the next.
        fast.chat_template = (
            "<s> {% for m in messages %}{% if loop.last %}<|system|> {{ m.content }} "
            "{% else %}<|{{ m.role }}|> {{ m.content }} <|end|> {% endif %}{% endfor %}"
        )
        text = fast.apply_chat_template(CHAT, tokenize=False, add_generation_prompt=True)
        served = fast(text, add_special_tokens=False)["input_ids"]
        assert last_user_token_span(fast, CHAT, served) == (None, SPAN_UNRESOLVED)


# ── Review round 1, H1/H2: the span is pinned by a case file BOTH repos test ──

import json as _json
import os as _os
from pathlib import Path as _Path

_CASES_PATH = _Path(__file__).resolve().parents[3] / "docs" / "schemas" / "last-user-span-cases.json"
_STUDIO_CASES = _Path(_os.environ.get("MISTUDIO_REPO", "/home/x-sean/app/miStudio")) / "docs" / "schemas" / "last-user-span-cases.json"
_CASES = _json.loads(_CASES_PATH.read_text())


def _case_tokenizer(prepend_bos: bool, template: str | None = None, case: dict | None = None):
    from tokenizers import processors

    vocab = (case or {}).get("vocab") or _CASES["vocab"]
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(vocab)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    if (case or {}).get("pretokenizer") == "metaspace_first":
        from tokenizers import Regex

        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Metaspace(replacement="\u2581", prepend_scheme="first", split=True),
            pre_tokenizers.Split(Regex("\u2581?<\\|[a-z]+\\|>"), behavior="isolated"),
        ])
    if prepend_bos:
        tok.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 0)])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>", unk_token="[UNK]")
    fast.chat_template = template or _CASES["template"]
    return fast


@pytest.mark.parametrize("case", _CASES["cases"], ids=lambda c: c["name"])
def test_the_served_span_matches_the_shared_cases(case):
    """What this server scores — tokenized as `InferenceService` does, specials ON."""
    tok = _case_tokenizer(case["prepend_bos"], case.get("template"), case)
    prompt = tok.apply_chat_template(case["messages"], tokenize=False, add_generation_prompt=True)
    served = tok(prompt)["input_ids"]
    span, _reason = last_user_token_span(tok, case["messages"], served)
    got = tok.convert_ids_to_tokens(served[span[0]: span[1]]) if span else None
    assert got == case["expected_span"]


def test_the_case_file_is_identical_in_mistudio():
    if not _STUDIO_CASES.exists():
        if _os.environ.get("MILLM_REQUIRE_CROSS_REPO_CHECKS") == "1":
            pytest.fail(f"miStudio's copy of the span cases is missing at {_STUDIO_CASES}")
        pytest.skip("miStudio checkout not present")
    assert _STUDIO_CASES.read_bytes() == _CASES_PATH.read_bytes()


class TestTheSpanIsOnlyComputedWhenSomethingReadsIt:
    """Review round 1 (M3): three renders and tokenizations per request are not free."""

    def _service(self, monkeypatch, calls):
        from millm.services import inference_service, probe_turns

        monkeypatch.setattr(probe_turns, "last_user_token_span", lambda *a, **k: calls.append(1) or ((1, 2), None))
        from types import SimpleNamespace

        svc = inference_service.InferenceService.__new__(inference_service.InferenceService)
        # `_tokenizer` is a property over the loaded model state; stand that up, not the property.
        svc._model_state = SimpleNamespace(
            is_loaded=True, current=SimpleNamespace(tokenizer=object())
        )
        return svc

    def test_not_computed_without_a_last_user_window(self, monkeypatch):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        calls = []
        ctx = MagicMock()
        ctx.probes = [SimpleNamespace(windows=("prompt", "response"))]
        self._service(monkeypatch, calls)._probe_note_last_user_span(ctx, [], [1, 2, 3], None)
        assert calls == [] and not ctx.set_last_user_span.called

    def test_computed_and_recorded_when_a_probe_reads_it(self, monkeypatch):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        calls = []
        ctx = MagicMock()
        ctx.probes = [SimpleNamespace(windows=("last_user",))]
        self._service(monkeypatch, calls)._probe_note_last_user_span(ctx, [], [1, 2, 3], None)
        assert calls == [1]
        ctx.set_last_user_span.assert_called_once_with((1, 2), None)


class TestTheCachedPreambleCannotGoStale:
    """Review round 3 (M1): templates that call `strftime_now` stamp today's date into the
    preamble, so a prefix cached before midnight matched no request after it and every
    single-turn `last_user` verdict went dark until a restart."""

    def test_a_stale_cached_prefix_is_recomputed_not_refused(self):
        from millm.services import probe_turns

        tok = _case_tokenizer(False)
        messages = [{"role": "user", "content": "virus spreads fast"}]
        served = tok(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))["input_ids"]
        good, _ = last_user_token_span(tok, messages, served)
        prefix, start = probe_turns.first_user_header(tok)
        stale = list(prefix)
        stale[1] = tok.convert_tokens_to_ids("brief")
        [key] = [k for k in probe_turns._cache[tok] if k.startswith("first:")]
        probe_turns._cache[tok][key] = (stale, start)
        span, reason = last_user_token_span(tok, messages, served)
        assert (span, reason) == (good, None)
        assert probe_turns.first_user_header(tok) == (prefix, start)

    def test_the_cache_is_bounded_against_client_kwargs(self):
        from millm.services import probe_turns

        tok = _case_tokenizer(False)
        for i in range(probe_turns._MAX_ENTRIES * 2):
            probe_turns.user_header_ids(tok, {"client_value": i})
        assert len(probe_turns._cache[tok]) <= probe_turns._MAX_ENTRIES
