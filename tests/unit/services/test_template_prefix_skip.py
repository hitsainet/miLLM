"""G38: a probe whose definition says `aggregation.skip_template_prefix` is served without the chat
template's fixed prefix (operator decisions G38-1..3, 2026-11-08).

The first P positions of every request are the same tokens in the same context — the start token,
the template's preamble, the first user header — so their scores are a constant, and under `max`
that constant is a floor the content never rises above. miStudio now trains, calibrates and exports
such probes without those positions; this server must read the same positions, or the bar applies
to another distribution. What is held here:

* the SHARED CASE FILE (`docs/schemas/template-prefix-cases.json`, byte-identical with miStudio):
  P and every window's mask, through `ProbeInputPreparer` (the `/api/probes/score` wiring) and the
  runtime's own windows, not only the helper;
* streaming == batch: the same verdicts whether the request is observed in one pass or as a
  prefill and one pass per token;
* a probe WITHOUT the flag is served exactly as before, whatever P is stated;
* an unstated P refuses only the windows that hold the prefix (`template_prefix_unknown`);
* parity of a flagged probe, from the vector's own stated P and on a tiny REAL model;
* the wiring: arming reads the flag from the stored document, and every serving path that scores
  states P (driven, not read).

Mutation controls are recorded in `0xcc/reviews/g38_template_prefix_2026-11-08.md`.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from miserver.ml.probe_head import ProbeHead
from miserver.services.probe_parity import ProbeParityEngine
from miserver.services.probe_runtime import ArmedProbe, ProbeRequestContext
from miserver.services.probe_scope import TEMPLATE_PREFIX_WINDOWS, scored_mask
from miserver.services.probe_scoring import ProbeInputPreparer, served_render, template_renderer
from miserver.services.probe_turns import template_prefix_ids, template_prefix_length
from tests.support.env_flags import flag
from tests.support.sibling_repos import cross_repo_checks_required, workbench_repo

REPO = Path(__file__).resolve().parents[3]
CASES_PATH = REPO / "docs" / "schemas" / "template-prefix-cases.json"
STUDIO_CASES = workbench_repo() / "docs" / "schemas" / "template-prefix-cases.json"
CASES = json.loads(CASES_PATH.read_text(encoding="utf-8"))
STRUCTURAL = CASES["structural"]
REAL = CASES["real"]
D = 4


def _env(name: str, default=None):
    """The ONE place this file reads a non-flag variable (`HF_HOME`). The suite's own switches go
    through `tests.support.env_flags` and the sibling helpers (040 FR-040.9, FR-040.13)."""
    return os.environ.get(name, default)


def _probe(probe_id="pr_g38", scope="all", rule="max", skip=True, windows=(), **over):
    """A head that reads channel 0 straight through: a position's score IS its activation."""
    return ArmedProbe(
        probe_id=probe_id, name=probe_id,
        head=ProbeHead(weight=torch.tensor([1.0, 0.0, 0.0, 0.0]), bias=0.0),
        rule=rule, scope=scope, layer=1, rung=2, rung_language="held-out", threshold=None,
        windows=tuple(windows), skip_template_prefix=skip, **over,
    )


def _acts(values):
    row = torch.zeros(1, len(values), D)
    for i, v in enumerate(values):
        row[0, i, 0] = float(v)
    return row


def _verdicts(ctx):
    return {(v.probe_id, v.window): v for v in ctx.finish()}


# ── the shared case file ─────────────────────────────────────────────────────────────────────


def _structural_tokenizer(case):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocab = case.get("vocab") or STRUCTURAL["vocab"]
    backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(vocab)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    if case["tokenizer_adds_bos"]:
        backend.post_processor = processors.TemplateProcessing(
            single="<s> $A", special_tokens=[("<s>", vocab.index("<s>"))]
        )
    fast = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<s>", unk_token="[UNK]")
    fast.chat_template = STRUCTURAL["templates"][case["template"]]
    if "stale_cached_prefix" in case:
        # The state a date-stamped preamble leaves after midnight: yesterday's prefix in the cache.
        from miserver.services import probe_turns

        template_prefix_ids(fast)
        [key] = [k for k in probe_turns._cache[fast] if k.startswith("template_prefix:")]
        probe_turns._cache[fast][key] = fast.convert_tokens_to_ids(case["stale_cached_prefix"])
    return fast


def _messages_item(messages):
    return SimpleNamespace(
        kinds=lambda: ["messages"],
        messages=[SimpleNamespace(role=m["role"], content=m["content"]) for m in messages],
    )


def _bits(mask):
    return "".join("1" if keep else "0" for keep in mask)


def _runtime_masks(prepared, windows):
    """Each contract window's mask AS THE RUNTIME BUILDS IT: one probe per scope, observed over
    position-encoded activations, read back through `token_scores_for` (scored positions only)."""
    n = len(prepared.ids)
    out = {}
    for window in windows:
        if window == "last_user":
            # Not a scope: read through the runtime's own window builder for a skipping `all` probe.
            probe = _probe("pr_lu", scope="all", rule="mean", windows=("last_user",))
            ctx = ProbeRequestContext("case", [probe])
            ctx.set_prompt_length(prepared.prompt_tokens)
            ctx.set_template_prefix(prepared.template_prefix)
            ctx.set_last_user_span(prepared.last_user_span, prepared.last_user_reason)
            ctx.observe(1, _acts(range(1, n + 1)))
            mask = ctx._mask_for_window(probe, "last_user", ctx._mask[probe.probe_id], n)
            out[window] = _bits([False] * n if mask is None else mask.tolist())
            continue
        probe = _probe(f"pr_{window}", scope=window, rule="mean")
        ctx = ProbeRequestContext("case", [probe])
        ctx.set_prompt_length(prepared.prompt_tokens)
        ctx.set_template_prefix(prepared.template_prefix)
        ctx.observe(1, _acts(range(1, n + 1)))           # position i carries i + 1
        scored = {int(v) - 1 for v in ctx.token_scores_for(probe.probe_id)}
        out[window] = _bits([i in scored for i in range(n)])
    return out


def _check(tokenizer, case, ids_key):
    expected = case["expected"]
    render = template_renderer(tokenizer)
    served = served_render(tokenizer, case["messages"], render)
    if ids_key == "tokens":
        assert tokenizer.convert_ids_to_tokens(served.ids) == expected["tokens"]
    else:
        assert served.ids == expected["ids"]
    assert served.prompt_tokens == expected["prompt_tokens"]
    assert template_prefix_length(tokenizer, served.ids) == expected["template_prefix"]
    # THE WIRING `/api/probes/score` runs.
    prepared = ProbeInputPreparer(tokenizer, render).prepare(0, _messages_item(case["messages"]))
    assert prepared.error is None and prepared.ids == served.ids
    assert prepared.template_prefix == expected["template_prefix"]
    assert _runtime_masks(prepared, expected["masks"]) == expected["masks"]
    # And `scored_mask` itself, from the counts alone.
    n, boundary = len(served.ids), served.prompt_tokens
    for window in ("all", "prompt", "response"):
        if window in expected["masks"]:
            assert _bits(scored_mask(scope=window, n_prompt_tokens=boundary, n_generated=n - boundary,
                                     skip_prefix=expected["template_prefix"])) == expected["masks"][window]


@pytest.mark.parametrize("case", STRUCTURAL["cases"], ids=lambda c: c["name"])
def test_structural_cases(case):
    _check(_structural_tokenizer(case), case, "tokens")


def _real_tokenizer(hf_id):
    sha = REAL["tokenizers"][hf_id]["chat_template_sha256"]
    dirs = []
    configured = flag("REAL_TOKENIZERS")
    if configured:
        dirs += [Path(p) for p in configured.split(os.pathsep) if p]
    hub = Path(_env("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    dirs += [Path(p) for p in sorted(glob.glob(str(hub / f"models--{hf_id.replace('/', '--')}" / "snapshots" / "*")))]
    from transformers import AutoTokenizer

    for path in dirs:
        if not (path / "tokenizer_config.json").exists():
            continue
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(path))
        except Exception:  # noqa: BLE001 - a snapshot this transformers cannot read
            continue
        if hashlib.sha256((getattr(tokenizer, "chat_template", None) or "").encode()).hexdigest() == sha:
            return tokenizer
    pytest.skip(
        f"NO {hf_id} TOKENIZER WITH CHAT TEMPLATE {sha[:12]} ON THIS MACHINE — the real-tokenizer "
        "template-prefix cases DID NOT RUN. Set MISERVER_REAL_TOKENIZERS to check them; the structural "
        "cases ran."
    )


_REAL = {}


@pytest.mark.parametrize("case", REAL["cases"], ids=lambda c: c["name"])
def test_real_tokenizer_cases(case):
    if case["tokenizer"] not in _REAL:
        _REAL[case["tokenizer"]] = _real_tokenizer(case["tokenizer"])
    _check(_REAL[case["tokenizer"]], case, "ids")


def test_the_case_file_is_identical_in_mistudio():
    """⚠ TWO IMPLEMENTATIONS, ONE PREFIX. A differing copy means the bar was cut over one set of
    positions and is served over another, with both suites green."""
    if not STUDIO_CASES.exists():
        if cross_repo_checks_required():
            pytest.fail(f"miStudio's copy of the template-prefix cases is missing at {STUDIO_CASES}")
        pytest.skip("miStudio checkout not present")
    assert STUDIO_CASES.read_bytes() == CASES_PATH.read_bytes()


def test_the_prefix_is_the_start_tokens_and_first_user_header():
    """One machinery: `template_prefix_ids` ends with `first_user_header`'s prefix."""
    from miserver.services.probe_turns import first_user_header

    checked = 0
    for case in STRUCTURAL["cases"]:
        if "vocab" in case:
            continue        # probes it cannot tell apart: the prefix is deliberately empty
        tokenizer = _structural_tokenizer(case)
        found = first_user_header(tokenizer, refresh=True)
        served = template_prefix_ids(tokenizer, refresh=True)
        if found is None:
            continue
        assert served[len(served) - len(found[0]):] == found[0]
        checked += 1
    assert checked >= 5


def test_a_stale_cached_prefix_is_re_read_once():
    """A date-stamped preamble moves at midnight; a request must not be measured against
    yesterday's prefix (the `last_user` window's review round 3, M1, applied to P)."""
    from miserver.services import probe_turns

    case = STRUCTURAL["cases"][0]
    tokenizer = _structural_tokenizer(case)
    served = served_render(tokenizer, case["messages"], template_renderer(tokenizer)).ids
    good = template_prefix_ids(tokenizer, refresh=True)
    [key] = [k for k in probe_turns._cache[tokenizer] if k.startswith("template_prefix:")]
    stale = list(good)
    stale[3] = tokenizer.convert_tokens_to_ids("brief")    # "yesterday's" preamble
    probe_turns._cache[tokenizer][key] = stale
    assert template_prefix_length(tokenizer, served) == case["expected"]["template_prefix"]
    assert probe_turns._cache[tokenizer][key] == good


# ── the runtime ──────────────────────────────────────────────────────────────────────────────


PREFIX = [50, 50, 50, 50]          # the floor: identical, high, in every request


class TestTheFloorIsGone:
    def test_max_without_the_flag_is_the_prefix_for_every_request(self):
        rows = [[3, 7, 1], [9, 2], [4, 4, 6, 8]]
        for skip, expected in ((False, [50, 50, 50]), (True, [7, 9, 8])):
            got = []
            for content in rows:
                ctx = ProbeRequestContext("req", [_probe(skip=skip)])
                ctx.set_template_prefix(len(PREFIX))
                ctx.observe(1, _acts(PREFIX + content))
                got.append(ctx.finish()[0].score)
            assert got == expected, skip


class TestStreamingEqualsBatch:
    """The served form: a prefill pass then one pass per generated token. The skip is positional
    (absolute positions), so the windows of the two forms must be identical, rule by rule."""

    VALUES = [50, 50, 50, 3, 9, 1, 4, 100, 2, 6]
    PROMPT = 7          # 3 prefix + 4 content, then 3 generated

    def _run(self, probes, passes):
        ctx = ProbeRequestContext("req", probes)
        ctx.set_prompt_length(self.PROMPT)
        ctx.set_template_prefix(3)
        start = 0
        for size in passes:
            ctx.observe(1, _acts(self.VALUES[start: start + size]))
            start += size
        return {key: (v.scored, v.score, v.n_scored_tokens) for key, v in _verdicts(ctx).items()}

    @pytest.mark.parametrize("rule,params", [
        ("max", {}), ("mean", {}), ("softmax", {"tau": 1.0}), ("rolling_mean_max", {"window": 2}),
    ])
    def test_every_window_and_rule(self, rule, params):
        def probes():
            return [
                _probe("pr_all", "all", rule, windows=("all", "prompt", "response"), rule_params=params),
                # `all` is NOT this probe's own scope, so it is rebuilt in `_mask_for_window`.
                _probe("pr_prompt", "prompt", rule, windows=("prompt", "all"), rule_params=params),
                _probe("pr_plain", "all", rule, skip=False, windows=("all", "prompt"), rule_params=params),
            ]

        batch = self._run(probes(), [len(self.VALUES)])
        stream = self._run(probes(), [self.PROMPT, 1, 1, 1])
        assert batch == stream
        assert batch[("pr_all", "all")][2] == len(self.VALUES) - 3
        assert batch[("pr_all", "prompt")][2] == self.PROMPT - 3
        assert batch[("pr_prompt", "prompt")][2] == self.PROMPT - 3
        assert batch[("pr_prompt", "all")][2] == len(self.VALUES) - 3
        assert batch[("pr_all", "response")][2] == len(self.VALUES) - self.PROMPT
        assert batch[("pr_plain", "all")][2] == len(self.VALUES)
        assert batch[("pr_plain", "prompt")][2] == self.PROMPT


class TestOldProbesAreServedAsBefore:
    def test_a_probe_without_the_flag_ignores_a_stated_prefix(self):
        for stated in (None, 0, 4):
            ctx = ProbeRequestContext("req", [_probe(skip=False, windows=("all", "prompt"))])
            ctx.set_prompt_length(6)
            if stated is not None:
                ctx.set_template_prefix(stated)
            ctx.observe(1, _acts(PREFIX + [3, 7]))
            got = _verdicts(ctx)
            assert got[("pr_g38", "all")].score == 50 and got[("pr_g38", "all")].n_scored_tokens == 6
            assert got[("pr_g38", "prompt")].n_scored_tokens == 6

    def test_arming_reads_the_flag_from_the_stored_document(self):
        from tests.unit.probe_fixtures import probe_definition

        from miserver.services.probe_arming import armed_probe_from_row

        def row(definition):
            return SimpleNamespace(
                id="pr_1", name="p", definition=definition, rule="max", scope="all", layer=1,
                rung=2, threshold=None, threshold_revision=1,
            )

        old = probe_definition()
        assert "skip_template_prefix" not in old["aggregation"]
        assert armed_probe_from_row(row(old)).skip_template_prefix is False
        new = probe_definition()
        new["aggregation"]["skip_template_prefix"] = True
        assert armed_probe_from_row(row(new)).skip_template_prefix is True


class TestTheLastUserWindowSkipsThePrefix:
    """Operator decision (2026-11-08): `last_user` also leaves out the template prefix, as
    `[max(start, P), end)`, for a probe that skips it — and only for one that does."""

    def _last_user(self, skip, span, prefix=4, values=PREFIX + [3, 7, 9, 1, 2]):
        probe = _probe(skip=skip, windows=("last_user",))
        ctx = ProbeRequestContext("req", [probe])
        ctx.set_prompt_length(len(values))
        ctx.set_template_prefix(prefix)
        ctx.set_last_user_span(span)
        ctx.observe(1, _acts(values))
        return _verdicts(ctx)[("pr_g38", "last_user")]

    def test_a_single_turn_span_starting_inside_the_prefix(self):
        """The newest (only) user header starts at 2, inside P = 4: the skipping probe reads [4, 7)."""
        skipping, plain = self._last_user(True, (2, 7)), self._last_user(False, (2, 7))
        assert skipping.n_scored_tokens == 3 and skipping.score == 9     # positions 4..6: 3, 7, 9
        assert plain.n_scored_tokens == 5 and plain.score == 50

    def test_a_multi_turn_span_after_the_prefix_is_unchanged(self):
        skipping, plain = self._last_user(True, (6, 9)), self._last_user(False, (6, 9))
        assert (skipping.score, skipping.n_scored_tokens) == (plain.score, plain.n_scored_tokens) == (9, 3)


class TestAnUnstatedPrefixIsSaid:
    def test_only_the_windows_that_hold_the_prefix_refuse(self):
        assert TEMPLATE_PREFIX_WINDOWS == ("all", "prompt", "last_user")
        ctx = ProbeRequestContext("req", [_probe(windows=("all", "prompt", "response", "last_user"))])
        ctx.set_prompt_length(5)
        ctx.set_last_user_span((3, 5))
        ctx.observe(1, _acts(PREFIX + [3, 7, 9]))
        got = _verdicts(ctx)
        for window in ("all", "prompt", "last_user"):
            assert got[("pr_g38", window)].scored is False
            assert got[("pr_g38", window)].not_scored_reason == "template_prefix_unknown"
        assert got[("pr_g38", "response")].scored is True
        assert got[("pr_g38", "response")].n_scored_tokens == 2      # positions 5 and 6

    def test_the_reason_is_published(self):
        from miserver.services.probe_reasons import OUTCOME_OF

        assert OUTCOME_OF["template_prefix_unknown"] == "skipped"


# ── parity ───────────────────────────────────────────────────────────────────────────────────


def _parity_forward(values):
    def forward(input_ids, context):
        context.observe(1, _acts(values[: input_ids.shape[1]]))
    return forward


def _flagged_definition(token_ids, scores, prefix, **vector):
    return {"test_vectors": {"tolerance": 0.001, "vectors": [{
        "messages": [{"role": "user", "content": "x"}], "token_ids": token_ids,
        "token_scores": scores, "score": max(scores), **(
            {"template_prefix_tokens": prefix} if prefix is not None else {}
        ), **vector,
    }]}}


class TestParityOfAFlaggedProbe:
    VALUES = [50, 50, 50, 3, 9, 1]

    def test_it_passes_on_the_vectors_own_prefix(self):
        report = ProbeParityEngine(_parity_forward(self.VALUES)).run(
            _probe(), _flagged_definition(list(range(6)), [3.0, 9.0, 1.0], 3), tolerance=0.001,
        )
        assert report.passed is True, report.as_details()
        assert report.vectors[0].n_tokens == 3

    def test_the_stated_prefix_wins_over_a_re_measured_one(self):
        """A date-stamped preamble re-read on another day measures a different P from the same ids;
        the vector's own stated P is what the recorded scores belong to."""
        tokenizer = _structural_tokenizer(STRUCTURAL["cases"][0])
        report = ProbeParityEngine(_parity_forward(self.VALUES)).run(
            _probe(), _flagged_definition(list(range(6)), [3.0, 9.0, 1.0], 3), tolerance=0.001,
            tokenizer=tokenizer,
        )
        assert report.passed is True, report.as_details()

    def test_without_a_stated_prefix_or_a_tokenizer_it_is_not_compared(self):
        report = ProbeParityEngine(_parity_forward(self.VALUES)).run(
            _probe(), _flagged_definition(list(range(6)), [3.0, 9.0, 1.0], None), tolerance=0.001,
        )
        assert report.passed is False
        assert report.vectors[0].comparable is False
        assert report.vectors[0].reason == "template_prefix_unknown"

    def test_a_flagged_probe_on_a_tiny_real_model(self):
        """A REAL model and the case file's tokenizer: the reference is computed by hand from the
        hidden states (head over positions >= P, max), and parity reproduces it through the
        runtime. Dropping the skip anywhere on the served path moves the scored count."""
        from transformers import LlamaConfig, LlamaForCausalLM

        case = STRUCTURAL["cases"][0]
        tokenizer = _structural_tokenizer(case)
        torch.manual_seed(0)
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=len(STRUCTURAL["vocab"]), hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=64,
        )).eval()
        weight = torch.randn(16)
        probe = ArmedProbe(
            probe_id="pr_real", name="real", head=ProbeHead(weight=weight, bias=0.25), rule="max",
            scope="all", layer=1, rung=2, rung_language="held-out", threshold=None,
            skip_template_prefix=True,
        )
        ids = served_render(tokenizer, case["messages"], template_renderer(tokenizer)).ids
        prefix = template_prefix_length(tokenizer, ids)
        with torch.no_grad():
            hidden = model(torch.tensor([ids]), output_hidden_states=True).hidden_states[2][0]
        expected = (hidden @ weight + 0.25)[prefix:].tolist()

        def forward(input_ids, context):
            with torch.no_grad():
                out = model(input_ids, output_hidden_states=True).hidden_states[2]
            context.observe(1, out)

        definition = _flagged_definition(ids, expected, prefix)
        definition["test_vectors"]["vectors"][0]["score"] = max(expected)
        report = ProbeParityEngine(forward).run(probe, definition, tolerance=1e-4, tokenizer=tokenizer)
        assert prefix == case["expected"]["template_prefix"] > 0
        assert report.passed is True, report.as_details()


# ── the serving wiring, driven ───────────────────────────────────────────────────────────────


class TestEveryScoringPathStatesThePrefix:
    """Each serving path that scores is DRIVEN with a skipping probe and a plain twin armed on a
    real tiny model. The skipping probe must score exactly P fewer positions than its twin — never
    `template_prefix_unknown`, which is what a path that forgot to state P would report."""

    @pytest.fixture
    def served(self, monkeypatch):
        from contextlib import asynccontextmanager
        from unittest.mock import patch

        from tests.unit.f25_fixtures import WORDS, clear_loaded, make_service, word_model, word_tokenizer

        from miserver.services.probe_runtime import ProbeRuntimeState

        ProbeRuntimeState.reset_for_tests()

        @asynccontextmanager
        async def no_db():
            raise RuntimeError("no database here")
            yield  # pragma: no cover

        monkeypatch.setattr("miserver.db.base.async_session_factory", no_db)
        model = word_model()
        tokenizer = word_tokenizer()
        # The content-free probes `x`/`y` must be distinct tokens for the prefix to exist: rename two
        # unused words (same ids, same model) so this tokenizer can tell them apart.
        vocab = {("x" if w == "w28" else "y" if w == "w29" else w): i for i, w in enumerate(WORDS)}
        tokenizer.backend_tokenizer.model = type(tokenizer.backend_tokenizer.model)(vocab, unk_token="[UNK]")
        svc = make_service(model, tokenizer)
        recorded = []

        async def record(context, verdicts=None, full_ids=None, **kwargs):
            if context is not None:
                recorded.append(verdicts if verdicts is not None else context.finish())

        svc._probe_record = record
        state = ProbeRuntimeState()
        for probe_id, skip in (("pr_skip", True), ("pr_plain", False)):
            state.arm(ArmedProbe(
                probe_id=probe_id, name=probe_id, head=ProbeHead(weight=torch.ones(16), bias=0.0),
                rule="mean", scope="all", layer=0, rung=2, rung_language="held-out", threshold=None,
                skip_template_prefix=skip,
            ), model)
        with patch("miserver.services.inference_service.torch.cuda.is_available", return_value=False):
            yield svc, tokenizer, recorded
        ProbeRuntimeState.reset_for_tests()
        clear_loaded()

    async def _drive(self, svc, call):
        from tests.support.generation_entry_points import _chat, _text

        if call == "chat":
            await svc.create_chat_completion(_chat())
        elif call == "stream":
            async for _ in svc.stream_chat_completion(_chat(stream=True)):
                pass
        else:
            await svc.create_text_completion(_text())

    @pytest.mark.parametrize("call,prefix", [("chat", 2), ("stream", 2), ("text", 1)])
    async def test_the_path_states_P(self, served, call, prefix):
        svc, tokenizer, recorded = served
        assert len(template_prefix_ids(tokenizer, refresh=True)) == 2   # "<s> user"
        await self._drive(svc, call)
        assert len(recorded) == 1, recorded
        by_id = {v.probe_id: v for v in recorded[0]}
        skip, plain = by_id["pr_skip"], by_id["pr_plain"]
        assert skip.scored, skip.not_scored_reason
        assert plain.scored, plain.not_scored_reason
        assert plain.n_scored_tokens - skip.n_scored_tokens == prefix

    def test_the_helper_measures_only_when_a_probe_skips(self, monkeypatch):
        from unittest.mock import MagicMock

        from miserver.services import inference_service, probe_turns

        calls = []
        monkeypatch.setattr(probe_turns, "template_prefix_length",
                            lambda tok, ids: calls.append(list(ids)) or 2)
        svc = inference_service.InferenceService.__new__(inference_service.InferenceService)
        svc._model_state = SimpleNamespace(is_loaded=True, current=SimpleNamespace(tokenizer=object()))
        ctx = MagicMock()
        ctx.probes = [SimpleNamespace(skip_template_prefix=False)]
        svc._probe_note_template_prefix(ctx, torch.tensor([[1, 2, 3]]))
        assert calls == [] and not ctx.set_template_prefix.called
        ctx.probes = [SimpleNamespace(skip_template_prefix=True)]
        svc._probe_note_template_prefix(ctx, torch.tensor([[1, 2, 3]]))
        assert calls == [[1, 2, 3]]
        ctx.set_template_prefix.assert_called_once_with(2)


def test_the_scoring_api_states_P_on_its_context():
    """`/api/probes/score`: the prepared input's P reaches the context `_score_one` builds."""
    from tests.unit.f25_fixtures import clear_loaded, make_service, word_model, word_tokenizer

    from miserver.ml.model_loader import LoadedModelState
    from miserver.services.probe_scoring import PreparedInput, ProbeScoringService, model_pin

    make_service(word_model(), word_tokenizer())
    try:
        svc = ProbeScoringService.__new__(ProbeScoringService)
        values = PREFIX + [3, 7]
        prepared = PreparedInput(index=0, input_kind="messages", ids=list(range(6)), prompt_tokens=6,
                                 template_prefix=4)
        pin = model_pin(LoadedModelState().current)
        verdicts = svc._score_one(prepared, [_probe()], _parity_forward(values), pin)
        assert verdicts[0].scored and verdicts[0].score == 7 and verdicts[0].n_scored_tokens == 2
    finally:
        clear_loaded()


# ── the hand-written mirror of the contract ──────────────────────────────────────────────────


class TestTheMirrorCarriesTheFields:
    """memory `circuit-contract-mirror-drift`: a field the vendored schema has and the mirror lacks
    is kept by `extra="allow"` and READ by nobody. These fields are acted on, so they are typed."""

    @staticmethod
    def _flagged():
        from tests.unit.probe_fixtures import probe_definition

        doc = probe_definition()
        doc["aggregation"]["skip_template_prefix"] = True
        vector = doc["test_vectors"]["vectors"][0]
        vector["template_prefix_tokens"] = 1
        vector["token_scores"] = vector["token_scores"][1:]
        return doc

    def test_a_flagged_document_round_trips_typed(self):
        from miserver.api.schemas.probe import Aggregation, ProbeDefinitionV1, TestVector

        assert "skip_template_prefix" in Aggregation.model_fields
        assert "template_prefix_tokens" in TestVector.model_fields
        parsed = ProbeDefinitionV1.model_validate(self._flagged())
        assert parsed.aggregation.skip_template_prefix is True
        assert parsed.test_vectors.vectors[0].template_prefix_tokens == 1
        dumped = parsed.model_dump()
        assert dumped["aggregation"]["skip_template_prefix"] is True
        assert dumped["test_vectors"]["vectors"][0]["template_prefix_tokens"] == 1

    def test_an_old_document_reads_as_not_skipping(self):
        from tests.unit.probe_fixtures import probe_definition

        from miserver.api.schemas.probe import ProbeDefinitionV1

        assert ProbeDefinitionV1.model_validate(probe_definition()).aggregation.skip_template_prefix is False

    @pytest.mark.parametrize("breakage,match", [
        ("drop_prefix", "carries no template_prefix_tokens"),
        ("unflag", "two different readouts"),
        ("wrong_count", "skips 1 prefix positions"),
    ])
    def test_an_incoherent_document_is_refused(self, breakage, match):
        from pydantic import ValidationError

        from miserver.api.schemas.probe import ProbeDefinitionV1

        doc = self._flagged()
        vector = doc["test_vectors"]["vectors"][0]
        if breakage == "drop_prefix":
            del vector["template_prefix_tokens"]
        elif breakage == "unflag":
            doc["aggregation"]["skip_template_prefix"] = False
        else:
            vector["token_scores"] = [0.5, 1.5, 2.0, 1.0]
        with pytest.raises(ValidationError, match=match):
            ProbeDefinitionV1.model_validate(doc)

    def test_the_flag_moves_the_detector_digest_only(self):
        from tests.unit.probe_fixtures import probe_definition

        from miserver.services.probe_digest import digests_of

        plain = probe_definition()
        explicit_false = probe_definition()
        explicit_false["aggregation"]["skip_template_prefix"] = False
        assert digests_of(explicit_false) == digests_of(plain)
        flagged = digests_of(self._flagged())
        assert flagged[0] != digests_of(plain)[0] and flagged[1] == digests_of(plain)[1]


def test_unknown_probe_contents_give_no_prefix():
    """Mutation control ML16 SURVIVED the first set: nothing here pinned the identical-renders
    guard. A vocabulary that cannot tell `x` from `y` renders both content-free probes alike, and
    their "common prefix" then runs through the content: a request whose first word is also unknown
    would have its own content skipped. No prefix is known, so nothing is skipped."""
    from tests.unit.f25_fixtures import word_tokenizer

    tokenizer = word_tokenizer()            # its vocabulary has no `x` or `y`: both are [UNK]
    assert template_prefix_ids(tokenizer, refresh=True) == []
    unknown_first = tokenizer.convert_tokens_to_ids(["<s>", "user", "[UNK]", "w1"])
    assert template_prefix_length(tokenizer, unknown_first) == 0


class TestReviewRound2:
    def test_the_mirror_flag_is_strict(self):
        """Arming reads the raw stored value (`is True`); a coerced `1`/`"true"` would validate as
        skipping and be armed as not skipping. The mirror refuses both."""
        from pydantic import ValidationError

        from miserver.api.schemas.probe import Aggregation

        for value in (1, "true", "1", 0):
            with pytest.raises(ValidationError):
                Aggregation(rule="max", streamable=True, skip_template_prefix=value)
        assert Aggregation(rule="max", streamable=True, skip_template_prefix=True).skip_template_prefix is True

    def test_push_order_guard_deployed_millm_refuses_a_flagged_document_with_identity(self):
        """THE PUSH-ORDER GUARD. A miLLM built before G38 projects the detector WITHOUT the added-after
        table, so the detector digest a flagged document carries (the case file's
        `added-after-aggregation.skip_template_prefix`) cannot be reproduced: import answers 409
        `identity_mismatch` rather than serving the probe without the skip. A flagged document
        stripped of `identity` is not checked at all — that gap is why "miLLM first" stays the rule.
        The same projection still reproduces every unflagged case, so old documents are unaffected."""
        import hashlib as _hashlib

        from miserver.services import probe_digest

        def pre_g38(document):
            payload = probe_digest._project(document, "", probe_digest.DETECTOR_V1, "detector")
            return _hashlib.sha256(probe_digest.canonical_bytes("detector", payload)).hexdigest()

        cases = {c["name"]: c for c in json.loads(
            (REPO / "docs" / "schemas" / "probe-identity-cases.json").read_text())["cases"]}
        flagged = cases["added-after-aggregation.skip_template_prefix"]
        assert flagged["document"]["aggregation"]["skip_template_prefix"] is True
        assert pre_g38(flagged["document"]) != flagged["detector_digest"]
        assert probe_digest.detector_digest(flagged["document"]) == flagged["detector_digest"]
        for name in ("residual", "added-after-default-aggregation.skip_template_prefix"):
            assert pre_g38(cases[name]["document"]) == cases[name]["detector_digest"], name

    def test_a_decode_pass_after_the_prefix_reuses_the_cached_all_window(self, monkeypatch):
        """Review round 2: a skipping `all` probe built an O(position) list on EVERY decode pass. Once
        the pass starts at or after P the cached all-true tensor is used, and `scored_mask` is not
        called; the prefill (which overlaps the prefix) still builds its window."""
        from miserver.services import probe_runtime

        calls = []
        real = probe_runtime.scored_mask
        monkeypatch.setattr(probe_runtime, "scored_mask", lambda **kw: calls.append(kw) or real(**kw))
        probe = _probe()
        ctx = ProbeRequestContext("req", [probe])
        ctx.set_prompt_length(6)
        ctx.set_template_prefix(4)
        ctx.observe(1, _acts(PREFIX + [3, 7]))     # prefill: overlaps the prefix
        assert len(calls) == 1
        for value in (9, 1, 2):
            ctx.observe(1, _acts([value]))         # decode passes, all after P
        assert len(calls) == 1, "a decode pass after the prefix rebuilt its window"
        cached = ctx._all_windows[(1, "cpu")]
        assert all(part is cached for part in ctx._mask[probe.probe_id][1:])
        verdict = ctx.finish()[0]
        assert verdict.n_scored_tokens == 5 and verdict.score == 9
