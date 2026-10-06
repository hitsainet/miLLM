"""`ProbeScoringService` (Feature 27, FR-27.4 – FR-27.7): resolution, the loop, the response.

Every test runs a TINY REAL Llama, real probe rows in SQLite, and the real construction, forward
and decision code. The one thing patched is `loaded_identity`, which needs a model row and a
download directory to establish a revision.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
import torch
from pydantic import ValidationError

import millm.services.probe_arming as probe_arming
import millm.services.probe_runtime as probe_runtime
from millm.api.schemas.probe_scoring import ProbeScoreRequest
from millm.core.errors import (
    ContextLengthExceededError,
    ProbeHookUnsupportedError,
    ProbeModelMismatchError,
    ProbeNoModelLoadedError,
    ProbeNotFoundError,
    ProbeSaeMissingError,
    ProbeScoreRequestError,
)
from millm.db.repositories.probe_repository import ProbeRepository
from millm.ml.model_loader import ENGINE_LLAMACPP, LoadedModel, LoadedModelState
from millm.services.probe_runtime import ProbeRequestContext, ProbeRuntimeState, Verdict
from millm.services.probe_scoring import (
    ProbeScoringService,
    parity_status,
    verdict_payload,
)
from millm.services.probe_service import ProbeService
from tests.unit.f25_fixtures import clear_loaded, make_service, word_model, word_tokenizer
from tests.unit.probe_fixtures import sae_probe_definition
from tests.unit.score_fixtures import tiny_definition, tiny_identity


@pytest.fixture(autouse=True)
def clean():
    ProbeRuntimeState.reset_for_tests()
    yield
    ProbeRuntimeState.reset_for_tests()
    clear_loaded()


@pytest.fixture
def setup(test_session, monkeypatch):
    model, tokenizer = word_model(), word_tokenizer()
    inference = make_service(model, tokenizer)
    monkeypatch.setattr(
        "millm.services.probe_arm_bridge.loaded_identity",
        AsyncMock(return_value=(tiny_identity(), model, tokenizer)),
    )
    repo = ProbeRepository(test_session)
    return model, tokenizer, inference, repo, test_session


async def _import(repo, definition):
    return await ProbeService(repo).import_definition(definition)


def req(**body) -> ProbeScoreRequest:
    body.setdefault("inputs", [{"token_ids": [2, 6, 7, 8], "prompt_tokens": 2}])
    return ProbeScoreRequest(**body)


async def score(setup, **body):
    _model, _tok, inference, repo, session = setup
    return await ProbeScoringService(repo, inference).score(req(**body), session)


# ── 4.1 shape refusals ───────────────────────────────────────────────────────────


class TestShapeRefusals:
    @pytest.mark.parametrize(("body", "fragment"), [
        ({"inputs": []}, "at least one input"),
        ({"inputs": [{"token_ids": [1], "text": "w1"}]}, "exactly one of"),
        ({"inputs": [{}]}, "exactly one of"),
        ({"inputs": [{"messages": [{"role": "user", "content": "w1"}], "prompt_tokens": 1}]},
         "only accepted with token_ids"),
        ({"inputs": [{"token_ids": [1, 2], "prompt_tokens": 3}]}, "exceeds the input's length"),
        ({"inputs": [{"token_ids": [1, -2]}]}, "non-negative"),
        ({"inputs": [{"text": "w1 w2"}]}, "T-49"),
    ])
    async def test_refused_with_INVALID_PROBE_SCORE_REQUEST(self, setup, body, fragment):
        await _import(setup[3], tiny_definition())
        with pytest.raises(ProbeScoreRequestError) as exc:
            await score(setup, **body)
        assert exc.value.code == "INVALID_PROBE_SCORE_REQUEST"
        assert exc.value.status_code == 400
        assert fragment in str(exc.value)

    async def test_too_many_inputs(self, setup, monkeypatch):
        from millm.core.config import settings

        monkeypatch.setattr(settings, "PROBE_SCORE_MAX_INPUTS", 2)
        with pytest.raises(ProbeScoreRequestError, match="limit of 2"):
            await score(setup, inputs=[{"token_ids": [2]}] * 3)

    async def test_too_many_probes(self, setup, monkeypatch):
        from millm.core.config import settings

        monkeypatch.setattr(settings, "PROBE_SCORE_MAX_PROBES", 1)
        with pytest.raises(ProbeScoreRequestError, match="limit of 1"):
            await score(setup, probe_ids=["a", "b"])

    def test_a_misspelt_field_is_a_422_not_a_drop(self):
        with pytest.raises(ValidationError):
            ProbeScoreRequest(inputs=[{"token_idz": [1]}])
        with pytest.raises(ValidationError):
            ProbeScoreRequest(inputs=[{"token_ids": [1]}], window=["all"])

    def test_an_unknown_window_is_refused(self):
        with pytest.raises(ValidationError, match="unknown window"):
            ProbeScoreRequest(inputs=[{"token_ids": [1]}], windows=["everything"])

    async def test_a_token_id_outside_the_vocabulary(self, setup):
        await _import(setup[3], tiny_definition())
        with pytest.raises(ProbeScoreRequestError, match="outside the loaded model's vocabulary"):
            await score(setup, inputs=[{"token_ids": [2, 999]}])

    async def test_text_refused_until_verified_names_the_check(self, setup):
        """0.3: refused, naming the pending verification, never scored under an unverified
        render. Remove with the gate when 027 FTASKS 0.2 passes."""
        await _import(setup[3], tiny_definition())
        with pytest.raises(ProbeScoreRequestError) as exc:
            await score(setup, inputs=[{"text": "w1"}])
        assert exc.value.details["pending_verification"] == "T-49 (027 FTASKS 0.2)"


# ── 4.3 probe resolution ─────────────────────────────────────────────────────────


class TestResolution:
    async def test_an_unknown_given_id_is_404(self, setup):
        with pytest.raises(ProbeNotFoundError):
            await score(setup, probe_ids=["pr_nope"])

    async def test_a_given_mismatched_probe_refuses_naming_the_field(self, setup):
        row = await _import(setup[3], tiny_definition(hf_id="other/model"))
        with pytest.raises(ProbeModelMismatchError) as exc:
            await score(setup, probe_ids=[row.id])
        assert any(m["field"] == "hf_id" for m in exc.value.details["mismatches"])

    async def test_omitted_ids_skip_the_mismatch_with_its_code(self, setup):
        good = await _import(setup[3], tiny_definition("good"))
        bad = await _import(setup[3], tiny_definition("bad", hf_id="other/model"))
        out = await score(setup)
        assert [p["probe_id"] for p in out["probes"]] == [good.id]
        assert [(s["probe_id"], s["code"]) for s in out["skipped"]] == [
            (bad.id, "PROBE_MODEL_MISMATCH")
        ]
        assert "different model" in out["skipped"][0]["reason"]

    async def test_every_probe_skipped_is_a_400(self, setup):
        await _import(setup[3], tiny_definition("bad", hf_id="other/model"))
        with pytest.raises(ProbeScoreRequestError, match="every probe was skipped"):
            await score(setup)

    async def test_an_sae_probe_whose_sae_is_missing(self, setup, tmp_path, monkeypatch):
        from millm.core.config import settings

        monkeypatch.setattr(settings, "SAE_CACHE_DIR", str(tmp_path))
        definition = sae_probe_definition()
        base = tiny_definition()
        definition["model"], definition["read"] = base["model"], base["read"]
        definition["sae"]["d_model"] = 16
        row = await _import(setup[3], definition)
        with pytest.raises(ProbeSaeMissingError):
            await score(setup, probe_ids=[row.id])
        await _import(setup[3], tiny_definition("dense"))
        out = await score(setup)
        assert [s["code"] for s in out["skipped"]] == ["PROBE_SAE_MISSING"]


# ── 4.4 the loop calls the live code ─────────────────────────────────────────────


class TestTheLoopRunsTheLiveCode:
    async def test_by_call_and_one_slot_per_input(self, setup, monkeypatch):
        _model, _tok, inference, repo, _ = setup
        await _import(repo, tiny_definition())
        built = []
        real_build = probe_arming.armed_probe_from_row

        def spy_build(*a, **k):
            built.append(1)
            return real_build(*a, **k)

        monkeypatch.setattr(probe_arming, "armed_probe_from_row", spy_build)
        decided = []
        real_verdict = ProbeRequestContext._verdict_for

        def spy_verdict(self, probe, window):
            decided.append(window)
            return real_verdict(self, probe, window)

        monkeypatch.setattr(ProbeRequestContext, "_verdict_for", spy_verdict)
        banded = []
        real_band = probe_runtime.threshold_for_length

        def spy_band(*a, **k):
            banded.append(1)
            return real_band(*a, **k)

        monkeypatch.setattr(probe_runtime, "threshold_for_length", spy_band)
        admitted = {"n": 0}
        real_admit = inference._admit

        @asynccontextmanager
        async def counting_admit(*a, **k):
            admitted["n"] += 1
            async with real_admit(*a, **k) as refusal:
                yield refusal

        inference._admit = counting_admit
        await score(setup, windows=["all"],
                    inputs=[{"token_ids": [2, 6, 7]}, {"token_ids": [2, 8]}, {"token_ids": [9]}])
        assert built == [1], "the runtime probe must be built by armed_probe_from_row"
        assert decided == ["all", "all", "all"], "_verdict_for must decide every input"
        assert banded, "threshold_for_length must be consulted"
        assert admitted["n"] == 3, "one admission slot per input (FR-27.6a)"

    async def test_offline_equals_the_live_runtime(self, setup):
        """The armed path and the stateless path on the same ids give the same score."""
        model, _tok, _inference, repo, _ = setup
        row = await _import(repo, tiny_definition())
        out = await score(setup, windows=["all"], inputs=[{"token_ids": [2, 6, 7, 8]}])
        offline = out["results"][0]["verdicts"][0]["score"]

        armed = probe_arming.armed_probe_from_row(row, windows=["all"])
        state = ProbeRuntimeState()
        state.arm(armed, model)
        state.begin_request("live")
        with torch.inference_mode():
            model(input_ids=torch.tensor([[2, 6, 7, 8]]), use_cache=False)
        live = state.end_request().finish()[0].score
        assert offline == live

    async def test_one_forward_scores_probes_on_two_layers(self, setup):
        _model, _tok, _inference, repo, _ = setup
        await _import(repo, tiny_definition("l0", layer=0))
        await _import(repo, tiny_definition("l1", layer=1))
        out = await score(setup, windows=["all"])
        assert {v["name"] for v in out["results"][0]["verdicts"]} == {"l0", "l1"}
        assert all(v["n_scored_tokens"] == 4 for v in out["results"][0]["verdicts"])


# ── 4.5 response mapping ─────────────────────────────────────────────────────────


class TestResponse:
    async def test_every_field(self, setup):
        row = await _import(setup[3], tiny_definition())
        out = await score(setup, windows=["all", "response"],
                          inputs=[{"token_ids": [2, 6, 7, 8], "prompt_tokens": 2}])
        result = out["results"][0]
        assert result["index"] == 0 and result["input_kind"] == "token_ids"
        assert result["n_tokens"] == 4 and result["prompt_tokens"] == 2
        assert result["token_ids"] is None and result["error"] is None
        by_window = {v["window"]: v for v in result["verdicts"]}
        all_ = by_window["all"]
        assert set(all_) == {"probe_id", "name", "window", "score", "threshold", "verdict", "rung",
                             "rung_language", "provisional", "threshold_revision",
                             "n_scored_tokens", "not_scored_reason"}
        assert all_["probe_id"] == row.id and all_["threshold"] == 1.0
        assert all_["verdict"] is (all_["score"] >= 1.0)
        assert all_["rung_language"] == "detects on unseen tasks"
        assert all_["threshold_revision"] == 1
        # P-20: the response window's weights never saw a reply — provisional, and it says so.
        assert by_window["response"]["provisional"] is True
        assert by_window["response"]["n_scored_tokens"] == 2

    async def test_token_ids_echoed_only_when_asked(self, setup):
        await _import(setup[3], tiny_definition())
        out = await score(setup, return_token_ids=True,
                          inputs=[{"messages": [{"role": "user", "content": "w1"}]}])
        tok = word_tokenizer()
        rendered = tok.apply_chat_template([{"role": "user", "content": "w1"}], tokenize=False,
                                           add_generation_prompt=True)
        # Exactly as live serving tokenizes its rendered prompt (`tokenizer(prompt)`).
        assert out["results"][0]["token_ids"] == list(tok(rendered)["input_ids"])

    def test_a_null_verdict_stays_null(self):
        v = Verdict(probe_id="p", name="n", rung=2, rung_language="x", scored=True, score=3.0,
                    threshold=None, fires=None)
        assert verdict_payload(v)["verdict"] is None

    async def test_parity_status_is_reported(self, setup):
        await _import(setup[3], tiny_definition())
        out = await score(setup)
        assert out["probes"][0]["parity"] == {"status": "never_run", "checked_against": None,
                                              "checked_at": None}
        assert out["model"]["hf_id"] == "tiny/llama"


class TestParityStatus:
    class Row:
        def __init__(self, parity):
            self.parity = parity

    def test_each_state(self):
        assert parity_status(self.Row(None))["status"] == "never_run"
        passed = parity_status(self.Row({"passed": True, "model": {"hf_id": "x"},
                                         "checked_at": "t"}))
        assert passed == {"status": "passed", "checked_against": {"hf_id": "x"},
                          "checked_at": "t"}
        assert parity_status(self.Row({"passed": False}))["status"] == "failed"

    def test_an_old_report_reads_unknown(self):
        assert parity_status(self.Row({"passed": True}))["checked_against"] == "unknown"


# ── 4.2 boundaries through the service ──────────────────────────────────────────


class TestBoundaries:
    async def test_bare_token_ids_cannot_score_response(self, setup):
        await _import(setup[3], tiny_definition())
        out = await score(setup, windows=["response"], inputs=[{"token_ids": [2, 6, 7]}])
        v = out["results"][0]["verdicts"][0]
        assert v["not_scored_reason"] == "prompt_boundary_unknown" and v["verdict"] is None

    async def test_last_user_on_token_ids_says_why(self, setup):
        await _import(setup[3], tiny_definition())
        out = await score(setup, windows=["last_user"], inputs=[{"token_ids": [2, 6, 7]}])
        assert out["results"][0]["verdicts"][0]["not_scored_reason"] == "token_ids_have_no_turns"

    async def test_last_user_on_messages_scores(self, setup):
        await _import(setup[3], tiny_definition())
        out = await score(setup, windows=["last_user"], inputs=[{"messages": [
            {"role": "user", "content": "w1 w2"}, {"role": "assistant", "content": "w3"},
            {"role": "user", "content": "w4 w5 w6"},
        ]}])
        v = out["results"][0]["verdicts"][0]
        assert v["not_scored_reason"] is None and v["n_scored_tokens"] >= 3


# ── 4.6 pinning ──────────────────────────────────────────────────────────────────


class TestPinning:
    async def test_a_reload_between_inputs_fails_the_rest_and_keeps_the_first(self, setup):
        from datetime import datetime

        _model, _tok, inference, repo, _ = setup
        await _import(repo, tiny_definition())
        real = inference.run_model_work
        calls = {"n": 0}

        async def reloading(fn):
            result = await real(fn)
            calls["n"] += 1
            if calls["n"] == 1:  # a reload lands between input 0 and input 1
                current = LoadedModelState().current
                LoadedModelState().set(LoadedModel(
                    model_id=current.model_id, model_name=current.model_name, model=current.model,
                    tokenizer=current.tokenizer, loaded_at=datetime(2026, 10, 7),
                ))
            return result

        inference.run_model_work = reloading
        out = await score(setup, inputs=[{"token_ids": [2, 6]}, {"token_ids": [2, 7]},
                                         {"token_ids": [2, 8]}])
        first, *rest = out["results"]
        assert first["verdicts"] and first["error"] is None
        assert [r["error"]["code"] for r in rest] == ["MODEL_CHANGED", "MODEL_CHANGED"]
        assert all(r["verdicts"] == [] for r in rest)


# ── 4.7 whole-request refusals ───────────────────────────────────────────────────


class TestWholeRequestRefusals:
    async def test_no_model_loaded(self, test_session):
        from millm.services.inference_service import InferenceService

        clear_loaded()
        with pytest.raises(ProbeNoModelLoadedError):
            await ProbeScoringService(ProbeRepository(test_session), InferenceService()).score(
                req(), test_session
            )

    async def test_a_gguf_model(self, test_session):
        from datetime import datetime

        from millm.services.inference_service import InferenceService

        LoadedModelState().set(LoadedModel(
            model_id=1, model_name="g", model=object(), tokenizer=None,
            loaded_at=datetime(2026, 10, 6), engine=ENGINE_LLAMACPP,
        ))
        with pytest.raises(ProbeHookUnsupportedError):
            await ProbeScoringService(ProbeRepository(test_session), InferenceService()).score(
                req(), test_session
            )

    async def test_an_input_over_the_context_window(self, setup):
        await _import(setup[3], tiny_definition())
        with pytest.raises(ContextLengthExceededError):
            await score(setup, inputs=[{"token_ids": [2] * 300}])


# ── 4.10 privacy ─────────────────────────────────────────────────────────────────


class TestPrivacy:
    async def test_no_input_text_or_token_list_in_the_logs(self, setup, caplog, monkeypatch):
        """Stdlib records via caplog; the scoring module's structlog logger via a recorder.

        ⚠ NOT `structlog.testing.capture_logs`: structlog caches each logger on first use, so a
        logger first used inside `capture_logs` keeps writing to THAT capture for the rest of the
        session and blinds every later test that captures it (five unrelated tests went red when
        this one ran first)."""
        import millm.services.probe_scoring as module

        recorded: list[tuple[str, str, dict]] = []

        class Recorder:
            def __getattr__(self, level):
                return lambda event, **kw: recorded.append((level, event, kw))

        monkeypatch.setattr(module, "logger", Recorder())
        await _import(setup[3], tiny_definition())
        caplog.set_level("DEBUG")
        await score(setup, inputs=[
            {"messages": [{"role": "user", "content": "w17 w18 w19"}]},
            {"token_ids": [2, 21, 22, 23]},
        ])
        above_debug = " ".join(r.getMessage() for r in caplog.records if r.levelno > 10)
        above_debug += " ".join(f"{e} {kw}" for lvl, e, kw in recorded if lvl != "debug")
        assert "w17 w18 w19" not in above_debug
        assert "21, 22, 23" not in above_debug
        lines = [(lvl, kw) for lvl, e, kw in recorded if e == "probe_score"]
        assert lines == [("info", lines[0][1])], "exactly one info line per request"
        assert set(lines[0][1]) == {"probes", "skipped", "inputs", "elapsed_ms"}


# ── 4.13 no routing change ───────────────────────────────────────────────────────


class TestNoRoutingChange:
    async def test_cbm_routing_is_unchanged_after_scoring(self, setup):
        class _CBM:
            is_running = True

            def sampling_params_match(self, *a):
                return True

        _model, _tok, inference, repo, _ = setup
        await _import(repo, tiny_definition())
        inference._cbm_backend = _CBM()
        before = inference._use_cbm_for_request()
        await score(setup)
        assert inference._use_cbm_for_request() is before is True
        assert ProbeRuntimeState().has_armed() is False

