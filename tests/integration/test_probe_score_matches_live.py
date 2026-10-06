"""Offline equals live, end to end on a tiny real transformer (Feature 27, task 8.4; US-2).

import → score UNARMED through `/api/probes/score`'s service → arm → a live chat request with
`max_tokens: 1` → the prompt-window score is the same number. With one generated token nothing
generated is fed back, so the live forward reads exactly the prompt the offline call read.

Hardware acceptance (BRD-04 acceptance 11) repeats this on LFM2.5-1.2B against a real definition's
test vectors; this proves the plumbing agrees by running the real code on both sides.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from millm.api.schemas.openai import ChatCompletionRequest, ChatMessage
from millm.api.schemas.probe_scoring import ProbeScoreRequest
from millm.db.repositories.probe_repository import ProbeEventRepository, ProbeRepository
from millm.services.inference_service import get_probe_verdicts
from millm.services.probe_arming import armed_probe_from_row
from millm.services.probe_runtime import ProbeRuntimeState
from millm.services.probe_scoring import ProbeScoringService
from millm.services.probe_service import ProbeService
from tests.unit.f25_fixtures import clear_loaded, make_service, word_model, word_tokenizer
from tests.unit.score_fixtures import tiny_definition, tiny_identity

MESSAGES = [{"role": "user", "content": "w3 w4 w5 w6"}]


@pytest.fixture(autouse=True)
def clean():
    ProbeRuntimeState.reset_for_tests()
    yield
    ProbeRuntimeState.reset_for_tests()
    clear_loaded()


async def test_offline_prompt_window_equals_live(test_session, monkeypatch):
    model, tokenizer = word_model(), word_tokenizer()
    inference = make_service(model, tokenizer)
    monkeypatch.setattr(
        "millm.services.probe_arm_bridge.loaded_identity",
        AsyncMock(return_value=(tiny_identity(), model, tokenizer)),
    )
    repo = ProbeRepository(test_session)
    row = await ProbeService(repo).import_definition(tiny_definition())
    events_before = await ProbeEventRepository(test_session).count()

    # 1. score UNARMED
    offline = await ProbeScoringService(repo, inference).score(
        ProbeScoreRequest(inputs=[{"messages": MESSAGES}], windows=["all", "prompt"]),
        test_session,
    )
    by_window = {v["window"]: v for v in offline["results"][0]["verdicts"]}
    assert ProbeRuntimeState().has_armed() is False
    assert await ProbeEventRepository(test_session).count() == events_before

    # 2. arm, 3. live chat with max_tokens 1
    ProbeRuntimeState().arm(armed_probe_from_row(row, windows=["all", "prompt"]), model)
    request = ChatCompletionRequest(
        model="tiny", messages=[ChatMessage(**m) for m in MESSAGES], max_tokens=1,
        temperature=0.0,
    )
    with patch("millm.services.inference_service.torch.cuda.is_available", return_value=False):
        await inference.create_chat_completion(request)
    live = {v.window: v for v in get_probe_verdicts()}

    # 4. the same numbers
    assert live["prompt"].scored and by_window["prompt"]["n_scored_tokens"] > 0
    assert live["prompt"].score == by_window["prompt"]["score"]
    assert live["prompt"].n_scored_tokens == by_window["prompt"]["n_scored_tokens"]
    assert live["all"].score == by_window["all"]["score"]
    assert live["prompt"].fires == by_window["prompt"]["verdict"]
