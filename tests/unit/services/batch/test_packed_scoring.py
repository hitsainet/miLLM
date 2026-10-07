"""Feature 26 task 6.7: packed scoring on a TINY REAL Llama — never a stub.

Rows differ in length by at least two tokens, so a gather off by one (or at -1) cannot match by
coincidence (FTID §8). M13's target is `test_mixed_length_packed_rows_score_their_own_last_token`.
"""

from __future__ import annotations

import torch

from millm.core.errors import ContextLengthExceededError, GenerationOutOfMemoryError
from millm.services.inference_service import ScoreSpec
from millm.services.next_token_scores import next_token_scores
from tests.unit.batch_fixtures import (  # noqa: F401
    batch_db,
    batch_dir,
    client_for,
    completion_line,
    harness,
)
from tests.unit.f25_fixtures import clear_loaded, make_service, word_model, word_tokenizer

PROMPTS = ["w1", "w1 w2 w3", "w4 w5 w6 w7 w8", "w9 w10 w11 w12 w13 w14 w15"]


def _service():
    return make_service(word_model(seed=3), word_tokenizer())


def _spec(text: str, top_k: int = 5) -> ScoreSpec:
    return ScoreSpec(text, True, None, 1.0, top_k)


async def test_single_path_scores_equal_a_direct_forward():
    """The synchronous scorer (pack_size=1) is still exactly one forward's last position."""
    svc = _service()
    try:
        async with svc._admit():
            scored = await svc._score_prompts(
                PROMPTS, add_special_tokens=True, allowed=None, temperature=1.0, top_k=5,
                pack_size=1,
            )
        for text, (scores, n) in zip(PROMPTS, scored):
            ids = svc._tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                logits = svc._model(**ids, use_cache=False).logits[0, -1].float()
            direct = next_token_scores(logits, allowed=None, temperature=1.0, top_k=5)
            assert scores.top == direct.top and n == ids.input_ids.shape[1]
    finally:
        clear_loaded()


async def test_mixed_length_packed_rows_score_their_own_last_token():
    svc = _service()
    try:
        async with svc._admit():
            single = await svc._score_prompts(
                PROMPTS, add_special_tokens=True, allowed=None, temperature=1.0, top_k=5,
                pack_size=1,
            )
            packed = await svc._score_specs_packed([_spec(p) for p in PROMPTS], max_rows=16)
        for (s_scores, s_n), p in zip(single, packed):
            assert p.prompt_tokens == s_n
            assert [t for t, _ in p.scores.top] == [t for t, _ in s_scores.top]
            for (_, a), (_, b) in zip(p.scores.top, s_scores.top):
                assert abs(a - b) < 1e-4
    finally:
        clear_loaded()


async def test_pack_size_routes_score_prompts_through_the_packed_path(monkeypatch):
    svc = _service()
    calls: list[int] = []
    real = svc._packed_next_token_logits

    def spy(rows):
        calls.append(len(rows))
        return real(rows)

    monkeypatch.setattr(svc, "_packed_next_token_logits", spy)
    try:
        async with svc._admit():
            await svc._score_prompts(PROMPTS, add_special_tokens=True, allowed=None,
                                     temperature=1.0, top_k=2, pack_size=1)
            assert calls == []
            await svc._score_prompts(PROMPTS, add_special_tokens=True, allowed=None,
                                     temperature=1.0, top_k=2, pack_size=4)
        assert calls == [4]
    finally:
        clear_loaded()


async def test_bounds_split_packs_by_rows_and_by_padded_tokens(monkeypatch):
    svc = _service()
    sizes: list[int] = []
    real = svc._packed_next_token_logits
    monkeypatch.setattr(svc, "_packed_next_token_logits", lambda rows: sizes.append(len(rows)) or real(rows))
    try:
        async with svc._admit():
            await svc._score_specs_packed([_spec(p) for p in PROMPTS], max_rows=3)
            assert sizes == [3, 1]
            sizes.clear()
            # 8 tokens is the longest prompt (with BOS); a 16-token budget fits two short rows.
            await svc._score_specs_packed([_spec(p) for p in PROMPTS], max_rows=16, max_tokens=16)
            assert sum(sizes) == 4 and max(sizes) <= 2
    finally:
        clear_loaded()


async def test_out_of_memory_halves_the_pack_and_a_single_row_gets_its_own_error(monkeypatch):
    svc = _service()
    real = svc._packed_next_token_logits
    attempts: list[int] = []

    def flaky(rows):
        attempts.append(len(rows))
        if len(rows) > 1 or rows[0] == svc._tokenizer(PROMPTS[3])["input_ids"]:
            raise GenerationOutOfMemoryError("oom", details={})
        return real(rows)

    monkeypatch.setattr(svc, "_packed_next_token_logits", flaky)
    try:
        async with svc._admit():
            out = await svc._score_specs_packed([_spec(p) for p in PROMPTS], max_rows=16)
        assert attempts[0] == 4 and attempts.count(1) == 4
        assert all(not isinstance(o, Exception) for o in out[:3])
        assert isinstance(out[3], GenerationOutOfMemoryError)
    finally:
        clear_loaded()


async def test_a_too_long_row_fails_alone_in_a_pack():
    svc = _service()
    try:
        async with svc._admit():
            out = await svc._score_specs_packed(
                [_spec("w1 w2"), _spec(" ".join(["w1"] * 400)), _spec("w3 w4 w5")], max_rows=16
            )
        assert isinstance(out[1], ContextLengthExceededError)
        assert not isinstance(out[0], Exception) and not isinstance(out[2], Exception)
    finally:
        clear_loaded()


async def test_a_packed_batch_marks_its_lines_and_matches_single_rows(harness):
    lines = [completion_line(f"c{i}", prompt=p) for i, p in enumerate(PROMPTS)]
    async with client_for(harness.app()) as client:
        packed_id = (await harness.create(client, lines, pack=True)).json()["id"]
        single_id = (await harness.create(client, lines, pack=False)).json()["id"]
    await harness.drain()
    async with client_for(harness.app()) as client:
        packed = await harness.lines(client, (await harness.batch(packed_id)).output_file_id)
        single = await harness.lines(client, (await harness.batch(single_id)).output_file_id)
    assert [l["response"]["millm"]["packed"] for l in packed] == [True] * 4
    assert [l["response"]["millm"]["packed"] for l in single] == [False] * 4
    for p, s in zip(packed, single):
        pc, sc = p["response"]["body"]["choices"][0], s["response"]["body"]["choices"][0]
        assert pc["logprobs"]["tokens"] == sc["logprobs"]["tokens"]
        assert abs(pc["logprobs"]["token_logprobs"][0] - sc["logprobs"]["token_logprobs"][0]) < 1e-4
        assert p["response"]["millm"]["headers"] == s["response"]["millm"]["headers"]


async def test_generation_rows_are_never_packed_whatever_pack_says(harness):
    line = {"custom_id": "g", "method": "POST", "url": "/v1/chat/completions",
            "body": {"model": "tiny", "messages": [{"role": "user", "content": "w1"}],
                     "max_tokens": 2}}
    lines = [line, {**line, "custom_id": "g2"}]
    async with client_for(harness.app()) as client:
        batch_id = (await harness.create(client, lines, endpoint="/v1/chat/completions",
                                         pack=True)).json()["id"]
    await harness.drain()
    async with client_for(harness.app()) as client:
        out = await harness.lines(client, (await harness.batch(batch_id)).output_file_id)
    assert [l["response"]["millm"]["packed"] for l in out] == [False, False]


async def test_activation_rows_run_single_inside_a_packed_batch(harness):
    with_acts = completion_line("acts", prompt="w1 w2", return_sae_activations={"top_k": 1})
    lines = [completion_line("a", prompt="w1"), with_acts, completion_line("b", prompt="w3 w4 w5")]
    async with client_for(harness.app()) as client:
        batch_id = (await harness.create(client, lines, pack=True)).json()["id"]
    await harness.drain()
    async with client_for(harness.app()) as client:
        batch = await harness.batch(batch_id)
        out = await harness.lines(client, batch.output_file_id)
        err = await harness.lines(client, batch.error_file_id)
    flags = {l["custom_id"]: l["response"]["millm"]["packed"] for l in out}
    assert flags.get("a") is True and flags.get("b") is True
    assert "acts" not in flags or flags["acts"] is False
    assert all(e["custom_id"] == "acts" for e in err)  # refused (no SAE attached) or single
