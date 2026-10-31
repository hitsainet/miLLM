"""026 F-4: packed embeddings on a TINY REAL Llama — never a stub.

A pack is right-padded and each row is pooled over its own real positions, so a packed vector must
equal the single forward's within float noise for every pooling mode, at mixed lengths. A row the
synchronous endpoint would refuse fails ITS line with the same error; its neighbours are served.
"""

from __future__ import annotations

import math

import pytest

from millm.api.schemas.openai import EmbeddingRequest
from millm.core.errors import EmbeddingInputTooLongError, GenerationOutOfMemoryError
from millm.ml.embedding_pooling import EmbeddingOptions
from tests.unit.batch_fixtures import (  # noqa: F401
    batch_db,
    batch_dir,
    client_for,
    harness,
)
from tests.unit.f25_fixtures import clear_loaded, make_service, word_model, word_tokenizer

TEXTS = ["w1", "w1 w2 w3", "w4 w5 w6 w7 w8", "w9 w10 w11 w12 w13 w14 w15"]


def _service():
    return make_service(word_model(seed=5), word_tokenizer())


def _req(text, pooling="mean", normalize=False):
    return EmbeddingRequest(model="tiny", input=text, pooling=pooling, normalize=normalize)


def _cos_distance(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return 1.0 - dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


@pytest.mark.parametrize("pooling", ["mean", "last", "cls"])
@pytest.mark.parametrize("normalize", [False, True])
async def test_packed_vectors_equal_single_ones_at_mixed_lengths(pooling, normalize):
    svc = _service()
    try:
        async with svc._admit():
            single = [svc._embed_inputs([t], EmbeddingOptions(pooling, normalize)) for t in TEXTS]
            packed = await svc._embed_requests_packed(
                [_req(t, pooling, normalize) for t in TEXTS], max_rows=16
            )
        for (s_vecs, s_counts), (p_vecs, p_counts) in zip(single, packed):
            assert p_counts == s_counts
            assert _cos_distance(s_vecs[0], p_vecs[0]) <= 1e-5
            assert max(abs(x - y) for x, y in zip(s_vecs[0], p_vecs[0])) <= 1e-4
    finally:
        clear_loaded()


async def test_a_list_input_row_keeps_its_order_and_counts():
    svc = _service()
    try:
        async with svc._admit():
            single = svc._embed_inputs(TEXTS[1:], EmbeddingOptions())
            out = await svc._embed_requests_packed(
                [_req(TEXTS[0]), EmbeddingRequest(model="tiny", input=TEXTS[1:])], max_rows=16
            )
        vectors, counts = out[1]
        assert counts == single[1] and len(vectors) == 3
        for a, b in zip(vectors, single[0]):
            assert _cos_distance(a, b) <= 1e-5
    finally:
        clear_loaded()


async def test_a_too_long_row_fails_alone_with_the_synchronous_refusal():
    svc = _service()
    long = " ".join(["w1"] * 400)
    try:
        async with svc._admit():
            with pytest.raises(EmbeddingInputTooLongError) as sync:
                svc._embed_inputs([long], EmbeddingOptions(), param_for_string=True)
            out = await svc._embed_requests_packed(
                [_req("w1 w2"), _req(long), _req("w3 w4 w5")], max_rows=16
            )
        assert isinstance(out[1], EmbeddingInputTooLongError)
        assert str(out[1]) == str(sync.value) and out[1].details == sync.value.details
        assert not isinstance(out[0], Exception) and not isinstance(out[2], Exception)
    finally:
        clear_loaded()


async def test_bounds_split_packs_by_rows_and_by_padded_tokens(monkeypatch):
    svc = _service()
    real = svc._packed_pooled_hidden
    sizes: list[int] = []

    def spy(rows, modes):
        sizes.append(len(rows))
        return real(rows, modes)

    monkeypatch.setattr(svc, "_packed_pooled_hidden", spy)
    try:
        async with svc._admit():
            await svc._embed_requests_packed([_req(t) for t in TEXTS], max_rows=3)
            assert sizes == [3, 1]
            sizes.clear()
            await svc._embed_requests_packed([_req(t) for t in TEXTS], max_rows=16, max_tokens=16)
            assert sum(sizes) == 4 and max(sizes) <= 2
    finally:
        clear_loaded()


async def test_out_of_memory_halves_the_pack_and_a_single_input_fails_its_row(monkeypatch):
    svc = _service()
    real = svc._packed_pooled_hidden
    poison = list(svc._tokenizer(TEXTS[3])["input_ids"])
    attempts: list[int] = []

    def flaky(rows, modes):
        attempts.append(len(rows))
        if len(rows) > 1 or rows[0] == poison:
            raise GenerationOutOfMemoryError("oom", details={})
        return real(rows, modes)

    monkeypatch.setattr(svc, "_packed_pooled_hidden", flaky)
    try:
        async with svc._admit():
            out = await svc._embed_requests_packed([_req(t) for t in TEXTS], max_rows=16)
        assert attempts[0] == 4 and attempts.count(1) == 4
        assert all(not isinstance(o, Exception) for o in out[:3])
        assert isinstance(out[3], GenerationOutOfMemoryError)
    finally:
        clear_loaded()


def _line(custom_id, text, **body):
    return {"custom_id": custom_id, "method": "POST", "url": "/v1/embeddings",
            "body": {"model": "tiny", "input": text, **body}}


async def test_a_packed_embeddings_batch_marks_its_lines_and_matches_single_rows(harness):
    lines = [_line(f"e{i}", t) for i, t in enumerate(TEXTS)]
    async with client_for(harness.app()) as client:
        packed_id = (await harness.create(client, lines, endpoint="/v1/embeddings",
                                          pack=True)).json()["id"]
        single_id = (await harness.create(client, lines, endpoint="/v1/embeddings",
                                          pack=False)).json()["id"]
    await harness.drain()
    async with client_for(harness.app()) as client:
        packed = await harness.lines(client, (await harness.batch(packed_id)).output_file_id)
        single = await harness.lines(client, (await harness.batch(single_id)).output_file_id)
    assert [l["response"]["millm"]["packed"] for l in packed] == [True] * 4
    assert [l["response"]["millm"]["packed"] for l in single] == [False] * 4
    for p, s in zip(packed, single):
        pb, sb = p["response"]["body"], s["response"]["body"]
        assert pb["usage"] == sb["usage"] and pb["model"] == sb["model"]
        assert _cos_distance(pb["data"][0]["embedding"], sb["data"][0]["embedding"]) <= 1e-5
        assert p["response"]["millm"]["headers"] == s["response"]["millm"]["headers"]


async def test_an_over_long_row_fails_alone_inside_a_packed_batch(harness):
    lines = [_line("ok1", "w1 w2"), _line("long", " ".join(["w1"] * 400)), _line("ok2", "w3")]
    async with client_for(harness.app()) as client:
        batch_id = (await harness.create(client, lines, endpoint="/v1/embeddings",
                                         pack=True)).json()["id"]
    await harness.drain()
    async with client_for(harness.app()) as client:
        batch = await harness.batch(batch_id)
        out = await harness.lines(client, batch.output_file_id)
        err = await harness.lines(client, batch.error_file_id)
    assert {l["custom_id"]: l["response"]["millm"]["packed"] for l in out} == {
        "ok1": True, "ok2": True,
    }
    assert [e["custom_id"] for e in err] == ["long"]


@pytest.mark.parametrize("path", ["single", "packed"])
async def test_the_lm_head_projects_one_position_not_every_one(path):
    """Embeddings read hidden states only; a full-width LM head made a 16-row pack SLOWER than
    single forwards on the 3090 (Qwen2.5-7B: 23.8 vs 30.8 inputs/s)."""
    svc = _service()
    widths: list[int] = []
    handle = svc._model.lm_head.register_forward_hook(
        lambda _m, args, _out: widths.append(int(args[0].shape[1]))
    )
    try:
        async with svc._admit():
            if path == "single":
                svc._embed_inputs([TEXTS[3]], EmbeddingOptions())
            else:
                await svc._embed_requests_packed([_req(t) for t in TEXTS], max_rows=16)
        assert widths and all(w == 1 for w in widths), widths
    finally:
        handle.remove()
        clear_loaded()


async def test_inputs_are_packed_by_length_not_by_arrival(monkeypatch):
    """In arrival order a pack is as wide as its longest member; sorted, neighbours share a width.
    Measured on the 3090: unsorted 16-row packs of 5- to 400-token inputs ran SLOWER than single
    forwards."""
    svc = _service()
    real = svc._packed_pooled_hidden
    widths: list[list[int]] = []

    def spy(rows, modes):
        widths.append([len(r) for r in rows])
        return real(rows, modes)

    monkeypatch.setattr(svc, "_packed_pooled_hidden", spy)
    shuffled = [TEXTS[3], TEXTS[0], TEXTS[2], TEXTS[1]]
    try:
        async with svc._admit():
            out = await svc._embed_requests_packed([_req(t) for t in shuffled], max_rows=2)
        flat = [n for pack in widths for n in pack]
        assert flat == sorted(flat), widths
        single = [svc._embed_inputs([t], EmbeddingOptions())[0][0] for t in shuffled]
        for (vectors, _), s in zip(out, single):
            assert _cos_distance(vectors[0], s) <= 1e-5
    finally:
        clear_loaded()


def test_a_packed_embedding_batch_takes_a_full_pack_per_chunk():
    from types import SimpleNamespace

    from millm.core.config import settings
    from millm.services.batch.runner import BatchRunner

    runner = BatchRunner.__new__(BatchRunner)
    packed, single = SimpleNamespace(pack=True), SimpleNamespace(pack=False)
    assert runner.chunk_size(packed, "embedding") == settings.BATCH_PACK_MAX_ROWS
    assert runner.chunk_size(single, "embedding") == settings.BATCH_CHUNK_ROWS
