"""The probe served-render rule, pinned against miStudio by a shared case file (2026-10-08).

`probe_scoring.served_render` (with `prompt_encoding.rendered_chat_ids`) renders every input
`/api/probes/score` and parity read; miStudio's `probe_monitor_render.served_render` renders every
probe input it trains, calibrates and exports. The two are mirrored BY HAND, and until this file
nothing tied them together — this week's defects were exactly that class.

`docs/schemas/served-render-cases.json` is byte-identical in both repos. This runs miLLM's OWN
renderer over every case — through `ProbeInputPreparer.prepare`, the `/api/probes/score` wiring,
not only the helper — and asserts the ids, the generation-prompt branch, the prompt/response
boundary and the `last_user` span. The `structural` set builds WordLevel tokenizers from the file
and ALWAYS runs; the `real` set needs the Llama-3.1-8B-Instruct tokenizer identified by its
chat-template hash and SKIPS LOUDLY when it is absent (`MILLM_REAL_TOKENIZERS`).

A case carrying `known_divergence.mistudio` is one miStudio is KNOWN not to meet; miStudio runs it
as a strict xfail, and `expected` is still asserted HERE like any other case, so this side cannot
drift into agreeing with a divergence. TODAY THERE IS NONE: the one there was (F1, a template that
writes no BOS while its tokenizer adds one — this server served one BOS, miStudio trained on none)
closed on 2026-10-08 when miStudio adopted this server's start-of-text rule.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from millm.services.probe_parity import ProbeParityEngine
from millm.services.probe_scoring import ProbeInputPreparer, served_render, template_renderer

REPO = Path(__file__).resolve().parents[3]
CASES_PATH = REPO / "docs" / "schemas" / "served-render-cases.json"
STUDIO_CASES = (
    Path(os.environ.get("MISTUDIO_REPO", "/home/x-sean/app/miStudio"))
    / "docs" / "schemas" / "served-render-cases.json"
)
CASES = json.loads(CASES_PATH.read_text(encoding="utf-8"))
STRUCTURAL = CASES["structural"]
REAL = CASES["real"]
SERVED = {"generation_prompt": True, "add_special_tokens": False}


def _structural_tokenizer(case):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocab = STRUCTURAL["vocab"]
    backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(vocab)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    if case["tokenizer_adds_bos"]:
        backend.post_processor = processors.TemplateProcessing(
            single="<s> $A", special_tokens=[("<s>", vocab.index("<s>"))]
        )
    bos = case.get("bos_token", "<s>")
    kwargs = {"bos_token": bos} if bos else {}
    fast = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", **kwargs)
    fast.chat_template = STRUCTURAL["templates"][case["template"]]
    return fast


def _messages_item(messages):
    return SimpleNamespace(
        kinds=lambda: ["messages"],
        messages=[SimpleNamespace(role=m["role"], content=m["content"]) for m in messages],
    )


def _check(tokenizer, case, ids_key):
    expected = case["expected"]
    messages = case["messages"]
    render = template_renderer(tokenizer)
    served = served_render(tokenizer, messages, render)
    assert served.generation_prompt is expected["generation_prompt"]
    if ids_key == "tokens":
        assert tokenizer.convert_ids_to_tokens(served.ids) == expected["tokens"]
    else:
        assert served.ids == expected["ids"]
    assert served.prompt_tokens == expected["prompt_tokens"]

    # THE WIRING `/api/probes/score` runs: ids, boundary AND the span over the same render.
    prepared = ProbeInputPreparer(tokenizer, render).prepare(0, _messages_item(messages))
    assert prepared.error is None
    assert prepared.ids == served.ids
    assert prepared.prompt_tokens == expected["prompt_tokens"]
    span = list(prepared.last_user_span) if prepared.last_user_span else None
    assert span == expected["last_user_span"]

    if "max_length" in case:
        # miStudio caps a vector keeping the TAIL; parity must name it as truncated, not mismatched.
        drift = ProbeParityEngine._drift(
            [{"messages": messages, "token_ids": expected["truncated_ids"]}], tokenizer, SERVED
        )
        assert drift["truncated_vectors"] == [0] and drift["mismatched"] == 0
        assert served.ids[-case["max_length"]:] == expected["truncated_ids"]


@pytest.mark.parametrize("case", STRUCTURAL["cases"], ids=lambda c: c["name"])
def test_structural_cases(case):
    _check(_structural_tokenizer(case), case, "tokens")


def test_the_structural_set_covers_every_branch():
    expected = [c["expected"] for c in STRUCTURAL["cases"]]
    assert {e["generation_prompt"] for e in expected} == {True, False}
    assert any(e["prompt_tokens"] is None for e in expected)
    assert any(e["last_user_span"] is None for e in expected)
    assert any(
        c["tokenizer_adds_bos"] and STRUCTURAL["templates"][c["template"]].startswith("<s>")
        for c in STRUCTURAL["cases"]
    ), "no case where the template AND the tokenizer would each supply a BOS"
    assert any(c["messages"][-1]["role"] == "tool" for c in STRUCTURAL["cases"])


@pytest.fixture(scope="module")
def llama():
    from tests.unit.services.test_probe_parity_render import _real_tokenizer

    return _real_tokenizer(REAL["tokenizer"]["chat_template_sha256"])


@pytest.mark.parametrize("case", REAL["cases"], ids=lambda c: c["name"])
def test_real_tokenizer_cases(case, llama):
    _check(llama, case, "ids")


def test_the_case_file_is_identical_in_mistudio():
    """⚠ TWO RENDERERS, ONE RULE. If miStudio's copy differs, the two sides are pinned to
    different rules and each suite stays green while they drift."""
    if not STUDIO_CASES.exists():
        if os.environ.get("MILLM_REQUIRE_CROSS_REPO_CHECKS") == "1":
            pytest.fail(f"miStudio's copy of the served-render cases is missing at {STUDIO_CASES}")
        pytest.skip("miStudio checkout not present")
    assert STUDIO_CASES.read_bytes() == CASES_PATH.read_bytes()
