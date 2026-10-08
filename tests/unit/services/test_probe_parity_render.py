"""Parity's `messages` round-trip re-renders the way the DEFINITION says it was rendered (2026-10-08).

THE DEFECT. miStudio now renders every probe input in miLLM's served form — the chat template WITH
the generation prompt for a conversation that does not end on an assistant turn, exactly one BOS —
and records it as the definition's optional `render` block, `{"generation_prompt": true,
"add_special_tokens": false}`. Parity's informational round-trip re-rendered every vector WITHOUT
the generation prompt, so on the first two served-render probes imported (`pr_f90227264893`,
`pr_7de1e85e2d16`) it reported **0 of 16** while miStudio's own build-time check recorded
`messages_reproduce_token_ids: true`. The verdict was unaffected (parity scores `token_ids`); the
report told an operator the document does not reproduce, when it does.

A second, older cause made the same report unconditional: the round-trip called
`apply_chat_template(..., tokenize=True)`, which on transformers 5 returns a `BatchEncoding`, and
`list()` of that is its KEYS — `['input_ids', 'attention_mask']`, never equal to any recorded ids.

THE FIX re-renders through `probe_scoring.served_render` — the SAME function `/api/probes/score`
uses — when the block records the served form; keeps the old no-generation-prompt render when the
block is absent, and says so; and names keep-tail-truncated vectors instead of calling them
mismatches.

Fixtures: (1) a WordLevel tokenizer under a real `PreTrainedTokenizerFast` and a real Jinja template
whose generation prompt is a distinct token — always runs; (2) the two REAL served-render
definitions (reduced, `tests/fixtures/probe_served_render_definitions.json`) against the real
Llama-3.1 tokenizer whose template hash the definitions record — skips LOUDLY when no such
tokenizer is on this machine (`MILLM_REAL_TOKENIZERS`, or the HuggingFace cache).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from millm.ml.probe_head import ProbeHead
from millm.services.probe_parity import ProbeParityEngine
from millm.services.probe_runtime import ArmedProbe
from millm.services.probe_scoring import served_render, template_renderer

SERVED = {"generation_prompt": True, "add_special_tokens": False}
WORDS = ["<s>", "[UNK]", "<|user|>", "<|assistant|>", "<|system|>", "<|end|>"] + [
    f"w{i}" for i in range(40)
]
TEMPLATE = (
    "<s> {% for m in messages %}<|{{ m.role }}|> {{ m.content }} <|end|> {% endfor %}"
    "{% if add_generation_prompt %}<|assistant|> {% endif %}"
)


@pytest.fixture(scope="module")
def tok() -> PreTrainedTokenizerFast:
    """Template writes `<s>` AND the tokenizer adds one (the Llama 3 / gemma / LFM2.5 shape), so a
    round-trip that tokenized with the default `add_special_tokens=True` would get two."""
    backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.post_processor = processors.TemplateProcessing(
        single="<s> $A", special_tokens=[("<s>", 0)]
    )
    fast = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<s>", unk_token="[UNK]")
    fast.chat_template = TEMPLATE
    return fast


USER_ENDED = [{"role": "system", "content": "w1 w2"}, {"role": "user", "content": "w3 w4 w5"}]
ASSISTANT_ENDED = USER_ENDED + [{"role": "assistant", "content": "w6 w7"}]
MULTI = ASSISTANT_ENDED + [{"role": "user", "content": "w8 w9"}]
LONG = [{"role": "user", "content": " ".join(f"w{i}" for i in range(10, 40))}]


def _served_ids(tok, messages) -> list[int]:
    """The served form, written out longhand rather than through the code under test."""
    gp = messages[-1]["role"] != "assistant"
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=gp)
    return list(tok(text, add_special_tokens=False)["input_ids"])


def _legacy_ids(tok, messages) -> list[int]:
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return list(tok(text, add_special_tokens=False)["input_ids"])


def _definition(vectors, render=SERVED) -> dict:
    doc = {
        "test_vectors": {
            "tolerance": 0.05,
            "vectors": [
                {"messages": m, "token_ids": ids, "token_scores": [4.0] * len(ids), "score": 4.0}
                for m, ids in vectors
            ],
        }
    }
    if render is not None:
        doc["render"] = render
    return doc


def _probe() -> ArmedProbe:
    return ArmedProbe(
        probe_id="pr_render", name="render", head=ProbeHead(weight=torch.ones(4), bias=0.0, layer=1),
        rule="mean", scope="all", layer=1, rung=2, rung_language="detects on unseen tasks",
        threshold=1.0,
    )


def _forward(input_ids, context):
    context.observe(1, torch.full((1, input_ids.shape[1], 4), 1.0))


def _drift(definition, tokenizer) -> dict:
    """Through `ProbeParityEngine.run`, never `_drift` directly: the `render` block must be READ
    off the definition by the engine, and a test calling the helper would not see that wiring."""
    report = ProbeParityEngine(_forward).run(
        _probe(), definition, tolerance=0.05, tokenizer=tokenizer
    )
    assert report.passed is True, "the round-trip is informational and must never gate"
    return report.tokenization_drift


class TestServedRenderDefinitions:
    def test_user_ended_assistant_ended_and_multi_turn_all_reproduce(self, tok):
        vectors = [(m, _served_ids(tok, m)) for m in (USER_ENDED, ASSISTANT_ENDED, MULTI)]
        drift = _drift(_definition(vectors), tok)
        assert drift["messages_reproduce_token_ids"] == 3
        assert drift["mismatched"] == 0 and drift["mismatched_vectors"] == []
        assert drift["truncated_vectors"] == []
        assert drift["rendered_with"] == "served_form"
        assert drift["render"] == SERVED
        assert "render_note" not in drift

    def test_the_generation_prompt_is_what_distinguishes_them(self, tok):
        """Guard on the fixture itself: a user-ended served render differs from the old one,
        so a round-trip that dropped the generation prompt cannot pass the test above."""
        assert _served_ids(tok, USER_ENDED) != _legacy_ids(tok, USER_ENDED)
        assert _served_ids(tok, ASSISTANT_ENDED) == _legacy_ids(tok, ASSISTANT_ENDED)

    def test_exactly_one_bos(self, tok):
        """The template writes `<s>` and the tokenizer adds one; the round-trip must use the
        one-BOS rule, not the tokenizer default, or every vector mismatches by a leading BOS."""
        ids = served_render(tok, USER_ENDED, template_renderer(tok)).ids
        assert ids[:2] != [0, 0] and ids[0] == 0
        assert ids == _served_ids(tok, USER_ENDED)


class TestAbsentRenderBlock:
    def test_an_old_document_keeps_the_old_render_and_says_so(self, tok):
        """A definition with no `render` block predates render recording: re-rendered WITHOUT the
        generation prompt — the way such documents were produced — and never read as served."""
        vectors = [(m, _legacy_ids(tok, m)) for m in (USER_ENDED, MULTI, ASSISTANT_ENDED)]
        drift = _drift(_definition(vectors, render=None), tok)
        assert drift["messages_reproduce_token_ids"] == 3
        assert drift["mismatched"] == 0
        assert drift["render"] is None
        assert drift["rendered_with"] == "no_generation_prompt"
        assert "predates render recording" in drift["render_note"]

    def test_an_absent_block_is_never_read_as_the_served_form(self, tok):
        """Served-form ids under a document that does NOT say it is served: the user-ended
        vectors must not reproduce — if they did, the absent block was read as served."""
        vectors = [(m, _served_ids(tok, m)) for m in (USER_ENDED, MULTI)]
        drift = _drift(_definition(vectors, render=None), tok)
        assert drift["messages_reproduce_token_ids"] == 0
        assert drift["mismatched_vectors"] == [0, 1]

    def test_a_block_that_does_not_record_the_served_form_is_not_served(self, tok):
        vectors = [(USER_ENDED, _legacy_ids(tok, USER_ENDED))]
        drift = _drift(
            _definition(vectors, render={"generation_prompt": False, "add_special_tokens": False}),
            tok,
        )
        assert drift["messages_reproduce_token_ids"] == 1
        assert drift["rendered_with"] == "no_generation_prompt"
        assert "does not record the served form" in drift["render_note"]


class TestKeepTailTruncation:
    def test_a_keep_tail_vector_is_named_truncated_not_mismatched(self, tok):
        full = _served_ids(tok, LONG)
        assert len(full) > 12
        vectors = [(USER_ENDED, _served_ids(tok, USER_ENDED)), (LONG, full[-12:])]
        drift = _drift(_definition(vectors), tok)
        assert drift["messages_reproduce_token_ids"] == 1
        assert drift["truncated_vectors"] == [1]
        assert drift["mismatched"] == 0 and drift["mismatched_vectors"] == []
        assert "TAIL" in drift["truncation_note"]

    def test_a_head_cut_is_a_mismatch(self, tok):
        """Only the TAIL is what miStudio keeps; ids equal to the render's HEAD are not a
        truncation miStudio produces, so they are reported as a mismatch."""
        full = _served_ids(tok, LONG)
        drift = _drift(_definition([(LONG, full[:12])]), tok)
        assert drift["truncated_vectors"] == []
        assert drift["mismatched_vectors"] == [0]

    def test_a_wrong_tail_is_a_mismatch(self, tok):
        full = _served_ids(tok, LONG)
        wrong = full[-12:]
        wrong[3] = 1
        drift = _drift(_definition([(LONG, wrong)]), tok)
        assert drift["truncated_vectors"] == [] and drift["mismatched_vectors"] == [0]

    def test_no_truncation_note_when_nothing_was_truncated(self, tok):
        drift = _drift(_definition([(USER_ENDED, _served_ids(tok, USER_ENDED))]), tok)
        assert "truncation_note" not in drift


# ── the real definitions, against the real tokenizer they were rendered with ─────────────────

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "probe_served_render_definitions.json"
DEFINITIONS = json.loads(FIXTURE.read_text())["definitions"]


def _real_tokenizer(sha: str):
    """A tokenizer whose chat template hashes to the definitions' `chat_template_sha256`, from
    `MILLM_REAL_TOKENIZERS` (os.pathsep-separated directories) or the HuggingFace cache. Matched
    by TEMPLATE HASH, so a tokenizer that merely shares a name cannot stand in for it."""
    dirs: list[Path] = []
    configured = os.environ.get("MILLM_REAL_TOKENIZERS")
    if configured:
        dirs += [Path(p) for p in configured.split(os.pathsep) if p]
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    for repo in ("models--meta-llama--Llama-3.1-8B-Instruct", "models--unsloth--Llama-3.1-8B-Instruct"):
        dirs += sorted((hub / repo / "snapshots").glob("*"))
    from transformers import AutoTokenizer

    for path in dirs:
        if not (path / "tokenizer_config.json").exists():
            continue
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(path))
        except Exception:  # noqa: BLE001 - a snapshot this transformers cannot read
            continue
        template = getattr(tokenizer, "chat_template", None) or ""
        if hashlib.sha256(template.encode("utf-8")).hexdigest() == sha:
            return tokenizer
    pytest.skip(
        "NO Llama-3.1-8B-Instruct TOKENIZER WITH TEMPLATE " + sha[:12] + " ON THIS MACHINE — set "
        "MILLM_REAL_TOKENIZERS to check the real served-render definitions; the WordLevel "
        "fixtures above still ran"
    )


@pytest.fixture(scope="module")
def llama():
    shas = {d["model"]["chat_template_sha256"] for d in DEFINITIONS.values()}
    assert len(shas) == 1
    return _real_tokenizer(shas.pop())


class TestTheRealDefinitions:
    def test_high_stakes_reproduces_with_its_two_truncated_vectors_named(self, llama):
        """`pm_f736aa73969d` (imported as `pr_f90227264893`): 16 vectors, seven assistant-ended,
        two cut to the 1,024-token cap keeping the tail. Live parity reported 0 of 16."""
        d = DEFINITIONS["pm_f736aa73969d"]
        drift = _drift(_definition_from(d), llama)
        assert drift["checked"] == 16
        assert drift["messages_reproduce_token_ids"] == 14
        assert drift["truncated_vectors"] == [10, 15]
        assert drift["mismatched"] == 0

    def test_humor_reproduces_sixteen_of_sixteen(self, llama):
        """`pm_1b1f8c50d6d7` (imported as `pr_7de1e85e2d16`): sixteen single user turns."""
        drift = _drift(_definition_from(DEFINITIONS["pm_1b1f8c50d6d7"]), llama)
        assert drift["messages_reproduce_token_ids"] == 16
        assert drift["truncated_vectors"] == [] and drift["mismatched"] == 0

    def test_without_its_render_block_the_same_document_does_not_reproduce(self, llama):
        """The negative control on real data: strip the block and the user-ended vectors fail,
        which is what the absent-block default must do rather than guess the served form."""
        d = DEFINITIONS["pm_1b1f8c50d6d7"]
        drift = _drift(_definition_from(d, render=None), llama)
        assert drift["messages_reproduce_token_ids"] == 0
        assert "predates render recording" in drift["render_note"]


def _definition_from(d: dict, render: object = "keep") -> dict:
    doc = {
        "test_vectors": {
            "tolerance": 0.05,
            "vectors": [
                {**v, "token_scores": [4.0] * len(v["token_ids"]), "score": 4.0}
                for v in d["test_vectors"]["vectors"]
            ],
        }
    }
    block = d["render"] if render == "keep" else render
    if block is not None:
        doc["render"] = block
    return doc
