"""The constrained-decoding module on its own (Feature 25, FR-25.10, FR-25.12): output validation,
the logits processor's refusal path, the grammar cache, stop ids and the header value."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from millm.api.schemas.openai import ChatCompletionRequest
from millm.core.errors import ConstrainedOutputInvalidError
from millm.ml.constrained_decoding import (
    GrammarCache,
    JsonConstraintProcessor,
    constrained_header,
    stop_token_ids,
    validate_output,
)
from tests.unit.f25_fixtures import CHAR_VOCAB, char_tokenizer

SCHEMA = {"type": "object", "properties": {"label": {"enum": ["a", "b"]}}, "required": ["label"],
          "additionalProperties": False}


def fmt(kind="json_schema", schema=SCHEMA, name="judge_v1"):
    body = {"type": kind}
    if kind == "json_schema":
        body["json_schema"] = {"name": name, "schema": schema}
    return ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "x"}],
                                 response_format=body).response_format


class TestValidateOutput:
    def test_valid_outputs_pass(self):
        validate_output('{"label": "a"}', fmt())
        validate_output('{"anything": [1]}', fmt("json_object"))

    @pytest.mark.parametrize("text, response_format", [
        ('{"label": "c"}', "schema"),
        ('{"label": "a", "x": 1}', "schema"),
        ('{"label": ', "schema"),
        ("[1, 2]", "object"),
        ("not json", "object"),
    ])
    def test_invalid_outputs_raise_without_their_content(self, text, response_format):
        f = fmt() if response_format == "schema" else fmt("json_object")
        with pytest.raises(ConstrainedOutputInvalidError) as exc:
            validate_output(text, f)
        assert exc.value.status_code == 500
        assert text not in exc.value.message and text not in repr(exc.value.details)
        assert exc.value.details["length"] == len(text)


class TestProcessor:
    @pytest.fixture
    def cache(self):
        return GrammarCache(char_tokenizer(), len(CHAR_VOCAB), [0], capacity=4)

    def test_it_masks_every_token_the_grammar_forbids(self, cache):
        proc = JsonConstraintProcessor(cache.compile(fmt()), 1, len(CHAR_VOCAB))
        scores = torch.zeros(1, len(CHAR_VOCAB))
        out = proc(torch.tensor([[1]]), scores)
        allowed = [CHAR_VOCAB[i] for i in range(len(CHAR_VOCAB)) if out[0, i] > float("-inf")]
        assert "{" in allowed and "a" not in allowed and "}" not in allowed

    def test_a_rejected_token_raises_the_typed_error_not_an_assert(self, cache):
        """`assert` is stripped by python -O, and the token would pass silently."""
        proc = JsonConstraintProcessor(cache.compile(fmt()), 1, len(CHAR_VOCAB))
        proc(torch.tensor([[1]]), torch.zeros(1, len(CHAR_VOCAB)))
        bad = CHAR_VOCAB.index("a")  # the document must open with "{"
        with pytest.raises(ConstrainedOutputInvalidError):
            proc(torch.tensor([[1, bad]]), torch.zeros(1, len(CHAR_VOCAB)))

    def test_one_matcher_per_row(self, cache):
        proc = JsonConstraintProcessor(cache.compile(fmt()), 2, len(CHAR_VOCAB))
        out = proc(torch.tensor([[1], [1]]), torch.zeros(2, len(CHAR_VOCAB)))
        assert torch.equal(out[0] > float("-inf"), out[1] > float("-inf"))
        brace = CHAR_VOCAB.index("{")
        proc(torch.tensor([[1, brace], [1, brace]]), torch.zeros(2, len(CHAR_VOCAB)))
        assert proc.mask_ms > 0


class TestGrammarCache:
    def test_identical_schemas_hit_and_the_lru_is_bounded(self):
        cache = GrammarCache(char_tokenizer(), len(CHAR_VOCAB), [0], capacity=2)
        g1 = cache.compile(fmt())
        assert cache.compile(fmt(name="other")) is g1, "the name does not change the grammar"
        assert (cache.hits, cache.misses) == (1, 1)
        cache.compile(fmt("json_object"))
        cache.compile(fmt(schema={"type": "integer"}))
        assert len(cache._grammars) == 2
        cache.compile(fmt())
        assert cache.misses == 4, "the oldest grammar was evicted"


def test_stop_ids_are_the_models_declared_eos_plus_the_tokenizers():
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[1, 106, 50]))
    tok = SimpleNamespace(eos_token_id=50)
    assert stop_token_ids(model, tok) == [1, 106, 50]
    model.generation_config.eos_token_id = 7
    assert stop_token_ids(model, SimpleNamespace(eos_token_id=9)) == [7, 9]
    model.generation_config.eos_token_id = True  # a bool is not an id
    assert stop_token_ids(model, SimpleNamespace(eos_token_id=None)) == []


def test_header_values():
    assert constrained_header(fmt()) == 'json_schema;name="judge_v1"'
    assert constrained_header(fmt("json_object")) == "json_object"
    assert constrained_header(fmt("text")) is None
