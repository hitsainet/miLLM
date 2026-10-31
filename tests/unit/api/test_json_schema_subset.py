"""The declared JSON Schema subset and its honesty probe (Feature 25, FR-25.10.9, FR-25.11.2).

HONESTY: every allowlisted keyword has a violating example, and the INSTALLED xgrammar must reject
it. A keyword added to `ALLOWED_KEYWORDS` without an example fails the coverage check; one the
library compiles without enforcing (xgrammar 0.2.8: `uniqueItems`, `not`) fails the probe. An
xgrammar upgrade that stops enforcing a keyword goes red here, before it reaches a client.

MUTATION CONTROL M14: add `uniqueItems` to the allowlist -> red.
"""

from __future__ import annotations

import json

import pytest

from millm.api.json_schema_subset import (
    ALLOWED_KEYWORDS,
    ANNOTATIONS,
    MAX_SCHEMA_BYTES,
    check,
    offending_keywords,
)
from millm.core.errors import ResponseFormatUnsupportedError

#: keyword -> (schema, a conforming document, a violating document)
PROBES = {
    "type": ({"type": "integer"}, "5", '"x"'),
    "properties": ({"type": "object", "properties": {"a": {"type": "integer"}}},
                   '{"a":1}', '{"a":"x"}'),
    "required": ({"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]},
                 '{"a":1}', "{}"),
    "additionalProperties": ({"type": "object", "properties": {"a": {"type": "integer"}},
                              "additionalProperties": False}, '{"a":1}', '{"a":1,"b":2}'),
    "items": ({"type": "array", "items": {"type": "integer"}}, "[1]", '["x"]'),
    "minItems": ({"type": "array", "items": {"type": "integer"}, "minItems": 2}, "[1,2]", "[1]"),
    "maxItems": ({"type": "array", "items": {"type": "integer"}, "maxItems": 1}, "[1]", "[1,2]"),
    "enum": ({"enum": ["a", "b"]}, '"a"', '"c"'),
    "const": ({"const": "a"}, '"a"', '"b"'),
    "minimum": ({"type": "integer", "minimum": 3}, "5", "1"),
    "maximum": ({"type": "integer", "maximum": 3}, "2", "9"),
    "minLength": ({"type": "string", "minLength": 3}, '"abc"', '"a"'),
    "maxLength": ({"type": "string", "maxLength": 2}, '"ab"', '"abcd"'),
    "pattern": ({"type": "string", "pattern": "^a+$"}, '"aa"', '"b"'),
    "anyOf": ({"anyOf": [{"type": "integer"}, {"type": "boolean"}]}, "true", '"x"'),
    "$ref": ({"$defs": {"i": {"type": "integer"}}, "$ref": "#/$defs/i"}, "3", '"x"'),
    "$defs": ({"type": "object", "properties": {"a": {"$ref": "#/$defs/i"}},
               "$defs": {"i": {"type": "integer"}}}, '{"a":1}', '{"a":"x"}'),
    # Not allowlisted; kept so a later attempt to add it has its probe ready.
    "uniqueItems": ({"type": "array", "items": {"type": "integer"}, "uniqueItems": True},
                    "[1,2]", "[1,1]"),
    "not": ({"not": {"type": "integer"}}, '"x"', "3"),
}


@pytest.fixture(scope="module")
def compiler():
    import xgrammar as xgr

    from tests.unit.f25_fixtures import CHAR_VOCAB, char_tokenizer

    info = xgr.TokenizerInfo.from_huggingface(char_tokenizer(), vocab_size=len(CHAR_VOCAB),
                                              stop_token_ids=[0])
    return xgr.GrammarCompiler(info)


def _accepts(compiler, schema, document) -> bool:
    import xgrammar as xgr

    matcher = xgr.GrammarMatcher(compiler.compile_json_schema(json.dumps(schema)))
    return bool(matcher.accept_string(document)) and bool(matcher.is_completed())


def test_every_allowlisted_keyword_has_a_probe():
    missing = ALLOWED_KEYWORDS - set(PROBES)
    assert not missing, f"allowlisted without an enforcement probe: {sorted(missing)}"


@pytest.mark.parametrize("keyword", sorted(ALLOWED_KEYWORDS))
def test_the_installed_library_enforces_each_allowlisted_keyword(compiler, keyword):
    schema, good, bad = PROBES[keyword]
    assert _accepts(compiler, schema, good), f"{keyword}: the conforming example was rejected"
    assert not _accepts(compiler, schema, bad), (
        f"{keyword}: xgrammar accepted a violating document — it compiles this keyword without "
        "enforcing it, so it must not be in ALLOWED_KEYWORDS")


@pytest.mark.parametrize("keyword", ["uniqueItems", "not", "multipleOf", "allOf", "oneOf",
                                     "format", "patternProperties", "if", "dependentRequired",
                                     "$schema"])
def test_refused_keywords_are_outside_the_allowlist(keyword):
    assert keyword not in ALLOWED_KEYWORDS and keyword not in ANNOTATIONS


class TestCheck:
    def test_a_supported_schema_passes(self):
        check({"type": "object", "title": "T", "description": "d",
               "properties": {"label": {"type": "string", "enum": ["a", "b"], "default": "a"},
                              "n": {"type": ["integer", "null"], "minimum": 0},
                              "tags": {"type": "array", "items": {"$ref": "#/$defs/tag"},
                                       "maxItems": 3}},
               "required": ["label"], "additionalProperties": False,
               "$defs": {"tag": {"type": "string", "pattern": "^[a-z]+$"}}})

    @pytest.mark.parametrize("schema, keyword, pointer", [
        ({"type": "integer", "multipleOf": 2}, "multipleOf", "/multipleOf"),
        ({"type": "object", "properties": {"a": {"type": "array", "uniqueItems": True}}},
         "uniqueItems", "/properties/a/uniqueItems"),
        ({"not": {"type": "string"}}, "not", "/not"),
        ({"$ref": "https://example.com/s.json"}, "$ref (only local #/$defs/ references)", "/$ref"),
        ({"type": "array", "items": [{"type": "string"}]}, "items (tuple form)", "/items"),
        ({"type": "object", "properties": {"a/b": {"format": "date"}}}, "format",
         "/properties/a~1b/format"),
        ({"type": "uuid"}, "type", "/type"),
    ])
    def test_a_keyword_outside_the_subset_is_named_with_its_pointer(self, schema, keyword,
                                                                    pointer):
        with pytest.raises(ResponseFormatUnsupportedError) as exc:
            check(schema)
        assert exc.value.details["param"] == "response_format"
        assert {"keyword": keyword, "pointer": pointer} in exc.value.details["keywords"]
        assert pointer in exc.value.message

    def test_every_offender_is_listed_up_to_twenty(self):
        schema = {"type": "object", "properties": {f"p{i}": {"format": "x"} for i in range(30)}}
        assert len(offending_keywords(schema)) == 20

    def test_size_cap(self):
        schema = {"type": "object", "description": "x" * (MAX_SCHEMA_BYTES + 1)}
        with pytest.raises(ResponseFormatUnsupportedError, match="bytes"):
            check(schema)

    def test_depth_cap(self):
        schema: dict = {"type": "integer"}
        for _ in range(40):
            schema = {"type": "array", "items": schema}
        with pytest.raises(ResponseFormatUnsupportedError, match="depth"):
            check(schema)

    def test_property_cap(self):
        schema = {"type": "object", "properties": {f"p{i}": {"type": "integer"}
                                                   for i in range(257)}}
        with pytest.raises(ResponseFormatUnsupportedError, match="256"):
            check(schema)

    def test_annotation_values_are_not_walked(self):
        check({"type": "string", "examples": [{"multipleOf": 3}], "default": {"not": 1}})
