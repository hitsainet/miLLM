"""Every field the adapter loader reads is guaranteed by the vendored schema (033 review finding 16).

Every field the loader reads unconditionally is a REQUIRED property (resolving `$ref` into
`$defs`), and the fields it reads with a default are present, optional, and carry that default. A
required field that became optional would let a manifest validate and then fail with a KeyError at
load; an optional field the loader indexed without its default would do the same (addendum
2026-10-14).

Split out of `tests/unit/api/test_adapter_manifest_schema_sync.py` (suite tracker 0.9): this half
needs the loader's own lists, so it imports `millm`, and it needs only the vendored file — it runs
in the main unit job. The byte-identity half needs miForge and runs, `millm`-free, in the
`adapter-manifest-sync` CI job.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# The loader's OWN lists (read unconditionally / read with a default), imported rather than
# copied: a copy here could agree with the schema while the loader read something else.
from millm.services.adapter_manifest import FIELDS_OPTIONAL_WITH_DEFAULT, FIELDS_REQUIRED

VENDORED = Path(__file__).resolve().parents[3] / "docs" / "schemas" / "adapter-manifest-v1.json"


@pytest.fixture(scope="module")
def frozen() -> dict:
    assert VENDORED.exists(), f"vendored adapter manifest schema missing at {VENDORED}"
    return json.loads(VENDORED.read_text())


def _resolve(schema: dict, node: dict) -> dict:
    """Follow a local `$ref` (`#/$defs/Name`) to the object it names."""
    while "$ref" in node:
        ref = node["$ref"]
        assert ref.startswith("#/$defs/"), f"non-local $ref {ref!r}"
        node = schema["$defs"][ref.removeprefix("#/$defs/")]
    return node


def _lookup(schema: dict, dotted: str) -> tuple[dict, str, dict]:
    """(parent object schema, leaf name, leaf property schema) for a dotted field path."""
    *parents, leaf = dotted.split(".")
    node = _resolve(schema, schema)
    for part in parents:
        assert part in node.get("properties", {}), f"{dotted}: {part!r} is not a property"
        node = _resolve(schema, node["properties"][part])
    assert leaf in node.get("properties", {}), f"{dotted}: {leaf!r} is not a property"
    return node, leaf, node["properties"][leaf]


def _required_chain(schema: dict, dotted: str) -> bool:
    """True when every segment of the path is required by its parent object."""
    parts = dotted.split(".")
    node = _resolve(schema, schema)
    for i, part in enumerate(parts):
        if part not in node.get("required", []):
            return False
        if i < len(parts) - 1:
            node = _resolve(schema, node["properties"][part])
    return True


class TestWhatMiLLMReadsIsWhatTheSchemaGuarantees:
    @pytest.mark.parametrize("dotted", FIELDS_REQUIRED)
    def test_each_field_read_unconditionally_is_required(self, frozen, dotted):
        _lookup(frozen, dotted)
        assert _required_chain(frozen, dotted), (
            f"{dotted} is read unconditionally by the adapter loader but is not required by the "
            "schema — a valid manifest could omit it and the load would fail with a KeyError"
        )

    @pytest.mark.parametrize("dotted", sorted(FIELDS_OPTIONAL_WITH_DEFAULT))
    def test_each_field_read_with_a_default_is_optional_with_that_default(self, frozen, dotted):
        parent, leaf, spec = _lookup(frozen, dotted)
        assert leaf not in parent.get("required", []), (
            f"{dotted} became required; the loader's default is now dead code — check the schema "
            "change was intended"
        )
        assert "default" in spec, f"{dotted} has no default in the schema"
        assert spec["default"] == FIELDS_OPTIONAL_WITH_DEFAULT[dotted]

    def test_the_only_scaling_rule_is_alpha_over_r(self, frozen):
        """A new rule must reach the loader before it reaches a manifest: the loader refuses any
        value but `alpha_over_r`, and this test goes red the day the enum grows."""
        _, _, spec = _lookup(frozen, "adapter.scaling")
        assert spec["enum"] == ["alpha_over_r"]

    def test_the_helpers_agree_with_the_schema(self, frozen):
        """The structural helpers must be able to fail: an optional field reads as not required."""
        assert not _required_chain(frozen, "base.manifest_sha256")
        assert _required_chain(frozen, "base.name")
