"""The vendored adapter manifest schema must stay byte-identical to miForge's (FR-33.6).

miForge owns `miforge.adapter-manifest` v1 and is normative for it (BRD-01 R-01.28); miLLM vendors
it because it serves adapters (R-01.27, Feature 33). If this fails, someone edited the copy
instead of the source, or miForge republished v1 without a version bump. Re-vendor with `cp`;
never edit the copy.

The byte check is skipped LOUDLY when miForge is not checked out, and REQUIRED under
`MILLM_REQUIRE_CROSS_REPO_CHECKS=1` — which the CI job `adapter-manifest-sync` sets, checking
miForge out with the read-only deploy key `MIFORGE_DEPLOY_KEY`.

⚠ IMPORTS NOTHING FROM `millm`. That job installs pytest alone, and importing `millm` pulls in
the package's whole runtime (structlog, torch, transformers) through the package `__init__`s —
which turned the job into an ImportError on every push, a verdict about neither contract (suite
tracker 0.9). `test_cross_repo_sync_files_need_only_pytest.py` holds this. The check that every
field the LOADER reads is guaranteed by the schema needs the loader, so it lives in
`tests/unit/services/test_adapter_manifest_fields_match_schema.py` and runs in the main unit job
(it needs only the vendored file, not miForge).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.support.sibling_repos import sibling_repo

VENDORED = Path(__file__).resolve().parents[3] / "docs" / "schemas" / "adapter-manifest-v1.json"
MIFORGE = sibling_repo("miForge", "MIFORGE_REPO") / "docs/schemas/adapter-manifest-v1.json"
REQUIRED = os.environ.get("MILLM_REQUIRE_CROSS_REPO_CHECKS") == "1"

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


class TestTheVendoredCopyIsMiForges:
    def test_it_is_byte_identical(self):
        if not MIFORGE.exists():
            if REQUIRED:
                pytest.fail(
                    f"MILLM_REQUIRE_CROSS_REPO_CHECKS=1 but miForge's schema is not at {MIFORGE}"
                )
            pytest.skip(
                f"miForge not checked out at {MIFORGE}: the adapter manifest byte-identity check "
                "did NOT run (set MIFORGE_REPO, or MILLM_REQUIRE_CROSS_REPO_CHECKS=1 to fail)"
            )
        assert VENDORED.read_bytes() == MIFORGE.read_bytes(), (
            "the vendored adapter manifest schema has drifted from miForge's. Re-vendor with cp; "
            "never edit the copy — the two repos would then disagree about what an adapter is."
        )

    def test_the_vendored_file_is_the_v1_contract(self, frozen):
        assert frozen["$ref"] == "#/$defs/AdapterManifestV1"
        root = _resolve(frozen, frozen)
        assert root["properties"]["kind"]["const"] == "miforge.adapter-manifest"
        assert root["properties"]["schema_version"]["const"] == "1"
