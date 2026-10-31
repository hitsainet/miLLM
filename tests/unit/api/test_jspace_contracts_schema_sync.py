"""miStudio's J-space contract (027 task 4.2, miLLM half): the vendored copy must stay byte-identical.

miStudio publishes `docs/schemas/jspace-contracts-v1.json` (027 tasks 4.1/4.2, generated from its
`src/schemas/jspace_contracts.py`) with four kinds: `jlens-artifact/v1`, `workspace-annotation/v1`,
`jlens-readout/v1` and `jspace-watchlist/v1`. miLLM vendors it so the runtime consumer that BRD-MIS-JSPACE-001
sequences after miStudio (BRD-MILLM-JSPACE-001, watchlist evaluation at inference time) starts from a
frozen surface it can diff against.

Three guarantees, each failing for a different reason:

**Byte identity.** The vendored file equals miStudio's. If this fails, someone edited the copy instead
of the source, or miStudio republished the contract. Re-vendor by copying; never hand-edit. Skipped when
miStudio is not checked out (the public mirror), REQUIRED under `MILLM_REQUIRE_CROSS_REPO_CHECKS=1`,
which the `cross-repo-contract` CI job sets with miStudio checked out.

**The kinds are the four this repo has acknowledged.** A fifth kind, or a kind's version moving, has to
be read here deliberately (027 FTDD §5: "new kinds added deliberately").

**No hand-written mirror yet, so no consumer yet.** miLLM's other miStudio contracts have a pydantic mirror
in `millm/api/schemas/*.py`, and re-vendoring without updating it once dropped a field silently (suite
memory `circuit-contract-mirror-drift`). On 2026-10-20 miLLM has NO code that reads a J-space document,
so a mirror would be a model nothing calls. This test instead fails the day production code names one of
these kinds while `MIRRORED_KINDS` is empty for it: the consumer and its mirror (with the field-level
conformance check `test_probe_schema_sync.py` uses) must land together.

This file imports nothing from `millm` and needs pytest alone, so the `cross-repo-contract` job can run it
without the ML stack (`test_cross_repo_sync_files_need_only_pytest.py` holds that).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from tests.support.sibling_repos import REPO_ROOT, mistudio_repo

NAME = "jspace-contracts-v1.json"
VENDORED = REPO_ROOT / "docs" / "schemas" / NAME
MISTUDIO = mistudio_repo() / "docs" / "schemas" / NAME
REQUIRED = os.environ.get("MILLM_REQUIRE_CROSS_REPO_CHECKS") == "1"

#: The kinds this repo has read and acknowledged, by `$defs` name.
KINDS = {
    "JLensArtifactDocument": "mistudio.jlens-artifact/v1",
    "WorkspaceAnnotationDocument": "mistudio.workspace-annotation/v1",
    "ReadoutRecordDocument": "mistudio.jlens-readout/v1",
    "WatchlistDocument": "mistudio.jspace-watchlist/v1",
}

#: kind -> the `millm/` module holding its hand-written mirror. Empty until BRD-MILLM-JSPACE-001
#: adds a consumer; registering a kind here is the moment its field-level conformance test is owed.
MIRRORED_KINDS: dict[str, str] = {}


@pytest.fixture(scope="module")
def frozen() -> dict:
    assert VENDORED.is_file(), f"vendored J-space schema missing at {VENDORED}"
    return json.loads(VENDORED.read_text(encoding="utf-8"))


class TestTheVendoredCopyIsMiStudios:
    def test_it_is_byte_identical(self):
        if not MISTUDIO.is_file():
            if REQUIRED:
                pytest.fail(
                    f"MILLM_REQUIRE_CROSS_REPO_CHECKS=1 but miStudio's J-space schema is not at {MISTUDIO}"
                )
            pytest.skip(f"miStudio not checked out at {MISTUDIO} (set MISTUDIO_REPO)")
        assert VENDORED.read_bytes() == MISTUDIO.read_bytes(), (
            "docs/schemas/jspace-contracts-v1.json has drifted from miStudio's. Re-vendor by copying "
            "miStudio's file; never edit the copy."
        )

    def test_it_is_the_v1_jspace_contract(self, frozen):
        assert frozen["title"] == "miStudio J-space interchange kinds v1"
        assert frozen["$id"].endswith("/docs/schemas/jspace-contracts-v1.json")


class TestTheKindsAreTheAcknowledgedFour:
    def test_the_document_choices_are_exactly_these_models(self, frozen):
        assert sorted(alt["$ref"] for alt in frozen["oneOf"]) == sorted(
            f"#/$defs/{name}" for name in KINDS
        )

    @pytest.mark.parametrize("defname", sorted(KINDS))
    def test_each_kind_is_pinned_by_its_const(self, frozen, defname):
        """The version lives IN the kind: a consumer rejects an unknown kind rather than reading v2 as
        v1 with fields missing (027 FTDD §2). So the const is the version pin."""
        kind = frozen["$defs"][defname]["properties"]["kind"]
        assert kind.get("const") == KINDS[defname]
        assert defname in frozen["$defs"][defname].get("title", defname)


def _production_sources() -> list[Path]:
    root = REPO_ROOT / "millm"
    files = sorted(root.rglob("*.py"))
    assert len(files) > 50, f"found only {len(files)} files under {root}: the scan would assert nothing"
    return files


class TestNoConsumerWithoutAMirror:
    def test_production_code_names_no_unmirrored_jspace_kind(self):
        """A literal-string scan: it cannot see a kind assembled at runtime, which is recorded here
        rather than hidden. What it does catch is the ordinary way a consumer arrives (comparing a
        document's `kind` to the constant)."""
        unmirrored = {k for k in KINDS.values() if k not in MIRRORED_KINDS}
        pattern = re.compile("|".join(re.escape(k) for k in sorted(unmirrored)))
        hits = [
            f"{path.relative_to(REPO_ROOT)}: {m.group(0)}"
            for path in _production_sources()
            for m in pattern.finditer(path.read_text(encoding="utf-8"))
        ]
        assert not hits, (
            "production code reads a J-space kind that has no hand-written mirror. Add the mirror to "
            "millm/api/schemas/, register it in MIRRORED_KINDS with a field-level conformance test "
            f"(see test_probe_schema_sync.py), and only then consume it: {hits}"
        )

    def test_every_registered_mirror_module_exists(self):
        for kind, module in MIRRORED_KINDS.items():
            assert kind in KINDS.values(), kind
            assert (REPO_ROOT / module).is_file(), module
