"""The probe mirror must stay conformant with the frozen contract, and the vendored copy
must stay byte-identical to miStudio's.

Two separate guarantees, and they fail for different reasons:

**Structural conformance** — every field the frozen schema defines exists on the pydantic mirror,
and every required field is still required. If this fails the mirror drifted; fix the mirror, never
the vendored file. v1 is frozen and a change needs a v2.

**Byte identity** — the vendored file equals miStudio's. If this fails, someone edited the copy
instead of the source, or miStudio republished the contract without a version bump. A contract that
two repos disagree about is not a contract, and the disagreement shows up as a probe that imports
here and scores differently there.

The byte check is skipped when miStudio is not checked out, and REQUIRED under
`MILLM_REQUIRE_CROSS_REPO_CHECKS=1`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.support.sibling_repos import mistudio_repo
from tests.unit.f28_fixtures import db  # noqa: F401 - fixture
from millm.api.schemas.probe import (
    AUTHORITATIVE_INPUT,
    BASES,
    DISTRIBUTIONS,
    HOOK_POINT,
    KIND,
    RULES,
    SCOPES,
    STREAMABLE_RULES,
    Aggregation,
    BehaviourRef,
    Decision,
    Evidence,
    ModelIdentity,
    ProbeDefinitionV1,
    ProbeHeadSpec,
    ProbeIdentity,
    ReadPoint,
    SaeReference,
    TestVector,
    TestVectors,
)

VENDORED = Path(__file__).resolve().parents[3] / "docs" / "schemas" / "probe-definition-v1.json"
MISTUDIO = mistudio_repo() / (
    "docs/schemas/probe-definition-v1.json"
)
REQUIRED = os.environ.get("MILLM_REQUIRE_CROSS_REPO_CHECKS") == "1"

#: frozen `$defs` name -> the mirror model that must carry its fields.
MIRRORED = {
    "ProbeDefinitionV1": ProbeDefinitionV1,
    "ModelIdentity": ModelIdentity,
    "ReadPoint": ReadPoint,
    "ProbeHead": ProbeHeadSpec,
    "Aggregation": Aggregation,
    "Decision": Decision,
    "Evidence": Evidence,
    "SaeReference": SaeReference,
    "TestVector": TestVector,
    "TestVectors": TestVectors,
    "BehaviourRef": BehaviourRef,
    "ProbeIdentity": ProbeIdentity,
}


@pytest.fixture
def valid_definition() -> dict:
    from tests.unit.probe_fixtures import probe_definition

    return probe_definition()


@pytest.fixture(scope="module")
def frozen() -> dict:
    assert VENDORED.exists(), f"vendored probe schema missing at {VENDORED}"
    return json.loads(VENDORED.read_text())


class TestTheVendoredCopyIsMiStudios:
    def test_it_is_byte_identical(self):
        if not MISTUDIO.exists():
            if REQUIRED:
                pytest.fail(
                    f"MILLM_REQUIRE_CROSS_REPO_CHECKS=1 but miStudio's schema is not at {MISTUDIO}"
                )
            pytest.skip(f"miStudio not checked out at {MISTUDIO}")
        assert VENDORED.read_bytes() == MISTUDIO.read_bytes(), (
            "the vendored contract has drifted from miStudio's. Re-vendor from miStudio; never "
            "edit the copy — the two repos would then disagree about what a probe means."
        )

    def test_the_vendored_file_is_the_v1_contract(self, frozen):
        assert frozen["title"] == "miStudio Probe Definition v1"
        assert frozen["$ref"].endswith("ProbeDefinitionV1")


class TestStructuralConformance:
    @pytest.mark.parametrize("defname", sorted(MIRRORED))
    def test_every_frozen_field_exists_on_the_mirror(self, frozen, defname):
        spec = frozen["$defs"][defname]
        mirror = MIRRORED[defname]
        missing = set(spec.get("properties", {})) - set(mirror.model_fields)
        assert not missing, f"{defname}: the mirror is missing {sorted(missing)}"

    @pytest.mark.parametrize("defname", sorted(MIRRORED))
    def test_required_stays_required(self, frozen, defname):
        spec = frozen["$defs"][defname]
        mirror = MIRRORED[defname]
        for name in spec.get("required", []):
            field = mirror.model_fields.get(name)
            assert field is not None, f"{defname}.{name} is required by the contract but absent"
            assert field.is_required(), (
                f"{defname}.{name} is required by the contract but optional on the mirror — a "
                f"document missing it would import and then fail at score time"
            )

    def test_additive_fields_survive_a_round_trip(self, valid_definition):
        """`extra='allow'` is the point: a newer miStudio may emit fields this build has never
        heard of, and stripping them would silently downgrade a document on re-export."""
        payload = dict(valid_definition)
        payload["some_future_field"] = {"added": "later"}
        parsed = ProbeDefinitionV1.model_validate(payload)
        assert parsed.model_dump()["some_future_field"] == {"added": "later"}


class TestEveryEnumMatchesTheContract:
    """⚠ ADDED AFTER THE TEST ABOVE FAILED TO CATCH A REAL BUG.

    The structural checks verify that a field EXISTS and stays required. They say nothing about
    what values it accepts — so `scope` was declared with miStudio's internal 032 vocabulary
    (`all | assistant | user | last_assistant`, lifted from a TypeScript type) while the contract's
    enum is `all | prompt | response`. The mirror would have rejected every real document carrying
    `prompt` or `response`, and accepted three values no valid document contains, with a green
    sync test.

    This walks every `enum` and `const` in the frozen schema and requires the mirror to agree.
    """

    def _enums(self, frozen: dict) -> dict[str, list]:
        found: dict[str, list] = {}
        for defname, spec in frozen["$defs"].items():
            for fieldname, field in (spec.get("properties") or {}).items():
                # ⚠ AN OPTIONAL ENUM IS WRAPPED IN `anyOf` ([{enum}, {type: null}]). Reading only
                # the top level missed two: `LengthBand.threshold_source` from the day it shipped,
                # and `ModelIdentity.load_dtype` on 2026-10-03 — so "this test must learn about
                # it rather than pass" was untrue of every optional enum in the contract.
                for alt in [field, *field.get("anyOf", [])]:
                    if "enum" in alt:
                        found[f"{defname}.{fieldname}"] = sorted(alt["enum"])
                        break
                    if "const" in alt:
                        found[f"{defname}.{fieldname}"] = [alt["const"]]
                        break
        return found

    def test_the_frozen_schema_still_has_the_enums_this_pins(self, frozen):
        """If the contract gains an enum, this test must learn about it rather than pass."""
        assert set(self._enums(frozen)) == {
            "ProbeDefinitionV1.kind",
            "ProbeDefinitionV1.scope",
            "ProbeDefinitionV1.basis",
            "ReadPoint.hook_point",
            "Aggregation.rule",
            "EvaluationEntry.distribution",
            "TestVectors.authoritative_input",
            "LengthBand.threshold_source",
            "ModelIdentity.load_dtype",
            "ModelIdentity.quantization",
        }

    def test_load_dtype_is_this_servers_precision_rule(self, frozen):
        """The precisions a definition may declare are exactly the ones this server's resolver
        produces — so a definition can never name a precision miLLM has no way to load at."""
        from millm.ml.native_dtype import LOAD_DTYPES

        assert sorted(LOAD_DTYPES) == self._enums(frozen)["ModelIdentity.load_dtype"]

    def test_threshold_source(self, frozen):
        """`length_bands_from_definition` distinguishes a cut band from an inherited one by these."""
        assert self._enums(frozen)["LengthBand.threshold_source"] == ["band", "global"]

    def test_scope(self, frozen):
        assert sorted(SCOPES) == self._enums(frozen)["ProbeDefinitionV1.scope"]

    def test_basis(self, frozen):
        assert sorted(BASES) == self._enums(frozen)["ProbeDefinitionV1.basis"]

    def test_rule(self, frozen):
        assert sorted(RULES) == self._enums(frozen)["Aggregation.rule"]

    def test_kind(self, frozen):
        assert [KIND] == self._enums(frozen)["ProbeDefinitionV1.kind"]

    def test_hook_point(self, frozen):
        assert [HOOK_POINT] == self._enums(frozen)["ReadPoint.hook_point"]

    def test_distribution(self, frozen):
        assert sorted(DISTRIBUTIONS) == self._enums(frozen)["EvaluationEntry.distribution"]

    def test_authoritative_input(self, frozen):
        assert [AUTHORITATIVE_INPUT] == self._enums(frozen)["TestVectors.authoritative_input"]


class TestTheConstantsMatchTheContract:
    def test_kind(self):
        assert KIND == "mistudio.probe-definition/v1"

    def test_hook_point_is_resid_post_only(self):
        assert HOOK_POINT == "resid_post"

    def test_rules_scopes_and_bases(self):
        assert RULES == ("mean", "max", "last", "softmax", "attention", "rolling_mean_max")
        assert SCOPES == ("all", "prompt", "response")
        assert BASES == ("residual", "sae_features")

    def test_last_is_the_only_non_streamable_rule(self):
        assert set(RULES) - STREAMABLE_RULES == {"last"}

    def test_the_rule_list_matches_the_runtime(self):
        """The mirror and the head must agree on what a rule is, or a definition validates and
        then cannot be scored."""
        from millm.ml.probe_head import RULES as RUNTIME_RULES, STREAMABLE as RUNTIME_STREAMABLE

        assert tuple(RULES) == tuple(RUNTIME_RULES)
        assert set(STREAMABLE_RULES) == set(RUNTIME_STREAMABLE)


class TestTheBehaviourFieldSurvives:
    """miStudio 037's additive `behaviour` (re-vendored 2026-10-21). Through the REAL import and
    re-export routes: the field must come back byte-for-byte and project `behaviour_key` (memory
    `circuit-contract-mirror-drift`: a mirror that forgets a field drops it on re-export)."""

    BEHAVIOUR = {"key": "pii", "name": "PII presence", "version": 3}

    def test_import_then_get_keeps_it_and_projects_the_key(self, db):  # noqa: F811
        from fastapi.testclient import TestClient

        from millm.api.dependencies import get_db
        from millm.main import create_app
        from tests.unit.probe_fixtures import probe_definition

        app = create_app()

        async def _db():
            async with db() as session:
                yield session

        app.dependency_overrides[get_db] = _db
        client = TestClient(app)
        doc = probe_definition(name="with-behaviour")
        doc["behaviour"] = dict(self.BEHAVIOUR)
        imported = client.post("/api/probes/import", json=doc).json()
        assert imported["success"], imported
        pid = imported["data"]["id"]
        detail = client.get(f"/api/probes/{pid}").json()["data"]
        assert detail["definition"]["behaviour"] == self.BEHAVIOUR
        assert detail["behaviour"] == self.BEHAVIOUR
        parsed = ProbeDefinitionV1.model_validate(detail["definition"])
        assert parsed.behaviour is not None
        assert parsed.model_dump(mode="json", exclude_none=False)["behaviour"] == self.BEHAVIOUR

    def test_an_uncatalogued_key_has_a_null_version(self):
        from tests.unit.probe_fixtures import probe_definition

        doc = probe_definition()
        doc["behaviour"] = {"key": "pii", "name": "pii"}
        assert ProbeDefinitionV1.model_validate(doc).behaviour.version is None

    def test_a_malformed_behaviour_is_refused(self):
        import pydantic

        from tests.unit.probe_fixtures import probe_definition

        for bad in ({"key": "PII", "name": "x"}, {"key": "pii"}, {"key": "pii", "name": "x",
                                                                   "archived": True}):
            doc = probe_definition()
            doc["behaviour"] = bad
            with pytest.raises(pydantic.ValidationError):
                ProbeDefinitionV1.model_validate(doc)


class TestTheIdentityBlockIsTyped:
    """miStudio 039's additive `identity` (re-vendored 2026-10-23): typed on the mirror, validated
    on import, and carried through `model_dump` unchanged (memory: circuit-contract-mirror-drift)."""

    IDENTITY = {"algorithm": "mistudio.probe-identity/1", "detector_digest": "a" * 64,
                "decision_digest": "b" * 64, "decision_revision": 3}

    def test_it_round_trips(self, valid_definition):
        doc = dict(valid_definition)
        doc["identity"] = dict(self.IDENTITY)
        parsed = ProbeDefinitionV1.model_validate(doc)
        assert isinstance(parsed.identity, ProbeIdentity)
        assert parsed.model_dump(mode="json")["identity"] == self.IDENTITY

    def test_a_document_without_it_still_imports(self, valid_definition):
        assert ProbeDefinitionV1.model_validate(dict(valid_definition)).identity is None

    @pytest.mark.parametrize("field,value", [("detector_digest", "XYZ"), ("decision_revision", 0),
                                             ("algorithm", "other/1")])
    def test_a_malformed_block_is_refused(self, valid_definition, field, value):
        from pydantic import ValidationError

        doc = dict(valid_definition)
        doc["identity"] = {**self.IDENTITY, field: value}
        with pytest.raises(ValidationError):
            ProbeDefinitionV1.model_validate(doc)
