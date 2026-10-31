"""The adapter manifest validator seam (`millm/services/adapter_manifest.py`), against the REAL
vendored schema (`docs/schemas/adapter-manifest-v1.json`, miForge 30648d4).

FTASKS 3.2's hand-made test schema is not needed: the contract is vendored, so every test here
validates against what production reads.
"""

from __future__ import annotations

import copy
import json

import pytest

from millm.core.errors import (
    AdapterInvalidError,
    AdapterManifestInvalidError,
    AdapterSchemaNotVendoredError,
)
from millm.services import adapter_manifest
from millm.services.adapter_manifest import (
    FIELDS_OPTIONAL_WITH_DEFAULT,
    FIELDS_REQUIRED,
    VENDORED,
    extract_fields,
    schema_validator,
    validate_manifest,
    vendored_validator,
)

HEX = "a" * 64


def manifest(**overrides) -> dict:
    m = {
        "kind": "miforge.adapter-manifest",
        "schema_version": "1",
        "name": "meta-llama/Llama-3.1-8B-Instruct:r9:adapter-200",
        "run_id": "r9",
        "step": 200,
        "base": {"name": "meta-llama/Llama-3.1-8B-Instruct", "manifest_sha256": None},
        "adapter": {"rank": 4, "alpha": 6.0, "target_modules": ["q_proj"],
                    "scaling": "alpha_over_r"},
        "weights_sha256": HEX,
        "created_at": "2026-10-14T12:00:00Z",
    }
    m.update(overrides)
    return m


def _delete(m: dict, dotted: str) -> None:
    *parents, leaf = dotted.split(".")
    node = m
    for p in parents:
        node = node[p]
    del node[leaf]


def paths_of(exc: AdapterManifestInvalidError) -> list[str]:
    return [e["path"] for e in exc.details["errors"]]


class TestTheVendoredValidator:
    def test_a_valid_manifest_has_no_errors(self):
        assert vendored_validator().errors(manifest()) == []

    @pytest.mark.parametrize(
        "dotted", list(FIELDS_REQUIRED) + ["kind", "schema_version", "created_at"]
    )
    def test_each_required_field_missing_is_refused_by_path(self, dotted):
        m = manifest()
        _delete(m, dotted)
        with pytest.raises(AdapterManifestInvalidError) as e:
            validate_manifest(m, vendored_validator())
        parent = "$" + "".join("." + p for p in dotted.split(".")[:-1])
        assert parent in paths_of(e.value)
        assert dotted.split(".")[-1] in json.dumps(e.value.details)

    @pytest.mark.parametrize(
        "field,value,path",
        [
            ("step", "200", "$.step"),
            ("step", -1, "$.step"),
            ("adapter", {"rank": "4", "alpha": 6.0, "target_modules": ["q"]}, "$.adapter.rank"),
            ("adapter", {"rank": 4, "alpha": 0, "target_modules": ["q"]}, "$.adapter.alpha"),
            ("adapter", {"rank": 4, "alpha": 6.0, "target_modules": []},
             "$.adapter.target_modules"),
            ("weights_sha256", "A" * 64, "$.weights_sha256"),
            ("base", {"name": 7}, "$.base.name"),
        ],
    )
    def test_wrong_types_are_refused_by_path(self, field, value, path):
        with pytest.raises(AdapterManifestInvalidError) as e:
            validate_manifest(manifest(**{field: value}), vendored_validator())
        assert path in paths_of(e.value)

    def test_an_unknown_top_level_key_is_refused_and_named(self):
        with pytest.raises(AdapterManifestInvalidError) as e:
            validate_manifest(manifest(extra=1), vendored_validator())
        assert paths_of(e.value) == ["$.extra"]

    def test_an_unknown_nested_key_is_refused_and_named(self):
        m = manifest()
        m["adapter"]["use_rslora"] = True
        with pytest.raises(AdapterManifestInvalidError) as e:
            validate_manifest(m, vendored_validator())
        assert paths_of(e.value) == ["$.adapter.use_rslora"]

    def test_an_unknown_scaling_rule_is_refused(self):
        m = manifest()
        m["adapter"]["scaling"] = "rslora"
        with pytest.raises(AdapterManifestInvalidError) as e:
            validate_manifest(m, vendored_validator())
        assert "$.adapter.scaling" in paths_of(e.value)

    @pytest.mark.parametrize("bad", ["not-a-date", "2026-10-14", "2026-13-01T00:00:00Z"])
    def test_created_at_is_actually_checked(self, bad):
        """D-29: jsonschema skips date-time without rfc3339-validator; ours is registered."""
        with pytest.raises(AdapterManifestInvalidError) as e:
            validate_manifest(manifest(created_at=bad), vendored_validator())
        assert paths_of(e.value) == ["$.created_at"]

    def test_every_problem_is_reported(self):
        m = manifest(step="x", extra=1)
        with pytest.raises(AdapterManifestInvalidError) as e:
            validate_manifest(m, vendored_validator())
        assert paths_of(e.value) == ["$.extra", "$.step"]

    def test_a_non_object_is_refused(self):
        with pytest.raises(AdapterManifestInvalidError):
            validate_manifest([1, 2], vendored_validator())


class TestTheSchemaIsReadAtCallTime:
    def test_an_absent_file_refuses(self, tmp_path):
        with pytest.raises(AdapterSchemaNotVendoredError) as e:
            vendored_validator(tmp_path / "absent.json")
        assert e.value.status_code == 409
        assert str(tmp_path) not in json.dumps(e.value.details)

    def test_the_default_path_is_read_now_not_at_import(self, monkeypatch, tmp_path):
        monkeypatch.setattr(adapter_manifest, "VENDORED", tmp_path / "absent.json")
        with pytest.raises(AdapterSchemaNotVendoredError):
            vendored_validator()

    def test_an_unreadable_schema_refuses(self, tmp_path):
        bad = tmp_path / "s.json"
        bad.write_text("{not json")
        with pytest.raises(AdapterSchemaNotVendoredError):
            vendored_validator(bad)

    def test_the_default_path_is_the_vendored_contract(self):
        assert VENDORED.name == "adapter-manifest-v1.json"
        assert VENDORED.parent.name == "schemas"
        assert VENDORED.is_file()

    def test_schema_validator_takes_an_explicit_schema(self):
        schema = json.loads(VENDORED.read_text())
        assert schema_validator(schema).errors(manifest()) == []


class TestFieldExtraction:
    def test_fields_and_scaling(self):
        f = extract_fields(manifest())
        assert (f.rank, f.alpha, f.scaling, f.scaling_rule) == (4, 6.0, 1.5, "alpha_over_r")
        assert f.target_modules == ("q_proj",)
        assert f.base_manifest_sha256 is None
        assert (f.name, f.run_id, f.step, f.weights_sha256) == (
            "meta-llama/Llama-3.1-8B-Instruct:r9:adapter-200", "r9", 200, HEX
        )

    def test_absent_scaling_takes_the_schema_default(self):
        m = manifest()
        del m["adapter"]["scaling"]
        validate_manifest(m, vendored_validator())
        f = extract_fields(m)
        assert f.scaling_rule == "alpha_over_r"
        assert f.scaling == 1.5

    def test_absent_base_hash_reads_as_none(self):
        m = manifest(base={"name": "x"})
        validate_manifest(m, vendored_validator())
        assert extract_fields(m).base_manifest_sha256 is None

    def test_an_unknown_scaling_is_refused_even_past_a_permissive_validator(self):
        m = manifest()
        m["adapter"]["scaling"] = "rslora"
        with pytest.raises(AdapterInvalidError) as e:
            extract_fields(m)
        assert e.value.details["reason"] == "scaling"


class TestTheFieldListsAreTheSyncTests:
    def test_they_agree_with_the_schema_sync_test(self):
        """The sync test pins its lists against the vendored schema; these must be the same
        lists, or the module reads a field the sync test never checked."""
        # The field half of the sync test moved here in 8067bf7 (tracker 0.9).
        from tests.unit.services import test_adapter_manifest_fields_match_schema as sync

        assert FIELDS_REQUIRED == sync.FIELDS_REQUIRED
        assert FIELDS_OPTIONAL_WITH_DEFAULT == sync.FIELDS_OPTIONAL_WITH_DEFAULT

    def test_extraction_reads_only_listed_fields(self):
        """Every field `extract_fields` indexes unconditionally is in FIELDS_REQUIRED: removing
        one optional field the module defaults must not raise."""
        m = manifest()
        for dotted in FIELDS_OPTIONAL_WITH_DEFAULT:
            _delete(m, dotted)
        extract_fields(copy.deepcopy(m))
