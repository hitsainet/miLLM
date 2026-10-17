"""The checkpoint manifest contract and canonical naming (Feature 32, FTASKS 1.7).

Every rule is tested against the REAL vendored schema; scratch schemas are deep copies of it with
one change, so a fixture cannot agree with the code by construction.
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest

from millm.core.errors import (
    InvalidManifestError,
    InvalidRunIdError,
    ManifestFieldNotCallersError,
)
from millm.services import checkpoint_manifest as cm
from millm.services.rfc3339 import is_rfc3339

REPO = Path(__file__).resolve().parents[3]

SHA = "a" * 64


def caller_fields(**overrides):
    fields = {
        "run_id": "run-01",
        "step": 500,
        "domain": "finance",
        "signal_source": "corpus",
        "parent": None,
        "optimizer": {"name": "adamw-8bit", "state": "fresh"},
    }
    fields.update(overrides)
    return fields


def full_manifest(**overrides):
    doc = cm.build_manifest(
        caller_fields(), base_model="meta-llama/Llama-3.1-8B", weights_sha256=SHA,
        created_at="2026-10-13T12:00:00Z",
    )
    doc.update(overrides)
    return doc


def errors_of(exc_info) -> list[dict]:
    return exc_info.value.details["errors"]


# --------------------------------------------------------------------------- ownership


class TestOwnershipIsDerivedFromTheSchema:
    def test_the_split_is_what_the_vendored_schema_says(self):
        own = cm.ownership()
        assert own.caller == {
            "domain", "grader_hash", "merged_adapter", "optimizer", "parent", "retention",
            "run_id", "selector_hash", "signal_source", "step", "test",
        }
        assert own.writer == {"base_model", "created_at", "name", "weights_sha256"}
        assert own.caller_required == {
            "run_id", "step", "domain", "signal_source", "parent", "optimizer",
        }

    def test_it_is_a_partition_of_the_properties_minus_the_two_constants(self):
        props = cm.manifest_definition(cm.load_schema())["properties"]
        own = cm.ownership()
        assert own.caller.isdisjoint(own.writer)
        assert own.caller | own.writer | cm.UNMARKED == set(props)
        assert cm.UNMARKED <= set(props)

    def test_the_startup_assertion_passes_on_the_vendored_schema(self):
        assert cm.assert_schema_ownership() == cm.ownership()

    def test_an_undescribed_property_fails_the_startup_assertion(self):
        schema = cm.load_schema()
        cm.manifest_definition(schema)["properties"]["novel"] = {"type": "string"}
        with pytest.raises(cm.SchemaOwnershipError, match="novel"):
            cm.derive_ownership(schema)

    def test_a_writer_description_without_the_exact_sentence_fails(self):
        """A paraphrase (hyphen for the em dash) is not the sentence: a field must be marked by
        the schema's own words, or it is not miLLM's to write."""
        schema = cm.load_schema()
        prop = cm.manifest_definition(schema)["properties"]["created_at"]
        prop["description"] = prop["description"].replace("—", "-")
        with pytest.raises(cm.SchemaOwnershipError, match="created_at"):
            cm.derive_ownership(schema)

    def test_nested_descriptions_are_ignored(self):
        """`ScoreRecord.provider` also begins 'Written by the trainer.'; it is not a manifest
        field and must not enter either set."""
        schema = cm.load_schema()
        provider = schema["$defs"]["ScoreRecord"]["properties"]["provider"]
        assert provider["description"].startswith(cm.TRAINER)
        own = cm.derive_ownership(schema)
        assert "provider" not in own.caller | own.writer
        # And a nested property WITHOUT a mark does not fail the assertion either.
        schema["$defs"]["ScoreRecord"]["properties"]["unmarked"] = {"type": "string"}
        cm.derive_ownership(schema)

    def test_load_schema_returns_a_copy(self):
        a = cm.load_schema()
        a["$defs"].clear()
        assert cm.load_schema()["$defs"]

    def test_the_assertion_is_not_run_at_import(self):
        """032 review 5.13: nothing at module level calls the derivation."""
        tree = ast.parse((REPO / "millm/services/checkpoint_manifest.py").read_text())
        for node in tree.body:
            if isinstance(node, (ast.Expr, ast.Assign)):
                for call in ast.walk(node):
                    if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                        assert call.func.id not in {
                            "ownership", "assert_schema_ownership", "derive_ownership",
                        }


# --------------------------------------------------------------------------- naming


class TestCanonicalStep:
    @pytest.mark.parametrize(
        "step, path, name",
        [
            (5, "step-000005", "step-5"),
            (50, "step-000050", "step-50"),
            (500_000, "step-500000", "step-500000"),
            (1_000_000, "step-1000000", "step-1000000"),
        ],
    )
    def test_render_and_parse(self, step, path, name):
        assert cm.step_path_segment(step) == path
        assert cm.step_name_segment(step) == name
        rendered = cm.render_name("base", "run", step)
        assert rendered == f"base:run:{name}"
        assert cm.parse_name(rendered) == ("base", "run", step)

    def test_paths_sort_lexicographically_below_one_million(self):
        steps = [5, 50, 500, 999_999, 7, 70_000]
        assert sorted(cm.step_path_segment(s) for s in steps) == [
            cm.step_path_segment(s) for s in sorted(steps)
        ]

    def test_a_base_name_containing_colons_parses(self):
        assert cm.parse_name("hf:org/model:rev:run-1:step-40") == ("hf:org/model:rev", "run-1", 40)

    @pytest.mark.parametrize(
        "bad",
        ["run:step-5", "base::step-5", ":run:step-5", "base:run:step-", "base:run:step-x",
         "base:run:step-05", "base:run:step-²", "base:run:5", "base:run:step-5\n", ""],
    )
    def test_malformed_names_refused(self, bad):
        with pytest.raises(ValueError):
            cm.parse_name(bad)

    @pytest.mark.parametrize("bad", [-1, True, 1.0, "5"])
    def test_a_step_is_a_non_negative_integer(self, bad):
        with pytest.raises(ValueError):
            cm.step_path_segment(bad)

    def test_only_the_two_step_functions_build_step_strings(self):
        """FR-32.4.1: no other code in `millm/` formats a step (f-string, `+`, `%` or
        `.format` over a literal containing `step-`)."""
        allowed = {("millm/services/checkpoint_manifest.py", "step_path_segment"),
                   ("millm/services/checkpoint_manifest.py", "step_name_segment")}
        offenders = []
        for path in sorted((REPO / "millm").rglob("*.py")):
            rel = path.relative_to(REPO).as_posix()
            tree = ast.parse(path.read_text())
            parents = {}
            for node in ast.walk(tree):
                for child in ast.iter_child_nodes(node):
                    parents[child] = node

            def enclosing(node):
                while node in parents:
                    node = parents[node]
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        return node.name
                return None

            for node in ast.walk(tree):
                builds = False
                if isinstance(node, ast.JoinedStr):
                    builds = any(isinstance(v, ast.Constant) and "step-" in str(v.value)
                                 for v in node.values)
                elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod)):
                    builds = any(isinstance(side, ast.Constant) and isinstance(side.value, str)
                                 and "step-" in side.value for side in (node.left, node.right))
                elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                      and node.func.attr == "format" and isinstance(node.func.value, ast.Constant)
                      and "step-" in str(node.func.value.value)):
                    builds = True
                if builds and (rel, enclosing(node)) not in allowed:
                    offenders.append(f"{rel}:{node.lineno}")
        assert offenders == []


class TestRunIdRules:
    @pytest.mark.parametrize("good", ["r", "run-01", "A.b_c-d", "x" * 128])
    def test_save_accepts(self, good):
        assert cm.validate_run_id(good, mode="save") == good

    @pytest.mark.parametrize(
        "bad", ["a/b", "..", "../x", "a:b", "x" * 129, "", ".hidden", "-x", "run\n", "a b",
                None, 5],
    )
    def test_save_refuses(self, bad):
        with pytest.raises(InvalidRunIdError):
            cm.validate_run_id(bad, mode="save")

    @pytest.mark.parametrize("good", ["run_2026-10-13T12", "Run With Spaces", "-x", ".hidden",
                                      "x" * 200, "ünï"])
    def test_load_accepts_any_safe_segment(self, good):
        """miForge's run_id format is unpinned (D-22): a load must not refuse it for style."""
        assert cm.validate_run_id(good, mode="load") == good

    @pytest.mark.parametrize("bad", ["a/b", "..", ".", "", "a:b", "a\x00b", "../etc",
                                     "x" * 256, None])
    def test_load_refuses_traversal(self, bad):
        with pytest.raises(InvalidRunIdError):
            cm.validate_run_id(bad, mode="load")

    def test_the_name_limit_is_the_model_row_column(self):
        from millm.db.models.model import Model

        assert cm.model_name_limit() == Model.__table__.c.name.type.length == 255

    def test_a_name_too_long_for_the_row_is_refused_before_any_insert(self):
        base = "b" * 230
        assert cm.check_name_length(base, "r" * 10, 5) == f"{base}:{'r' * 10}:step-5"
        with pytest.raises(InvalidRunIdError) as exc:
            cm.check_name_length(base, "r" * 30, 5)
        assert exc.value.details["limit"] == 255


# --------------------------------------------------------------------------- save validation


class TestCallerFields:
    def test_valid_fields_pass(self):
        assert cm.validate_caller_fields(caller_fields()) == caller_fields()

    @pytest.mark.parametrize("field", sorted({"base_model", "created_at", "name",
                                              "weights_sha256"}))
    def test_every_writer_field_refused_with_its_name(self, field):
        with pytest.raises(ManifestFieldNotCallersError) as exc:
            cm.validate_caller_fields(caller_fields(**{field: "x"}))
        assert exc.value.details["fields"] == [field]

    def test_writer_fields_named_together_and_before_schema_errors(self):
        with pytest.raises(ManifestFieldNotCallersError) as exc:
            cm.validate_caller_fields({"weights_sha256": "x", "name": "y", "bogus": 1})
        assert exc.value.details["fields"] == ["name", "weights_sha256"]

    def test_a_missing_required_field_is_named_by_json_path(self):
        fields = caller_fields()
        del fields["domain"]
        with pytest.raises(InvalidManifestError) as exc:
            cm.validate_caller_fields(fields)
        assert [e["json_path"] for e in errors_of(exc)] == ["$"]
        assert "domain" in errors_of(exc)[0]["message"]

    def test_all_errors_reported_sorted_by_json_path(self):
        fields = caller_fields(step=-1, signal_source="nope",
                               optimizer={"name": "a", "state": "bad"}, zzz=1, aaa=2)
        with pytest.raises(InvalidManifestError) as exc:
            cm.validate_caller_fields(fields)
        paths = [e["json_path"] for e in errors_of(exc)]
        assert paths == sorted(paths)
        assert {"$.aaa", "$.zzz", "$.signal_source", "$.optimizer.state"} <= set(paths)

    def test_an_unknown_field_is_refused_on_save_and_named(self):
        with pytest.raises(InvalidManifestError) as exc:
            cm.validate_caller_fields(caller_fields(learning_rate=1e-4))
        assert errors_of(exc) == [
            {"json_path": "$.learning_rate",
             "message": "'learning_rate' is not a checkpoint manifest field"},
        ]

    def test_an_unknown_nested_field_is_refused(self):
        with pytest.raises(InvalidManifestError) as exc:
            cm.validate_caller_fields(
                caller_fields(optimizer={"name": "a", "state": "fresh", "lr": 1})
            )
        assert errors_of(exc)[0]["json_path"] == "$.optimizer"

    def test_constants_equal_to_the_schema_are_accepted_and_popped(self):
        """D-30: checked, then removed BEFORE the projection (whose properties are the caller's
        only, so an un-popped `kind` would be refused as unknown)."""
        out = cm.validate_caller_fields(
            caller_fields(kind="miforge.checkpoint-manifest", schema_version="1")
        )
        assert "kind" not in out and "schema_version" not in out

    @pytest.mark.parametrize("field, value", [("kind", "other"), ("schema_version", "2"),
                                              ("schema_version", 1)])
    def test_a_constant_unequal_to_the_schema_is_refused(self, field, value):
        with pytest.raises(InvalidManifestError) as exc:
            cm.validate_caller_fields(caller_fields(**{field: value}))
        assert [e["json_path"] for e in errors_of(exc)] == [f"$.{field}"]

    def test_the_strict_run_id_rule_applies_on_save(self):
        with pytest.raises(InvalidRunIdError):
            cm.validate_caller_fields(caller_fields(run_id="has space"))

    def test_not_an_object(self):
        with pytest.raises(InvalidManifestError):
            cm.validate_caller_fields(["run_id"])


class TestBuildAndFullValidation:
    def test_a_manifest_milm_builds_validates_and_is_ordered(self):
        doc = full_manifest()
        assert doc["kind"] == "miforge.checkpoint-manifest"
        assert doc["schema_version"] == "1"
        assert doc["name"] == "meta-llama/Llama-3.1-8B:run-01:step-500"
        assert doc["weights_sha256"] == SHA
        props = list(cm.manifest_definition(cm.load_schema())["properties"])
        assert list(doc) == [p for p in props if p in doc]
        assert cm.validate_full(doc) == doc

    def test_created_at_defaults_to_now_in_utc_with_z(self):
        doc = cm.build_manifest(caller_fields(), base_model="b", weights_sha256=SHA)
        assert doc["created_at"].endswith("Z") and is_rfc3339(doc["created_at"])

    @pytest.mark.parametrize(
        "bad",
        ["yesterday", "2026-10-13", "20261013T120000Z", "2026-10-13T12:00Z",
         "2026-10-13 12:00:00Z", "2026-10-13T12:00:00", "2026-13-01T00:00:00Z",
         "2026-02-30T00:00:00Z", "2026-10-13T12:00:00Z\n",
         # fromisoformat accepts a seconds offset; RFC 3339's offset is hours:minutes only, so
         # only a FULL match of the grammar refuses it (a prefix match would not).
         "2026-10-13T12:00:00+05:30:00"],
    )
    def test_a_malformed_created_at_fails(self, bad):
        """Control M7: without the custom checker jsonschema skips `date-time` entirely."""
        with pytest.raises(InvalidManifestError) as exc:
            cm.validate_full(full_manifest(created_at=bad))
        assert [e["json_path"] for e in errors_of(exc)] == ["$.created_at"]

    @pytest.mark.parametrize("good", ["2026-10-13T12:00:00Z", "2026-10-13t12:00:00.123z",
                                      "2026-10-13T12:00:00+05:30"])
    def test_rfc3339_forms_pass(self, good):
        cm.validate_full(full_manifest(created_at=good))

    def test_a_miforge_manifest_without_the_constants_validates_on_load(self):
        doc = full_manifest()
        del doc["kind"], doc["schema_version"]
        assert cm.validate_full(doc) == doc

    def test_an_unknown_key_is_refused_on_load_and_named(self):
        with pytest.raises(InvalidManifestError) as exc:
            cm.validate_full(full_manifest(trainer_notes="x", another=1))
        assert [e["json_path"] for e in errors_of(exc)] == ["$.another", "$.trainer_notes"]

    def test_a_wrong_constant_is_refused_on_load(self):
        with pytest.raises(InvalidManifestError):
            cm.validate_full(full_manifest(schema_version="2"))

    def test_a_missing_writer_field_is_refused_on_load(self):
        doc = full_manifest()
        del doc["weights_sha256"]
        with pytest.raises(InvalidManifestError):
            cm.validate_full(doc)

    def test_build_refuses_a_name_too_long_for_the_row(self):
        with pytest.raises(InvalidRunIdError):
            cm.build_manifest(caller_fields(), base_model="b" * 250, weights_sha256=SHA)

    def test_serialisation_is_utf8_indented_with_a_final_newline(self):
        doc = full_manifest(domain="finanças")
        raw = cm.serialise_manifest(doc)
        assert raw.endswith(b"}\n") and "finanças".encode() in raw
        import json

        assert json.loads(raw) == doc

    def test_the_projection_keeps_defs(self):
        projection = cm._projection()
        assert set(projection["$defs"]) == set(cm.load_schema()["$defs"])
        assert set(cm.manifest_definition(projection)["properties"]) == cm.ownership().caller
        # and the vendored schema itself was not modified by building it
        assert "name" in cm.manifest_definition(cm.load_schema())["properties"]

    def test_a_scratch_copy_is_not_the_cache(self):
        schema = copy.deepcopy(cm.load_schema())
        cm.manifest_definition(schema)["properties"].pop("name")
        assert "name" in cm.manifest_definition(cm.load_schema())["properties"]


def test_a_name_exactly_at_the_limit_fits_and_one_more_does_not():
    """R32 B43: the boundary of `check_name_length` (a name of 255 characters fits)."""
    import millm.services.checkpoint_manifest as cm
    from millm.core.errors import InvalidRunIdError

    run = "r" * 10
    base = "b" * (255 - len(f":{run}:step-5"))
    assert len(cm.check_name_length(base, run, 5)) == 255
    with pytest.raises(InvalidRunIdError):
        cm.check_name_length(base + "b", run, 5)
