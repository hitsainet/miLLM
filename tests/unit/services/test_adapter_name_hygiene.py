"""An adapter may not be served under a base's name, or under an unservable one (R-01.27 bullet 4).

Refusal: `ADAPTER_MANIFEST_INVALID` with `details.reason == "name"`, `details.problem` one of
`empty`, `not_printable`, `base_name`, `registered_name`, raised before any tensor exists and
before any pause (it is a `prepare_adapter` refusal). The manifest tool refuses the same empty /
not-printable names and a name equal to `--base-name`.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from millm.core.errors import AdapterManifestInvalidError
from millm.ml.adapter_hooks import LoraHookSet
from millm.services.adapter_artifact import (
    ResidentBase,
    name_problem,
    prepare_adapter,
    require_adapter_name,
    resident_base,
)
from millm.tools.write_adapter_manifest import ManifestToolError, build_manifest
from tests.unit.adapter_fixtures import BASE_NAME, tiny_llama, write_adapter
from tests.unit.tools.test_write_adapter_manifest import write_peft_dir

ROW = "Llama-3.1-8B-Instruct"
SERVED = "llama-ft:run-7:step-900"
RESIDENT = ResidentBase(
    manifest_sha256=None, names=frozenset({ROW, BASE_NAME}),
    served_names=frozenset({ROW, BASE_NAME, SERVED}),
)


@pytest.fixture(scope="module")
def llama():
    return tiny_llama()


def refuse(name, resident=RESIDENT, registered=()):
    with pytest.raises(AdapterManifestInvalidError) as e:
        require_adapter_name(name, resident, registered)
    assert e.value.details["reason"] == "name"
    assert e.value.details["path"] == "$.name"
    return e.value.details["problem"]


@pytest.mark.parametrize("name", ["", None])
def test_an_empty_name_is_refused(name):
    assert refuse(name) == "empty"


@pytest.mark.parametrize("name", ["a\nb", "a b", "a\tb", "a\x1b[31mb", "a b", " a", "a\x00"])
def test_whitespace_and_control_characters_are_refused(name):
    assert refuse(name) == "not_printable"


@pytest.mark.parametrize("name", [ROW, BASE_NAME, SERVED])
def test_each_name_of_the_resident_base_is_refused(name):
    assert refuse(name) == "base_name"


def test_another_registered_row_is_refused():
    assert refuse("other-model", registered=["other-model", "x"]) == "registered_name"


def test_an_ordinary_name_passes():
    require_adapter_name(f"{BASE_NAME}:r9:adapter-200", RESIDENT, ["other-model"])
    assert name_problem("a:b:adapter-1") is None


def test_a_name_longer_than_the_served_name_columns_is_refused():
    """R2-C: the served name is written to `probe_events.served_model` (String(255), migration
    022) on every probe event while the adapter is applied; a longer name makes every insert fail
    on PostgreSQL and monitoring goes quiet. Refused at load. Control: drop the length check ->
    no refusal."""
    from millm.services.adapter_artifact import MAX_ADAPTER_NAME_CHARS

    assert refuse("a" * (MAX_ADAPTER_NAME_CHARS + 1)) == "too_long"
    require_adapter_name("a" * MAX_ADAPTER_NAME_CHARS, RESIDENT)
    # Characters, not bytes: PostgreSQL's VARCHAR(n) counts characters.
    require_adapter_name("é" * MAX_ADAPTER_NAME_CHARS, RESIDENT)


def test_the_name_bound_is_the_width_of_every_column_the_served_name_is_written_to():
    from millm.db.models.model import Model
    from millm.db.models.probe import ProbeEvent
    from millm.services.adapter_artifact import MAX_ADAPTER_NAME_CHARS

    assert ProbeEvent.__table__.c.served_model.type.length == MAX_ADAPTER_NAME_CHARS
    assert Model.__table__.c.name.type.length == MAX_ADAPTER_NAME_CHARS


def test_unicode_letters_and_slashes_are_servable_names():
    """No false refusals for names a Hub base produces: '/' and non-ASCII letters."""
    for name in ("meta-llama/Llama-3.1-8B-Instruct:r9:adapter-200", "modèle/日本語:r:adapter-1"):
        require_adapter_name(name, RESIDENT, ["other-model"])


def test_resident_base_collects_every_served_name():
    row = SimpleNamespace(name=ROW, repo_id=BASE_NAME, source="huggingface", local_path=None)
    loaded = SimpleNamespace(model_name="loaded-name",
                             weights_identity=SimpleNamespace(name=SERVED, manifest_sha256=None))
    resident = resident_base(row, loaded)
    assert resident.served_names == {ROW, BASE_NAME, "loaded-name", SERVED}
    assert resident.names == {ROW, BASE_NAME}


class TestInPrepare:
    def _prepare(self, tmp_path, llama, name, monkeypatch, **kw):
        root = tmp_path / "checkpoints"
        write_adapter(root / "a", llama, manifest_overrides={"name": name})
        built = []
        real = LoraHookSet.build
        monkeypatch.setattr(LoraHookSet, "build",
                            staticmethod(lambda *a, **k: built.append(1) or real(*a, **k)))
        try:
            return prepare_adapter("a", llama, RESIDENT, roots=[("primary", str(root))], **kw)
        finally:
            self.built = built

    @pytest.mark.parametrize("name", [BASE_NAME, ROW, SERVED, "", "x\ny"])
    def test_refused_before_any_tensor(self, tmp_path, llama, monkeypatch, name):
        with pytest.raises(AdapterManifestInvalidError) as e:
            self._prepare(tmp_path, llama, name, monkeypatch)
        assert e.value.details["reason"] == "name"
        assert self.built == []

    def test_registered_names_are_passed_through(self, tmp_path, llama, monkeypatch):
        with pytest.raises(AdapterManifestInvalidError) as e:
            self._prepare(tmp_path, llama, "other-model", monkeypatch,
                          registered_names=["other-model"])
        assert e.value.details["problem"] == "registered_name"

    def test_a_good_name_prepares(self, tmp_path, llama, monkeypatch):
        art = self._prepare(tmp_path, llama, f"{BASE_NAME}:r9:adapter-200", monkeypatch)
        assert art.fields.name == f"{BASE_NAME}:r9:adapter-200"


class TestTheTool:
    def _build(self, tmp_path: Path, **kw):
        d = write_peft_dir(tmp_path / "a")
        args = dict(base_name=BASE_NAME, base_manifest_sha256=None, run_id=None, step=0, name=None)
        args.update(kw)
        return build_manifest(d, **args)

    @pytest.mark.parametrize("name", ["", "a b", "a\nb", "a\x07", "a" * 256])
    def test_an_unservable_name_is_refused(self, tmp_path, name):
        with pytest.raises(ManifestToolError, match="cannot be served"):
            self._build(tmp_path, name=name)

    def test_the_base_name_as_adapter_name_is_refused(self, tmp_path):
        with pytest.raises(ManifestToolError, match="base's name"):
            self._build(tmp_path, name=BASE_NAME)

    @pytest.mark.parametrize("base", [" " + BASE_NAME, BASE_NAME + "\n", "a\x07b"])
    def test_a_base_name_with_whitespace_is_refused(self, tmp_path, base):
        # An explicit, valid adapter name, so only the --base-name check can refuse.
        with pytest.raises(ManifestToolError, match="--base-name"):
            self._build(tmp_path, base_name=base, name="good:r:adapter-1")

    def test_the_default_name_passes(self, tmp_path):
        assert self._build(tmp_path)["name"] == f"{BASE_NAME}:a:adapter-0"
