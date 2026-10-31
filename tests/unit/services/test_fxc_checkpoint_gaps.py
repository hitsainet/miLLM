"""Feature 032 requirement gaps left open by review RX (fix round FX-C), over the REAL service.

Harness: `tests/unit/checkpoint_service_fixtures.py` (real barrier, weight service, model service,
checkpoint store and resident identity; only the admission slot and the loader's CUDA `load` are
faked).

* FR-32.1.5 — `auto_map` files are copied into the step directory; a module that is not a bare
  file stem is refused (R32 F9's rule) and not copied, even when the file exists inside the source.
* FR-32.2.2 — `weights_state` is read UNDER the pause. A tear that lands while the save waits for
  the admission slot refuses the save; with `require_clean` moved above the pause the save would
  snapshot torn weights.
* FR-32.3.8 — a caller `parent` that disagrees with the resident's identity is logged
  (`manifest_parent_mismatch`) and reported, never refused; a match logs nothing.
* FR-32.4.4 — `base_model` of a save over a checkpoint-identity resident is the identity's base,
  not the carrier row's repo id.
* FR-32.7.3 — every lineage link reports `run_id`, `step` and `merged_adapter` as recorded.

Mutation controls (each run against this file, restored, `git diff` verified clean):
  C1  `shutil.copy2(...)` of auto_map files removed            -> copy test red
  C2  the bare-stem rule reduced to NUL only                   -> refusal test red
  C3  `barrier.require_clean("save")` moved above the pause     -> FR-32.2.2 test red
  C4  `manifest_parent_mismatch` warning removed               -> FR-32.3.8 test red
  C5  `_resident_base`'s identity branch removed               -> FR-32.4.4 test red
  C6  lineage link `run_id` / `step` / `merged_adapter` -> None -> FR-32.7.3 test red (each)
"""

from __future__ import annotations

import json

import pytest
from structlog.testing import capture_logs

from millm.api.schemas.weights_checkpoints import LoadRequest
from millm.core.errors import WeightsNotCleanError
from millm.services.resident_identity import identity_of
from millm.services.weights_checkpoint_service import WeightsCheckpointService
from tests.unit.checkpoint_service_fixtures import (  # noqa: F401 - harness is a fixture
    ATTACHMENT_ID,
    RESIDENT_NAME,
    RESIDENT_REPO,
    TRAINER,
    attach_holder,
    harness,
    manifest_fields,
    resident_llama,
    write_foreign_checkpoint,
)
from tests.unit.weights_fixtures import tiny_llama


def _service(h) -> WeightsCheckpointService:
    return WeightsCheckpointService(h.models)


def _manifest(h, relative: str) -> dict:
    return json.loads((h.root / relative / "manifest.json").read_text())


# ----------------------------------------------------------------------------- FR-32.1.5


def _remote_code_source(tmp_path):
    """The row's cache directory holding a top-level module and a nested one."""
    source = tmp_path / "base"
    (source / "sub").mkdir(parents=True)
    (source / "modeling_x.py").write_text("# top-level remote code\n")
    (source / "sub" / "mod.py").write_text("# nested remote code\n")
    return source


async def test_save_copies_the_auto_map_module_into_the_step_directory(harness, tmp_path):
    h = harness
    _remote_code_source(tmp_path)
    loaded = resident_llama()
    loaded.model.config.auto_map = {
        "AutoConfig": "modeling_x.XConfig",
        "AutoModelForCausalLM": "org/repo--modeling_x.XForCausalLM",
    }
    attach_holder(h.weights, h.clock)
    saved = await _service(h).save(TRAINER, ATTACHMENT_ID, manifest_fields())
    step = h.root / saved.path
    copied = step / "modeling_x.py"
    assert copied.is_file(), sorted(p.name for p in step.iterdir())
    assert copied.read_text() == "# top-level remote code\n"
    assert json.loads((step / "config.json").read_text())["auto_map"]["AutoConfig"] == (
        "modeling_x.XConfig")


async def test_save_refuses_an_auto_map_module_that_is_not_a_bare_stem(harness, tmp_path):
    """`sub/mod` exists INSIDE the source, so confinement passes; only the bare-stem rule
    refuses it (transformers resolves one top-level file)."""
    h = harness
    _remote_code_source(tmp_path)
    loaded = resident_llama()
    loaded.model.config.auto_map = {"AutoModel": "sub/mod.Model",
                                    "AutoConfig": "modeling_x.XConfig"}
    attach_holder(h.weights, h.clock)
    with capture_logs() as logs:
        saved = await _service(h).save(TRAINER, ATTACHMENT_ID, manifest_fields())
    names = sorted(p.name for p in (h.root / saved.path).iterdir())
    assert "mod.py" not in names and "modeling_x.py" in names
    refused = [e for e in logs if e["event"] == "checkpoint_auto_map_module_refused"]
    assert [e["module"] for e in refused] == ["sub/mod"]


# ----------------------------------------------------------------------------- FR-32.2.2


async def test_a_tear_while_the_save_waits_for_the_slot_refuses_the_save(harness):
    """The state is read once the pause HOLDS the slot: a tear landing in the wait refuses."""
    h = harness
    resident_llama()
    attach_holder(h.weights, h.clock)
    h.inference.on_take = lambda: h.barrier.mark_torn("test_tear_during_slot_wait")
    with pytest.raises(WeightsNotCleanError) as info:
        await _service(h).save(TRAINER, ATTACHMENT_ID, manifest_fields())
    assert info.value.details == {"operation": "save", "weights_state": "torn"}
    assert not (h.root / "run").exists(), "a torn set was snapshotted and written"
    assert h.barrier.snapshot().last_save is None


# ----------------------------------------------------------------------------- FR-32.3.8


async def test_a_parent_mismatch_is_logged_and_reported_not_refused(harness):
    h = harness
    resident_llama()
    attach_holder(h.weights, h.clock)
    service = _service(h)
    with capture_logs() as logs:
        first = await service.save(TRAINER, ATTACHMENT_ID, manifest_fields("run", 1))
    assert first.parent_check == "match"
    assert not [e for e in logs if e["event"] == "manifest_parent_mismatch"]

    with capture_logs() as logs:
        second = await service.save(
            TRAINER, ATTACHMENT_ID, manifest_fields("run", 2, parent="org/other:run:step-9"))
    assert second.parent_check == "mismatch"
    lines = [e for e in logs if e["event"] == "manifest_parent_mismatch"]
    assert len(lines) == 1
    line = lines[0]
    assert line["log_level"] == "warning"
    assert (line["run_id"], line["step"]) == ("run", 2)
    assert line["parent"] == "org/other:run:step-9"
    assert line["expected_parent"] == first.name
    # Written as the caller wrote it (not refused, not corrected).
    assert _manifest(h, second.path)["parent"] == "org/other:run:step-9"


# ----------------------------------------------------------------------------- FR-32.4.4


async def test_a_save_over_a_checkpoint_identity_keeps_the_identitys_base(harness):
    """The resident was loaded in place from a checkpoint whose `base_model` is the ROW NAME;
    the row's repo id differs. The next save's base is the identity's, not the row's."""
    h = harness
    resident_llama()
    foreign = write_foreign_checkpoint(h.root, "forge", 3, tiny_llama(seed=4),
                                       base_model=RESIDENT_NAME)
    foreign_name = f"{RESIDENT_NAME}:forge:step-3"
    loaded_result = await _service(h).load(TRAINER, LoadRequest(name=foreign_name))
    assert loaded_result.mode == "in_place"
    from millm.ml.model_loader import LoadedModelState

    identity = identity_of(LoadedModelState().current)
    assert identity.base_model == RESIDENT_NAME and identity.base_model != RESIDENT_REPO

    attach_holder(h.weights, h.clock)
    saved = await _service(h).save(
        TRAINER, ATTACHMENT_ID, manifest_fields("next", 1, parent=foreign_name))
    assert saved.parent_check == "match"
    assert saved.name == f"{RESIDENT_NAME}:next:step-1"
    assert _manifest(h, saved.path)["base_model"] == RESIDENT_NAME
    assert (await _service(h).lineage())["base_model"] == RESIDENT_NAME
    assert foreign.is_dir()


# ----------------------------------------------------------------------------- FR-32.7.3


async def test_each_lineage_link_reports_run_id_step_and_merged_adapter(harness):
    h = harness
    resident_llama()
    attach_holder(h.weights, h.clock)
    service = _service(h)
    s1 = await service.save(TRAINER, ATTACHMENT_ID, manifest_fields("run-a", 1))
    s2 = await service.save(TRAINER, ATTACHMENT_ID, manifest_fields("run-a", 2, parent=s1.name))
    merged = {"name": "org/tiny-llama:adapter-x", "step": 40, "manifest_sha256": "d" * 64}
    path = h.root / s1.path / "manifest.json"
    doc = json.loads(path.read_text())
    doc["merged_adapter"] = merged
    path.write_text(json.dumps(doc))

    body = await service.lineage()
    assert body["complete"] is True
    chain = body["chain"]
    assert [(link["run_id"], link["step"]) for link in chain] == [("run-a", 2), ("run-a", 1)]
    assert chain[0]["merged_adapter"] is None
    assert chain[1]["merged_adapter"] == merged
    assert chain[0]["name"] == s2.name and chain[0]["parent"] == s1.name
