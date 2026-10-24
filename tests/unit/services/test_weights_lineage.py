"""`WeightsCheckpointService.lineage` (032 FTASKS 8.3; review 5.8).

Mutation controls, each red in this file:
  M20 a missing manifest treated as complete  -> test_each_break_is_named[manifest_missing]
  M21 cycle detection removed                 -> test_each_break_is_named[cycle] (bounded by
                                                 the depth limit, so it fails, never hangs)
  lineage confinement removed                 -> test_parent_outside_the_root_is_a_break
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

from millm.core.config import settings
from millm.core.errors import ModelNotLoadedError
from millm.services import checkpoint_store
from millm.services.weights_checkpoint_service import WeightsCheckpointService
from tests.unit.checkpoint_service_fixtures import (  # noqa: F401 - harness is a fixture
    ATTACHMENT_ID,
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


async def _branch(h):
    """US-10: run-a step 1 -> run-a step 2 -> run-b step 3 (a branch), each saved by miLLM."""
    resident_llama()
    attach_holder(h.weights, h.clock)
    service = _service(h)
    s1 = await service.save(TRAINER, ATTACHMENT_ID, manifest_fields("run-a", 1))
    s2 = await service.save(TRAINER, ATTACHMENT_ID, manifest_fields("run-a", 2, parent=s1.name))
    s3 = await service.save(TRAINER, ATTACHMENT_ID, manifest_fields("run-b", 3, parent=s2.name))
    return s1, s2, s3


def _rewrite(h, run, step, **changes):
    path = h.root / run / f"step-{step:06d}" / "manifest.json"
    doc = json.loads(path.read_text())
    doc.update(changes)
    path.write_text(json.dumps(doc))


class TestChain:
    async def test_three_link_branch_in_order_without_reading_weights(self, harness):
        h = harness
        s1, s2, s3 = await _branch(h)
        with patch.object(checkpoint_store, "sha256_file",
                          side_effect=AssertionError("lineage hashed a file")) as spy, \
             patch("millm.services.weights_checkpoint_service.sha256_file", spy):
            body = await _service(h).lineage()
        assert spy.call_count == 0
        assert body["complete"] is True and body["break"] is None
        assert body["weights_verified"] is False
        assert body["base_model"] == RESIDENT_REPO
        assert [link["name"] for link in body["chain"]] == [s3.name, s2.name, s1.name]
        assert [link["path"] for link in body["chain"]] == [
            "run-b/step-000003", "run-a/step-000002", "run-a/step-000001"]
        assert [link["manifest_sha256"] for link in body["chain"]] == [
            s3.manifest_sha256, s2.manifest_sha256, s1.manifest_sha256]
        assert body["chain"][0]["weights_sha256"] == s3.weights_sha256
        assert body["chain"][-1]["parent"] is None
        assert body["resident"] == {"name": s3.name, "origin": "checkpoint",
                                    "manifest_sha256": s3.manifest_sha256,
                                    "commits_since_identity": 0}

    async def test_commits_since_identity_after_a_commit(self, harness):
        h = harness
        await _branch(h)
        attachment = h.weights._attachment
        await h.barrier.quiesce(attachment.digest, TRAINER)
        h.barrier.commit_begin(attachment.digest)
        h.barrier.commit_end(attachment.digest)
        body = await _service(h).lineage()
        assert body["resident"]["commits_since_identity"] == 1

    async def test_row_loaded_resident_has_an_empty_chain(self, harness):
        h = harness
        resident_llama()
        body = await _service(h).lineage()
        assert body["chain"] == [] and body["complete"] is True
        assert body["base_model"] == RESIDENT_REPO
        assert body["resident"]["origin"] == "model" and body["resident"]["name"] == "tiny-llama"

    async def test_no_model_is_refused(self, harness):
        with pytest.raises(ModelNotLoadedError):
            await _service(harness).lineage()

    async def test_walks_a_miforge_chain_across_a_read_root(self, harness, tmp_path, monkeypatch):
        """Review 6.3: lineage follows foreign manifests (no kind/schema_version)."""
        h = harness
        resident_llama()
        forge = tmp_path / "forge"
        first = write_foreign_checkpoint(forge, "house-sql", 200, tiny_llama(seed=1))
        parent = f"{RESIDENT_REPO}:house-sql:step-200"
        write_foreign_checkpoint(forge, "house-sql", 400, tiny_llama(seed=2), parent=parent)
        monkeypatch.setattr(settings, "WEIGHTS_CHECKPOINT_READ_ROOTS", str(forge))
        from millm.api.schemas.weights_checkpoints import LoadRequest

        await _service(h).load(TRAINER, LoadRequest(path="house-sql/latest"))
        body = await _service(h).lineage()
        assert [link["name"] for link in body["chain"]] == [
            f"{RESIDENT_REPO}:house-sql:step-400", parent]
        assert body["complete"] is True and body["chain"][1]["root"] == str(forge)
        assert first.is_dir()


class TestBreaks:
    @pytest.mark.parametrize("reason", [
        "manifest_missing", "manifest_invalid", "name_mismatch", "cycle", "depth_limit"])
    async def test_each_break_is_named(self, harness, monkeypatch, reason):
        h = harness
        s1, s2, s3 = await _branch(h)
        at = s2.name
        if reason == "manifest_missing":
            (h.root / "run-a" / "step-000002" / "manifest.json").unlink()
        elif reason == "manifest_invalid":
            (h.root / "run-a" / "step-000002" / "manifest.json").write_text("{not json")
        elif reason == "name_mismatch":
            _rewrite(h, "run-a", 2, name=f"{RESIDENT_REPO}:run-a:step-9")
        elif reason == "cycle":
            _rewrite(h, "run-a", 1, parent=s3.name)
            at = s3.name
        else:
            monkeypatch.setattr(settings, "WEIGHTS_LINEAGE_MAX_DEPTH", 1)
        body = await _service(h).lineage()
        assert body["complete"] is False
        assert body["break"] == {"at": at, "reason": reason}

    async def test_a_parent_run_that_does_not_exist_is_missing(self, harness):
        """M20: no root holds the parent's run at all."""
        h = harness
        s1, s2, s3 = await _branch(h)
        _rewrite(h, "run-a", 1, parent=f"{RESIDENT_REPO}:gone:step-1")
        body = await _service(h).lineage()
        assert body["complete"] is False
        assert body["break"] == {"at": f"{RESIDENT_REPO}:gone:step-1",
                                 "reason": "manifest_missing"}

    async def test_parent_outside_the_root_is_a_break(self, harness, tmp_path):
        """Review 5.8: a `parent` naming a traversal, or a run directory that is a link out of
        the root, never reads a manifest outside it."""
        h = harness
        s1, s2, s3 = await _branch(h)
        outside = tmp_path / "outside"
        write_foreign_checkpoint(outside, "evil", 1, tiny_llama())
        os.symlink(outside / "evil", h.root / "evil")
        _rewrite(h, "run-a", 1, parent=f"{RESIDENT_REPO}:evil:step-1")
        body = await _service(h).lineage()
        assert body["break"] == {"at": f"{RESIDENT_REPO}:evil:step-1",
                                 "reason": "manifest_invalid"}
        _rewrite(h, "run-a", 1, parent="x:../../etc:step-1")
        body = await _service(h).lineage()
        assert body["break"] == {"at": "x:../../etc:step-1", "reason": "manifest_invalid"}

    async def test_a_parent_run_under_two_roots_is_ambiguous(self, harness, tmp_path, monkeypatch):
        """D-21 inside the walk (R2-B): the parent's run exists under the primary root AND a
        read root, so the walk refuses to guess which manifest is the parent. Named
        `ambiguous`, at the parent; the links before it are kept."""
        h = harness
        s1, s2, s3 = await _branch(h)
        forge = tmp_path / "forge"
        write_foreign_checkpoint(forge, "run-a", 2, tiny_llama(seed=4))
        monkeypatch.setattr(settings, "WEIGHTS_CHECKPOINT_READ_ROOTS", str(forge))
        body = await _service(h).lineage()
        assert body["complete"] is False
        assert body["break"] == {"at": s2.name, "reason": "ambiguous"}
        assert [link["name"] for link in body["chain"]] == [s3.name]
