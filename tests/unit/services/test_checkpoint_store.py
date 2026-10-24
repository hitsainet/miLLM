"""Durable checkpoint storage (Feature 32, FTASKS 2.7): atomic publish, confinement, hashing,
roots and startup cleanup, all over a real `tmp_path`."""

from __future__ import annotations

import errno
import hashlib
import os
import uuid
from pathlib import Path

import pytest
import torch
from structlog.testing import capture_logs

from millm.core.config import settings
from millm.core.errors import (
    CheckpointAmbiguousError,
    CheckpointExistsError,
    CheckpointNotFoundError,
    CheckpointWriteFailedError,
    InsufficientStorageError,
    InvalidCheckpointPathError,
    InvalidRunIdError,
)
from millm.services import checkpoint_store as cs


@pytest.fixture()
def roots(tmp_path, monkeypatch):
    primary = tmp_path / "primary"
    forge = tmp_path / "forge"
    primary.mkdir()
    forge.mkdir()
    monkeypatch.setattr(settings, "WEIGHTS_CHECKPOINT_DIR", str(primary))
    monkeypatch.setattr(settings, "WEIGHTS_CHECKPOINT_READ_ROOTS", f"{forge}, ,{primary}")
    return primary, forge


def write_files(files: dict[str, bytes]):
    def write(tmp: Path):
        for name, data in files.items():
            (tmp / name).write_bytes(data)
        return sorted(files)
    return write


def make_step(root: Path, run_id: str, step: int) -> Path:
    path = root / run_id / cs.step_path_segment(step)
    path.mkdir(parents=True)
    return path


# --------------------------------------------------------------------------- roots


class TestRoots:
    def test_primary_then_read_roots_without_the_primary_or_blanks(self, roots):
        primary, forge = roots
        assert cs.checkpoint_root() == primary
        assert cs.read_roots() == [forge]
        assert cs.checkpoint_roots() == [primary, forge]

    def test_a_run_in_one_root_is_found_there(self, roots):
        primary, forge = roots
        make_step(forge, "forge-run", 5)
        assert cs.find_run_root("forge-run") == forge
        assert cs.find_run_root("absent") is None

    def test_a_run_in_two_roots_is_ambiguous(self, roots):
        primary, forge = roots
        make_step(primary, "dup", 5)
        make_step(forge, "dup", 7)
        with pytest.raises(CheckpointAmbiguousError) as exc:
            cs.resolve_checkpoint("dup", 5)
        assert exc.value.details["roots"] == [str(primary), str(forge)]

    def test_resolve_checkpoint_and_by_name_across_roots(self, roots):
        _, forge = roots
        step = make_step(forge, "forge-run", 500)
        assert cs.resolve_checkpoint("forge-run", 500) == (forge, step)
        assert cs.resolve_by_name("org/m:x:forge-run:step-500") == (forge, step)
        with pytest.raises(CheckpointNotFoundError):
            cs.resolve_checkpoint("forge-run", 400)
        with pytest.raises(CheckpointNotFoundError):
            cs.resolve_by_name("not-a-name")

    @pytest.mark.parametrize("parent", ["x:../../etc:step-1", "x:..:step-1", "x:a/b:step-1"])
    def test_traversal_in_a_parent_name_is_refused_before_any_read(self, roots, parent):
        """032 review 5.8: lineage feeds an untrusted `parent` here."""
        with pytest.raises(InvalidRunIdError):
            cs.resolve_by_name(parent)


# --------------------------------------------------------------------------- confinement


class TestConfinement:
    def test_relative_and_absolute_inside_are_accepted(self, roots):
        primary, _ = roots
        step = make_step(primary, "r", 5)
        assert cs.resolve_inside_root(primary, "r/step-000005", expect="dir") == step
        assert cs.resolve_inside_root(primary, step, expect="dir") == step

    @pytest.mark.parametrize("bad", ["../x", "r/../../x", "/etc", "/etc/passwd", "r/\x00"])
    def test_outside_refused_without_echoing_the_resolved_path(self, roots, bad):
        primary, _ = roots
        with pytest.raises(InvalidCheckpointPathError) as exc:
            cs.resolve_inside_root(primary, bad)
        assert exc.value.details == {"input": bad}

    def test_a_latest_link_escaping_the_root_is_refused(self, roots, tmp_path):
        primary, _ = roots
        outside = tmp_path / "outside"
        outside.mkdir()
        (primary / "r").mkdir()
        os.symlink(os.path.relpath(outside, primary / "r"), primary / "r" / "latest")
        with pytest.raises(InvalidCheckpointPathError):
            cs.resolve_inside_root(primary, "r/latest", expect="dir")

    def test_a_latest_link_inside_resolves(self, roots):
        primary, _ = roots
        step = make_step(primary, "r", 5)
        cs.replace_latest(primary / "r", step.name)
        assert cs.resolve_inside_root(primary, "r/latest", expect="dir") == step

    def test_a_fifo_is_not_a_regular_file(self, roots):
        primary, _ = roots
        step = make_step(primary, "r", 5)
        os.mkfifo(step / "model.safetensors")
        with pytest.raises(InvalidCheckpointPathError) as exc:
            cs.resolve_inside_root(primary, step / "model.safetensors", expect="file")
        assert exc.value.details["reason"] == "not_regular_file"

    def test_missing_is_not_found(self, roots):
        primary, _ = roots
        with pytest.raises(CheckpointNotFoundError):
            cs.resolve_inside_root(primary, "r/step-000005", expect="dir")

    def test_resolve_path_finds_the_root(self, roots):
        primary, forge = roots
        step = make_step(forge, "f", 5)
        assert cs.resolve_path(step) == (forge, step)
        assert cs.resolve_path("f/step-000005") == (forge, step)
        with pytest.raises(InvalidCheckpointPathError):
            cs.resolve_path("../forge/f/step-000005")
        with pytest.raises(InvalidCheckpointPathError):
            cs.resolve_path("/etc")
        make_step(primary, "f", 6)
        with pytest.raises(CheckpointAmbiguousError):
            cs.resolve_path("f/step-000005")


# --------------------------------------------------------------------------- sizing, hashing


class TestSizingAndHashing:
    def test_sha256_bytes_matches_hashlib_and_sha256_file(self, tmp_path):
        import hashlib

        data = b"x" * 1000 + b"tail"
        (tmp_path / "f").write_bytes(data)
        assert cs.sha256_bytes(data) == hashlib.sha256(data).hexdigest()
        assert cs.sha256_bytes(data) == cs.sha256_file(tmp_path / "f")
        assert cs.sha256_bytes(b"") == hashlib.sha256(b"").hexdigest()

    def test_estimate_counts_a_tied_tensor_once_plus_the_overhead(self, monkeypatch):
        model = torch.nn.Module()
        model.embed = torch.nn.Embedding(10, 4)
        model.head = torch.nn.Linear(4, 10, bias=False)
        model.head.weight = model.embed.weight
        model.register_buffer("scale", torch.ones(3, dtype=torch.float16))
        model.register_buffer("rope", torch.ones(100), persistent=False)
        monkeypatch.setattr(settings, "WEIGHTS_SAVE_OVERHEAD_MB", 2)
        assert cs.estimate_bytes(model) == 10 * 4 * 4 + 3 * 2 + 2 * 1024 * 1024
        assert cs.estimate_bytes(model, overhead_mb=0) == 166

    def test_free_space_refusal_names_both_figures(self, roots, monkeypatch):
        primary, _ = roots
        monkeypatch.setattr(settings, "WEIGHTS_SAVE_FREE_SPACE_FACTOR", 1.1)
        monkeypatch.setattr(cs.shutil, "disk_usage",
                            lambda p: type("U", (), {"free": 1000})())
        cs.check_free_space(primary, 909)
        with pytest.raises(InsufficientStorageError) as exc:
            cs.check_free_space(primary / "not" / "yet", 1000)
        assert "1100" in exc.value.message and "1000 bytes are free" in exc.value.message
        assert exc.value.details == {"need_bytes": 1000, "free_bytes": 1000, "factor": 1.1,
                                     "required_bytes": 1100}

    @pytest.mark.parametrize("size", [0, 1, 8 * 1024 * 1024, 8 * 1024 * 1024 + 1])
    def test_sha256_file_hashes_the_whole_file(self, tmp_path, size):
        """Control M3: dropping the final partial chunk changes the 1-byte and 8 MiB+1 cases."""
        data = os.urandom(size)
        path = tmp_path / "f"
        path.write_bytes(data)
        assert cs.sha256_file(path) == hashlib.sha256(data).hexdigest()
        assert cs.sha256_file(path, chunk=3) == hashlib.sha256(data).hexdigest()

    def test_fingerprint_changes_on_replace(self, tmp_path):
        path = tmp_path / "f"
        path.write_bytes(b"a")
        before = cs.FileFingerprint.of(path)
        replacement = tmp_path / "g"
        replacement.write_bytes(b"a")
        os.replace(replacement, path)
        assert cs.FileFingerprint.of(path) != before


# --------------------------------------------------------------------------- publish


def no_temp(run_dir: Path) -> bool:
    return not [p for p in run_dir.iterdir() if ".tmp-" in p.name]


class TestAtomicPublish:
    def test_a_save_publishes_the_step_and_points_latest_at_it(self, roots):
        primary, _ = roots
        published = cs.write_checkpoint_atomically(
            primary, "run", 5, write_files({"model.safetensors": b"w", "manifest.json": b"{}"})
        )
        final = primary / "run" / "step-000005"
        assert published.path == final and published.written == ["manifest.json",
                                                                  "model.safetensors"]
        assert (final / "model.safetensors").read_bytes() == b"w"
        assert os.readlink(primary / "run" / "latest") == "step-000005"  # relative
        assert no_temp(primary / "run")

    def test_the_temp_dir_is_named_for_the_step_and_inside_the_run(self, roots):
        primary, _ = roots
        seen = []
        cs.write_checkpoint_atomically(primary, "run", 7, lambda tmp: seen.append(tmp))
        assert seen[0].parent == primary / "run"
        assert cs._STEP_TMP.fullmatch(seen[0].name)
        assert seen[0].name.startswith(".step-000007.tmp-")

    def test_a_failure_while_writing_leaves_nothing_and_latest_unchanged(self, roots):
        """Control M4 (no temporary directory): a partial step directory would be published."""
        primary, _ = roots
        cs.write_checkpoint_atomically(primary, "run", 5, write_files({"a": b"1"}))

        def boom(tmp: Path):
            (tmp / "model.safetensors").write_bytes(b"partial")
            raise RuntimeError("disk on fire")

        with pytest.raises(RuntimeError):
            cs.write_checkpoint_atomically(primary, "run", 10, boom)
        assert not (primary / "run" / "step-000010").exists()
        assert no_temp(primary / "run")
        assert os.readlink(primary / "run" / "latest") == "step-000005"

    def test_an_oserror_while_writing_is_a_write_failure_with_its_stage(self, roots):
        primary, _ = roots

        def boom(tmp: Path):
            raise OSError(errno.ENOSPC, "No space left on device")

        with pytest.raises(CheckpointWriteFailedError) as exc:
            cs.write_checkpoint_atomically(primary, "run", 10, boom)
        assert exc.value.details["stage"] == "write"
        assert no_temp(primary / "run")

    def test_a_rename_failure_leaves_latest_unchanged(self, roots, monkeypatch):
        """Control M5: `latest` must move only after the rename succeeded."""
        primary, _ = roots
        cs.write_checkpoint_atomically(primary, "run", 5, write_files({"a": b"1"}))

        def fail(src, dst):
            raise OSError(errno.EIO, "I/O error")

        monkeypatch.setattr(cs.os, "rename", fail)
        with pytest.raises(CheckpointWriteFailedError) as exc:
            cs.write_checkpoint_atomically(primary, "run", 10, write_files({"a": b"2"}))
        assert exc.value.details["stage"] == "rename"
        assert os.readlink(primary / "run" / "latest") == "step-000005"
        assert no_temp(primary / "run")

    def test_an_existing_step_is_refused_before_writing(self, roots):
        primary, _ = roots
        make_step(primary, "run", 5)
        calls = []
        with pytest.raises(CheckpointExistsError) as exc:
            cs.write_checkpoint_atomically(primary, "run", 5, calls.append)
        assert calls == [] and exc.value.details == {"path": "run/step-000005"}

    def test_a_step_appearing_during_the_write_is_refused_before_the_rename(self, roots):
        primary, _ = roots

        def racing(tmp: Path):
            make_step(primary, "run", 5)
            (primary / "run" / "step-000005" / "theirs").write_bytes(b"x")

        with pytest.raises(CheckpointExistsError):
            cs.write_checkpoint_atomically(primary, "run", 5, racing)
        assert [p.name for p in (primary / "run" / "step-000005").iterdir()] == ["theirs"]
        assert no_temp(primary / "run")

    def test_enotempty_from_the_rename_is_checkpoint_exists(self, roots, monkeypatch):
        """032 review 5.7: a concurrent save's rename onto a non-empty directory."""
        primary, _ = roots

        def fail(src, dst):
            raise OSError(errno.ENOTEMPTY, "Directory not empty")

        monkeypatch.setattr(cs.os, "rename", fail)
        with pytest.raises(CheckpointExistsError):
            cs.write_checkpoint_atomically(primary, "run", 5, write_files({"a": b"1"}))
        assert no_temp(primary / "run")

    def test_the_strict_run_id_rule_applies(self, roots):
        primary, _ = roots
        for bad in ("../x", "a:b", "has space"):
            with pytest.raises(InvalidRunIdError):
                cs.write_checkpoint_atomically(primary, bad, 5, write_files({}))

    def test_a_read_root_is_never_written(self, roots):
        _, forge = roots
        with pytest.raises(InvalidCheckpointPathError) as exc:
            cs.write_checkpoint_atomically(forge, "run", 5, write_files({"a": b"1"}))
        assert exc.value.details["reason"] == "read_only_root"
        assert list(forge.iterdir()) == []

    def test_a_symlinked_run_dir_escaping_the_root_is_refused(self, roots, tmp_path):
        primary, _ = roots
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        os.symlink(elsewhere, primary / "run")
        with pytest.raises(InvalidCheckpointPathError):
            cs.write_checkpoint_atomically(primary, "run", 5, write_files({"a": b"1"}))
        assert list(elsewhere.iterdir()) == []


class TestDurabilityOrder:
    """The order of the durable steps is the contract (FR-32.1.2/3, 032 review 5.7)."""

    @pytest.fixture()
    def events(self, monkeypatch):
        log: list[tuple[str, str]] = []
        real_fsync_dir, real_fsync_file = cs._fsync_dir, cs._fsync_file
        real_rename, real_replace = os.rename, os.replace

        def fsync_dir(path):
            log.append(("fsync_dir", str(path)))
            real_fsync_dir(path)

        def fsync_file(path):
            log.append(("fsync_file", Path(path).name))
            real_fsync_file(path)

        def rename(src, dst):
            log.append(("rename", Path(dst).name))
            real_rename(src, dst)

        def replace(src, dst):
            log.append(("replace", Path(dst).name))
            real_replace(src, dst)

        monkeypatch.setattr(cs, "_fsync_dir", fsync_dir)
        monkeypatch.setattr(cs, "_fsync_file", fsync_file)
        monkeypatch.setattr(cs.os, "rename", rename)
        monkeypatch.setattr(cs.os, "replace", replace)
        return log

    def test_new_run_dir_fsyncs_root_then_files_then_rename_then_run_dir_then_latest(
        self, roots, events
    ):
        primary, _ = roots
        cs.write_checkpoint_atomically(
            primary, "run", 5, write_files({"model.safetensors": b"w", "manifest.json": b"{}"})
        )
        run = str(primary / "run")
        kinds = [e for e in events]
        assert kinds[0] == ("fsync_dir", str(primary))  # the new run directory's entry
        files = {name for kind, name in kinds if kind == "fsync_file"}
        assert files == {"model.safetensors", "manifest.json"}
        rename_at = kinds.index(("rename", "step-000005"))
        assert all(i < rename_at for i, e in enumerate(kinds) if e[0] == "fsync_file")
        # the temp directory itself is fsynced before the rename
        assert any(e[0] == "fsync_dir" and ".step-000005.tmp-" in e[1]
                   for e in kinds[:rename_at])
        assert kinds[rename_at + 1] == ("fsync_dir", run)
        assert kinds[rename_at + 2] == ("replace", "latest")
        assert kinds[rename_at + 3] == ("fsync_dir", run)
        assert len(kinds) == rename_at + 4

    def test_an_existing_run_dir_does_not_refsync_the_root(self, roots, events):
        primary, _ = roots
        (primary / "run").mkdir()
        cs.write_checkpoint_atomically(primary, "run", 5, write_files({"a": b"1"}))
        assert ("fsync_dir", str(primary)) not in events


class TestReplaceLatest:
    def test_replaces_an_existing_link_atomically_and_relatively(self, roots):
        primary, _ = roots
        a, b = make_step(primary, "r", 1), make_step(primary, "r", 2)
        cs.replace_latest(primary / "r", a.name)
        cs.replace_latest(primary / "r", b.name)
        assert os.readlink(primary / "r" / "latest") == "step-000002"
        assert sorted(p.name for p in (primary / "r").iterdir()) == [
            "latest", "step-000001", "step-000002"]

    @pytest.mark.parametrize("bad", ["", "a/b", "..", "/abs"])
    def test_refuses_a_non_sibling_target(self, roots, bad):
        primary, _ = roots
        (primary / "r").mkdir()
        with pytest.raises(ValueError):
            cs.replace_latest(primary / "r", bad)


# --------------------------------------------------------------------------- cleanup


class TestStaleTempCleanup:
    def test_removes_only_this_modules_leftovers_and_logs_each(self, roots, tmp_path):
        primary, forge = roots
        run = primary / "run"
        step = make_step(primary, "run", 5)
        hexid = uuid.uuid4().hex
        stale_step = run / f".step-000010.tmp-{hexid}"
        stale_step.mkdir()
        (stale_step / "model.safetensors").write_bytes(b"x")
        stale_link = run / f"latest.tmp-{uuid.uuid4().hex}"
        os.symlink("step-000005", stale_link)
        cs.replace_latest(run, step.name)
        forge_tmp = run / f".tmp-save-{uuid.uuid4()}"          # miForge's own temp name
        forge_tmp.mkdir()
        lookalike = run / ".step-000010.tmp-notahexuuid"
        lookalike.mkdir()
        # a symlinked run directory is never descended into
        other = tmp_path / "other"
        other.mkdir()
        (other / f".step-000001.tmp-{uuid.uuid4().hex}").mkdir()
        os.symlink(other, primary / "linked-run")

        with capture_logs() as logs:
            removed = cs.remove_stale_temp_dirs()

        assert sorted(removed) == sorted([stale_step, stale_link])
        assert not stale_step.exists() and not os.path.lexists(stale_link)
        assert step.is_dir() and forge_tmp.is_dir() and lookalike.is_dir()
        assert os.readlink(run / "latest") == "step-000005"
        assert len(list(other.iterdir())) == 1
        removal_logs = [e for e in logs if e["event"] == "checkpoint_temp_removed"]
        assert sorted(e["path"] for e in removal_logs) == sorted([str(stale_step),
                                                                 str(stale_link)])

    def test_never_runs_on_a_read_root(self, roots):
        _, forge = roots
        stale = forge / "run" / f".step-000010.tmp-{uuid.uuid4().hex}"
        stale.mkdir(parents=True)
        cs.remove_stale_temp_dirs()           # primary only
        assert stale.is_dir()
        assert cs.remove_stale_temp_dirs(forge) == []   # even when named explicitly
        assert stale.is_dir()

    def test_a_missing_root_is_nothing_to_do(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "WEIGHTS_CHECKPOINT_DIR", str(tmp_path / "absent"))
        monkeypatch.setattr(settings, "WEIGHTS_CHECKPOINT_READ_ROOTS", "")
        assert cs.remove_stale_temp_dirs() == []
