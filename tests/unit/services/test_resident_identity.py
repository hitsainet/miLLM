"""The resident identity, the base epoch and the base-replaced chokepoints (addendum §2.1).

The chokepoints are `LoadedModelState.set` and `LoadedModelState.clear`. The rules pinned here:
the epoch advances and the callbacks run BEFORE the resident slot changes and before any weight
moves; a raising callback aborts with the old state intact; callbacks run with no lock held;
`set_identity` is a compare-and-set that a newer load or a moved epoch defeats; and every
identity change writes exactly one `weights_identity_changed` line.
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any

import pytest

import millm.services.resident_identity as ri
from millm.ml.model_loader import LoadedModel, LoadedModelState, ModelLoader
from millm.services.resident_identity import (
    ResidentIdentity,
    advance_base_epoch,
    base_epoch,
    base_replaced_callbacks,
    base_served_name,
    identity_of,
    matches_resident,
    register_on_base_replaced,
    register_served_name_overlay,
    served_name,
    served_name_overlays,
    set_identity,
)


class _Weights:
    """A stand-in model that records when it is moved off the card."""

    def __init__(self) -> None:
        self.moved_to: list[str] = []

    def to(self, device: str) -> "_Weights":
        self.moved_to.append(device)
        return self


def _loaded(model_id: int = 1, name: str = "base-row") -> LoadedModel:
    return LoadedModel(model_id=model_id, model_name=name, model=_Weights(), tokenizer=None,
                       loaded_at=datetime(2026, 10, 14), device="cpu")


def _identity(name: str = "run-a:step-000005") -> ResidentIdentity:
    return ResidentIdentity(name=name, base_model="org/base", origin="checkpoint",
                            checkpoint_dir="run-a/step-000005", manifest_sha256="f" * 64,
                            expected_parent=None, commits_at_identity=0)


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __getattr__(self, level: str):
        def record(event: str, *args: Any, **kwargs: Any) -> None:
            self.calls.append((event, kwargs))

        return record

    def events(self, name: str) -> list[dict[str, Any]]:
        return [k for e, k in self.calls if e == name]


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    state = LoadedModelState()
    state._loaded, state._unloading = None, False
    monkeypatch.setattr(ri, "_overlays", [])
    monkeypatch.setattr(ri, "_base_replaced", [])
    yield
    state._loaded, state._unloading = None, False


@pytest.fixture
def rec(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(ri, "logger", recorder)
    return recorder


class TestReads:
    def test_identity_of(self):
        assert identity_of(None) is None
        plain = _loaded()
        assert plain.weights_identity is None and identity_of(plain) is None
        plain.weights_identity = _identity()
        assert identity_of(plain) == _identity()

    def test_served_name_is_the_identity_name_else_the_row_name(self):
        assert served_name() is None
        loaded = _loaded()
        LoadedModelState().set(loaded)
        assert served_name() == base_served_name() == "base-row"
        loaded.weights_identity = _identity("run-a:step-000005")
        assert served_name() == "run-a:step-000005"

    def test_overlays_apply_last_registered_first_and_may_decline(self):
        LoadedModelState().set(_loaded())

        def declines():
            return None

        def adapter():
            return "run-a:adapter-200"

        register_served_name_overlay(adapter)
        register_served_name_overlay(declines)
        register_served_name_overlay(adapter)  # idempotent
        assert served_name_overlays() == (adapter, declines)
        assert served_name() == "run-a:adapter-200"
        assert base_served_name() == "base-row"

    def test_matches_resident_accepts_served_and_bare_base_names(self):
        LoadedModelState().set(_loaded())
        register_served_name_overlay(lambda: "run-a:adapter-200")
        assert matches_resident("run-a:adapter-200")
        assert matches_resident("base-row")
        assert not matches_resident("other")
        assert not matches_resident("")

    def test_nothing_resident_matches_nothing(self):
        assert not matches_resident("base-row")


class TestChokepointSet:
    def test_set_advances_the_epoch_and_fires_before_the_slot_changes(self, rec):
        state = LoadedModelState()
        old, new = _loaded(1, "old"), _loaded(2, "new")
        state.set(old)
        seen: list[tuple[str, Any, int]] = []
        register_on_base_replaced(lambda reason: seen.append((reason, state.current, base_epoch())))
        before = base_epoch()
        state.set(new)
        assert seen == [("model_load", old, before + 1)]
        assert state.current is new and base_epoch() == before + 1

    def test_a_raising_callback_aborts_the_set_with_the_old_state(self):
        state = LoadedModelState()
        old = _loaded(1, "old")
        state.set(old)

        def refuse(reason: str) -> None:
            raise RuntimeError("drop failed")

        register_on_base_replaced(refuse)
        with pytest.raises(RuntimeError, match="drop failed"):
            state.set(_loaded(2, "new"))
        assert state.current is old

    def test_one_identity_line_per_set(self, rec):
        LoadedModelState().set(_loaded(1, "old"))
        LoadedModelState().set(_loaded(2, "new"))
        lines = rec.events("weights_identity_changed")
        assert [(l["old_name"], l["new_name"], l["reason"]) for l in lines] == [
            (None, "old", "model_load"), ("old", "new", "model_load")]
        assert lines[1]["new_origin"] == "model" and lines[1]["new_model_id"] == 2


class TestChokepointClear:
    def test_clear_fires_before_any_weight_moves(self, rec):
        state = LoadedModelState()
        loaded = _loaded()
        weights = loaded.model
        state.set(loaded)
        seen: list[tuple[Any, list[str], int]] = []
        register_on_base_replaced(
            lambda reason: seen.append((state.current, list(weights.moved_to), base_epoch())))
        before = base_epoch()
        state.clear()
        assert seen == [(loaded, [], before + 1)]
        assert weights.moved_to == ["cpu"] and state.current is None
        lines = rec.events("weights_identity_changed")
        assert [(l["old_name"], l["new_name"], l["reason"]) for l in lines][-1] == (
            "base-row", None, "model_unload")

    def test_a_raising_callback_aborts_the_unload_with_the_weights_untouched(self):
        state = LoadedModelState()
        loaded = _loaded()
        weights = loaded.model
        state.set(loaded)
        register_on_base_replaced(lambda reason: (_ for _ in ()).throw(RuntimeError("no")))
        with pytest.raises(RuntimeError):
            state.clear()
        assert state.current is loaded and loaded.model is weights
        assert weights.moved_to == []

    def test_clearing_nothing_replaces_nothing(self, rec):
        fired: list[str] = []
        register_on_base_replaced(fired.append)
        before = base_epoch()
        LoadedModelState().clear()
        assert fired == [] and base_epoch() == before
        assert rec.events("weights_identity_changed") == []

    def test_the_loader_unload_path_fires_once(self):
        state = LoadedModelState()
        state.set(_loaded())
        fired: list[str] = []
        register_on_base_replaced(fired.append)
        assert ModelLoader().unload() is True
        assert fired == ["model_unload"]

    @pytest.mark.parametrize("chokepoint", ["set", "clear"])
    def test_callbacks_run_with_the_state_lock_free(self, chokepoint):
        """Callbacks run with no lock held, so one that takes LoadedModelState._lock (as
        set_identity does) cannot deadlock. clear() runs in the unload worker's thread, so the
        chokepoint runs in a thread here too. The callback PROBES the lock with a timeout
        instead of blocking on it, so a regression fails this test rather than hanging the
        suite."""
        state = LoadedModelState()
        loaded = _loaded()
        state.set(loaded)
        results: list[Any] = []

        def probe(reason: str) -> None:
            free = LoadedModelState._lock.acquire(timeout=1)
            if free:
                LoadedModelState._lock.release()
            results.append((state.current, free))

        register_on_base_replaced(probe)
        target = state.clear if chokepoint == "clear" else (lambda: state.set(_loaded(2, "n")))
        worker = threading.Thread(target=target)
        worker.start()
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert results == [(loaded, True)]


class TestSetIdentity:
    def test_writes_the_resident_and_logs_one_line(self, rec):
        loaded = _loaded()
        LoadedModelState().set(loaded)
        rec.calls.clear()
        assert set_identity(loaded, _identity("run-a:step-000005")) is True
        assert loaded.weights_identity == _identity("run-a:step-000005")
        [line] = rec.events("weights_identity_changed")
        assert (line["old_name"], line["new_name"], line["new_origin"], line["reason"]) == (
            "base-row", "run-a:step-000005", "checkpoint", "set_identity")

    def test_a_model_no_longer_resident_is_not_relabelled(self, rec):
        stale = _loaded(1, "old")
        LoadedModelState().set(stale)
        LoadedModelState().set(_loaded(2, "new"))
        assert set_identity(stale, _identity()) is False
        assert stale.weights_identity is None
        assert rec.events("weights_identity_superseded")[0]["reason"] == "not_resident"

    def test_a_moved_epoch_defeats_the_write(self, rec):
        loaded = _loaded()
        LoadedModelState().set(loaded)
        captured = base_epoch()
        advance_base_epoch("in_place_load")  # 032's in-place copy, between snapshot and write
        assert set_identity(loaded, _identity(), expected_epoch=captured) is False
        assert loaded.weights_identity is None
        assert rec.events("weights_identity_superseded")[0]["reason"] == "epoch_moved"
        assert set_identity(loaded, _identity(), expected_epoch=base_epoch()) is True

    def test_the_save_vs_load_race(self):
        """032 review 5.1: a save captures the epoch and model under its pause; a full reload
        lands while it writes files; the save's identity must not label the new model."""
        old = _loaded(1, "base-row")
        LoadedModelState().set(old)
        captured_model, captured_epoch = LoadedModelState().current, base_epoch()
        LoadedModelState().set(_loaded(1, "base-row"))  # the reload, same row, new weights
        assert set_identity(captured_model, _identity(), expected_epoch=captured_epoch) is False
        assert LoadedModelState().current.weights_identity is None

    def test_the_compare_and_set_holds_the_state_lock(self):
        """The check and the write are one step under LoadedModelState._lock: while another
        thread holds it, set_identity waits rather than writing beside a replacement."""
        loaded = _loaded()
        LoadedModelState().set(loaded)
        done = threading.Event()

        def write():
            set_identity(loaded, _identity())
            done.set()

        with LoadedModelState._lock:
            worker = threading.Thread(target=write)
            worker.start()
            assert not done.wait(0.1), "set_identity wrote without the state lock"
        assert done.wait(5)
        worker.join()
        assert loaded.weights_identity == _identity()


class TestRegistries:
    def test_callbacks_are_registered_once_and_readable(self):
        def drop(reason: str) -> None:
            pass

        register_on_base_replaced(drop)
        register_on_base_replaced(drop)
        assert base_replaced_callbacks() == (drop,)

    def test_the_epoch_only_moves_forward(self):
        before = base_epoch()
        assert advance_base_epoch("x") == before + 1
        assert base_epoch() == before + 1
