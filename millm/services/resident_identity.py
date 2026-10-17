"""What the resident model IS, as served, and when its weights were last replaced.

Shared by Features 031, 032 and 033 (contract: 0xcc/reviews/attachable_weights_impl_addendum_2026-10-14.md §2.1).

- The **served name** is what every `/v1` response reports as its model. It is the
  resident identity's name, or a name an overlay supplies (033 registers one while an
  adapter is applied). Callers read it through `served_name()`, never from the row.
- The **base epoch** advances every time the resident parameter set is replaced (a load,
  an unload, an in-place copy) and never on a save or a commit. 033 compares it across
  its verification and its pause, because the base can change in between.
- **Base-replaced callbacks** run inside the replacement, before the first write, so a
  replaced base never serves state that belonged to the old one (033 drops its adapter).

The chokepoints are `LoadedModelState.set` and `LoadedModelState.clear` (`model_loader.py`),
which call `begin_base_replacement` before they touch the old weights, and 032's in-place copy,
which calls it before its first write. `set_identity` is a compare-and-set under
`LoadedModelState._lock`, so a save that finishes after a newer load cannot relabel the new
weights (032 review 5.1). Every identity change writes one `weights_identity_changed` line.

Lock order: `LoadedModelState._lock` may be held while this module's `_lock` is taken (to read
the epoch), never the reverse; callbacks run with NEITHER lock held, so a callback may read the
resident model or call `set_identity`.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Literal, Optional

import structlog

logger = structlog.get_logger()

Origin = Literal["model", "checkpoint", "instantiated"]


@dataclass(frozen=True)
class ResidentIdentity:
    """The identity of the resident parameter set (032 FTDD §4.4)."""

    name: str
    base_model: str
    origin: Origin
    checkpoint_dir: Optional[str]
    manifest_sha256: Optional[str]
    expected_parent: Optional[str]
    commits_at_identity: int
    seed: Optional[int] = None
    dtype: Optional[str] = None


_lock = threading.Lock()
_epoch = 0
_overlays: list[Callable[[], Optional[str]]] = []
_base_replaced: list[Callable[[str], object]] = []
#: The chokepoint calls, `(reason, model_id)`, an open `covered_base_replacement` already
#: accounts for. Each is CONSUMED by the first matching call: a second call for the same model
#: inside the scope is another operation (a concurrent unload) and advances as usual.
_covered: Optional[set[tuple[str, int]]] = None
_covered_reason: Optional[str] = None


def _current_loaded() -> Any:
    from millm.ml.model_loader import LoadedModelState

    return LoadedModelState().current


def identity_of(loaded: Any) -> Optional[ResidentIdentity]:
    """The resident identity of `loaded`, or None for a plain row load (origin `model`)."""
    if loaded is None:
        return None
    identity: Optional[ResidentIdentity] = loaded.weights_identity
    return identity


def _describe(loaded: Any) -> dict[str, Any]:
    """The fields one side of an identity-change line names."""
    if loaded is None:
        return {"name": None, "origin": None, "model_id": None}
    identity = identity_of(loaded)
    return {
        "name": identity.name if identity is not None else loaded.model_name,
        "origin": identity.origin if identity is not None else "model",
        "model_id": loaded.model_id,
    }


def _log_change(before: dict[str, Any], after: dict[str, Any], reason: str) -> None:
    logger.info(
        "weights_identity_changed",
        reason=reason,
        epoch=base_epoch(),
        old_name=before["name"],
        old_origin=before["origin"],
        old_model_id=before["model_id"],
        new_name=after["name"],
        new_origin=after["origin"],
        new_model_id=after["model_id"],
    )


def log_identity_change(old: Any, new: Any, reason: str) -> None:
    """Write the one `weights_identity_changed` line for a change from `old` to `new`.

    `old` and `new` are a `LoadedModel` (read for its identity, else its row name) or None.
    """
    _log_change(_describe(old), _describe(new), reason)


def set_identity(
    loaded: Any, identity: Optional[ResidentIdentity], *, expected_epoch: Optional[int] = None
) -> bool:
    """Set `loaded.weights_identity`, compare-and-set: True if written, False if superseded.

    Refused (False, nothing written) when `loaded` is no longer the resident model, or when
    `expected_epoch` is given and the base epoch has moved since the caller captured it — a
    load completed in between, so the identity the caller computed describes weights that are
    no longer served. Checked and written under `LoadedModelState._lock`, the lock every
    replacement of the resident model takes.
    """
    from millm.ml.model_loader import LoadedModelState

    state = LoadedModelState()
    superseded: Optional[str] = None
    with LoadedModelState._lock:
        if loaded is None or state.current is not loaded:
            superseded = "not_resident"
        elif expected_epoch is not None and base_epoch() != expected_epoch:
            superseded = "epoch_moved"
        else:
            before = _describe(loaded)
            loaded.weights_identity = identity
            after = _describe(loaded)
    if superseded is not None:
        logger.info(
            "weights_identity_superseded",
            reason=superseded,
            expected_epoch=expected_epoch,
            epoch=base_epoch(),
            name=identity.name if identity is not None else None,
        )
        return False
    _log_change(before, after, "set_identity")
    return True


def unsaved_commits(steps_since_load: Any, loaded: Any = None) -> int:
    """Commits the resident weights carry that no checkpoint records (addendum D-45, F-A).

    Derived from state that already exists, never kept beside it:

    * the barrier's `steps_since_load` counts commit-ends since the parameter set was last
      replaced (a load of any kind resets it to 0);
    * a save sets the resident identity with `commits_at_identity` = that counter at the save
      (`weights_checkpoint_service._save_locked`), and a checkpoint load sets it to 0 with the
      counter reset.

    So the unsaved count is the counter minus `commits_at_identity` (0 for a plain row load,
    which has no identity). When the counter is BELOW `commits_at_identity`, it was reset after
    the identity was recorded and no save has run since (a save would have recorded a value no
    larger than the counter), so every commit it counts is unsaved. Nothing resident, or a
    counter that is not an int, is 0: commits exist only through the barrier.
    """
    loaded = _current_loaded() if loaded is None else loaded
    if loaded is None:
        return 0
    if not isinstance(steps_since_load, int) or isinstance(steps_since_load, bool):
        return 0
    identity = identity_of(loaded)
    at_identity = identity.commits_at_identity if identity is not None else 0
    if not isinstance(at_identity, int) or isinstance(at_identity, bool):
        at_identity = 0
    if steps_since_load < at_identity:
        return max(steps_since_load, 0)
    return steps_since_load - at_identity


def base_served_name(loaded: Any = None) -> Optional[str]:
    """The base's served name, ignoring overlays: the identity name, else the row name."""
    loaded = _current_loaded() if loaded is None else loaded
    if loaded is None:
        return None
    identity = identity_of(loaded)
    return identity.name if identity is not None else loaded.model_name


def served_name(loaded: Any = None) -> Optional[str]:
    """The name `/v1` reports for the resident model, overlays applied (last registered wins)."""
    base = base_served_name(loaded)
    if base is None:
        return None
    with _lock:
        overlays = list(_overlays)
    for overlay in reversed(overlays):
        name = overlay()
        if name:
            return name
    return base


def register_served_name_overlay(fn: Callable[[], Optional[str]]) -> None:
    """Register a function that may replace the served name (returns None to decline)."""
    with _lock:
        if fn not in _overlays:
            _overlays.append(fn)


def served_name_overlays() -> tuple[Callable[[], Optional[str]], ...]:
    """The registered overlays, for live-registry reachability tests."""
    with _lock:
        return tuple(_overlays)


def matches_resident(requested: str, loaded: Any = None) -> bool:
    """Whether a `/v1` request naming `requested` means the resident model.

    True for the served name (overlays applied) and for the bare base name, so a client
    naming the base keeps working while an adapter is applied (033 D12). `loaded` defaults to
    the singleton's resident; callers holding their own state object pass it.
    """
    if not requested:
        return False
    return requested in {served_name(loaded), base_served_name(loaded)} - {None}


def base_epoch() -> int:
    """A counter that advances each time the resident parameter set is replaced."""
    with _lock:
        return _epoch


def advance_base_epoch(reason: str) -> int:
    """Advance the base epoch. Called at the replacement chokepoints only."""
    global _epoch
    with _lock:
        _epoch += 1
        value = _epoch
    logger.info("weights_base_epoch_advanced", epoch=value, reason=reason)
    return value


def register_on_base_replaced(callback: Callable[[str], object]) -> None:
    """Register a callback run inside every base replacement, before the first write."""
    with _lock:
        if callback not in _base_replaced:
            _base_replaced.append(callback)


def base_replaced_callbacks() -> tuple[Callable[[str], object], ...]:
    """The registered callbacks, for live-registry reachability tests."""
    with _lock:
        return tuple(_base_replaced)


def fire_base_replaced(reason: str) -> None:
    """Run every base-replaced callback. A raising callback propagates: the caller aborts
    before its first write, so the state is unchanged."""
    for callback in base_replaced_callbacks():
        callback(reason)


def begin_base_replacement(reason: str) -> int:
    """What every chokepoint calls BEFORE it moves, frees or overwrites the resident weights:
    advance the base epoch, then run the base-replaced callbacks. Returns the new epoch.

    The epoch moves first, so anything that observes a callback's effect (033's adapter
    dropped) also sees the new epoch. If a callback raises, the replacement aborts with the
    weights untouched; the epoch has still moved, which only makes a concurrent comparison
    refuse — conservative, never a mislabel. Call it with no `LoadedModelState` lock held.
    """
    epoch = advance_base_epoch(reason)
    fire_base_replaced(reason)
    return epoch


@contextmanager
def covered_base_replacement(
    reason: str, chokepoints: Iterable[tuple[str, int]]
) -> Iterator[int]:
    """One replacement made of several chokepoint calls (032's full reload: an unload then a
    load). The epoch advances and the callbacks run ONCE, here, before the caller's first
    write; inside the scope each expected chokepoint call — `("model_unload", id)` from
    `LoadedModelState.clear`, `("model_load", id)` from `.set` — does neither, ONCE. A raising
    callback raises before the scope opens, so the caller aborts with nothing changed.

    Scoped by call, not process-wide: a chokepoint for any other model, or a second call for a
    covered one (another operation's unload inside the scope), still advances and still runs
    the callbacks. Not reentrant; the refusal comes before any side effect. The cover is
    removed when the scope exits, however it exits."""
    global _covered, _covered_reason
    with _lock:
        if _covered is not None:
            raise RuntimeError("covered_base_replacement is not reentrant")
        # Reserved, covering nothing yet: chokepoints inside the callbacks still advance.
        _covered, _covered_reason = set(), reason
    try:
        epoch = begin_base_replacement(reason)
        with _lock:
            _covered = {(str(r), int(m)) for r, m in chokepoints}
        yield epoch
    finally:
        with _lock:
            _covered = _covered_reason = None


def begin_chokepoint_replacement(reason: str, model_id: Optional[int]) -> int:
    """What `LoadedModelState.set` and `.clear` call: `begin_base_replacement`, unless an open
    `covered_base_replacement` expects exactly this call and has not yet seen it (then the call
    consumes the cover and does nothing: the scope already advanced and ran the callbacks)."""
    with _lock:
        key = (reason, model_id)
        covered = False
        if _covered is not None and key in _covered:
            _covered.discard(key)
            covered = True
        by, epoch = _covered_reason, _epoch
    if covered:
        logger.info("weights_base_replacement_covered", reason=reason, by=by, model_id=model_id)
        return epoch
    return begin_base_replacement(reason)
