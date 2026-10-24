"""THE behaviour selector (039 FR-39.1, FTDD §6.1, FTID §3.1) and the per-key member map.

A probe set is a SELECTOR, not an artifact: no table holds membership. A behaviour's members are the
imported probes whose `behaviour_key` is the key (and whose `behaviour_version` is the version), and
whose identity matches the resident model. One function selects them for every caller — the set
arm route (and its dry run), the capacity preview and stateless scoring by behaviour — so the four
can never disagree about who is in a set.

⚠ CALLERS IMPORT THE MODULE, NEVER THE FUNCTION: `from millm.services import probe_selection` and
`probe_selection.select_behaviour(...)`. The call-count test patches the module attribute; a caller
that bound the function by name at import would be invisible to it (suite memory
`monkeypatches-that-silently-do-nothing`).

The refusals, in order (FTID §3.1): `INVALID_PROBE_SELECTION` (a `probe_ids` entry of another key)
→ `CATALOGUE_LIMIT` (`PROBE_MAX_PER_BEHAVIOUR`) → `BEHAVIOUR_VERSION_MIXED`.

`version` has THREE states, so it uses a sentinel: `UNSET` (not narrowed), `None` (the uncatalogued
version — a key whose definitions say `version: null` is a version of its own, OQ-39.3 (a)) and an
integer.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from millm.core.config import settings
from millm.core.errors import (
    BehaviourVersionMixedError,
    CatalogueLimitError,
    InvalidProbeSelectionError,
)

logger = logging.getLogger(__name__)


class _Unset:
    """The `version` argument was not given (distinct from `None`, the uncatalogued version)."""

    _instance: Optional["_Unset"] = None

    def __new__(cls) -> "_Unset":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "UNSET"


UNSET: Any = _Unset()


@dataclass(frozen=True)
class Selection:
    """What one call selected: the members, the version they share, and what was left out."""

    key: str
    #: The one version every member carries (`None` = uncatalogued).
    version: Optional[int]
    #: The member rows, sorted by id.
    probes: tuple[Any, ...]
    #: `{version: [probe_id, ...]}` over the key's probes for the resident model, BEFORE narrowing.
    versions_seen: dict[Optional[int], list[str]] = field(default_factory=dict)
    #: Probes of the key fitted on another model: not members here, and named so nobody wonders.
    skipped_identity: tuple[str, ...] = ()

    @property
    def probe_ids(self) -> list[str]:
        return [p.id for p in self.probes]


def _versions_payload(versions: dict[Optional[int], list[str]]) -> list[dict[str, Any]]:
    return [
        {"version": v, "probe_ids": sorted(ids)}
        for v, ids in sorted(versions.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))
    ]


async def select_behaviour(
    repository: Any,
    key: str,
    *,
    version: Any = UNSET,
    probe_ids: Optional[Iterable[str]] = None,
    loaded: Any = None,
) -> Selection:
    """Select one behaviour's probes for the resident model. Raises the three refusals above.

    `loaded` is the resident `LoadedIdentity`; `None` selects without the identity filter (the
    behaviour listing, which shows every imported probe). One query on `(behaviour_key,
    behaviour_version)` through `ProbeRepository.by_behaviour`.
    """
    from millm.services.probe_arming import identity_refusal

    rows = await repository.by_behaviour(key)
    members: list[Any] = []
    skipped: list[str] = []
    for row in rows:
        if loaded is not None and identity_refusal(row, loaded) is not None:
            skipped.append(row.id)
            continue
        members.append(row)

    if probe_ids is not None:
        wanted = list(dict.fromkeys(probe_ids))
        by_id = {row.id: row for row in members}
        foreign = [pid for pid in wanted if pid not in by_id]
        if foreign:
            raise InvalidProbeSelectionError(
                f"{len(foreign)} of the probe ids given are not imported probes of behaviour "
                f"{key!r} for the loaded model",
                details={"behaviour": key, "probe_ids": foreign},
            )
        members = [by_id[pid] for pid in wanted]

    versions: dict[Optional[int], list[str]] = {}
    for row in members:
        versions.setdefault(getattr(row, "behaviour_version", None), []).append(row.id)
    if version is not UNSET:
        members = [row for row in members if getattr(row, "behaviour_version", None) == version]

    limit = int(settings.PROBE_MAX_PER_BEHAVIOUR)
    if len(members) > limit:
        raise CatalogueLimitError(
            f"Behaviour {key!r} selects {len(members)} probes; at most {limit} may form one set",
            details={"limit": "PROBE_MAX_PER_BEHAVIOUR", "value": limit, "source": "global",
                     "behaviour": key, "selected": len(members)},
        )

    chosen = {getattr(row, "behaviour_version", None) for row in members}
    if len(chosen) > 1:
        raise BehaviourVersionMixedError(
            f"Behaviour {key!r} has probes of {len(chosen)} versions; arm one version at a time "
            f"(pass `version`, or `probe_ids`)",
            details={"behaviour": key, "versions": _versions_payload(
                {v: ids for v, ids in versions.items() if v in chosen})},
        )
    members.sort(key=lambda row: row.id)
    return Selection(
        key=key,
        version=next(iter(chosen)) if chosen else (None if version is UNSET else version),
        probes=tuple(members),
        versions_seen=versions,
        skipped_identity=tuple(sorted(skipped)),
    )


# ── the in-memory member map (read by `build_set_block` with no DB read per request) ──────────

_lock = threading.Lock()
#: `probe_id -> (hf_id, behaviour_key, behaviour_version)` for every imported probe with a key.
_members_of: dict[str, tuple[str, str, Optional[int]]] = {}
#: `(hf_id, key, version) -> sorted probe ids`.
_groups: dict[tuple[str, str, Optional[int]], tuple[str, ...]] = {}


def _rebuild_groups(members: dict[str, tuple[str, str, Optional[int]]]) -> None:
    groups: dict[tuple[str, str, Optional[int]], list[str]] = {}
    for probe_id, group in members.items():
        groups.setdefault(group, []).append(probe_id)
    global _groups, _members_of
    _members_of = members
    _groups = {group: tuple(sorted(ids)) for group, ids in groups.items()}


async def refresh_member_counts(session: Any) -> int:
    """Rebuild the member map from the `probes` table: ONE query. Never raises.

    Called at startup (`lifespan`) and after every import and delete. A failure keeps the previous
    map and is logged: a stale member count makes a `set` block say "complete" or "partial"
    wrongly, which is reported, never fatal to serving.
    """
    from sqlalchemy import select

    from millm.db.models.probe import Probe

    try:
        rows = (await session.execute(
            select(Probe.id, Probe.hf_id, Probe.behaviour_key, Probe.behaviour_version)
            .where(Probe.behaviour_key.is_not(None))
        )).all()
    except Exception as exc:  # noqa: BLE001 - bookkeeping must not break the caller
        logger.warning("probe_member_counts_refresh_failed error=%s", type(exc).__name__)
        return -1
    with _lock:
        _rebuild_groups({r.id: (r.hf_id, r.behaviour_key, r.behaviour_version) for r in rows})
    return len(rows)


def members_of_group(probe_id: str) -> Optional[tuple[str, ...]]:
    """Every imported probe id of `probe_id`'s own (model, key, version) group, or None."""
    group = _members_of.get(probe_id)
    if group is None:
        return None
    return _groups.get(group, (probe_id,))


def member_count(hf_id: str, key: str, version: Optional[int]) -> int:
    return len(_groups.get((hf_id, key, version), ()))


def forget_for_tests() -> None:
    with _lock:
        _rebuild_groups({})
