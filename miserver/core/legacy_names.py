"""Old (miLLM) and new (miServer) names (both honoured from P0 to P7), and the record of every
old-name use (040 FR-040.8, FR-040.10, FR-040.11; TID §3.1).

P7 (FR-040.40, FTDD §6.12): every old environment name is in `RETIRED_ENV` (no longer read; reported in
`retired_but_set` when set) and `HOST_ALIASES` is empty (the host middleware watches nothing). The
reader, the recorder and the health block stay until P8.

Hand-kept on purpose (CF-3): the suite's ``census.py`` (``ENV_RENAMES`` / ``HOST_RENAMES``) is the
authority and writes them into ``docs/legacy-identifiers.md``; ``tests/unit/core/test_legacy_names_published.py``
compares the two, so a row added to one and not the other is red. This module imports only the
standard library and reads no file: ``probe_result.py`` (a hot path) and ``tests/support/env_flags.py``
import it, so it must not pull the settings or the database.

The recorder is in memory only; ``miserver/services/legacy_names_store.py`` flushes it to
``legacy_name_uses`` once a minute. Names, times and counts only, never a value.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

Kind = Literal["env", "host"]


@dataclass(frozen=True)
class EnvAlias:
    old: str
    new: str
    scope: Literal["server", "test"]


@dataclass(frozen=True)
class HostAlias:
    old: str
    new: str
    via: Literal["ingress", "in_cluster", "public"]
    observed: bool = True


#: Every environment name the server or its tests read under the old brand (FR-040.8, FR-040.9).
#: The server reads exactly one (`STRICT_RESULTS`); the rest are test and CI switches. Kept after P7
#: so a retired name that is still set is reported with its new twin (`retired_but_set`).
ENV_ALIASES: tuple[EnvAlias, ...] = (
    EnvAlias("MILLM_STRICT_RESULTS", "MISERVER_STRICT_RESULTS", "server"),
    EnvAlias("MILLM_REQUIRE_CROSS_REPO_CHECKS", "MISERVER_REQUIRE_CROSS_REPO_CHECKS", "test"),
    EnvAlias("MILLM_REQUIRE_PG", "MISERVER_REQUIRE_PG", "test"),
    EnvAlias("MILLM_REQUIRE_RUNNER", "MISERVER_REQUIRE_RUNNER", "test"),
    EnvAlias("MILLM_WRITE_CONTRACTS", "MISERVER_WRITE_CONTRACTS", "test"),
    EnvAlias("MILLM_REAL_TOKENIZERS", "MISERVER_REAL_TOKENIZERS", "test"),
    EnvAlias("MILLM_TEST_PG_BASE", "MISERVER_TEST_PG_BASE", "test"),
    EnvAlias("MILLM_BATCH_HW_URL", "MISERVER_BATCH_HW_URL", "test"),
    EnvAlias("MILLM_BATCH_HW_FILE", "MISERVER_BATCH_HW_FILE", "test"),
)

#: Every old host a request could arrive under until P7 (FR-040.11). Emptied at P7 (FR-040.40) on the
#: evidence of S001's post-sweep reading (0xcc/reviews/040_P7_controls_2026-11-13.md): the old
#: ingress hosts, the in-cluster ExternalName and the old CORS origins are gone from the manifests.
HOST_ALIASES: tuple[HostAlias, ...] = ()

#: P7 moved every old environment name here (FR-040.40): no longer read, reported when set.
RETIRED_ENV: tuple[str, ...] = (
    "MILLM_STRICT_RESULTS",
    "MILLM_REQUIRE_CROSS_REPO_CHECKS",
    "MILLM_REQUIRE_PG",
    "MILLM_REQUIRE_RUNNER",
    "MILLM_WRITE_CONTRACTS",
    "MILLM_REAL_TOKENIZERS",
    "MILLM_TEST_PG_BASE",
    "MILLM_BATCH_HW_URL",
    "MILLM_BATCH_HW_FILE",
)

_NEW_PREFIX = "MISERVER_"


@dataclass
class Hit:
    new: str
    first: datetime
    last: datetime
    count: int


@dataclass(frozen=True)
class EnvView:
    """Name pairs only: a value is compared here and never returned (FR-040.12)."""

    also_set: tuple[tuple[str, str], ...]
    conflicts: tuple[tuple[str, str], ...]
    retired_but_set: tuple[tuple[str, str], ...]
    effective_old: tuple[tuple[str, str], ...]


_LOCK = threading.Lock()
_HITS: dict[tuple[str, str], Hit] = {}
_ENV_BY_NEW: dict[str, EnvAlias] = {}
_HOSTS: dict[str, HostAlias] = {}


def _rebuild() -> None:
    """Index the tables. Tests that monkeypatch `ENV_ALIASES` / `HOST_ALIASES` call it too."""
    _ENV_BY_NEW.clear()
    _ENV_BY_NEW.update({a.new: a for a in ENV_ALIASES})
    _HOSTS.clear()
    _HOSTS.update({a.old: a for a in HOST_ALIASES})


_rebuild()


def _now() -> datetime:
    return datetime.now(UTC)


def env(new: str, default: str | None = None) -> str | None:
    """The value of a `MISERVER_*` name, falling back to its old `MILLM_*` twin (FR-040.8).

    "Set" means present, even to "": the new name wins whenever it is present. The old name is read
    only while it is not retired (since P7 none is read), and each read through it is recorded. A `MISERVER_*` name with no
    alias is read new-only (ID-10). Any other name is a programming error.
    """
    if not new.startswith(_NEW_PREFIX):
        raise KeyError(f"legacy_names.env reads MISERVER_* names only, not {new!r}")
    if new in os.environ:
        return os.environ[new]
    alias = _ENV_BY_NEW.get(new)
    if alias is not None and alias.old in os.environ and alias.old not in RETIRED_ENV:
        record("env", alias.old, alias.new)
        return os.environ[alias.old]
    return default


def record(kind: str, old: str, new: str) -> None:
    """Count one use of an old name. Never raises and never logs: it runs on every request."""
    try:
        now = _now()
        with _LOCK:
            hit = _HITS.get((kind, old))
            if hit is None:
                _HITS[(kind, old)] = Hit(new, now, now, 1)
            else:
                hit.last = now
                hit.count += 1
    except Exception:  # noqa: BLE001 - observing a name must never fail the request that used it
        pass


def drain_hits() -> dict[tuple[str, str], Hit]:
    """Take every unflushed hit (the flush owns them until it writes or restores them)."""
    global _HITS
    with _LOCK:
        taken, _HITS = _HITS, {}
    return taken


def peek_hits() -> dict[tuple[str, str], Hit]:
    """A copy of the unflushed hits, for the health read (no drain)."""
    with _LOCK:
        return {k: Hit(h.new, h.first, h.last, h.count) for k, h in _HITS.items()}


def restore_hits(hits: Mapping[tuple[str, str], Hit]) -> None:
    """Merge hits back after a failed flush: earliest first, latest last, counts summed."""
    with _LOCK:
        for key, hit in hits.items():
            mine = _HITS.get(key)
            if mine is None:
                _HITS[key] = Hit(hit.new, hit.first, hit.last, hit.count)
            else:
                mine.first = min(mine.first, hit.first)
                mine.last = max(mine.last, hit.last)
                mine.count += hit.count


def environment_view(environ: Mapping[str, str] | None = None) -> EnvView:
    """Which old environment names are set, and how they relate to their new twins."""
    environ = os.environ if environ is None else environ
    also, conflicts, retired, effective = [], [], [], []
    for alias in ENV_ALIASES:
        pair = (alias.old, alias.new)
        old_set, new_set = alias.old in environ, alias.new in environ
        if alias.old in RETIRED_ENV:
            if old_set:
                retired.append(pair)
            continue
        if old_set and new_set:
            also.append(pair)
            if environ[alias.old] != environ[alias.new]:
                conflicts.append(pair)
        elif old_set:
            effective.append(pair)
    return EnvView(tuple(also), tuple(conflicts), tuple(retired), tuple(effective))


def host_alias(raw: str) -> HostAlias | None:
    """The alias for a `Host` / `X-Forwarded-Host` value: lower case, port and trailing dot removed."""
    host = raw.strip().lower()
    if host.startswith("["):
        end = host.find("]")
        host = host[: end + 1] if end >= 0 else host
    elif ":" in host:
        host = host.split(":", 1)[0]
    host = host.rstrip(".")
    return _HOSTS.get(host)
