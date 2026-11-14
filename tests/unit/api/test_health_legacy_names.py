"""T10: `GET /api/health/detailed` reports old names in use (040 FR-040.12; BR-023, PD-S1, CF-4,
CF-24). Exact key sets; values after recorded hits; the 24-hour window on a frozen clock; a store
failure still answers 200 with `legacy_names_meta.error` set; no value ever leaks (the sentinel is
searched for as text and in its base64 form).

Since P7 (FR-040.40) the real tables watch nothing: `test_p7_watches_nothing_and_reports_a_retired_name`
holds that. The block's rules are still tested on tables planted back into the transition state
(`transition` fixture), since the block stays until P8.
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from miserver.core import legacy_names
from miserver.services import legacy_names_store as store
from tests.unit.f28_fixtures import db  # noqa: F401 - fixture

NEW = "MISERVER_STRICT_RESULTS"
(OLD,) = [a.old for a in legacy_names.ENV_ALIASES if a.new == NEW]
HOST = "k8s-millm.hitsai.local"
#: The pre-P7 host table, as it stood at P6 (eight old hosts, one unobserved).
TRANSITION_HOSTS = (
    legacy_names.HostAlias(HOST, "k8s-miserver.hitsai.local", "ingress"),
    legacy_names.HostAlias("k8s-millm.hitsai.net", "miserver.hitsai.net", "ingress"),
    legacy_names.HostAlias("millm.hitsai.local", "miserver.hitsai.local", "public"),
    legacy_names.HostAlias("millm.hitsai.net", "miserver.hitsai.net", "public"),
    legacy_names.HostAlias("millm-backend.millm", "miserver-backend.miserver", "in_cluster"),
    legacy_names.HostAlias("millm-backend.millm.svc", "miserver-backend.miserver.svc", "in_cluster"),
    legacy_names.HostAlias("millm-backend.millm.svc.cluster.local",
                           "miserver-backend.miserver.svc.cluster.local", "in_cluster"),
    legacy_names.HostAlias("k8s-millm-otel.hitsai.local", "k8s-miserver-otel.hitsai.local", "ingress",
                           observed=False),
)
NOW = datetime(2026, 11, 20, 15, 4, tzinfo=UTC)
BLOCK_KEYS = {"in_use", "also_set", "conflicts", "retired_but_set"}
META_KEYS = {"observing_since", "last_flush_at", "watching", "in_use_window_hours",
             "transition_hosts", "unobserved_hosts", "new_host_for", "retire_on", "error"}


@pytest.fixture
def client(db, monkeypatch):  # noqa: F811
    from miserver.main import create_app

    for name in (NEW, OLD):
        monkeypatch.delenv(name, raising=False)
    legacy_names.drain_hits()
    yield TestClient(create_app())
    legacy_names.drain_hits()


@pytest.fixture
def transition(monkeypatch):
    """The tables as they stood before P7: nothing retired, eight old hosts."""
    monkeypatch.setattr(legacy_names, "RETIRED_ENV", ())
    monkeypatch.setattr(legacy_names, "HOST_ALIASES", TRANSITION_HOSTS)


def _detailed(client) -> dict:
    r = client.get("/api/health/detailed")
    assert r.status_code == 200, r.text
    return r.json()


def test_exact_key_sets(client):
    body = _detailed(client)
    assert set(body["legacy_names"]) == BLOCK_KEYS
    assert set(body["legacy_names_meta"]) == META_KEYS
    assert body["legacy_names_meta"]["error"] is None
    assert body["legacy_names_meta"]["in_use_window_hours"] == 24


def test_p7_watches_nothing_and_reports_a_retired_name(client, monkeypatch):
    """T33 (040 FR-040.40), on the real tables: the block is present and watches nothing old; an old
    name set alone is not in use, it is in `retired_but_set`."""
    meta = _detailed(client)["legacy_names_meta"]
    assert meta["watching"] == {"env": 0, "host": 0}
    assert meta["transition_hosts"] == [] and meta["unobserved_hosts"] == [] and meta["new_host_for"] == {}
    monkeypatch.setenv(OLD, "1")
    block = _detailed(client)["legacy_names"]
    assert block["retired_but_set"] == [OLD]
    assert block["in_use"] == [] and block["also_set"] == [] and block["conflicts"] == []


def test_watching_before_p7(client, transition):
    assert _detailed(client)["legacy_names_meta"]["watching"] == {"env": 1, "host": 7}


def test_values_after_hits(client, monkeypatch, transition):
    monkeypatch.setenv(OLD, "1")  # the old name alone: effective now
    legacy_names.record("host", HOST, "k8s-miserver.hitsai.local")
    block = _detailed(client)["legacy_names"]
    kinds = {(e["kind"], e["old"], e["new"]) for e in block["in_use"]}
    assert kinds == {("env", OLD, NEW), ("host", HOST, "k8s-miserver.hitsai.local")}
    for entry in block["in_use"]:
        assert set(entry) == {"kind", "old", "new", "last_seen"}
    assert block["also_set"] == [] and block["conflicts"] == [] and block["retired_but_set"] == []
    monkeypatch.setenv(NEW, "0")
    block = _detailed(client)["legacy_names"]
    assert block["also_set"] == [{"old": OLD, "new": NEW}]
    assert block["conflicts"] == [{"old": OLD, "new": NEW}]


async def test_23h_in_use_25h_not(client, db, monkeypatch):  # noqa: F811
    for hours, host in ((23, "millm.hitsai.net"), (25, "millm.hitsai.local")):
        monkeypatch.setattr(legacy_names, "_now", lambda h=hours: NOW - timedelta(hours=h))
        legacy_names.record("host", host, "x")
    await store.flush_once(db)
    monkeypatch.setattr(store, "_now", lambda: NOW)
    in_use = {e["old"] for e in _detailed(client)["legacy_names"]["in_use"]}
    assert in_use == {"millm.hitsai.net"}


def test_store_failure_answers_200_with_error(client, monkeypatch):
    async def boom(_factory):
        raise ConnectionError("database down")

    monkeypatch.setattr(store, "_select_all", boom)
    legacy_names.record("host", HOST, "k8s-miserver.hitsai.local")
    body = _detailed(client)
    assert body["legacy_names_meta"]["error"] == "ConnectionError"
    assert [e["old"] for e in body["legacy_names"]["in_use"]] == [HOST], "memory still reported"


def test_meta_always_present_with_error_key(client, monkeypatch):
    async def shape_failure(_factory):
        raise ValueError("bad")

    monkeypatch.setattr(store, "read_block", shape_failure)
    raw = client.get("/api/health/detailed")
    assert raw.status_code == 200
    meta = raw.json()["legacy_names_meta"]
    assert set(meta) == META_KEYS and meta["error"] == "ValueError"
    assert '"error":' in raw.text.replace(" ", "")
    ok = _detailed_raw(client, monkeypatch)
    assert '"error":null' in ok.replace(" ", ""), "a null error must be emitted, not dropped"


def _detailed_raw(client, monkeypatch) -> str:
    monkeypatch.undo()
    for name in (NEW, OLD):
        monkeypatch.delenv(name, raising=False)
    return client.get("/api/health/detailed").text


def test_unobserved_hosts_is_the_otel_alias(client, transition):
    meta = _detailed(client)["legacy_names_meta"]
    assert meta["unobserved_hosts"] == ["k8s-millm-otel.hitsai.local"]
    assert HOST in meta["transition_hosts"] and len(meta["transition_hosts"]) == 8


def test_new_host_for_pairs_every_transition_host_from_the_table(client, monkeypatch, transition):
    """040 FR-040.31: the admin UI's old-address notice reads each old host's new host from the
    server (never a second copy in the UI). The pairs are the alias table's, read at answer time."""
    meta = _detailed(client)["legacy_names_meta"]
    assert meta["new_host_for"] == {a.old: a.new for a in legacy_names.HOST_ALIASES}
    assert set(meta["new_host_for"]) == set(meta["transition_hosts"])
    assert meta["new_host_for"][HOST] == "k8s-miserver.hitsai.local"
    extra = legacy_names.HostAlias("zz-old.example", "zz-new.example", "ingress")
    monkeypatch.setattr(legacy_names, "HOST_ALIASES", (*legacy_names.HOST_ALIASES, extra))
    assert _detailed(client)["legacy_names_meta"]["new_host_for"]["zz-old.example"] == "zz-new.example"


def test_retire_on_is_the_setting(client, monkeypatch):
    """040 FR-040.31, ID-8: the date the old-address notice shows is `LEGACY_ALIASES_RETIRE_ON`,
    read at answer time; unset, it is null (the notice then gives no date)."""
    from datetime import date

    from miserver.core.config import settings

    monkeypatch.setattr(settings, "LEGACY_ALIASES_RETIRE_ON", None)
    assert _detailed(client)["legacy_names_meta"]["retire_on"] is None
    monkeypatch.setattr(settings, "LEGACY_ALIASES_RETIRE_ON", date(2026, 12, 11))
    assert _detailed(client)["legacy_names_meta"]["retire_on"] == "2026-12-11"


@pytest.mark.parametrize("tables", ["p7", "transition"])
def test_no_value_leaks(client, monkeypatch, request, tables):
    if tables == "transition":
        request.getfixturevalue("transition")  # also_set / conflicts compare the two values
    sentinel = "SENTINEL-040-VALUE-zq81"
    monkeypatch.setenv(OLD, sentinel)
    monkeypatch.setenv(NEW, sentinel + "-other")
    text = client.get("/api/health/detailed").text
    for needle in (sentinel, base64.b64encode(sentinel.encode()).decode()[:16]):
        assert needle not in text
    assert OLD in text, "the NAME is reported"
