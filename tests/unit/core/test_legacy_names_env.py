"""T05: the dual environment reader (040 FR-040.8, FR-040.9), and T33's P7 inverse (FR-040.40).

Since P7 every old name is in `RETIRED_ENV`: `test_retired_old_name_is_not_read_and_is_reported`
holds that on the REAL tables. The reader and the health block stay until P8 (FTDD §6.12): they
report a retired name that is still set, so the reader's rules are still tested, on a table planted
back into the transition state (`transition` fixture).
"""

from __future__ import annotations

import pytest

from miserver.core import legacy_names

NEW = "MISERVER_STRICT_RESULTS"
(OLD,) = [a.old for a in legacy_names.ENV_ALIASES if a.new == NEW]


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for alias in legacy_names.ENV_ALIASES:
        monkeypatch.delenv(alias.old, raising=False)
        monkeypatch.delenv(alias.new, raising=False)
    legacy_names.drain_hits()
    yield
    legacy_names.drain_hits()


@pytest.fixture
def transition(monkeypatch):
    """The reader's rules before P7: nothing retired."""
    monkeypatch.setattr(legacy_names, "RETIRED_ENV", ())


def test_retired_old_name_is_not_read_and_is_reported(monkeypatch):
    """T33 (040 FR-040.40), on the real tables: every old name set ALONE is not read, records no use,
    and is listed in `retired_but_set`; none counts as effective, also set or a conflict."""
    pairs = {(a.old, a.new) for a in legacy_names.ENV_ALIASES}
    assert len(pairs) >= 9, "the pairs stay after P7 so a retired name can still be reported"
    for old, _new in pairs:
        monkeypatch.setenv(old, "1")
    for old, new in sorted(pairs):
        assert legacy_names.env(new) is None, f"{old} is retired but was read for {new}"
        assert legacy_names.env(new, "d") == "d", old
    assert legacy_names.drain_hits() == {}, "a retired name is not a use"
    view = legacy_names.environment_view()
    assert set(view.retired_but_set) == pairs
    assert view.effective_old == () and view.also_set == () and view.conflicts == ()
    assert set(legacy_names.RETIRED_ENV) == {old for old, _new in pairs}, "P7 retires every old name"


def test_the_new_name_is_read_after_p7(monkeypatch):
    monkeypatch.setenv(OLD, "0")
    monkeypatch.setenv(NEW, "1")
    assert legacy_names.env(NEW) == "1"
    assert legacy_names.drain_hits() == {}
    assert legacy_names.environment_view().retired_but_set == ((OLD, NEW),), (
        "set beside its new twin, a retired name is still reported")


def test_new_name_wins(monkeypatch, transition):
    monkeypatch.setenv(NEW, "1")
    monkeypatch.setenv(OLD, "0")
    assert legacy_names.env(NEW) == "1"
    assert legacy_names.drain_hits() == {}, "the new name won: no old-name use to record"
    monkeypatch.setenv(NEW, "")
    assert legacy_names.env(NEW, "d") == "", "present-but-empty still wins"


def test_old_name_alone_works_and_records(monkeypatch, transition):
    monkeypatch.setenv(OLD, "1")
    assert legacy_names.env(NEW) == "1"
    assert legacy_names.env(NEW) == "1"
    hits = legacy_names.drain_hits()
    assert list(hits) == [("env", OLD)]
    assert hits[("env", OLD)].new == NEW and hits[("env", OLD)].count == 2


def test_neither_gives_default():
    assert legacy_names.env(NEW) is None
    assert legacy_names.env(NEW, "0") == "0"
    assert legacy_names.drain_hits() == {}


def test_also_set_and_conflicts_never_carry_values(transition):
    view = legacy_names.environment_view({OLD: "sentinel-a", NEW: "sentinel-b"})
    assert view.also_set == ((OLD, NEW),)
    assert view.conflicts == ((OLD, NEW),)
    assert view.effective_old == ()
    assert "sentinel" not in repr(view)
    same = legacy_names.environment_view({OLD: "1", NEW: "1"})
    assert same.also_set == ((OLD, NEW),) and same.conflicts == ()
    only_old = legacy_names.environment_view({OLD: "1"})
    assert only_old.effective_old == ((OLD, NEW),) and only_old.also_set == ()


def test_one_retired_name_is_not_read_while_the_others_are(monkeypatch):
    monkeypatch.setattr(legacy_names, "RETIRED_ENV", (OLD,))
    monkeypatch.setenv(OLD, "1")
    assert legacy_names.env(NEW) is None
    assert legacy_names.drain_hits() == {}
    view = legacy_names.environment_view()
    assert view.retired_but_set == ((OLD, NEW),)
    assert view.effective_old == () and view.also_set == ()


def test_new_only_name_without_alias(monkeypatch):
    """ID-10: a MISERVER_* name with no old spelling (041's TEST_PACKAGE_ROOT) reads new-only."""
    name = "MISERVER_TEST_PACKAGE_ROOT"
    monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MILLM_TEST_PACKAGE_ROOT", "/x")
    assert legacy_names.env(name) is None
    monkeypatch.setenv(name, "/y")
    assert legacy_names.env(name) == "/y"
    with pytest.raises(KeyError):
        legacy_names.env(OLD)


def test_host_alias_normalises_case_port_and_trailing_dot(monkeypatch):
    planted = (legacy_names.HostAlias("k8s-millm.hitsai.local", "k8s-miserver.hitsai.local", "ingress"),
               legacy_names.HostAlias("millm-backend.millm.svc.cluster.local",
                                      "miserver-backend.miserver.svc.cluster.local", "in_cluster"))
    monkeypatch.setattr(legacy_names, "HOST_ALIASES", planted)
    legacy_names._rebuild()
    try:
        for raw in ("k8s-millm.hitsai.local", "K8S-MILLM.hitsai.LOCAL", "k8s-millm.hitsai.local:443",
                    "k8s-millm.hitsai.local.", " k8s-millm.hitsai.local.:8080 "):
            alias = legacy_names.host_alias(raw)
            assert alias is not None and alias.new == "k8s-miserver.hitsai.local", raw
        assert legacy_names.host_alias("k8s-miserver.hitsai.local") is None
        assert legacy_names.host_alias("[::1]:8000") is None
        assert legacy_names.host_alias("millm-backend.millm.svc.cluster.local:8000").via == "in_cluster"
    finally:
        monkeypatch.undo()
        legacy_names._rebuild()
    assert legacy_names.host_alias("k8s-millm.hitsai.local") is None, "P7: the real table is empty"


def test_restore_merges_after_a_failed_flush(monkeypatch, transition):
    monkeypatch.setenv(OLD, "1")
    legacy_names.env(NEW)
    taken = legacy_names.drain_hits()
    legacy_names.env(NEW)
    legacy_names.restore_hits(taken)
    merged = legacy_names.drain_hits()[("env", OLD)]
    assert merged.count == 2 and merged.first <= merged.last


def test_the_recorder_never_raises(monkeypatch):
    def boom():
        raise RuntimeError("clock")

    monkeypatch.setattr(legacy_names, "_now", boom)
    legacy_names.record("env", OLD, NEW)  # swallowed
