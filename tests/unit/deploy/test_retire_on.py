"""The retirement date of the old names is set when P4 closes, not before (040 FR-040.38, ID-8;
FTDD_DECISIONS T3).

`LEGACY_ALIASES_RETIRE_ON` (P4 close + 30 days) is what `legacy_names_meta.retire_on` and the admin
UI's old-address notice show. The date is known only when P4 closes, so `k8s/base/backend.yaml`
carries it exactly when the committed export's header says P4 is closed: the push that closes P4
(P4-E) adds both in one commit, and a later edit cannot drop the date without this failing.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml

from tests.support import legacy_identifiers as li

ROOT = Path(__file__).resolve().parents[3]
NAME = "LEGACY_ALIASES_RETIRE_ON"


def backend_env(text: str) -> dict[str, str | None]:
    """The `millm`-container environment of the `miserver-backend` Deployment."""
    for document in yaml.safe_load_all(text):
        if isinstance(document, dict) and document.get("kind") == "Deployment" \
                and document["metadata"]["name"] == "miserver-backend":
            containers = document["spec"]["template"]["spec"]["containers"]
            env = [e for c in containers for e in c.get("env") or []]
            assert len(env) > 10, "the backend's env was not read: the walk is failing open"
            return {e["name"]: e.get("value") for e in env}
    raise AssertionError("no miserver-backend Deployment in k8s/base/backend.yaml")


def retire_on_state(env: dict[str, str | None], p4_closed: bool) -> str | None:
    """None when the manifest agrees with the export's header; else what is wrong."""
    if not p4_closed:
        return f"{NAME} is set before P4 is closed" if NAME in env else None
    if NAME not in env:
        return f"P4 is closed and {NAME} is not set (P4 close + 30 days, ID-8)"
    try:
        date.fromisoformat(str(env[NAME]))
    except ValueError:
        return f"{NAME}={env[NAME]!r} is not an ISO date"
    return None


def test_backend_sets_the_retirement_date_exactly_when_p4_is_closed():
    env = backend_env((ROOT / "k8s" / "base" / "backend.yaml").read_text())
    assert retire_on_state(env, li.load().closed_at_least("P4")) is None


def test_the_rule_both_ways():
    with_date, without = {NAME: "2026-12-11", "X": "1"}, {"X": "1"}
    assert retire_on_state(without, False) is None
    assert retire_on_state(with_date, True) is None
    assert "before P4 is closed" in retire_on_state(with_date, False)
    assert "is not set" in retire_on_state(without, True)
    assert "not an ISO date" in retire_on_state({NAME: "soon"}, True)


def test_the_setting_reads_the_manifest_form():
    """The value as the manifest would carry it parses to the `date` the health block serves."""
    from miserver.core.config import Settings

    assert Settings(LEGACY_ALIASES_RETIRE_ON="2026-12-11").LEGACY_ALIASES_RETIRE_ON == date(2026, 12, 11)
