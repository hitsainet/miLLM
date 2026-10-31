"""Feature 36's settings load with the FTDD §7 defaults, and `.env.example` names each one.

⚠ `PROBE_TRANSCRIPTS` defaults to `off` (D5). Recording user text is the operator's explicit
choice; a default of `on` would store personal data on every install that never asked for it.
Mutation control: flipping the code default to `"on"` turns `test_recording_is_off_by_default` red.
"""

from pathlib import Path

from millm.core.config import Settings

DEFAULTS = {
    "PROBE_TRANSCRIPTS": "off",
    "PROBE_TRANSCRIPT_MAX_AGE_DAYS": 7,
    "PROBE_TRANSCRIPT_MAX_TURNS": 20000,
    "PROBE_SESSION_IDLE_MINUTES": 30,
    "PROBE_SESSION_MAX_AGE_DAYS": 30,
    "PROBE_RETENTION_INTERVAL_S": 300,
}


def _clean(monkeypatch):
    for name in DEFAULTS:
        monkeypatch.delenv(name, raising=False)


def test_defaults_load(monkeypatch):
    _clean(monkeypatch)
    settings = Settings()
    assert {name: getattr(settings, name) for name in DEFAULTS} == DEFAULTS


def test_recording_is_off_by_default(monkeypatch):
    _clean(monkeypatch)
    assert Settings().PROBE_TRANSCRIPTS == "off"


def test_env_example_lists_each_setting():
    text = (Path(__file__).resolve().parents[2] / ".env.example").read_text()
    for name, value in DEFAULTS.items():
        assert f"#{name}={value}" in text, name
