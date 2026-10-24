"""Feature 27's settings load with the FTID §9 defaults, and `.env.example` names each one."""

from pathlib import Path

from millm.core.config import Settings

DEFAULTS = {
    "PROBE_SCORE_MAX_INPUTS": 64,
    "SAE_ACTIVATIONS_MAX_TOP_K": 64,
    "SAE_ACTIVATIONS_MAX_ENTRIES": 65536,
    "SAE_ACTIVATIONS_ENCODE_CHUNK": 512,
}


def test_defaults_load(monkeypatch):
    for name in DEFAULTS:
        monkeypatch.delenv(name, raising=False)
    settings = Settings()
    assert {name: getattr(settings, name) for name in DEFAULTS} == DEFAULTS


def test_the_armed_cap_default_is_the_measured_one(monkeypatch):
    """037 FR-37.4 set it from the hardware record (4); 039 (D13, TD39-14) demoted it to the
    armed-PROBE backstop at 64, because the count limit now counts BEHAVIOURS
    (`PROBE_MAX_ARMED_BEHAVIOURS`, 2) and the cost gate charges every probe of every set."""
    monkeypatch.delenv("PROBE_MAX_ARMED", raising=False)
    monkeypatch.delenv("PROBE_MAX_ARMED_BEHAVIOURS", raising=False)
    assert Settings().PROBE_MAX_ARMED == 64
    assert Settings().PROBE_MAX_ARMED_BEHAVIOURS == 2


def test_the_score_probe_cap_is_retired_into_the_armed_cap():
    """037 FR-37.6 / TC-37.3: no separate constant to drift; an old .env naming it still starts."""
    from millm.core.config import RETIRED_SETTINGS

    assert "PROBE_SCORE_MAX_PROBES" not in Settings.model_fields
    assert "PROBE_SCORE_MAX_PROBES" in RETIRED_SETTINGS


def test_env_example_lists_each_setting():
    text = (Path(__file__).resolve().parents[2] / ".env.example").read_text()
    for name, value in DEFAULTS.items():
        assert f"#{name}={value}" in text, name
