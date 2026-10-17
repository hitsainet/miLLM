"""Feature 27's settings load with the FTID §9 defaults, and `.env.example` names each one."""

from pathlib import Path

from millm.core.config import Settings

DEFAULTS = {
    "PROBE_SCORE_MAX_INPUTS": 64,
    "PROBE_SCORE_MAX_PROBES": 8,
    "SAE_ACTIVATIONS_MAX_TOP_K": 64,
    "SAE_ACTIVATIONS_MAX_ENTRIES": 65536,
    "SAE_ACTIVATIONS_ENCODE_CHUNK": 512,
}


def test_defaults_load(monkeypatch):
    for name in DEFAULTS:
        monkeypatch.delenv(name, raising=False)
    settings = Settings()
    assert {name: getattr(settings, name) for name in DEFAULTS} == DEFAULTS


def test_probe_score_cap_matches_the_armed_cap():
    assert Settings().PROBE_SCORE_MAX_PROBES == Settings().PROBE_MAX_ARMED


def test_env_example_lists_each_setting():
    text = (Path(__file__).resolve().parents[2] / ".env.example").read_text()
    for name, value in DEFAULTS.items():
        assert f"#{name}={value}" in text, name
