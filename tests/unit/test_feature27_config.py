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
    armed-PROBE backstop at 64. ENH-006 D83 removed the behaviour count and made the budget 5% of
    the model's own time; the backstop stays 64 (the cost gate binds first on every measured
    model)."""
    monkeypatch.delenv("PROBE_MAX_ARMED", raising=False)
    monkeypatch.delenv("PROBE_SLOWDOWN_BUDGET_PCT", raising=False)
    assert Settings().PROBE_MAX_ARMED == 64
    assert Settings().PROBE_SLOWDOWN_BUDGET_PCT == 5.0


def test_the_behaviour_count_and_recommendation_ceilings_are_retired():
    """D83: an environment that still names them starts, with a warning, and they do nothing."""
    from millm.core.config import RETIRED_SETTINGS

    for name in ("PROBE_MAX_ARMED_BEHAVIOURS", "PROBE_RECOMMEND_TARGET_SETS",
                 "PROBE_RECOMMEND_MAX_DECODE_SHARE", "PROBE_RECOMMEND_MAX_PREFILL_SHARE"):
        assert name not in Settings.model_fields and name in RETIRED_SETTINGS


def test_the_score_probe_cap_is_retired_into_the_armed_cap():
    """037 FR-37.6 / TC-37.3: no separate constant to drift; an old .env naming it still starts."""
    from millm.core.config import RETIRED_SETTINGS

    assert "PROBE_SCORE_MAX_PROBES" not in Settings.model_fields
    assert "PROBE_SCORE_MAX_PROBES" in RETIRED_SETTINGS


def test_env_example_lists_each_setting():
    text = (Path(__file__).resolve().parents[2] / ".env.example").read_text()
    for name, value in DEFAULTS.items():
        assert f"#{name}={value}" in text, name
