"""Feature 19's switch defaults ON (operator decision D19, 2026-10-16) and is documented where an
operator looks: `.env.example` and the configuration reference."""

from pathlib import Path

from millm.core.config import Settings

ROOT = Path(__file__).resolve().parents[2]


def test_concurrent_circuit_serving_is_on_by_default(monkeypatch):
    monkeypatch.delenv("CIRCUIT_ALLOW_CONCURRENT", raising=False)
    assert Settings().CIRCUIT_ALLOW_CONCURRENT is True


def test_it_can_still_be_switched_off(monkeypatch):
    monkeypatch.setenv("CIRCUIT_ALLOW_CONCURRENT", "false")
    assert Settings().CIRCUIT_ALLOW_CONCURRENT is False


def test_env_example_names_the_default():
    assert "#CIRCUIT_ALLOW_CONCURRENT=true" in (ROOT / ".env.example").read_text()


def test_the_configuration_reference_names_the_default():
    text = (ROOT / "manual" / "docs" / "reference" / "configuration.md").read_text()
    assert "| `CIRCUIT_ALLOW_CONCURRENT` | `true` |" in text
