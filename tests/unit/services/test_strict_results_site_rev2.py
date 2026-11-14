"""Rev2 review of 040 P0 (2026-11-08): the converted strict-mode site in `inference_service.py`.

Control C1 (`if strict_results():` → `if False:`) left every test green: nothing drove a builder
defect through `InferenceService._probe_publish_results`, the one place that decides whether a
defect is raised (tests, `MISERVER_STRICT_RESULTS=1`) or logged and dropped (production). With the
line broken, every inference-path test would swallow a broken result builder.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from miserver.core import legacy_names
from miserver.services import probe_result
from miserver.services.inference_service import InferenceService

NEW = "MISERVER_STRICT_RESULTS"
(OLD,) = [a.old for a in legacy_names.ENV_ALIASES if a.new == NEW]


class _Defect(RuntimeError):
    pass


@pytest.fixture
def broken_builder(monkeypatch):
    def build(*_a, **_k):
        raise _Defect("builder defect")

    monkeypatch.setattr(probe_result, "build_probe_results", build)
    for name in (NEW, OLD):
        monkeypatch.delenv(name, raising=False)
    legacy_names.drain_hits()
    yield
    legacy_names.drain_hits()


def test_strict_raises_a_builder_defect(broken_builder, monkeypatch):
    monkeypatch.setenv(NEW, "1")
    with pytest.raises(_Defect):
        InferenceService._probe_publish_results(SimpleNamespace(results="sentinel"), [])


def test_the_retired_old_name_no_longer_makes_it_strict(broken_builder, monkeypatch):
    """040 P7 (FR-040.40): the old name alone is retired, so production behaviour (log and drop)
    holds, and nothing is recorded."""
    monkeypatch.setenv(OLD, "1")
    context = SimpleNamespace(results="sentinel")
    assert InferenceService._probe_publish_results(context, []) is None
    assert ("env", OLD) not in legacy_names.peek_hits()


def test_production_logs_and_drops(broken_builder):
    context = SimpleNamespace(results="sentinel")
    assert InferenceService._probe_publish_results(context, []) is None
    assert context.results is None


def test_the_suite_runs_strict():
    """Control C7: `tests/conftest.py` setting the switch to "0" (or to a name nothing reads) left
    every test green; the suite would then stop raising builder defects everywhere."""
    assert probe_result.strict_results(), "tests/conftest.py must turn strict results on"
