"""
Unit tests for the OWUI Cluster Dial Filter Function (Feature 10, R3: the
filter shipped with logic worth pinning but no tests — dict-vs-model valve
shapes, precedence, degradation). The file imports only pydantic, so it
runs in the normal suite with no Open WebUI dependency.
"""

import importlib.util
from pathlib import Path

import pytest

FILTER_PATH = (Path(__file__).resolve().parents[3]
               / "integrations" / "openwebui" / "millm_dial_filter.py")


@pytest.fixture(scope="module")
def filter_module():
    spec = importlib.util.spec_from_file_location("millm_dial_filter", FILTER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def f(filter_module):
    return filter_module.Filter()


class TestValveShapes:
    """R1 finding: OWUI versions deliver __user__['valves'] as a pydantic
    model OR a plain dict — getattr on a dict silently dropped the dial."""

    def test_pydantic_model_valves(self, filter_module, f):
        uv = filter_module.Filter.UserValves(dial="max")
        assert f.inlet({}, {"valves": uv})["steering_intensity"] == "max"

    def test_dict_valves(self, f):
        assert f.inlet({}, {"valves": {"dial": "off"}})["steering_intensity"] == "off"

    def test_dict_custom_lambda(self, f):
        body = f.inlet({}, {"valves": {"dial": "custom", "custom_lambda": 1.5}})
        assert body["steering_intensity"] == 1.5

    def test_missing_user_entirely(self, f):
        assert "steering_intensity" not in f.inlet({"messages": []})


class TestPrecedence:
    def test_user_dial_beats_operator_default(self, f):
        f.valves.default_dial = "max"
        assert f.inlet({}, {"valves": {"dial": "off"}})["steering_intensity"] == "off"

    def test_operator_default_fills_in(self, f):
        f.valves.default_dial = "min"
        assert f.inlet({}, {"valves": {"dial": "default"}})["steering_intensity"] == "min"

    def test_operator_custom_default(self, f):
        f.valves.default_dial = "custom"
        f.valves.default_custom_lambda = 0.5
        assert f.inlet({})["steering_intensity"] == 0.5

    def test_server_position_ignores_operator_default(self, f):
        """R3: users need a position meaning 'server state governs' even
        after the operator sets a default."""
        f.valves.default_dial = "max"
        body = f.inlet({"steering_intensity": 0.7},
                       {"valves": {"dial": "server"}})
        assert body["steering_intensity"] == 0.7  # untouched

    def test_resolved_dial_replaces_preexisting_body_field(self, f):
        body = f.inlet({"steering_intensity": 0.7},
                       {"valves": {"dial": "max"}})
        assert body["steering_intensity"] == "max"

    def test_default_valves_leave_scripted_field_alone(self, f):
        body = f.inlet({"steering_intensity": 0.7},
                       {"valves": {"dial": "default"}})
        assert body["steering_intensity"] == 0.7


class TestDegradation:
    """A broken valve must never break the user's chat."""

    def test_master_switch_off(self, f):
        f.valves.enabled = False
        assert "steering_intensity" not in f.inlet({}, {"valves": {"dial": "max"}})

    def test_unknown_dial_string_degrades(self, f):
        assert "steering_intensity" not in f.inlet({}, {"valves": {"dial": "bogus"}})

    def test_out_of_range_custom_degrades(self, f):
        assert "steering_intensity" not in f.inlet(
            {}, {"valves": {"dial": "custom", "custom_lambda": 9}})

    def test_non_numeric_custom_degrades(self, f):
        assert "steering_intensity" not in f.inlet(
            {}, {"valves": {"dial": "custom", "custom_lambda": "wat"}})

    def test_custom_zero_is_off(self, f):
        body = f.inlet({}, {"valves": {"dial": "custom", "custom_lambda": 0.0}})
        assert body["steering_intensity"] == 0.0

    def test_model_rejects_out_of_range_at_valve_level(self, filter_module):
        with pytest.raises(Exception):
            filter_module.Filter.UserValves(custom_lambda=3.0)

    def test_no_millm_imports(self):
        source = FILTER_PATH.read_text()
        assert "from millm" not in source and "import millm" not in source
