"""
Unit tests for the OWUI Cluster Dial Filter Function (Feature 10, R3: the
filter shipped with logic worth pinning but no tests — dict-vs-model valve
shapes, precedence, degradation). The file imports only pydantic, so it
runs in the normal suite with no Open WebUI dependency.
"""

import asyncio
import importlib.util
from pathlib import Path

import pytest


def inlet(f, *args, **kwargs):
    """v1.3.0: inlet is async (status emitter) — run it to completion."""
    return asyncio.run(f.inlet(*args, **kwargs))

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
        assert inlet(f, {}, {"valves": uv})["steering_intensity"] == "max"

    def test_dict_valves(self, f):
        assert inlet(f, {}, {"valves": {"dial": "off"}})["steering_intensity"] == "off"

    def test_dict_custom_lambda(self, f):
        body = inlet(f, {}, {"valves": {"dial": "custom", "custom_lambda": 1.5}})
        assert body["steering_intensity"] == 1.5

    def test_missing_user_entirely(self, f):
        assert "steering_intensity" not in inlet(f, {"messages": []})


class TestPrecedence:
    def test_user_dial_beats_operator_default(self, f):
        f.valves.default_dial = "max"
        assert inlet(f, {}, {"valves": {"dial": "off"}})["steering_intensity"] == "off"

    def test_operator_default_fills_in(self, f):
        f.valves.default_dial = "min"
        assert inlet(f, {}, {"valves": {"dial": "default"}})["steering_intensity"] == "min"

    def test_operator_custom_default(self, f):
        f.valves.default_dial = "custom"
        f.valves.default_custom_lambda = 0.5
        assert inlet(f, {})["steering_intensity"] == 0.5

    def test_server_position_ignores_operator_default(self, f):
        """R3: users need a position meaning 'server state governs' even
        after the operator sets a default."""
        f.valves.default_dial = "max"
        body = inlet(f, {"steering_intensity": 0.7},
                       {"valves": {"dial": "server"}})
        assert body["steering_intensity"] == 0.7  # untouched

    def test_resolved_dial_replaces_preexisting_body_field(self, f):
        body = inlet(f, {"steering_intensity": 0.7},
                       {"valves": {"dial": "max"}})
        assert body["steering_intensity"] == "max"

    def test_default_valves_leave_scripted_field_alone(self, f):
        body = inlet(f, {"steering_intensity": 0.7},
                       {"valves": {"dial": "default"}})
        assert body["steering_intensity"] == 0.7


class TestDegradation:
    """A broken valve must never break the user's chat."""

    def test_master_switch_off(self, f):
        f.valves.enabled = False
        assert "steering_intensity" not in inlet(f, {}, {"valves": {"dial": "max"}})

    def test_unknown_dial_string_degrades(self, f):
        assert "steering_intensity" not in inlet(f, {}, {"valves": {"dial": "bogus"}})

    def test_out_of_range_custom_degrades(self, f):
        assert "steering_intensity" not in inlet(f, 
            {}, {"valves": {"dial": "custom", "custom_lambda": 9}})

    def test_non_numeric_custom_degrades(self, f):
        assert "steering_intensity" not in inlet(f, 
            {}, {"valves": {"dial": "custom", "custom_lambda": "wat"}})

    def test_custom_zero_is_off(self, f):
        body = inlet(f, {}, {"valves": {"dial": "custom", "custom_lambda": 0.0}})
        assert body["steering_intensity"] == 0.0

    def test_model_rejects_out_of_range_at_valve_level(self, filter_module):
        with pytest.raises(Exception):
            filter_module.Filter.UserValves(custom_lambda=3.0)

    def test_no_millm_imports(self):
        source = FILTER_PATH.read_text()
        assert "from millm" not in source and "import millm" not in source


class TestChipAndStatus:
    """v1.3.0 UX: per-chat toggle chip + status line in the chat."""

    def test_toggle_chip_and_icon_declared(self, f):
        assert f.toggle is True
        assert f.icon.startswith("data:image/svg+xml;base64,")

    def _emitter(self, sink):
        async def emit(event):
            sink.append(event)
        return emit

    def test_status_reports_the_injected_dial(self, f):
        events = []
        body = inlet(f, {}, {"valves": {"dial": "max"}},
                     __event_emitter__=self._emitter(events))
        assert body["steering_intensity"] == "max"
        assert events and events[0]["type"] == "status"
        assert "max" in events[0]["data"]["description"]

    def test_status_off_and_custom_wordings(self, f):
        events = []
        inlet(f, {}, {"valves": {"dial": "off"}},
              __event_emitter__=self._emitter(events))
        inlet(f, {}, {"valves": {"dial": "custom", "custom_lambda": 1.5}},
              __event_emitter__=self._emitter(events))
        assert "off for this reply" in events[0]["data"]["description"]
        assert "λ=1.5" in events[1]["data"]["description"]

    def test_idle_dial_emits_teaching_hint_and_injects_nothing(self, f):
        events = []
        body = inlet(f, {}, {"valves": {"dial": "default"}},
                     __event_emitter__=self._emitter(events))
        assert "steering_intensity" not in body
        assert "server steering governs" in events[0]["data"]["description"]

    def test_show_status_false_silences(self, f):
        f.valves.show_status = False
        events = []
        inlet(f, {}, {"valves": {"dial": "max"}},
              __event_emitter__=self._emitter(events))
        assert events == []

    def test_broken_emitter_never_breaks_the_chat(self, f):
        async def broken(event):
            raise RuntimeError("boom")
        body = inlet(f, {}, {"valves": {"dial": "max"}},
                     __event_emitter__=broken)
        assert body["steering_intensity"] == "max"

    def test_no_emitter_still_works(self, f):
        assert inlet(f, {}, {"valves": {"dial": "min"}})["steering_intensity"] == "min"
