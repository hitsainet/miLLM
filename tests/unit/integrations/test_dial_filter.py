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

    def test_no_millm_imports(self, filter_module):
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


# =============================================================================
# Feature 14: circuit-aware status copy (filter v1.4.0)
# =============================================================================


class TestCircuitRungStatus:
    """The dial must say WHAT it is steering with, and must never describe a
    rung<2 circuit as causal."""

    @staticmethod
    def _mk(filter_module, **valve_overrides):
        f = filter_module.Filter()
        for k, v in valve_overrides.items():
            setattr(f.valves, k, v)
        return f

    def test_rung_language_map_mirrors_the_contract_verbatim(self, filter_module):
        f = self._mk(filter_module)
        assert f.RUNG_LANGUAGE[0] == "associated"
        assert f.RUNG_LANGUAGE[1] == "suggested (attribution-supported)"
        assert f.RUNG_LANGUAGE[2] == "causally validated (edge)"
        assert f.RUNG_LANGUAGE[3] == "faithfulness-tested (circuit)"

    def test_no_circuit_adds_no_suffix(self, filter_module):
        assert self._mk(filter_module)._circuit_suffix(None) == ""
        assert self._mk(filter_module)._circuit_suffix({}) == ""

    def test_validated_circuit_named_with_its_phrase(self, filter_module):
        s = self._mk(filter_module)._circuit_suffix(
            {"name": "fear→threat", "rung": 2,
             "rung_language": "causally validated (edge)"}
        )
        assert 'circuit "fear→threat"' in s
        assert "causally validated (edge)" in s
        assert "UNVALIDATED" not in s

    @pytest.mark.parametrize(
        "rung,phrase", [(0, "associated"), (1, "suggested (attribution-supported)")]
    )
    def test_unvalidated_circuit_is_marked_and_never_causal(self, filter_module, rung, phrase):
        s = self._mk(filter_module)._circuit_suffix(
            {"name": "c", "rung": rung, "rung_language": phrase}
        )
        assert "[UNVALIDATED]" in s
        assert "causal" not in s.lower()

    def test_server_phrase_used_when_it_matches_the_mirror(self, filter_module):
        s = self._mk(filter_module)._circuit_suffix(
            {"name": "c", "rung": 0, "rung_language": "associated"}
        )
        assert "associated" in s

    def test_a_spoofed_server_phrase_cannot_inject_causal_language(self, filter_module):
        """R1 security: the filter renders server text verbatim into the chat,
        so a spoofed/MITM'd endpoint could claim a rung-0 circuit is causal.
        The phrase is now validated against the mirrored vocabulary."""
        s = self._mk(filter_module)._circuit_suffix(
            {"name": "c", "rung": 0, "rung_language": "causally validated (edge)"}
        )
        assert "causal" not in s.lower()
        assert "associated" in s
        assert "[UNVALIDATED]" in s

    def test_missing_rung_language_falls_back_to_the_mirror(self, filter_module):
        s = self._mk(filter_module)._circuit_suffix({"name": "c", "rung": 1})
        assert "suggested (attribution-supported)" in s

    def test_malformed_rung_degrades_to_unvalidated_not_causal(self, filter_module):
        s = self._mk(filter_module)._circuit_suffix({"name": "c", "rung": "garbage"})
        assert "[UNVALIDATED]" in s
        assert "causal" not in s.lower()

    def test_slice_fallback_is_disclosed(self, filter_module):
        s = self._mk(filter_module)._circuit_suffix(
            {"name": "c", "rung": 2, "rung_language": "causally validated (edge)",
             "serving_mode": "slice_fallback"}
        )
        assert "per-layer SLICE" in s
        assert "not the whole circuit" in s

    async def test_probe_disabled_by_valve(self, filter_module):
        f = self._mk(filter_module, show_circuit_rung=False)
        assert await f._circuit_status() is None

    async def test_probe_failure_degrades_silently(self, filter_module):
        """miLLM down / older build without the route must not break chat."""
        f = self._mk(filter_module, millm_base_url="http://127.0.0.1:9")  # nothing listening
        assert await f._circuit_status() is None

    async def test_empty_base_url_skips_the_probe(self, filter_module):
        f = self._mk(filter_module, millm_base_url="")
        assert await f._circuit_status() is None
