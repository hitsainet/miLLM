"""A threshold that varies with input length, served.

⚠ **ONE CONSTANT THRESHOLD IS MISCALIBRATED AT EVERY LENGTH BUT THE ONE IT WAS CUT AT.** A
probe's score drifts with how many tokens were scored, in a direction that depends on the
corpus. miStudio measured, on a `mean` probe across five out-of-distribution sets:

    anthropic_hh     realised FPR 0.0030 -> 0.0161 across length quartiles (5.4x its budget)
    mental_health    recall       0.500  -> 0.297

The second is the direction that took a live monitor here silent on turn four of a real
conversation — 27.70 on the person's first message, 7.51 on turn four against a 7.96 bar, while
that same sentence still carried the probe's two highest-scoring tokens throughout.

`decision.length_bands` is additive: a document without it behaves exactly as before.
"""

from __future__ import annotations

import pytest

from millm.services.probe_arming import length_bands_from_definition
from millm.services.probe_runtime import threshold_for_length


def _doc(bands):
    return {"decision": {"threshold": 9.0, "length_bands": bands}}


BANDS = [
    {"min_tokens": 0, "max_tokens": 99, "threshold": 2.0},
    {"min_tokens": 100, "max_tokens": 499, "threshold": 5.0},
    {"min_tokens": 500, "max_tokens": None, "threshold": 11.0},
]


class TestReadingTheTableFromAnotherRepositorysDocument:
    def test_a_well_formed_table_is_read(self):
        assert length_bands_from_definition(_doc(BANDS)) == BANDS

    def test_absent_is_empty_not_an_error(self):
        """Every document written before 2026-10-02 has no table and must still arm."""
        assert length_bands_from_definition({"decision": {"threshold": 9.0}}) == []
        assert length_bands_from_definition({}) == []
        assert length_bands_from_definition(None) == []

    @pytest.mark.parametrize(
        "bands,why",
        [
            ([{"min_tokens": 0, "max_tokens": 99, "threshold": None}], "a null threshold"),
            ([{"min_tokens": 0, "max_tokens": 99, "threshold": True}], "a bool, not a number"),
            ([{"min_tokens": 0, "max_tokens": 99, "threshold": 1.0}], "a closed last band"),
            ([{"min_tokens": -1, "max_tokens": None, "threshold": 1.0}], "a negative bound"),
            (["not a dict"], "a non-dict entry"),
            ([], "an empty list"),
        ],
    )
    def test_a_torn_table_is_refused_WHOLE_rather_than_patched(self, bands, why):
        """⚠ A TABLE WITH A HOLE IS WORSE THAN NO TABLE.

        Dropping the bad entry would leave a gap, and a length falling in that gap would get
        no bar at all — silently, at serve time, on exactly the long inputs the drift hurts
        most. Returning `[]` falls back to the single threshold, which is a defined behaviour.
        """
        assert length_bands_from_definition(_doc(bands)) == [], why

    def test_a_closed_last_band_is_refused_because_long_inputs_would_escape(self):
        closed = [{"min_tokens": 0, "max_tokens": 99, "threshold": 2.0}]
        assert length_bands_from_definition(_doc(closed)) == []


class TestTheBarAVerdictIsJudgedAgainst:
    @pytest.mark.parametrize(
        "n_tokens,expected",
        [(0, 2.0), (99, 2.0), (100, 5.0), (499, 5.0), (500, 11.0), (10_000_000, 11.0)],
    )
    def test_each_length_lands_in_its_band(self, n_tokens, expected):
        assert threshold_for_length(BANDS, n_tokens, fallback=9.0) == expected

    def test_no_table_falls_back_to_the_single_threshold(self):
        assert threshold_for_length([], 250, fallback=9.0) == 9.0
        assert threshold_for_length(None, 250, fallback=9.0) == 9.0

    def test_a_null_threshold_in_a_band_falls_back_rather_than_firing_on_nothing(self):
        """`threshold: None` means FIRE ON NOTHING at the top level. A band must not be able to
        smuggle that in as a bar of its own."""
        bands = [{"min_tokens": 0, "max_tokens": None, "threshold": None}]
        assert threshold_for_length(bands, 50, fallback=9.0) == 9.0

    def test_the_production_case(self):
        """The conversation that started this: the same disclosure at turn 1 and turn 4.

        The prompt window grew 86 -> 985 scored tokens and the score fell 27.70 -> 7.51 against
        a constant 7.96 bar, so turn 4 went silent. With a bar cut for its own length band, the
        long turn is judged against a bar appropriate to it.
        """
        bands = [
            {"min_tokens": 0, "max_tokens": 200, "threshold": 7.96},
            {"min_tokens": 201, "max_tokens": None, "threshold": 6.50},
        ]
        assert 27.70 > threshold_for_length(bands, 86, 7.96)    # turn 1 fires, as it did
        assert 7.51 > threshold_for_length(bands, 985, 7.96)    # turn 4 now fires too
        assert not 7.51 > 7.96                                  # and did not, before


class TestTheRuntimeActuallyUsesIt:
    def test_the_armed_probe_carries_the_table(self):
        from millm.services.probe_runtime import ArmedProbe

        assert "length_bands" in ArmedProbe.__dataclass_fields__

    def test_arming_reads_the_table_off_the_definition(self):
        """Asserted on the CALL: a reader nothing calls is a column of empty lists."""
        import ast
        import inspect

        from millm.services import probe_arming

        tree = ast.parse(inspect.getsource(probe_arming))
        calls = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and getattr(n.func, "id", None) == "length_bands_from_definition"
        ]
        assert calls, "armed_probe_from_row never reads decision.length_bands"

    def test_the_verdict_refines_its_threshold_by_length(self):
        import ast
        import inspect

        from millm.services import probe_runtime

        tree = ast.parse(inspect.getsource(probe_runtime))
        calls = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and getattr(n.func, "id", None) == "threshold_for_length"
        ]
        assert calls, "the verdict path never consults the length bands"

    def test_the_verdict_reports_the_threshold_it_actually_used(self):
        """⚠ `base` IS BUILT BEFORE THE TOKEN COUNT IS KNOWN.

        Firing against a length-band bar while reporting the window bar would make every
        verdict's own `threshold` field a quiet lie — and it is the field a reader checks the
        score against.
        """
        import inspect

        from millm.services import probe_runtime

        src = inspect.getsource(probe_runtime.ProbeRequestContext._verdict_for)
        assert '**{**base, "threshold": threshold}' in src, (
            "the final Verdict spreads `base` unmodified, so it reports the pre-refinement bar"
        )
