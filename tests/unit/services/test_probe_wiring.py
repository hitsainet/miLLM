"""The serving-path wiring: where the probe lifecycle is called, and in what order.

⚠ ORDER IS THE WHOLE POINT HERE, not merely that the calls exist.

`_probe_finish` must run **before** the response is committed — before the non-streaming route
reads the ContextVar for its header, and before the streaming path yields its terminal chunk. A
verdict computed in a `finally` is a verdict computed after the stream has closed, with nothing
left to attach it to. The FTDD names that hazard explicitly; these tests are what hold it.

And `_probe_record` must run in the `finally`, so an event exists even when generation failed. A
request that crashed and a request nobody monitored look identical in the event table otherwise.
"""

from __future__ import annotations

import inspect
import re

import pytest

from millm.api.routes.openai.chat import build_probe_verdicts_header
from millm.services.inference_service import (
    InferenceService,
    _verdict_payload,
    get_probe_verdicts,
    reset_steering_memo,
    set_probe_verdicts,
)
from millm.services.probe_runtime import Verdict


def source_of(name: str) -> str:
    return inspect.getsource(getattr(InferenceService, name))


def verdict(**over) -> Verdict:
    base = dict(
        probe_id="pr_1", name="high-stakes", rung=3,
        rung_language="detects on unseen tasks, compared with a judge",
        scored=True, score=2.31, threshold=1.07, fires=True,
    )
    base.update(over)
    return Verdict(**base)


class TestEveryServingPathIsWired:
    """Each path must begin, finish and record. A path that begins and never finishes produces
    a context that is never closed, and the NEXT request's `begin_request` raises."""

    @pytest.mark.parametrize(
        "method",
        ["create_chat_completion", "stream_chat_completion", "create_text_completion",
         "_cbm_stream_chat_completion"],
    )
    def test_it_begins_finishes_and_records(self, method):
        code = source_of(method)
        assert "_probe_begin(" in code, f"{method} never opens a probe context"
        assert "_probe_finish(" in code, f"{method} never computes a verdict"
        assert "_probe_record(" in code, f"{method} never persists an event"

    def test_the_cbm_streaming_path_is_included(self):
        """⚠ Task 5.8 — no task in the original chain covered this path.

        `PROBE_FORCE_SERIAL` keeps continuous batching out while it is true, but it is a SETTING,
        and this path does no sensing at all. With the flag off and a probe armed, the verdict
        would simply be absent: no header, no chunk, no event, no reason.
        """
        code = source_of("_cbm_stream_chat_completion")
        assert 'mark_not_scored("continuous_batching")' in code


class TestFinishRunsBeforeTheResponseIsCommitted:
    def test_streaming_finishes_before_the_terminal_chunk(self):
        """A `finally` runs after the stream has closed. There would be nothing to attach to.

        ⚠ ANCHORED ON THE NORMAL TERMINAL PATH, NOT ON `index()`. My first version took the first
        `[DONE]` in the function and failed — `stream_chat_completion` has SEVEN, four of which
        are refusal or error returns that come earlier in the source. Taking the first was the
        "satisfied by the wrong occurrence" trap, and here it produced a false FAILURE; the same
        mistake in the other direction is a false pass.
        """
        code = source_of("stream_chat_completion")
        finish = code.index("_probe_finish(")
        final_chunk = code.index('yield f"data: {final_chunk.model_dump_json')
        done = code.index('yield "data: [DONE]', final_chunk)   # the one AFTER the final chunk
        assert finish < final_chunk < done, "the verdict is computed too late to be delivered"

    def test_the_probe_chunk_sits_between_the_final_chunk_and_DONE(self):
        code = source_of("stream_chat_completion")
        final_chunk = code.index('yield f"data: {final_chunk.model_dump_json')
        chunk = code.index("_probe_stream_chunk(", final_chunk)
        done = code.index('yield "data: [DONE]', chunk)
        assert final_chunk < chunk < done

    def test_an_error_path_still_writes_an_EVENT_even_though_it_sends_no_chunk(self):
        """⚠ A DELIBERATE ASYMMETRY, found by making the test above precise.

        Four `[DONE]` emissions in this function are refusals or error returns, and three of them
        occur after the probe context is open. None sends a verdict chunk, and that is right: the
        client is receiving an error, not a completion, and bolting a verdict onto it would
        suggest the request was scored and served normally.

        What must NOT happen is silence in the event log — a request that failed and a request
        nobody monitored would then be indistinguishable. `_probe_record` lives in the `finally`,
        so every one of those paths still records, carrying whatever the probe saw of the prompt.
        """
        code = source_of("stream_chat_completion")
        begin = code.index("_probe_begin(")
        record = code.index("_probe_record(")
        dones_after_begin = [
            m for m in range(len(code))
            if code.startswith('yield "data: [DONE]', m) and m > begin
        ]
        assert len(dones_after_begin) >= 3, "expected several post-begin exits to cover"
        assert code.index("finally:", begin) < record, "record must run for every exit"

    def test_non_streaming_finishes_before_the_finally(self):
        code = source_of("create_chat_completion")
        assert code.index("_probe_finish(") < code.index("finally:")

    def test_record_runs_in_the_finally_so_a_crash_still_leaves_an_event(self):
        """A request that crashed and a request nobody monitored must not look identical."""
        for method in ("create_chat_completion", "stream_chat_completion",
                       "create_text_completion"):
            code = source_of(method)
            assert code.index("finally:") < code.index("_probe_record("), method


class TestContinuousBatchingIsForcedSerial:
    def test_the_cbm_gate_asks_the_probe_registry(self):
        """⚠ Not the SAE registry.

        The sensing clause reads `AttachedSAEState`, which cannot see a probe — a probe sits on
        any layer and needs no attached SAE. Copying that shape would have produced a guard that
        never fires. The circuit clause asks its own service, and this follows it.
        """
        code = source_of("_use_cbm_for_request")
        assert "PROBE_FORCE_SERIAL" in code
        assert "ProbeRuntimeState().has_armed()" in code
        assert 'reason="probes_armed"' in code

    def test_the_probe_clause_does_not_read_AttachedSAEState(self):
        code = source_of("_use_cbm_for_request")
        probe_clause = code[code.index("PROBE_FORCE_SERIAL"):]
        probe_clause = probe_clause[: probe_clause.index("_cbm_force_serial_monitoring")]
        assert "AttachedSAEState" not in probe_clause


class TestBatchedRequestsSayWhy:
    def test_n_greater_than_one_is_marked_not_scored(self):
        code = source_of("create_chat_completion")
        assert 'mark_not_scored("batched_request")' in code

    def test_multi_prompt_completions_are_marked_not_scored(self):
        code = source_of("create_text_completion")
        assert 'mark_not_scored("batched_request")' in code


class TestTheHungThreadGuard:
    def test_probes_are_disarmed_where_sensing_is(self):
        """A woken hung thread's forward pass would call the probe hook into the NEXT request's
        context, reporting one conversation's verdict against another's id."""
        code = source_of("stream_chat_completion")
        assert 'disarm_all("generation_thread_hung")' in code

    def test_the_reason_is_recorded_not_just_the_flag(self):
        code = source_of("stream_chat_completion")
        assert 'mark_all_disarmed("generation_thread_hung")' in code


class TestTheVerdictContextVar:
    def test_it_is_cleared_by_the_existing_request_reset(self):
        """⚠ A route that has to remember TWO resets is a route that will one day remember one.

        A stale list would attach one request's verdicts to another's response — reporting a
        concept as detected in a conversation that never contained it.
        """
        set_probe_verdicts([verdict()])
        assert get_probe_verdicts()
        reset_steering_memo()
        assert get_probe_verdicts() == []


class TestTheHeader:
    def test_the_documented_example_is_produced_exactly(self):
        assert build_probe_verdicts_header([verdict()]) == (
            '"high-stakes";score=2.31;threshold=1.07;verdict=?1;rung=3'
        )

    def test_nothing_armed_means_no_header_at_all(self):
        """An unarmed server's response must be byte-identical to what it was before F024."""
        assert build_probe_verdicts_header([]) == ""

    def test_members_are_sorted_by_name_for_a_deterministic_header(self):
        header = build_probe_verdicts_header(
            [verdict(name="zz"), verdict(name="aa"), verdict(name="mm")]
        )
        assert [m.split(";")[0] for m in header.split(", ")] == ['"aa"', '"mm"', '"zz"']

    def test_not_scored_members_carry_their_reason(self):
        header = build_probe_verdicts_header(
            [verdict(scored=False, not_scored_reason="speculative_decoding",
                     score=None, threshold=None, fires=None)]
        )
        assert 'not-scored;reason="speculative_decoding"' in header
        assert "score=" not in header

    def test_no_threshold_means_NO_verdict_parameter(self):
        """⚠ `verdict=?0` would report a decision the probe never made.

        A probe with no threshold ranks without deciding, so the parameter is omitted entirely.
        """
        header = build_probe_verdicts_header([verdict(threshold=None, fires=None)])
        assert "verdict=" not in header
        assert "threshold=" not in header
        assert "score=2.31" in header

    def test_it_stays_a_single_header_line(self):
        header = build_probe_verdicts_header([verdict(name="a"), verdict(name="b")])
        assert "\n" not in header and "\r" not in header

    def test_quotes_in_a_name_cannot_break_the_structure(self):
        header = build_probe_verdicts_header([verdict(name='ev"il')])
        assert header.count('"') == 2


class TestTheStreamPayload:
    def test_a_scored_verdict_carries_its_numbers(self):
        payload = _verdict_payload(verdict())
        assert payload == {
            "name": "high-stakes", "scored": True, "score": 2.31, "threshold": 1.07,
            "verdict": True, "rung": 3,
            "rung_language": "detects on unseen tasks, compared with a judge",
        }

    def test_an_unscored_verdict_carries_NO_numbers(self):
        """Reporting a score of 0 for a request nobody scored would be a measurement never taken."""
        payload = _verdict_payload(
            verdict(scored=False, not_scored_reason="batched_request",
                    score=None, threshold=None, fires=None)
        )
        assert payload["scored"] is False
        assert payload["reason"] == "batched_request"
        assert "score" not in payload and "verdict" not in payload

    def test_the_rung_language_travels_so_no_client_maps_a_number_to_a_phrase(self):
        assert "rung_language" in _verdict_payload(verdict())
