"""The conversation that proved a constant threshold is wrong, kept as a regression case.

⚠ **THE SAME SENTENCE, BYTE FOR BYTE, FIRED AND THEN WENT SILENT.** On 2026-10-01 a person
sent one message, exchanged two short unrelated turns, and re-sent the identical message. The
probe, its weights and its threshold were unchanged throughout:

    turn 1    86 scored tokens   score 27.7037   FIRES
    turn 2   497                       15.2069   fires
    turn 3   713                        8.0504   fires by 1.1%
    turn 4   985                        7.5061   SILENT   (bar 7.9596)

Nothing about the input changed between turns 1 and 4 — same string, same sha256. The only
variable was the conversation accumulated around it. That is why this estate now carries a
threshold per input-length band rather than one constant bar.

It is a better experiment than the length-stratified AUROC that motivated the fix, because it
has no confounds at all: no rephrasing, no different topic, no change in how much the model
said. It happened by accident and it is worth keeping.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from millm.services.probe_runtime import threshold_for_length

FIXTURE = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "probe_length_drift_conversation.json"


@pytest.fixture(scope="module")
def case():
    return json.loads(FIXTURE.read_text())


def test_the_fixture_is_present(case):
    """Guard the guard: a missing file would make every assertion below vacuous."""
    assert case["messages"], "no transcript in the fixture"
    assert len(case["observed"]) == 4


def test_turn_four_repeats_turn_one_byte_for_byte(case):
    """The whole value of this case. If a future edit paraphrases either turn, it stops being
    a controlled experiment and becomes an anecdote."""
    msgs = case["messages"]
    first = msgs[0]["content"]
    repeat = msgs[6]["content"]
    assert msgs[0]["role"] == "user" and msgs[6]["role"] == "user"
    assert first == repeat
    assert hashlib.sha256(first.encode()).hexdigest() == case["turn_1_sha256"]
    assert case["turn_1_sha256"] == case["turn_4_sha256"]


def test_the_constant_threshold_inverts_the_verdict(case):
    """What was actually observed: identical input, opposite verdicts, one bar."""
    bar = case["threshold_at_capture"]
    turns = {t["turn"]: t for t in case["observed"]}
    assert turns[1]["score"] > bar, "turn 1 fired"
    assert turns[4]["score"] < bar, "turn 4 did not"
    assert turns[1]["scored_tokens"] < turns[4]["scored_tokens"]


def test_the_score_falls_monotonically_as_the_context_grows(case):
    """The drift itself, in one conversation."""
    scores = [t["score"] for t in sorted(case["observed"], key=lambda t: t["turn"])]
    assert scores == sorted(scores, reverse=True)


def test_a_length_band_table_recovers_the_missed_turn(case):
    """⚠ THE ACCEPTANCE CRITERION FOR THE FIX, written against the real numbers.

    The bands here are the ones the calibration corpus actually yields (OpenHermes negatives:
    0-203, 204-339, 340-518, 519+). A bar cut from band 3's own negatives must catch turn 4
    while band 0's bar still catches turn 1 — and must NOT be so low that it would fire on
    anything, which is what the realised-FPR check in miStudio's calibration enforces.
    """
    turns = {t["turn"]: t for t in case["observed"]}
    bands = [
        {"min_tokens": 0, "max_tokens": 203, "threshold": 7.96},
        {"min_tokens": 204, "max_tokens": 339, "threshold": 7.50},
        {"min_tokens": 340, "max_tokens": 518, "threshold": 7.10},
        {"min_tokens": 519, "max_tokens": None, "threshold": 6.50},
    ]
    for turn in (1, 2, 3, 4):
        bar = threshold_for_length(bands, turns[turn]["scored_tokens"], fallback=7.96)
        assert turns[turn]["score"] > bar, (
            f"turn {turn} ({turns[turn]['scored_tokens']} tokens, score "
            f"{turns[turn]['score']}) still misses its band's bar of {bar}"
        )


def test_the_constant_bar_is_what_failed_not_the_probe(case):
    """Turn 4's score still exceeds every band bar below the constant one — the detector was
    working the whole time, and only the bar was wrong."""
    turns = {t["turn"]: t for t in case["observed"]}
    assert turns[4]["score"] > 6.50
    assert turns[4]["score"] < case["threshold_at_capture"]
