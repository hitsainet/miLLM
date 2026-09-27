"""Probe rung language: identical to miStudio's, and never overclaiming.

Two guarantees.

**Identity with miStudio.** Both systems describe the same probe to a human. If the strings drift,
the same probe means one thing in the tool that trained it and another in the runtime that serves
it — and nobody would see a failure, only two sentences that disagree.

**The honesty contract.** A probe makes no causal claim. `PROBE_FORBIDDEN_WORDS` may never appear
in its language, enforced over the dictionaries rather than trusted to review, so the rule cannot be
broken by a rewording.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

from millm.core.circuit_evidence import RUNG_LANGUAGE as CIRCUIT_LANGUAGE
from millm.core.probe_evidence import (
    ARM_WITHOUT_ACKNOWLEDGEMENT_MIN_RUNG,
    PROBE_FORBIDDEN_WORDS,
    PROBE_RUNG_LANGUAGE,
    PROBE_RUNG_NEXT_STEP,
    ProbeRung,
    needs_arm_acknowledgement,
    probe_rung_language,
    probe_rung_next_step,
)

MISTUDIO_LADDER = Path(os.environ.get("MISTUDIO_REPO", "/home/x-sean/app/miStudio")) / (
    "backend/src/schemas/evidence_ladder.py"
)
REQUIRED = os.environ.get("MILLM_REQUIRE_CROSS_REPO_CHECKS") == "1"


def _mistudio():
    if not MISTUDIO_LADDER.exists():
        if REQUIRED:
            pytest.fail(
                f"MILLM_REQUIRE_CROSS_REPO_CHECKS=1 but miStudio's ladder is not at "
                f"{MISTUDIO_LADDER}"
            )
        pytest.skip(f"miStudio not checked out at {MISTUDIO_LADDER}")
    spec = importlib.util.spec_from_file_location("mistudio_evidence_ladder", MISTUDIO_LADDER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestIdenticalToMiStudio:
    def test_every_rung_phrase_matches(self):
        mis = _mistudio()
        ours = {int(k): v for k, v in PROBE_RUNG_LANGUAGE.items()}
        theirs = {int(k): v for k, v in mis.PROBE_RUNG_LANGUAGE.items()}
        assert ours == theirs, "probe rung language has drifted from miStudio"

    def test_every_next_step_matches(self):
        mis = _mistudio()
        ours = {int(k): v for k, v in PROBE_RUNG_NEXT_STEP.items()}
        theirs = {int(k): v for k, v in mis.PROBE_RUNG_NEXT_STEP.items()}
        assert ours == theirs

    def test_the_forbidden_word_list_matches(self):
        mis = _mistudio()
        assert set(PROBE_FORBIDDEN_WORDS) == set(mis.PROBE_FORBIDDEN_WORDS)

    def test_the_ladder_has_the_same_four_rungs(self):
        mis = _mistudio()
        assert [(r.name, int(r)) for r in ProbeRung] == [
            (r.name, int(r)) for r in mis.ProbeRung
        ]


class TestTheHonestyContract:
    @pytest.mark.parametrize("rung", list(ProbeRung))
    def test_no_rung_phrase_uses_a_forbidden_word(self, rung):
        phrase = PROBE_RUNG_LANGUAGE[rung].lower()
        for word in PROBE_FORBIDDEN_WORDS:
            assert word not in phrase, f"rung {int(rung)} says {phrase!r}, which contains {word!r}"

    @pytest.mark.parametrize("rung", list(ProbeRung))
    def test_no_next_step_uses_a_forbidden_word(self, rung):
        phrase = PROBE_RUNG_NEXT_STEP[rung].lower()
        for word in PROBE_FORBIDDEN_WORDS:
            assert word not in phrase, f"next step {int(rung)} contains {word!r}"

    def test_the_top_rung_says_COMPARED_not_BEATS(self):
        """⚠ Load-bearing wording, and measured.

        On the reference run the judge won: 0.8744 mean AUROC against the 1.2B probe's 0.7938, on
        all five sets individually. A rung-3 badge reading "beats a judge" would be false for that
        probe, and the probe's case on a small model is cost and latency, not accuracy.
        """
        top = PROBE_RUNG_LANGUAGE[ProbeRung.JUDGE_COMPARED]
        assert "compared with a judge" in top
        assert "beat" not in top.lower()

    def test_probe_language_is_not_circuit_language(self):
        """A probe borrowing the circuit ladder would call a rung-2 detector "causally validated
        (edge)" — not a weaker version of the truth, a different claim entirely."""
        assert set(PROBE_RUNG_LANGUAGE.values()).isdisjoint(set(CIRCUIT_LANGUAGE.values()))


class TestAccessors:
    def test_language_and_next_step_accept_a_plain_int(self):
        assert probe_rung_language(2) == "detects on unseen tasks"
        assert probe_rung_next_step(3) == "top rung — nothing further"

    def test_an_unknown_rung_raises_rather_than_returning_a_default(self):
        """A default phrase for an unrecognised rung would describe evidence nobody established."""
        with pytest.raises(ValueError):
            probe_rung_language(7)

    @pytest.mark.parametrize("rung,needed", [(0, True), (1, True), (2, False), (3, False)])
    def test_acknowledgement_is_needed_below_rung_two(self, rung, needed):
        assert needs_arm_acknowledgement(rung) is needed

    def test_the_threshold_is_unseen_tasks(self):
        assert ARM_WITHOUT_ACKNOWLEDGEMENT_MIN_RUNG == ProbeRung.UNSEEN_TASKS
