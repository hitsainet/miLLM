"""Probe evidence-rung vocabulary (Feature 24) — mirrored VERBATIM from miStudio.

This is the ONE source of user-facing probe rung language in miLLM: the runtime, the management
API, the MCP surface and the admin UI all render `rung_language(rung)` rather than hand-writing
copy per surface.

⚠ **A PROBE HAS ITS OWN LADDER; IT DOES NOT BORROW THE CIRCUIT ONE.** A circuit's rungs are claims
about *mechanism*, rising to a statement about intervention. A probe makes no such claim at any
rung: it is a detector, and its evidence is whether it detects its concept on data unlike its
training data, and how it compares with an LLM monitor. Rendering a probe through
`circuit_evidence.RUNG_LANGUAGE` would describe a rung-2 detector in the vocabulary of a validated
intervention — not a weaker version of the truth, a different claim entirely.

(The circuit phrases are deliberately not reproduced here. `test_circuit_evidence.py`'s copy audit
forbids hand-writing them on a runtime surface, and it caught this docstring doing exactly that on
2026-09-27 — which is the guard working, not a false positive: a phrase typed into a module is a
phrase that can drift from the one `rung_language()` returns.)

**The honesty contract.** `PROBE_FORBIDDEN_WORDS` may never appear in probe language: *causal*
because a probe makes no causal claim; *validated* and *guarantee* because an AUROC interval is
evidence of detection on some data, not a warrant; *safe* because a detector says what it detected,
never that a system is safe. `tests/unit/core/test_probe_evidence.py` enforces this over the
dictionaries below, so the rule cannot be broken by a rewording.

The strings are byte-identical to miStudio's `backend/src/schemas/evidence_ladder.py` — do not
paraphrase them. A divergence between the authoring tool and the serving runtime means the same
probe is described two different ways depending on which system you asked, and the cross-repo test
in `tests/unit/core/test_probe_evidence.py` pins them together.
"""

from __future__ import annotations

from enum import IntEnum


class ProbeRung(IntEnum):
    """A probe's evidence, from the evaluations ACTUALLY RUN.

    Each rung is a claim about generalization, never about mechanism:

      0 TRAINED         the probe exists
      1 HELD_OUT        the AUROC CI lower bound is above 0.5 on at least one IN-distribution
                        evaluation set
      2 UNSEEN_TASKS    the CI lower bound is above 0.5 on EVERY out-of-distribution set run,
                        and at least one was run
      3 JUDGE_COMPARED  rung 2, plus a completed judge run on those same sets

    ⚠ The rung is the highest PASSED, not a chain that must be walked. A probe with no
    in-distribution set can reach rung 2 without ever claiming rung 1 — which is the actual state
    of every probe on the miStudio estate today, and is why the reasons travel beside the number.
    """

    TRAINED = 0
    HELD_OUT = 1
    UNSEEN_TASKS = 2
    JUDGE_COMPARED = 3


#: The rung at or above which a probe may be armed without an explicit operator acknowledgement.
ARM_WITHOUT_ACKNOWLEDGEMENT_MIN_RUNG = ProbeRung.UNSEEN_TASKS

#: Words probe language may never use. See the module docstring.
PROBE_FORBIDDEN_WORDS: tuple[str, ...] = ("causal", "safe", "guarantee", "validated")

#: The ONLY source of user-facing probe rung language.
PROBE_RUNG_LANGUAGE: dict[ProbeRung, str] = {
    ProbeRung.TRAINED: "trained",
    ProbeRung.HELD_OUT: "detects on held-out data",
    ProbeRung.UNSEEN_TASKS: "detects on unseen tasks",
    ProbeRung.JUDGE_COMPARED: "detects on unseen tasks, compared with a judge",
}

#: What moves a probe up one rung.
PROBE_RUNG_NEXT_STEP: dict[ProbeRung, str] = {
    ProbeRung.TRAINED: "evaluate on an in-distribution held-out set",
    ProbeRung.HELD_OUT: "evaluate on every out-of-distribution set",
    ProbeRung.UNSEEN_TASKS: "run the judge baseline on the same out-of-distribution sets",
    ProbeRung.JUDGE_COMPARED: "top rung — nothing further",
}


def probe_rung_language(rung: "ProbeRung | int") -> str:
    """Server-rendered probe rung phrase — the single language source."""
    return PROBE_RUNG_LANGUAGE[ProbeRung(rung)]


def probe_rung_next_step(rung: "ProbeRung | int") -> str:
    return PROBE_RUNG_NEXT_STEP[ProbeRung(rung)]


def needs_arm_acknowledgement(rung: "ProbeRung | int") -> bool:
    """Whether arming this probe requires an explicit operator acknowledgement.

    ⚠ This is a SECOND consent, not a duplicate of the definition's. The person who exported a
    weak probe from miStudio and the person arming it against live traffic here are not
    necessarily the same person, and only the second one is choosing to monitor with it.
    """
    return int(rung) < int(ARM_WITHOUT_ACKNOWLEDGEMENT_MIN_RUNG)
