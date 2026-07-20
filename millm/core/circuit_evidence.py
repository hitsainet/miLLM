"""Evidence-rung vocabulary (Feature 13) — mirrored VERBATIM from miStudio.

This module is the ONE source of user-facing rung language in miLLM: the
runtime, the management API, the MCP surface and the admin UI all render
``rung_language(rung)`` rather than hand-writing per-surface copy.

**The honesty contract:** the word "causal" MUST NEVER describe an artifact
below rung 2 (``CAUSALLY_VALIDATED``). A mined or attribution-supported circuit
is an *association* or a *suggestion*, not a causal claim — presenting it as
causal at the point of live influence is exactly the overclaim the evidence
ladder exists to prevent. ``tests/unit/core/test_circuit_evidence.py`` carries
a copy-audit that greps the runtime + UI surfaces to enforce this.

The strings here are byte-identical to miStudio's
``backend/src/schemas/evidence_ladder.py`` — do not paraphrase them; a
divergence between the authoring tool and the serving runtime is a drift bug.
"""

from enum import IntEnum


class EvidenceRung(IntEnum):
    """Rungs are strictly ordered; an artifact's rung is the highest PASSED."""

    MINED = 0                    # statistical association survived null + support
    ATTRIBUTION_SUPPORTED = 1    # gradient attribution agrees
    CAUSALLY_VALIDATED = 2       # real intervention satisfied the criterion
    FAITHFULNESS_TESTED = 3      # circuit-level necessity (and, where run, sufficiency)


#: The rung at or above which causal language is permitted.
CAUSAL_LANGUAGE_MIN_RUNG = EvidenceRung.CAUSALLY_VALIDATED

# The ONLY source of user-facing rung language (UI, API, MCP, exports).
RUNG_LANGUAGE: dict[EvidenceRung, str] = {
    EvidenceRung.MINED: "associated",
    EvidenceRung.ATTRIBUTION_SUPPORTED: "suggested (attribution-supported)",
    EvidenceRung.CAUSALLY_VALIDATED: "causally validated (edge)",
    EvidenceRung.FAITHFULNESS_TESTED: "faithfulness-tested (circuit)",
}

# What moves an artifact up one rung — surfaced as UI tooltips and MCP hints.
RUNG_NEXT_STEP: dict[EvidenceRung, str] = {
    EvidenceRung.MINED: "run the attribution pass (sign agreement + magnitude percentile)",
    EvidenceRung.ATTRIBUTION_SUPPORTED: (
        "run intervention validation (effect size vs null + sign consistency)"
    ),
    EvidenceRung.CAUSALLY_VALIDATED: "run circuit-level faithfulness at promotion",
    EvidenceRung.FAITHFULNESS_TESTED: "top rung — nothing further",
}


def _coerce(rung: "EvidenceRung | int") -> EvidenceRung:
    """Coerce an int/enum to EvidenceRung, clamping out-of-range defensively.

    A malformed imported document must not crash a render: an unknown rung
    degrades to MINED (the most conservative claim), never to a higher one.
    """
    try:
        return EvidenceRung(int(rung))
    except (ValueError, TypeError):
        return EvidenceRung.MINED


def rung_language(rung: "EvidenceRung | int") -> str:
    """Server-rendered rung phrase — the single language source for all surfaces."""
    return RUNG_LANGUAGE[_coerce(rung)]


def rung_next_step(rung: "EvidenceRung | int") -> str:
    """What would raise this artifact one rung (tooltip / agent hint)."""
    return RUNG_NEXT_STEP[_coerce(rung)]


def circuit_rung(edge_rungs: "list[EvidenceRung | int]") -> EvidenceRung:
    """A circuit's displayed rung = MIN over its member edges' rungs.

    An edge-less circuit (hand-assembled, no mined evidence) is rung 0 by
    definition — defined, not a crash. Taking the MIN means a circuit is only
    as strong as its weakest edge: one unvalidated edge keeps the whole circuit
    out of causal language.
    """
    if not edge_rungs:
        return EvidenceRung.MINED
    return EvidenceRung(min(int(_coerce(r)) for r in edge_rungs))


def is_validated(rung: "EvidenceRung | int") -> bool:
    """True when causal language is permitted for this artifact (rung >= 2)."""
    return _coerce(rung) >= CAUSAL_LANGUAGE_MIN_RUNG


def requires_unvalidated_ack(rung: "EvidenceRung | int") -> bool:
    """True when activating this artifact needs an explicit user acknowledgement."""
    return not is_validated(rung)
