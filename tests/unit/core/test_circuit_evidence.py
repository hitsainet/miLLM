"""Evidence-rung vocabulary tests (Feature 13, task 3.0).

Pins the rung values and language VERBATIM against miStudio's ladder, the
MIN-over-edges circuit rung, and — most importantly — carries the **copy-audit**
that enforces the honesty contract: the word "causal" must never describe an
artifact below rung 2 on any runtime or UI surface.
"""

import re
from pathlib import Path

import pytest

from millm.core.circuit_evidence import (
    CAUSAL_LANGUAGE_MIN_RUNG,
    RUNG_LANGUAGE,
    RUNG_NEXT_STEP,
    EvidenceRung,
    circuit_rung,
    is_validated,
    requires_unvalidated_ack,
    rung_language,
    rung_next_step,
)

REPO = Path(__file__).resolve().parents[3]


class TestRungValues:
    def test_exact_ladder_values(self):
        """The integer values ARE the contract — a shift would silently
        re-grade every imported circuit."""
        assert EvidenceRung.MINED == 0
        assert EvidenceRung.ATTRIBUTION_SUPPORTED == 1
        assert EvidenceRung.CAUSALLY_VALIDATED == 2
        assert EvidenceRung.FAITHFULNESS_TESTED == 3

    def test_language_is_verbatim_miStudio(self):
        """Byte-identical to miStudio's RUNG_LANGUAGE — no paraphrasing."""
        assert RUNG_LANGUAGE[EvidenceRung.MINED] == "associated"
        assert (
            RUNG_LANGUAGE[EvidenceRung.ATTRIBUTION_SUPPORTED]
            == "suggested (attribution-supported)"
        )
        assert (
            RUNG_LANGUAGE[EvidenceRung.CAUSALLY_VALIDATED] == "causally validated (edge)"
        )
        assert (
            RUNG_LANGUAGE[EvidenceRung.FAITHFULNESS_TESTED]
            == "faithfulness-tested (circuit)"
        )

    def test_rung_language_accepts_int_and_enum(self):
        assert rung_language(0) == "associated"
        assert rung_language(EvidenceRung.MINED) == "associated"

    def test_next_step_covers_every_rung(self):
        for rung in EvidenceRung:
            assert RUNG_NEXT_STEP[rung]
            assert rung_next_step(int(rung))

    def test_out_of_range_rung_degrades_to_mined_not_crash(self):
        """A malformed document must not crash a render, and must never
        degrade UPWARD into a stronger claim."""
        assert rung_language(99) == "associated"
        assert rung_language(-5) == "associated"
        assert rung_language("garbage") == "associated"
        assert is_validated(99) is False


class TestCircuitRung:
    def test_min_over_edges(self):
        """A circuit is only as strong as its weakest edge."""
        assert circuit_rung([3, 2, 1]) == EvidenceRung.ATTRIBUTION_SUPPORTED
        assert circuit_rung([2, 2, 2]) == EvidenceRung.CAUSALLY_VALIDATED
        assert circuit_rung([3, 3]) == EvidenceRung.FAITHFULNESS_TESTED

    def test_single_unvalidated_edge_blocks_causal_language(self):
        """One rung-0 edge keeps the WHOLE circuit out of causal language."""
        rung = circuit_rung([3, 3, 3, 0])
        assert rung == EvidenceRung.MINED
        assert is_validated(rung) is False

    def test_empty_edges_is_mined_not_crash(self):
        assert circuit_rung([]) == EvidenceRung.MINED

    def test_accepts_enum_members(self):
        assert circuit_rung([EvidenceRung.FAITHFULNESS_TESTED, EvidenceRung.MINED]) == 0


class TestValidationGate:
    @pytest.mark.parametrize("rung,validated", [(0, False), (1, False), (2, True), (3, True)])
    def test_is_validated_threshold(self, rung, validated):
        assert is_validated(rung) is validated
        assert requires_unvalidated_ack(rung) is (not validated)

    def test_threshold_constant_is_rung_two(self):
        assert CAUSAL_LANGUAGE_MIN_RUNG == EvidenceRung.CAUSALLY_VALIDATED == 2


class TestCopyAudit:
    """The honesty contract, enforced by grep (mirrors miStudio's guard).

    Any surface that renders rung language must obtain it from RUNG_LANGUAGE.
    The only permitted occurrences of "causal" are: rung>=2 language itself,
    and prose that explicitly states the prohibition.
    """

    #: Files that legitimately contain the word (the vocabulary + its guards).
    ALLOWED = {
        "millm/core/circuit_evidence.py",
        "tests/unit/core/test_circuit_evidence.py",
    }

    #: "causal" in an unrelated sense — the transformer architecture term and
    #: HuggingFace class names have nothing to do with the evidence ladder.
    UNRELATED_SENSE = re.compile(
        r"causal[- ]?lm|causallm|autoregressive|architecture", re.IGNORECASE
    )

    #: The audit only fires when "causal" appears in an EVIDENCE-GRADING
    #: context — a circuit/edge/rung claim — because that is where the word
    #: asserts a validation level. ("the wrong causal influence" about steering
    #: generally is a different, legitimate sense and is not an overclaim.)
    EVIDENCE_CONTEXT = re.compile(
        r"circuit|rung|edge|evidence|validated", re.IGNORECASE
    )

    def _scan(self, roots: list[str], suffixes: tuple[str, ...]) -> list[tuple[str, int, str]]:
        hits: list[tuple[str, int, str]] = []
        for root in roots:
            base = REPO / root
            if not base.exists():
                continue
            for path in base.rglob("*"):
                if path.suffix not in suffixes or "node_modules" in path.parts:
                    continue
                rel = str(path.relative_to(REPO))
                if rel in self.ALLOWED:
                    continue
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                for i, line in enumerate(text.splitlines(), 1):
                    if not re.search(r"\bcausal", line, re.IGNORECASE):
                        continue
                    if self.UNRELATED_SENSE.search(line):
                        continue  # "causal LM" / CausalLMOutputWithPast etc.
                    if not self.EVIDENCE_CONTEXT.search(line):
                        continue  # not an evidence-ladder claim
                    hits.append((rel, i, line.strip()))
        return hits

    #: Comment prefixes stripped before auditing. A marker in a COMMENT must not
    #: exempt a claim in the code — that is how an allow-list quietly stops
    #: guarding (R3 finding: `const m = "...causally validated"; // rung_language`
    #: previously passed).
    _COMMENT = re.compile(r"(//.*$)|(#.*$)|(/\*.*?\*/)")

    @staticmethod
    def _code_only(line: str) -> str:
        """The line with trailing comments removed."""
        return TestCopyAudit._COMMENT.sub("", line)

    def test_no_handwritten_causal_language_on_runtime_surfaces(self):
        """No runtime/UI file may hand-write 'causal' — it must come from
        RUNG_LANGUAGE (which only yields it at rung>=2)."""
        hits = self._scan(
            ["millm", "admin-ui/src"], (".py", ".ts", ".tsx")
        )
        offenders = []
        for rel, line_no, line in hits:
            # Audit the CODE, not the comments: a marker in a trailing comment
            # must never exempt a claim made in a string literal.
            code = self._code_only(line)
            if not re.search(r"\bcausal", code, re.IGNORECASE):
                continue  # the only occurrence was in a comment — prose, not copy
            lowered = code.lower()
            # Permitted: prose that states the prohibition/contract, or a
            # reference to the ladder constant//vocabulary function.
            if any(
                marker in lowered
                for marker in (
                    # Negations / prohibitions — these DENY a causal claim.
                    "never",
                    "forbid",
                    "must not",
                    "not causal",
                    "no causal",
                    "isn't causal",
                    "is not causal",
                    # Explicit rung-gating context.
                    "rung >= 2",
                    "rung>=2",
                    "rung < 2",
                    "rung<2",
                    "below rung 2",
                    # The ladder enumerated in a doc comment: the phrase is
                    # attached to its own rung number, not to a lower one.
                    "2 causally validated",
                    # Copy stating the GATE ("only a causally validated circuit
                    # activates without an acknowledgement") — that is the rule,
                    # not a claim about a specific low-rung artifact.
                    "only a causally validated",
                    # References to the vocabulary itself, not hand-written copy.
                    "causally_validated",   # the enum member name
                    "rung_language",        # rendering through the vocabulary
                    "causal_language_min_rung",
                )
            ):
                continue
            offenders.append(f"{rel}:{line_no}: {line}")
        assert not offenders, (
            "Hand-written causal language found on a runtime/UI surface — render "
            "rung_language(rung) instead:\n" + "\n".join(offenders)
        )

    def test_rung_below_two_language_never_contains_causal(self):
        """The vocabulary itself must never emit 'causal' below rung 2."""
        for rung in (EvidenceRung.MINED, EvidenceRung.ATTRIBUTION_SUPPORTED):
            assert "causal" not in rung_language(rung).lower()

    def test_rung_two_and_above_may_use_causal(self):
        assert "causally validated" in rung_language(EvidenceRung.CAUSALLY_VALIDATED)
