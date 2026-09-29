"""Which positions a probe may score.

⚠ THIS FILE REPLACED A LARGER ONE THAT TESTED THE WRONG VOCABULARY. The first version exercised
role spans recovered by prefix-rendering the chat template, for scopes
`all | assistant | user | last_assistant` — 032's INTERNAL names, which I took from a miStudio
TypeScript type instead of the frozen schema. The contract's scopes are `all | prompt | response`,
so none of that machinery could ever have run against a real document, and every one of those 22
tests was green.

What is left is simpler because the contract is simpler, and that simplicity is the point: every
scope is computable from two counts, with no tokenizer, no template, and no way to be "unreliable".
"""

from __future__ import annotations

import pytest

from millm.api.schemas.probe import SCOPES as CONTRACT_SCOPES
from millm.services.probe_scope import SCOPES, scope_is_reproducible, scored_mask, window


class TestTheVocabularyIsTheContractS:
    def test_it_matches_the_mirror(self):
        """If these drift, a document validates at the door and is scored under a scope the
        runtime invented."""
        assert tuple(SCOPES) == tuple(CONTRACT_SCOPES)

    def test_an_unknown_scope_raises(self):
        with pytest.raises(ValueError, match="unknown scope"):
            scored_mask(scope="everything", n_prompt_tokens=1)

    @pytest.mark.parametrize("scope", ["assistant", "user", "last_assistant"])
    def test_miStudios_internal_names_are_not_accepted(self, scope):
        with pytest.raises(ValueError, match="unknown scope"):
            scored_mask(scope=scope, n_prompt_tokens=1)


class TestTheMasks:
    def test_all_covers_prompt_and_response(self):
        assert scored_mask(scope="all", n_prompt_tokens=3, n_generated=2) == [True] * 5

    def test_prompt_stops_where_generation_starts(self):
        assert scored_mask(scope="prompt", n_prompt_tokens=3, n_generated=2) == [
            True, True, True, False, False
        ]

    def test_response_is_only_what_the_model_produced(self):
        assert scored_mask(scope="response", n_prompt_tokens=3, n_generated=2) == [
            False, False, False, True, True
        ]

    def test_response_before_anything_is_generated_selects_nothing(self):
        """Not an error: a response-scoped probe simply has nothing to say yet. The caller reports
        `no_scored_tokens`, which is different from a score of 0."""
        assert scored_mask(scope="response", n_prompt_tokens=4) == [False] * 4

    @pytest.mark.parametrize("scope", SCOPES)
    def test_the_mask_always_covers_every_position(self, scope):
        mask = scored_mask(scope=scope, n_prompt_tokens=7, n_generated=5)
        assert len(mask) == 12

    def test_negative_counts_are_refused(self):
        with pytest.raises(ValueError, match="cannot be negative"):
            scored_mask(scope="all", n_prompt_tokens=-1)


class TestReproducibility:
    def test_only_all_is_reproducible_from_token_ids(self):
        """⚠ A `prompt` or `response` vector's recorded token_scores came from miStudio's NARROWER
        role mask, and the contract carries no record of which positions those were.

        miStudio's own export note calls the mapping lossy and widening-by-design. That is safe for
        serving and fatal for exact reproduction, which is why parity reports it instead of
        pretending.
        """
        assert scope_is_reproducible("all") is True
        assert scope_is_reproducible("prompt") is False
        assert scope_is_reproducible("response") is False


class TestWindow:
    def test_it_slices_one_forward_pass_out_of_the_mask(self):
        mask = [True, False, True, False, True]
        assert window(mask, 1, 3) == [False, True, False]

    def test_no_mask_means_every_position_in_the_pass(self):
        assert window(None, 0, 4) == [True] * 4

    def test_a_short_mask_pads_CLOSED_not_open(self):
        """Padding open would score positions nobody vouched for."""
        assert window([True, True], 0, 5) == [True, True, False, False, False]


class TestRuntimeAdmissionTracksTheWiring:
    """⚠ `RUNTIME_SCORABLE_SCOPES` is a claim about the CODE, so it is checked against the code.

    `scored_mask` in this module is complete and correct and has **no production caller**:
    `ProbeRequestContext` is built with `mask=None`, so `window(None, ...)` returns all-True and
    every armed probe scores every position. That is right for `all` and wrong for the other two —
    a `prompt`-scoped probe would score the model's own output with weights that never saw one.

    Until 2026-09-28 the only thing refusing those scopes was the parity gate, which cannot replay
    them. A safety property resting on another gate's side effect disappears the moment that gate
    changes, so `ProbeArmingService` now refuses on runtime capability directly.

    The failure this pins is the **drift** between the constant and the wiring, in both
    directions: widening the set without building the mask opens the arming path onto a runtime
    that ignores scope, and building the mask without widening the set leaves a working capability
    unreachable — the same defect this estate has shipped three times in this arc alone.
    """

    @staticmethod
    def _production_callers_of(name: str) -> list[str]:
        """Files under `millm/` containing a CALL to `name`, excluding its own definition.

        ⚠ A CALL, by AST — not a text search. A substring match here would be satisfied by the
        docstrings and comments that discuss `scored_mask` at length, which is exactly how a guard
        in this repo has passed for the wrong reason before.
        """
        import ast
        import pathlib

        hits: list[str] = []
        root = pathlib.Path(__file__).resolve().parents[3] / "millm"
        assert root.is_dir(), f"package root not found at {root}"
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                called = (
                    func.id
                    if isinstance(func, ast.Name)
                    else func.attr
                    if isinstance(func, ast.Attribute)
                    else None
                )
                if called == name:
                    hits.append(f"{path.relative_to(root.parent)}:{node.lineno}")
        return hits

    def test_the_ast_scan_can_see_a_call_at_all(self):
        """⚠ A source scan that matches nothing asserts nothing, and this repo has shipped two
        that failed open. Prove the scanner finds a call that is definitely there before
        believing it about one that should not be."""
        from millm.services.probe_scope import scope_is_reproducible  # noqa: F401

        found = self._production_callers_of("scope_is_reproducible")
        assert found, "the scanner found no call to scope_is_reproducible, so it is broken"
        assert any("probe_parity.py" in f for f in found), found

    def test_the_admitted_set_matches_what_is_wired(self):
        from millm.services.probe_scope import RUNTIME_SCORABLE_SCOPES, SCOPES

        wired = self._production_callers_of("scored_mask")
        if wired:
            assert set(RUNTIME_SCORABLE_SCOPES) == set(SCOPES), (
                f"scored_mask has production caller(s) {wired}, so the runtime can honour every "
                f"scope — widen RUNTIME_SCORABLE_SCOPES, or the capability is unreachable"
            )
        else:
            assert set(RUNTIME_SCORABLE_SCOPES) == {"all"}, (
                "nothing calls scored_mask, so ProbeRequestContext gets mask=None and every "
                "probe scores every position. Admitting a narrower scope here would arm a "
                "detector that reads positions it was never fitted on."
            )

    def test_it_is_wired_now(self):
        """⚠ Direction, not just consistency. The branch above is satisfied by BOTH states, so
        on its own it would stay green if the wiring were removed and the set narrowed back
        together — a consistent retreat. This pins which side we are on."""
        from millm.services.probe_scope import RUNTIME_SCORABLE_SCOPES, SCOPES

        wired = self._production_callers_of("scored_mask")
        assert wired, (
            "scored_mask has lost its production caller; the runtime is back to scoring every "
            "position regardless of scope"
        )
        assert set(RUNTIME_SCORABLE_SCOPES) == set(SCOPES)

    def test_the_request_context_is_still_built_without_a_mask(self):
        """The other half of the same fact, read from the construction site rather than the
        call site — so a caller added to `scored_mask` that never reaches the context cannot
        make the pair of tests agree that scope is honoured."""
        import ast
        import inspect

        from millm.services import probe_runtime

        tree = ast.parse(inspect.getsource(probe_runtime))
        masks = [
            kw.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "ProbeRequestContext"
            for kw in node.keywords
            if kw.arg == "mask"
        ]
        constructions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "ProbeRequestContext"
        ]
        assert constructions, "no ProbeRequestContext construction found — retarget this test"
        assert masks == [], (
            f"a mask is now passed at construction ({len(masks)} site(s)); scope may now be "
            f"honoured at runtime — re-check RUNTIME_SCORABLE_SCOPES"
        )
