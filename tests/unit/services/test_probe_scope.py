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
