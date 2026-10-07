"""Pooling and normalisation as pure functions (Feature 30, FR-30.2.2 – FR-30.2.6).

Padding positions hold 1e3, never zero: a fixture whose padding is zero agrees with a mask-blind
mean by construction (FPRD §12).

MUTATION CONTROLS (030_implementation_controls): M8 drop the mask branch in `mean` -> the padding
tests go red.
"""

from __future__ import annotations

import math

import pytest
import torch

from millm.ml.embedding_pooling import (
    EmbeddingOptions,
    NonFiniteEmbeddingError,
    finalize_vector,
    pool_hidden,
)

PAD = 1e3


def _rows() -> list[torch.Tensor]:
    """Two rows of real tokens, lengths 2 and 5, width 4, distinct known values."""
    torch.manual_seed(1)
    return [torch.randn(2, 4), torch.randn(5, 4)]


def _padded(rows: list[torch.Tensor], side: str) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(r.shape[0] for r in rows)
    hidden = torch.full((len(rows), width, rows[0].shape[1]), PAD)
    mask = torch.zeros(len(rows), width, dtype=torch.long)
    for b, r in enumerate(rows):
        n = r.shape[0]
        if side == "right":
            hidden[b, :n], mask[b, :n] = r, 1
        else:
            hidden[b, width - n:], mask[b, width - n:] = r, 1
    return hidden, mask


def _expected(row: torch.Tensor, mode: str) -> torch.Tensor:
    return {"mean": row.mean(dim=0), "cls": row[0], "last": row[-1]}[mode]


class TestKnownAnswers:
    def test_each_mode_on_hand_built_values(self):
        hidden = torch.tensor([[[1.0, 0.0], [3.0, 2.0], [5.0, 4.0]]])
        mask = torch.ones(1, 3, dtype=torch.long)
        assert pool_hidden(hidden, mask, "mean").tolist() == [[3.0, 2.0]]
        assert pool_hidden(hidden, mask, "cls").tolist() == [[1.0, 0.0]]
        assert pool_hidden(hidden, mask, "last").tolist() == [[5.0, 4.0]]

    def test_the_result_is_batch_by_width_and_never_squeezed(self):
        """Today's `.squeeze()` turned a width-1 vector into a float; the pool returns [B, D]."""
        hidden = torch.ones(1, 3, 1)
        out = pool_hidden(hidden, torch.ones(1, 3), "mean")
        assert tuple(out.shape) == (1, 1)
        assert finalize_vector(out[0], False) == [1.0]

    def test_unknown_mode_is_refused(self):
        with pytest.raises(ValueError, match="unknown pooling mode"):
            pool_hidden(torch.ones(1, 2, 2), torch.ones(1, 2), "max")  # type: ignore[arg-type]

    def test_shape_disagreement_and_empty_rows_are_refused(self):
        with pytest.raises(ValueError, match="disagree"):
            pool_hidden(torch.ones(1, 2, 2), torch.ones(1, 3), "mean")
        with pytest.raises(ValueError, match="no real tokens"):
            pool_hidden(torch.ones(2, 2, 2), torch.tensor([[1, 1], [0, 0]]), "mean")


class TestTheDefaultDoesNotMove:
    """FR-30.2.2: an unpadded `mean` is today's exact expression, element for element."""

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_unpadded_mean_is_bit_identical_to_hidden_mean(self, dtype):
        torch.manual_seed(3)
        hidden = torch.randn(1, 7, 16).to(dtype)
        mask = torch.ones(1, 7, dtype=torch.long)
        assert torch.equal(pool_hidden(hidden, mask, "mean"), hidden.mean(dim=1))

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_default_floats_equal_the_old_conversion(self, dtype):
        torch.manual_seed(4)
        hidden = torch.randn(1, 5, 8).to(dtype)
        old = hidden.mean(dim=1).squeeze().cpu().tolist()
        new = finalize_vector(pool_hidden(hidden, torch.ones(1, 5), "mean")[0], False)
        assert new == old


class TestPaddingIsMasked:
    """FR-30.2.3: each padded row gives its own unpadded answer, on either padding side."""

    @pytest.mark.parametrize("side", ["right", "left"])
    @pytest.mark.parametrize("mode", ["mean", "cls", "last"])
    def test_each_mode_equals_the_unpadded_per_row_answer(self, side, mode):
        rows = _rows()
        hidden, mask = _padded(rows, side)
        pooled = pool_hidden(hidden, mask, mode)  # type: ignore[arg-type]
        for b, row in enumerate(rows):
            assert torch.allclose(pooled[b], _expected(row, mode), atol=1e-6), (side, mode, b)
            assert pooled[b].abs().max() < PAD / 10, "a pad position entered the pool"

    def test_a_float_mask_works_as_well_as_an_int_one(self):
        rows = _rows()
        hidden, mask = _padded(rows, "left")
        assert torch.allclose(
            pool_hidden(hidden, mask.float(), "mean"), pool_hidden(hidden, mask, "mean")
        )


class TestClsOnACausalDecoder:
    """FR-30.2.4/30.2.5, T-92: `cls` is the first real position, which on a causal decoder sees
    only itself — so where every input starts with the same BOS, every `cls` vector is equal."""

    def test_identical_first_tokens_give_identical_cls_vectors(self):
        bos = torch.tensor([0.25, -1.0, 2.0])
        a = torch.stack([bos, torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0])])
        b = torch.stack([bos, torch.tensor([-7.0, 8.0, 0.5])])
        hidden, mask = _padded([a, b], "left")
        cls = pool_hidden(hidden, mask, "cls")
        last = pool_hidden(hidden, mask, "last")
        assert torch.equal(cls[0], cls[1]) and torch.equal(cls[0], bos)
        assert not torch.equal(last[0], last[1])


class TestFinalizeVector:
    def test_normalize_gives_unit_norm_within_1e5(self):
        torch.manual_seed(5)
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            vec = (torch.randn(64) * 37.0).to(dtype)
            out = finalize_vector(vec, True)
            assert abs(math.sqrt(sum(x * x for x in out)) - 1.0) < 1e-5, dtype

    def test_normalize_false_returns_the_same_list_as_today(self):
        vec = torch.tensor([0.1, -2.5, 3.0])
        assert finalize_vector(vec, False) == vec.cpu().tolist()

    def test_a_list_from_llamacpp_comes_back_unchanged(self):
        values = [0.1, 0.2, 0.30000000000000004]
        assert finalize_vector(values, False) == values

    def test_a_list_is_normalised_too(self):
        out = finalize_vector([3.0, 4.0], True)
        assert out == pytest.approx([0.6, 0.8], abs=1e-7)

    @pytest.mark.parametrize("normalize", [True, False])
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_values_raise_never_return(self, bad, normalize):
        with pytest.raises(NonFiniteEmbeddingError):
            finalize_vector(torch.tensor([1.0, bad]), normalize)
        with pytest.raises(NonFiniteEmbeddingError):
            finalize_vector([1.0, bad], normalize)

    def test_a_zero_vector_cannot_be_normalised(self):
        with pytest.raises(NonFiniteEmbeddingError, match="zero"):
            finalize_vector(torch.zeros(4), True)
        assert finalize_vector(torch.zeros(2), False) == [0.0, 0.0]

    def test_options_default_to_todays_behaviour(self):
        assert EmbeddingOptions() == EmbeddingOptions(pooling="mean", normalize=False)
