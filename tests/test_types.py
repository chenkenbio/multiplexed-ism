"""Tests for types.py: MultiplexedISMResult and aggregators."""

import torch
import pytest

from multiplexed_ism.types import (
    MeanAggregator,
    MultiplexedISMResult,
    SignedMaxAbsAggregator,
)


class TestMultiplexedISMResult:
    def test_frozen(self) -> None:
        result = MultiplexedISMResult(
            scores=torch.zeros(3, 10),
            ref_bases=torch.zeros(10, dtype=torch.long),
            start=0,
            end=10,
        )
        with pytest.raises(AttributeError):
            result.start = 5  # type: ignore[misc]

    def test_optional_fields_default_none(self) -> None:
        result = MultiplexedISMResult(
            scores=torch.zeros(3, 10),
            ref_bases=torch.zeros(10, dtype=torch.long),
            start=0,
            end=10,
        )
        assert result.baseline is None
        assert result.counts is None


class TestSignedMaxAbsAggregator:
    def test_keeps_largest_abs(self) -> None:
        agg = SignedMaxAbsAggregator((3, 5), device=torch.device("cpu"))
        pos = torch.tensor([0, 1, 2])

        # First update: positive deltas
        agg.update(0, pos, torch.tensor([1.0, 2.0, 3.0]))
        # Second update: negative deltas with larger abs at pos 0
        agg.update(0, pos, torch.tensor([-5.0, 1.0, -2.0]))

        result = agg.result()
        assert result[0, 0].item() == -5.0  # larger abs
        assert result[0, 1].item() == 2.0   # first was larger
        assert result[0, 2].item() == 3.0   # first was larger

    def test_preserves_sign(self) -> None:
        agg = SignedMaxAbsAggregator((3, 3), device=torch.device("cpu"))
        pos = torch.tensor([0])
        agg.update(1, pos, torch.tensor([-10.0]))
        result = agg.result()
        assert result[1, 0].item() == -10.0

    def test_multidim_task(self) -> None:
        # shape (3, L_mut, T1, T2)
        agg = SignedMaxAbsAggregator((3, 4, 2, 3), device=torch.device("cpu"))
        pos = torch.tensor([0, 1])
        deltas = torch.randn(2, 2, 3)
        agg.update(0, pos, deltas)
        result = agg.result()
        assert result.shape == (3, 4, 2, 3)


class TestSparseConversion:
    def _make_result(self, scores: torch.Tensor) -> MultiplexedISMResult:
        L_mut = scores.shape[1]
        return MultiplexedISMResult(
            scores=scores,
            ref_bases=torch.zeros(L_mut, dtype=torch.long),
            start=0,
            end=L_mut,
            baseline=torch.tensor([1.0]),
            counts=torch.ones(L_mut, dtype=torch.long),
        )

    def test_to_sparse_produces_sparse(self) -> None:
        scores = torch.tensor([[[0.5, 0.01], [-0.02, 0.8], [0.0, 0.3]]])
        # shape (1, 3, 2) — pretend alt_dim=1 for simplicity
        scores_3 = scores.expand(3, 3, 2).clone()
        result = self._make_result(scores_3)
        sparse = result.to_sparse(cutoff=0.1)
        assert sparse.is_sparse
        assert sparse.scores.is_sparse

    def test_to_sparse_correct_nnz(self) -> None:
        scores = torch.zeros(3, 5)
        scores[0, 0] = 1.0
        scores[1, 2] = -0.5
        scores[2, 4] = 0.3
        result = self._make_result(scores)
        sparse = result.to_sparse(cutoff=0.2)
        # |1.0| >= 0.2, |-0.5| >= 0.2, |0.3| >= 0.2 → 3 non-zeros
        assert sparse.scores._nnz() == 3

    def test_to_dense_roundtrip(self) -> None:
        scores = torch.randn(3, 10, 4)
        result = self._make_result(scores)
        cutoff = 0.5
        sparse = result.to_sparse(cutoff)
        dense = sparse.to_dense()
        assert not dense.is_sparse
        # Round-trip: dense result should match manually filtered original
        expected = scores.clone()
        expected[expected.abs() < cutoff] = 0.0
        assert torch.allclose(dense.scores, expected)

    def test_is_sparse_property(self) -> None:
        scores = torch.randn(3, 5)
        result = self._make_result(scores)
        assert not result.is_sparse
        sparse = result.to_sparse(cutoff=0.1)
        assert sparse.is_sparse

    def test_high_cutoff_nearly_empty(self) -> None:
        scores = torch.randn(3, 8) * 0.01  # all values small
        result = self._make_result(scores)
        sparse = result.to_sparse(cutoff=1.0)
        assert sparse.scores._nnz() == 0

    def test_zero_cutoff_keeps_all(self) -> None:
        torch.manual_seed(7)
        scores = torch.randn(3, 6)  # unlikely to have exact zeros
        result = self._make_result(scores)
        total = scores.numel()
        # cutoff just above 0 should keep all non-zero entries
        sparse = result.to_sparse(cutoff=1e-30)
        assert sparse.scores._nnz() == total

    def test_preserves_metadata(self) -> None:
        scores = torch.randn(3, 5)
        result = self._make_result(scores)
        sparse = result.to_sparse(cutoff=0.1)
        assert sparse.start == result.start
        assert sparse.end == result.end
        assert torch.equal(sparse.ref_bases, result.ref_bases)
        assert torch.equal(sparse.baseline, result.baseline)
        assert torch.equal(sparse.counts, result.counts)

    def test_to_dense_already_dense(self) -> None:
        scores = torch.randn(3, 5)
        result = self._make_result(scores)
        dense = result.to_dense()
        assert not dense.is_sparse
        assert torch.allclose(dense.scores, scores)


class TestMeanAggregator:
    def test_mean_single_update(self) -> None:
        agg = MeanAggregator((3, 5), device=torch.device("cpu"))
        pos = torch.tensor([0, 1])
        agg.update(0, pos, torch.tensor([4.0, 6.0]))
        result = agg.result()
        assert result[0, 0].item() == 4.0
        assert result[0, 1].item() == 6.0

    def test_mean_multiple_updates(self) -> None:
        agg = MeanAggregator((3, 5), device=torch.device("cpu"))
        pos = torch.tensor([0])
        agg.update(0, pos, torch.tensor([2.0]))
        agg.update(0, pos, torch.tensor([4.0]))
        agg.update(0, pos, torch.tensor([6.0]))
        result = agg.result()
        assert abs(result[0, 0].item() - 4.0) < 1e-6  # mean of 2,4,6 = 4

    def test_zero_count_safe(self) -> None:
        agg = MeanAggregator((3, 5), device=torch.device("cpu"))
        result = agg.result()
        assert torch.all(result == 0)  # no division by zero

    def test_multidim_task(self) -> None:
        agg = MeanAggregator((3, 4, 3), device=torch.device("cpu"))
        pos = torch.tensor([0, 1])
        deltas = torch.ones(2, 3) * 3.0
        agg.update(2, pos, deltas)
        agg.update(2, pos, deltas)
        result = agg.result()
        assert torch.allclose(result[2, 0], torch.tensor([3.0, 3.0, 3.0]))
