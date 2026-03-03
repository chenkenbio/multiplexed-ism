"""Integration tests for the public API."""

import torch
import pytest

from multiplexed_ism import multiplexed_ism, MultiplexedISMResult
from multiplexed_ism.mutation import ALT_TABLE

from tests.conftest import SumModel, LinearModel, DictModel, ConvModel


class TestBasicAPI:
    def test_return_type(self, short_seq: torch.Tensor) -> None:
        model = SumModel()
        result = multiplexed_ism(
            model, short_seq, n_var_per_run=2, n_samples=10,
            device="cpu", seed=42,
        )
        assert isinstance(result, MultiplexedISMResult)

    def test_scores_shape(self, short_seq: torch.Tensor) -> None:
        model = SumModel()
        result = multiplexed_ism(
            model, short_seq, n_var_per_run=2, n_samples=10,
            device="cpu", seed=42,
        )
        L = short_seq.shape[1]
        # SumModel outputs (B, 1) so T = (1,)
        assert result.scores.shape == (3, L, 1)

    def test_ref_bases_shape(self, short_seq: torch.Tensor) -> None:
        model = SumModel()
        result = multiplexed_ism(
            model, short_seq, n_var_per_run=2, n_samples=10,
            device="cpu", seed=42,
        )
        assert result.ref_bases.shape == (short_seq.shape[1],)

    def test_start_end(self, short_seq: torch.Tensor) -> None:
        model = SumModel()
        result = multiplexed_ism(
            model, short_seq, n_var_per_run=2, n_samples=10,
            start=5, end=15, device="cpu", seed=42,
        )
        assert result.scores.shape == (3, 10, 1)
        assert result.start == 5
        assert result.end == 15


class TestReturnOptions:
    def test_return_baseline(self, short_seq: torch.Tensor) -> None:
        model = SumModel()
        result = multiplexed_ism(
            model, short_seq, n_var_per_run=2, n_samples=10,
            device="cpu", seed=42, return_baseline=True,
        )
        assert result.baseline is not None
        assert result.baseline.ndim >= 1

    def test_return_counts(self, short_seq: torch.Tensor) -> None:
        model = SumModel()
        result = multiplexed_ism(
            model, short_seq, n_var_per_run=2, n_samples=50,
            device="cpu", seed=42, return_counts=True,
        )
        assert result.counts is not None
        assert result.counts.shape == (short_seq.shape[1],)
        # With 50 samples and 2 vars each, total mutations = 100
        assert result.counts.sum().item() == 50 * 2

    def test_no_baseline_by_default(self, short_seq: torch.Tensor) -> None:
        model = SumModel()
        result = multiplexed_ism(
            model, short_seq, n_var_per_run=2, n_samples=10,
            device="cpu", seed=42,
        )
        assert result.baseline is None
        assert result.counts is None


class TestSeedReproducibility:
    def test_same_seed_same_result(self, short_seq: torch.Tensor) -> None:
        model = SumModel()
        r1 = multiplexed_ism(
            model, short_seq, n_var_per_run=3, n_samples=20,
            device="cpu", seed=99,
        )
        r2 = multiplexed_ism(
            model, short_seq, n_var_per_run=3, n_samples=20,
            device="cpu", seed=99,
        )
        assert torch.allclose(r1.scores, r2.scores)

    def test_different_seed_different_result(self, short_seq: torch.Tensor) -> None:
        model = LinearModel(seq_len=20)
        r1 = multiplexed_ism(
            model, short_seq, n_var_per_run=3, n_samples=20,
            device="cpu", seed=1,
        )
        r2 = multiplexed_ism(
            model, short_seq, n_var_per_run=3, n_samples=20,
            device="cpu", seed=2,
        )
        # Very unlikely to be equal with different seeds
        assert not torch.allclose(r1.scores, r2.scores)


class TestDictModelOutput:
    def test_output_key(self, short_seq: torch.Tensor) -> None:
        model = DictModel(seq_len=20)
        result = multiplexed_ism(
            model, short_seq, n_var_per_run=2, n_samples=10,
            device="cpu", seed=42, output_key="logits",
        )
        assert result.scores.shape == (3, 20, 3)

    def test_target_fn(self, short_seq: torch.Tensor) -> None:
        model = DictModel(seq_len=20)
        result = multiplexed_ism(
            model, short_seq, n_var_per_run=2, n_samples=10,
            device="cpu", seed=42,
            target_fn=lambda d: d["logits"][:, :2],
        )
        assert result.scores.shape == (3, 20, 2)

    def test_target_fn_takes_priority(self, short_seq: torch.Tensor) -> None:
        model = DictModel(seq_len=20)
        # target_fn should override output_key
        result = multiplexed_ism(
            model, short_seq, n_var_per_run=2, n_samples=10,
            device="cpu", seed=42,
            output_key="logits",
            target_fn=lambda d: d["extra"][:, :1],
        )
        assert result.scores.shape == (3, 20, 1)


class TestAggregation:
    def test_mean_aggregation(self, short_seq: torch.Tensor) -> None:
        model = SumModel()
        result = multiplexed_ism(
            model, short_seq, n_var_per_run=2, n_samples=30,
            device="cpu", seed=42, aggregation="mean",
        )
        assert result.scores.shape == (3, 20, 1)

    def test_invalid_aggregation(self, short_seq: torch.Tensor) -> None:
        model = SumModel()
        with pytest.raises(ValueError, match="Unknown aggregation"):
            multiplexed_ism(
                model, short_seq, n_var_per_run=2, n_samples=10,
                device="cpu", seed=42, aggregation="invalid",
            )


class TestChunking:
    def test_chunked_shape(self, short_seq: torch.Tensor) -> None:
        model = SumModel()
        result = multiplexed_ism(
            model, short_seq, n_var_per_run=1, n_samples=20,
            device="cpu", seed=42, chunk_size=7,
        )
        assert result.scores.shape == (3, 20, 1)
        assert result.start == 0
        assert result.end == 20


class TestNVarPerRunOne:
    """When n_var_per_run=1, multiplexed ISM degenerates to standard ISM."""

    def test_sum_model_exact(self) -> None:
        """SumModel is position-independent, so n_var_per_run=1 with mean
        aggregation should recover the exact per-alt-base delta."""
        torch.manual_seed(0)
        L = 10
        bases = torch.randint(0, 4, (L,))
        seq = torch.zeros(4, L)
        seq[bases, torch.arange(L)] = 1.0

        model = SumModel()
        # n_samples = L * 10 ensures good coverage per position
        result = multiplexed_ism(
            model, seq, n_var_per_run=1, n_samples=L * 10,
            device="cpu", seed=42, aggregation="mean",
            return_baseline=True, return_counts=True,
        )

        # Compute expected deltas via brute force
        with torch.no_grad():
            baseline = model(seq.unsqueeze(0)).squeeze()  # (1,)

        ref_bases = seq.argmax(dim=0)
        for pos in range(L):
            ref = ref_bases[pos].item()
            for alt_idx in range(3):
                alt_base = ALT_TABLE[ref, alt_idx].item()
                mutated = seq.clone()
                mutated[ref, pos] = 0.0
                mutated[alt_base, pos] = 1.0
                with torch.no_grad():
                    pred = model(mutated.unsqueeze(0)).squeeze()
                expected_delta = pred - baseline
                actual = result.scores[alt_idx, pos]
                assert torch.allclose(actual, expected_delta, atol=1e-5), (
                    f"Mismatch at pos={pos}, alt_idx={alt_idx}: "
                    f"expected={expected_delta.item():.4f}, got={actual.item():.4f}"
                )


class TestValidation:
    def test_wrong_shape(self) -> None:
        model = SumModel()
        seq = torch.randn(3, 20)  # wrong: 3 channels instead of 4
        with pytest.raises(ValueError, match="shape \\(4, L\\)"):
            multiplexed_ism(model, seq, n_var_per_run=2, n_samples=10, device="cpu")

    def test_not_one_hot(self) -> None:
        model = SumModel()
        seq = torch.randn(4, 20)  # not one-hot
        with pytest.raises(ValueError, match="not valid one-hot"):
            multiplexed_ism(model, seq, n_var_per_run=2, n_samples=10, device="cpu")

    def test_bad_range(self) -> None:
        model = SumModel()
        seq = torch.zeros(4, 20)
        seq[0, :] = 1.0
        with pytest.raises(ValueError, match="Invalid range"):
            multiplexed_ism(
                model, seq, n_var_per_run=2, n_samples=10,
                start=15, end=10, device="cpu",
            )

    def test_bad_n_var(self) -> None:
        model = SumModel()
        seq = torch.zeros(4, 20)
        seq[0, :] = 1.0
        with pytest.raises(ValueError, match="n_var_per_run"):
            multiplexed_ism(
                model, seq, n_var_per_run=0, n_samples=10, device="cpu",
            )


class TestPrecisionValidation:
    def test_invalid_precision(self, short_seq: torch.Tensor) -> None:
        model = SumModel()
        with pytest.raises(ValueError, match="Invalid precision"):
            multiplexed_ism(
                model, short_seq, n_var_per_run=2, n_samples=10,
                device="cpu", precision="bad",  # type: ignore[arg-type]
            )


class TestChunkSizeValidation:
    def test_chunk_size_zero(self, short_seq: torch.Tensor) -> None:
        model = SumModel()
        with pytest.raises(ValueError, match="chunk_size"):
            multiplexed_ism(
                model, short_seq, n_var_per_run=2, n_samples=10,
                device="cpu", chunk_size=0,
            )

    def test_chunk_size_negative(self, short_seq: torch.Tensor) -> None:
        model = SumModel()
        with pytest.raises(ValueError, match="chunk_size"):
            multiplexed_ism(
                model, short_seq, n_var_per_run=2, n_samples=10,
                device="cpu", chunk_size=-1,
            )


class TestCallableModel:
    def test_lambda_model(self, medium_seq: torch.Tensor) -> None:
        """Test with a plain callable (not nn.Module)."""
        linear = torch.nn.Linear(400, 1)
        result = multiplexed_ism(
            lambda x: linear(x.reshape(x.shape[0], -1)),
            medium_seq, n_var_per_run=5, n_samples=50,
            device="cpu", seed=42, return_counts=True,
        )
        assert result.scores.shape == (3, 100, 1)
        assert result.counts is not None


class TestScoreCutoff:
    def test_produces_sparse_result(self, short_seq: torch.Tensor) -> None:
        model = SumModel()
        result = multiplexed_ism(
            model, short_seq, n_var_per_run=2, n_samples=20,
            device="cpu", seed=42, score_cutoff=0.01,
        )
        assert result.is_sparse
        assert result.scores.is_sparse

    def test_dense_roundtrip(self, short_seq: torch.Tensor) -> None:
        model = SumModel()
        result = multiplexed_ism(
            model, short_seq, n_var_per_run=2, n_samples=20,
            device="cpu", seed=42, score_cutoff=0.01,
        )
        dense = result.to_dense()
        assert not dense.is_sparse
        assert dense.scores.shape == (3, 20, 1)

    def test_chunked_with_cutoff(self, short_seq: torch.Tensor) -> None:
        model = SumModel()
        result = multiplexed_ism(
            model, short_seq, n_var_per_run=1, n_samples=20,
            device="cpu", seed=42, chunk_size=7, score_cutoff=0.01,
        )
        assert result.is_sparse
        assert result.scores.to_dense().shape == (3, 20, 1)

    def test_cutoff_zero_raises(self, short_seq: torch.Tensor) -> None:
        model = SumModel()
        with pytest.raises(ValueError, match="score_cutoff"):
            multiplexed_ism(
                model, short_seq, n_var_per_run=2, n_samples=10,
                device="cpu", score_cutoff=0.0,
            )

    def test_cutoff_negative_raises(self, short_seq: torch.Tensor) -> None:
        model = SumModel()
        with pytest.raises(ValueError, match="score_cutoff"):
            multiplexed_ism(
                model, short_seq, n_var_per_run=2, n_samples=10,
                device="cpu", score_cutoff=-0.5,
            )

    def test_no_cutoff_returns_dense(self, short_seq: torch.Tensor) -> None:
        model = SumModel()
        result = multiplexed_ism(
            model, short_seq, n_var_per_run=2, n_samples=10,
            device="cpu", seed=42,
        )
        assert not result.is_sparse


class TestConvModel:
    def test_conv_model(self, short_seq: torch.Tensor) -> None:
        model = ConvModel()
        result = multiplexed_ism(
            model, short_seq, n_var_per_run=3, n_samples=30,
            device="cpu", seed=42,
        )
        assert result.scores.shape == (3, 20, 1)
