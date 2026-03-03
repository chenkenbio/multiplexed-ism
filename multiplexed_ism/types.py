"""Result dataclass and online aggregators for multiplexed ISM."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass(frozen=True)
class MultiplexedISMResult:
    """Result container for multiplexed ISM.

    Parameters
    ----------
    scores : Tensor
        Aggregated deltas, shape ``(3, L_mut, *T)``.
        Index 0..2 corresponds to the 3 alternative bases (sorted order
        per reference base, see ``ALT_TABLE``).
    ref_bases : Tensor
        Reference base indices for each position, shape ``(L_mut,)``.
        Encoding: 0=A, 1=C, 2=G, 3=T.
    start : int
        Start position of the mutated region (inclusive).
    end : int
        End position of the mutated region (exclusive).
    baseline : Tensor or None
        Unmodified model prediction, shape ``(*T,)``.
    counts : Tensor or None
        Per-position sample counts, shape ``(L_mut,)``.
    """

    scores: Tensor
    ref_bases: Tensor
    start: int
    end: int
    baseline: Optional[Tensor] = None
    counts: Optional[Tensor] = None

    @property
    def is_sparse(self) -> bool:
        """Whether the scores tensor is in sparse COO format."""
        return self.scores.is_sparse

    def to_sparse(self, cutoff: float) -> "MultiplexedISMResult":
        """Zero entries below cutoff and convert scores to sparse COO format.

        Parameters
        ----------
        cutoff : float
            Absolute threshold. Entries with ``|score| < cutoff`` are zeroed.

        Returns
        -------
        MultiplexedISMResult
            New instance with sparse scores tensor.
        """
        filtered = self.scores.clone()
        filtered[filtered.abs() < cutoff] = 0.0
        return MultiplexedISMResult(
            scores=filtered.to_sparse(),
            ref_bases=self.ref_bases,
            start=self.start,
            end=self.end,
            baseline=self.baseline,
            counts=self.counts,
        )

    def to_dense(self) -> "MultiplexedISMResult":
        """Convert sparse scores back to dense format.

        Returns
        -------
        MultiplexedISMResult
            New instance with dense scores tensor.  If already dense,
            returns a new instance with cloned scores.
        """
        return MultiplexedISMResult(
            scores=self.scores.to_dense() if self.scores.is_sparse else self.scores.clone(),
            ref_bases=self.ref_bases,
            start=self.start,
            end=self.end,
            baseline=self.baseline,
            counts=self.counts,
        )


class BaseAggregator(abc.ABC):
    """Abstract base for online aggregators."""

    @abc.abstractmethod
    def update(
        self,
        alt_idx: int,
        pos_indices: Tensor,
        deltas: Tensor,
    ) -> None:
        """Incorporate a batch of deltas.

        Parameters
        ----------
        alt_idx : int
            Alternative base index (0, 1, or 2).
        pos_indices : Tensor
            Positions that were mutated, shape ``(n_positions,)``.
        deltas : Tensor
            Delta values to aggregate, shape ``(n_positions, *T)``.
        """

    @abc.abstractmethod
    def result(self) -> Tensor:
        """Return the aggregated scores, shape ``(3, L_mut, *T)``."""


class SignedMaxAbsAggregator(BaseAggregator):
    """Keep the delta with the largest absolute value, preserving sign.

    For each ``(alt_idx, position, *task)`` cell, retains whichever
    observed delta has the largest ``|value|``.
    """

    def __init__(self, shape_3_L_T: tuple[int, ...], device: torch.device) -> None:
        self._best = torch.zeros(shape_3_L_T, dtype=torch.float32, device=device)
        self._best_abs = torch.zeros(shape_3_L_T, dtype=torch.float32, device=device)

    def update(
        self,
        alt_idx: int,
        pos_indices: Tensor,
        deltas: Tensor,
    ) -> None:
        """Update with new deltas, keeping the one with largest |value|.

        Parameters
        ----------
        alt_idx : int
            Alternative base index (0, 1, or 2).
        pos_indices : Tensor
            Positions that were mutated, shape ``(n_positions,)``.
            Must contain unique indices; duplicate indices produce
            undefined results due to non-deterministic write ordering.
        deltas : Tensor
            Delta values, shape ``(n_positions, *T)``.
        """
        cur_abs = deltas.abs()
        existing_abs = self._best_abs[alt_idx, pos_indices]  # (n_pos, *T)
        mask = cur_abs > existing_abs  # element-wise
        self._best[alt_idx, pos_indices] = torch.where(mask, deltas, self._best[alt_idx, pos_indices])
        self._best_abs[alt_idx, pos_indices] = torch.where(mask, cur_abs, existing_abs)

    def result(self) -> Tensor:
        """Return scores with shape ``(3, L_mut, *T)``."""
        return self._best


class MeanAggregator(BaseAggregator):
    """Running mean via sum and count.

    Maintains fp32 running sum and int64 counts for numerical stability.
    """

    def __init__(self, shape_3_L_T: tuple[int, ...], device: torch.device) -> None:
        self._sum = torch.zeros(shape_3_L_T, dtype=torch.float32, device=device)
        self._count = torch.zeros(shape_3_L_T[:2], dtype=torch.int64, device=device)

    def update(
        self,
        alt_idx: int,
        pos_indices: Tensor,
        deltas: Tensor,
    ) -> None:
        """Accumulate deltas into running sum and count.

        Parameters
        ----------
        alt_idx : int
            Alternative base index (0, 1, or 2).
        pos_indices : Tensor
            Positions that were mutated, shape ``(n_positions,)``.
        deltas : Tensor
            Delta values, shape ``(n_positions, *T)``.
        """
        self._sum[alt_idx].index_add_(0, pos_indices, deltas.float())
        self._count[alt_idx].index_add_(
            0, pos_indices, torch.ones_like(pos_indices, dtype=torch.int64)
        )

    def result(self) -> Tensor:
        """Return mean scores with shape ``(3, L_mut, *T)``."""
        count = self._count.clamp(min=1)
        # Broadcast count to match _sum shape: (3, L_mut) -> (3, L_mut, *T)
        for _ in range(self._sum.ndim - 2):
            count = count.unsqueeze(-1)
        return self._sum / count
