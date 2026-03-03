"""Mutation generation: ALT_TABLE, position sampling, and sequence editing."""

from __future__ import annotations

import torch
from torch import Tensor


# ALT_TABLE[ref_base] gives the 3 alternative bases in sorted order.
# ref=0(A) -> [1(C), 2(G), 3(T)]
# ref=1(C) -> [0(A), 2(G), 3(T)]
# ref=2(G) -> [0(A), 1(C), 3(T)]
# ref=3(T) -> [0(A), 1(C), 2(G)]
ALT_TABLE = torch.tensor(
    [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]],
    dtype=torch.long,
)


def get_ref_bases(seq: Tensor, start: int, end: int) -> Tensor:
    """Extract reference base indices from a one-hot sequence.

    Parameters
    ----------
    seq : Tensor
        One-hot encoded sequence, shape ``(4, L)``.
    start : int
        Start position (inclusive).
    end : int
        End position (exclusive).

    Returns
    -------
    Tensor
        Reference base indices, shape ``(L_mut,)`` with values in {0,1,2,3}.
    """
    return seq[:, start:end].argmax(dim=0)


def sample_positions(
    L_mut: int,
    n_var_per_run: int,
    n_samples: int,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample mutation positions without replacement per sample.

    Parameters
    ----------
    L_mut : int
        Length of the mutable region.
    n_var_per_run : int
        Number of positions to mutate per sample.
    n_samples : int
        Number of multiplexed samples.
    generator : torch.Generator or None
        RNG for reproducibility.

    Returns
    -------
    Tensor
        Position indices, shape ``(n_samples, n_var_per_run)``.
    """
    positions = torch.empty(n_samples, n_var_per_run, dtype=torch.long)
    for i in range(n_samples):
        perm = torch.randperm(L_mut, generator=generator)
        positions[i] = perm[:n_var_per_run]
    return positions


def apply_mutations(
    seq: Tensor,
    ref_bases: Tensor,
    positions: Tensor,
    alt_idx: int,
    start: int,
) -> Tensor:
    """Apply mutations to a one-hot sequence.

    For each position in ``positions``, zero the reference channel and set
    the alternative channel to 1.0, producing a valid one-hot encoding.

    Parameters
    ----------
    seq : Tensor
        Original one-hot sequence, shape ``(4, L)``.
    ref_bases : Tensor
        Reference base indices for the mutable region, shape ``(L_mut,)``.
    positions : Tensor
        Positions to mutate (relative to mutable region), shape ``(n_var,)``.
    alt_idx : int
        Which alternative base to use (0, 1, or 2 indexing into ALT_TABLE).
    start : int
        Start of the mutable region in the full sequence.

    Returns
    -------
    Tensor
        Mutated copy of the sequence, shape ``(4, L)``.
    """
    mutated = seq.clone()
    abs_positions = positions + start
    ref_at_pos = ref_bases[positions]

    # Zero out reference channel
    mutated[ref_at_pos, abs_positions] = 0.0

    # Set alternative channel
    alt_table = ALT_TABLE.to(ref_at_pos.device)
    alt_bases = alt_table[ref_at_pos, alt_idx]
    mutated[alt_bases, abs_positions] = 1.0

    return mutated
