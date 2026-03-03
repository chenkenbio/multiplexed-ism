"""Public API for multiplexed in silico mutagenesis."""

from __future__ import annotations

import logging
from typing import Callable, Literal, Optional

import torch
from torch import Tensor

from .engine import run_ism
from .types import MultiplexedISMResult

logger = logging.getLogger(__name__)


def _validate_inputs(
    seq: Tensor,
    n_var_per_run: int,
    n_samples: int,
    start: int,
    end: int,
    inner_batch_size: int,
) -> None:
    """Validate inputs for multiplexed ISM.

    Parameters
    ----------
    seq : Tensor
        One-hot encoded sequence, expected shape ``(4, L)``.
    n_var_per_run : int
        Positions mutated per sample.
    n_samples : int
        Number of multiplexed samples.
    start : int
        Start of mutable region.
    end : int
        End of mutable region.
    inner_batch_size : int
        Sequences per forward pass.

    Raises
    ------
    ValueError
        If any input is invalid.
    """
    if seq.ndim != 2 or seq.shape[0] != 4:
        raise ValueError(
            f"input_seq must have shape (4, L), got {tuple(seq.shape)}"
        )

    L = seq.shape[1]

    # Check one-hot validity: each column should sum to 1 and contain only 0/1
    col_sums = seq.sum(dim=0)
    if not torch.allclose(col_sums, torch.ones_like(col_sums)):
        raise ValueError("input_seq is not valid one-hot: columns must sum to 1")

    unique_vals = seq.unique()
    if not all(v in (0.0, 1.0) for v in unique_vals.tolist()):
        raise ValueError("input_seq is not valid one-hot: values must be 0 or 1")

    if not (0 <= start < end <= L):
        raise ValueError(
            f"Invalid range: start={start}, end={end}, L={L}. "
            f"Require 0 <= start < end <= L."
        )

    if n_var_per_run < 1:
        raise ValueError(f"n_var_per_run must be >= 1, got {n_var_per_run}")

    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    if inner_batch_size < 1:
        raise ValueError(f"inner_batch_size must be >= 1, got {inner_batch_size}")


def multiplexed_ism(
    model: torch.nn.Module | Callable,
    input_seq: Tensor,
    n_var_per_run: int,
    n_samples: int,
    *,
    start: Optional[int] = None,
    end: Optional[int] = None,
    inner_batch_size: int = 32,
    output_key: Optional[str] = None,
    target_fn: Optional[Callable[[Tensor | dict[str, Tensor]], Tensor]] = None,
    aggregation: Literal["signed_max_abs", "mean"] = "signed_max_abs",
    precision: Literal["auto", "fp32", "bf16", "fp16"] = "auto",
    compile: bool = False,
    chunk_size: Optional[int] = None,
    score_cutoff: Optional[float] = None,
    seed: Optional[int] = None,
    device: Optional[torch.device | str] = None,
    return_baseline: bool = False,
    return_counts: bool = False,
    show_progress: bool = False,
) -> MultiplexedISMResult:
    """Multiplexed in silico mutagenesis.

    Mutates ``n_var_per_run`` positions simultaneously per sample,
    requiring only ``n_samples`` forward passes instead of the
    standard ``3 * L`` passes.

    Parameters
    ----------
    model : Module or callable
        PyTorch model accepting ``(batch, 4, L)`` input.
    input_seq : Tensor
        One-hot encoded sequence, shape ``(4, L)``.
    n_var_per_run : int
        Number of positions mutated per sample.
    n_samples : int
        Total multiplexed samples.
    start : int or None
        Start of mutable region (inclusive). Default: 0.
    end : int or None
        End of mutable region (exclusive). Default: L.
    inner_batch_size : int
        Max sequences per forward pass. Default: 32.
    output_key : str or None
        Key for dict model outputs.
    target_fn : callable or None
        Transform applied to model output before aggregation.
    aggregation : str
        ``"signed_max_abs"`` or ``"mean"``. Default: ``"signed_max_abs"``.
    precision : str
        ``"auto"``, ``"fp32"``, ``"bf16"``, or ``"fp16"``. Default: ``"auto"``.
    compile : bool
        Whether to ``torch.compile`` the model. Default: False.
    chunk_size : int or None
        Split mutable region into chunks to limit memory. Default: None.
    score_cutoff : float or None
        If set, zero entries where ``|score| < cutoff`` and return
        scores in sparse COO format.  Must be > 0.  Applied after
        aggregation so it is purely a storage optimisation.
    seed : int or None
        RNG seed for reproducibility.
    device : device or None
        Compute device. Default: inferred from model parameters.
    return_baseline : bool
        Include unmodified prediction in result.
    return_counts : bool
        Include per-position sample counts in result.
    show_progress : bool
        Show tqdm progress bar.

    Returns
    -------
    MultiplexedISMResult
    """
    L = input_seq.shape[-1]

    # Defaults
    if start is None:
        start = 0
    if end is None:
        end = L
    if device is None:
        try:
            device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            device = torch.device("cpu")
    device = torch.device(device)

    # Move seq to device
    seq = input_seq.to(device).float()
    L_mut = end - start

    # Clamp n_var_per_run
    n_var_per_run = min(n_var_per_run, L_mut)

    # Validate
    _validate_inputs(seq, n_var_per_run, n_samples, start, end, inner_batch_size)

    # Move model to device and eval mode
    if isinstance(model, torch.nn.Module):
        model = model.to(device).eval()

    # Optional compile
    if compile:
        try:
            model = torch.compile(model)
            logger.info("Model compiled with torch.compile")
        except Exception as e:
            logger.warning(f"torch.compile failed, falling back to eager: {e}")

    # Validate chunk_size
    if chunk_size is not None and chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    # Validate score_cutoff
    if score_cutoff is not None and score_cutoff <= 0:
        raise ValueError(f"score_cutoff must be > 0, got {score_cutoff}")

    # Chunked execution
    if chunk_size is not None and chunk_size < L_mut:
        result = _run_chunked(
            model=model,
            seq=seq,
            n_var_per_run=n_var_per_run,
            n_samples=n_samples,
            start=start,
            end=end,
            chunk_size=chunk_size,
            inner_batch_size=inner_batch_size,
            output_key=output_key,
            target_fn=target_fn,
            aggregation=aggregation,
            precision=precision,
            seed=seed,
            device=device,
            return_baseline=return_baseline,
            return_counts=return_counts,
            show_progress=show_progress,
        )
    else:
        # Single run
        result = run_ism(
            model=model,
            seq=seq,
            n_var_per_run=n_var_per_run,
            n_samples=n_samples,
            start=start,
            end=end,
            inner_batch_size=inner_batch_size,
            output_key=output_key,
            target_fn=target_fn,
            aggregation=aggregation,
            precision=precision,
            seed=seed,
            device=device,
            return_baseline=return_baseline,
            return_counts=return_counts,
            show_progress=show_progress,
        )

    # Apply sparse cutoff filter
    if score_cutoff is not None:
        result = result.to_sparse(score_cutoff)

    return result


def _run_chunked(
    model: torch.nn.Module | Callable,
    seq: Tensor,
    n_var_per_run: int,
    n_samples: int,
    start: int,
    end: int,
    chunk_size: int,
    inner_batch_size: int,
    output_key: Optional[str],
    target_fn: Optional[Callable],
    aggregation: str,
    precision: str,
    seed: Optional[int],
    device: torch.device,
    return_baseline: bool,
    return_counts: bool,
    show_progress: bool,
) -> MultiplexedISMResult:
    """Run ISM in positional chunks to limit memory usage.

    Parameters
    ----------
    chunk_size : int
        Max positions per chunk.

    Returns
    -------
    MultiplexedISMResult
        Concatenated results across all chunks.
    """
    all_scores = []
    all_ref_bases = []
    all_counts = []
    baseline = None

    chunks = list(range(start, end, chunk_size))

    for chunk_start in chunks:
        chunk_end = min(chunk_start + chunk_size, end)
        chunk_n_var = min(n_var_per_run, chunk_end - chunk_start)

        result = run_ism(
            model=model,
            seq=seq,
            n_var_per_run=chunk_n_var,
            n_samples=n_samples,
            start=chunk_start,
            end=chunk_end,
            inner_batch_size=inner_batch_size,
            output_key=output_key,
            target_fn=target_fn,
            aggregation=aggregation,
            precision=precision,
            seed=seed,
            device=device,
            return_baseline=return_baseline,
            return_counts=return_counts,
            show_progress=show_progress,
        )

        all_scores.append(result.scores)
        all_ref_bases.append(result.ref_bases)
        if result.counts is not None:
            all_counts.append(result.counts)
        if baseline is None and result.baseline is not None:
            baseline = result.baseline

    return MultiplexedISMResult(
        scores=torch.cat(all_scores, dim=1),
        ref_bases=torch.cat(all_ref_bases, dim=0),
        start=start,
        end=end,
        baseline=baseline,
        counts=torch.cat(all_counts, dim=0) if all_counts else None,
    )
