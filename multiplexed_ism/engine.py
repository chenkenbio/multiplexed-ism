"""Core streaming loop for multiplexed ISM."""

from __future__ import annotations

import logging
from typing import Any, Callable, Literal, Optional

import torch
from torch import Tensor

from .mutation import apply_mutations, get_ref_bases, sample_positions
from .types import (
    BaseAggregator,
    MeanAggregator,
    MultiplexedISMResult,
    SignedMaxAbsAggregator,
)

logger = logging.getLogger(__name__)


def _resolve_precision(
    precision: Literal["auto", "fp32", "bf16", "fp16"],
    device: torch.device,
) -> Optional[torch.dtype]:
    """Determine autocast dtype from precision setting.

    Parameters
    ----------
    precision : str
        Precision policy name.
    device : torch.device
        Target device.

    Returns
    -------
    torch.dtype or None
        Dtype for ``torch.autocast``, or None to disable autocast.
    """
    _VALID_PRECISIONS = {"auto", "fp32", "bf16", "fp16"}
    if precision not in _VALID_PRECISIONS:
        raise ValueError(
            f"Invalid precision {precision!r}, must be one of {_VALID_PRECISIONS}"
        )

    if precision == "fp32":
        return None
    if precision in ("bf16", "fp16"):
        return torch.bfloat16 if precision == "bf16" else torch.float16

    # auto: use bf16 on CUDA if supported, else fp32
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return None


def _extract_output(
    raw_output: Any,
    output_key: Optional[str],
    target_fn: Optional[Callable],
) -> Tensor:
    """Extract a tensor from model output.

    Priority: ``target_fn`` > ``output_key`` > raw tensor.

    Parameters
    ----------
    raw_output : Any
        Raw model output (Tensor or dict).
    output_key : str or None
        Key to index into a dict output.
    target_fn : callable or None
        Function to extract/transform the output.

    Returns
    -------
    Tensor
        Extracted output tensor.
    """
    if target_fn is not None:
        return target_fn(raw_output)
    if output_key is not None:
        return raw_output[output_key]
    return raw_output


def _make_aggregator(
    aggregation: str,
    shape: tuple[int, ...],
    device: torch.device,
) -> BaseAggregator:
    """Create an aggregator instance.

    Parameters
    ----------
    aggregation : str
        Aggregation method name.
    shape : tuple
        Shape ``(3, L_mut, *T)``.
    device : torch.device
        Device for aggregator buffers.

    Returns
    -------
    BaseAggregator
    """
    if aggregation == "signed_max_abs":
        return SignedMaxAbsAggregator(shape, device)
    elif aggregation == "mean":
        return MeanAggregator(shape, device)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation!r}")


def run_ism(
    model: torch.nn.Module | Callable,
    seq: Tensor,
    n_var_per_run: int,
    n_samples: int,
    *,
    start: int,
    end: int,
    inner_batch_size: int = 32,
    output_key: Optional[str] = None,
    target_fn: Optional[Callable] = None,
    aggregation: str = "signed_max_abs",
    precision: Literal["auto", "fp32", "bf16", "fp16"] = "auto",
    seed: Optional[int] = None,
    device: torch.device,
    return_baseline: bool = False,
    return_counts: bool = False,
    show_progress: bool = False,
) -> MultiplexedISMResult:
    """Execute the multiplexed ISM streaming loop.

    Parameters
    ----------
    model : Module or callable
        PyTorch model or callable ``(batch, 4, L) -> output``.
    seq : Tensor
        One-hot encoded input, shape ``(4, L)``, on ``device``.
    n_var_per_run : int
        Positions mutated simultaneously per sample.
    n_samples : int
        Total number of multiplexed samples.
    start : int
        Start of mutable region (inclusive).
    end : int
        End of mutable region (exclusive).
    inner_batch_size : int
        Max sequences per forward pass.
    output_key : str or None
        Key for dict model outputs.
    target_fn : callable or None
        Transform applied to model output.
    aggregation : str
        Aggregation method (``"signed_max_abs"`` or ``"mean"``).
    precision : str
        Precision policy.
    seed : int or None
        RNG seed for reproducibility.
    device : torch.device
        Compute device.
    return_baseline : bool
        Whether to include baseline prediction in result.
    return_counts : bool
        Whether to include per-position counts in result.
    show_progress : bool
        Whether to show a progress bar.

    Returns
    -------
    MultiplexedISMResult
    """
    L_mut = end - start
    autocast_dtype = _resolve_precision(precision, device)

    # --- baseline forward pass ---
    with torch.inference_mode():
        ctx = (
            torch.autocast(device_type=device.type, dtype=autocast_dtype)
            if autocast_dtype is not None
            else _nullcontext()
        )
        with ctx:
            baseline_pred = _extract_output(
                model(seq.unsqueeze(0)), output_key, target_fn
            )
            baseline_pred = baseline_pred.squeeze(0).float()  # (*T,)

    # --- setup ---
    ref_bases = get_ref_bases(seq, start, end)
    task_shape = baseline_pred.shape  # (*T,)
    agg_shape = (3, L_mut, *task_shape)
    aggregator = _make_aggregator(aggregation, agg_shape, device)

    # Per-position sample counts
    counts = torch.zeros(L_mut, dtype=torch.int64, device=device)

    # RNG
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)

    # Sample all positions upfront
    all_positions = sample_positions(L_mut, n_var_per_run, n_samples, generator)

    # samples_per_batch: how many multiplexed samples fit in one forward pass
    # Each sample produces 3 mutated sequences (one per alt)
    samples_per_batch = max(1, inner_batch_size // 3)

    # --- progress bar ---
    sample_indices = range(0, n_samples, samples_per_batch)
    if show_progress:
        try:
            from tqdm import tqdm
            sample_indices = tqdm(
                sample_indices, desc="Multiplexed ISM", total=len(sample_indices)
            )
        except ImportError:
            logger.warning("tqdm not installed; progress bar disabled")

    # --- streaming loop ---
    with torch.inference_mode():
        ctx = (
            torch.autocast(device_type=device.type, dtype=autocast_dtype)
            if autocast_dtype is not None
            else _nullcontext()
        )
        with ctx:
            for batch_start in sample_indices:
                batch_end = min(batch_start + samples_per_batch, n_samples)
                batch_positions = all_positions[batch_start:batch_end]  # (B, n_var)
                actual_batch = batch_end - batch_start

                # Build mutated sequences: for each sample, 3 alt bases
                mutated_seqs = []
                for s in range(actual_batch):
                    pos = batch_positions[s].to(device)
                    for alt_idx in range(3):
                        mutated = apply_mutations(
                            seq, ref_bases, pos, alt_idx, start
                        )
                        mutated_seqs.append(mutated)

                batch_input = torch.stack(mutated_seqs)  # (B*3, 4, L)
                raw_out = model(batch_input)
                output = _extract_output(raw_out, output_key, target_fn)
                output = output.float()  # (B*3, *T)

                # Compute deltas and update aggregator
                deltas = output - baseline_pred.unsqueeze(0)  # (B*3, *T)

                for s in range(actual_batch):
                    pos = batch_positions[s].to(device)
                    counts.index_add_(
                        0, pos, torch.ones_like(pos, dtype=torch.int64)
                    )
                    for alt_idx in range(3):
                        idx = s * 3 + alt_idx
                        delta = deltas[idx]  # (*T,)
                        # Expand delta to all mutated positions
                        delta_expanded = delta.unsqueeze(0).expand(
                            pos.shape[0], *task_shape
                        )
                        aggregator.update(alt_idx, pos, delta_expanded)

    scores = aggregator.result()
    return MultiplexedISMResult(
        scores=scores,
        ref_bases=ref_bases,
        start=start,
        end=end,
        baseline=baseline_pred if return_baseline else None,
        counts=counts if return_counts else None,
    )


class _nullcontext:
    """Minimal no-op context manager (avoids contextlib import)."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: Any) -> None:
        pass
