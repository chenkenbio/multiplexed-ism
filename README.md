# multiplexed-ism

GPU-first multiplexed in silico mutagenesis (ISM) for PyTorch genomic models.

Standard ISM requires `3 × L` forward passes to score every single-nucleotide variant. Multiplexed ISM mutates `n_var_per_run` positions simultaneously per sample, recovering per-position effect estimates from only `n_samples` forward passes — typically orders of magnitude fewer.

## Installation

```bash
pip install .

# with progress bar support
pip install ".[progress]"

# development
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.0, NumPy ≥ 1.24

## Quick start

```python
import torch
from multiplexed_ism import multiplexed_ism

# Any PyTorch model that accepts (batch, 4, L) one-hot input
model = MyGenomicModel()

# One-hot encoded DNA sequence, shape (4, L)
seq = torch.zeros(4, 1000)
seq[torch.randint(0, 4, (1000,)), torch.arange(1000)] = 1.0

result = multiplexed_ism(
    model,
    seq,
    n_var_per_run=10,   # positions mutated per sample
    n_samples=500,      # total forward passes (vs 3000 for standard ISM)
    device="cuda",
    seed=42,
)

# result.scores: (3, L, *T) — delta per alt base per position
# result.ref_bases: (L,) — reference base indices (0=A, 1=C, 2=G, 3=T)
```

## API reference

### `multiplexed_ism()`

```python
multiplexed_ism(
    model,                          # nn.Module or callable: (B, 4, L) -> output
    input_seq,                      # one-hot tensor, shape (4, L)
    n_var_per_run,                  # positions mutated per sample
    n_samples,                      # number of multiplexed samples
    *,
    start=None, end=None,           # mutable region [start, end)
    inner_batch_size=32,            # sequences per forward pass
    output_key=None,                # key for dict model outputs
    target_fn=None,                 # transform on model output before aggregation
    aggregation="signed_max_abs",   # "signed_max_abs" or "mean"
    precision="auto",               # "auto", "fp32", "bf16", "fp16"
    compile=False,                  # torch.compile the model
    chunk_size=None,                # split region into chunks (memory control)
    score_cutoff=None,              # sparsify: zero |score| < cutoff, return COO
    seed=None,                      # RNG seed for reproducibility
    device=None,                    # compute device (inferred from model if None)
    return_baseline=False,          # include unmodified prediction
    return_counts=False,            # include per-position sample counts
    show_progress=False,            # tqdm progress bar
)
```

Returns a `MultiplexedISMResult` with fields:
- **`scores`**: `Tensor` of shape `(3, L_mut, *T)` — aggregated deltas
- **`ref_bases`**: `Tensor` of shape `(L_mut,)` — reference base indices
- **`start`** / **`end`**: mutable region bounds
- **`baseline`**: unmodified prediction (if `return_baseline=True`)
- **`counts`**: per-position sample counts (if `return_counts=True`)

### Aggregation methods

| Method | Behaviour |
|---|---|
| `"signed_max_abs"` | Keep the delta with the largest absolute value, preserving sign |
| `"mean"` | Running mean of all observed deltas per cell |

### Sparse storage

For long sequences where most scores are near-zero, use `score_cutoff` to filter and compress:

```python
# Sparsify during computation
result = multiplexed_ism(model, seq, ..., score_cutoff=0.01)
assert result.is_sparse

# Convert back to dense when needed
dense_result = result.to_dense()

# Or sparsify an existing dense result
sparse_result = dense_result.to_sparse(cutoff=0.05)
```

Filtering is applied **after** aggregation, so the computation remains exact — sparsification is purely a storage optimisation.

### Chunked execution

For memory-constrained settings, split the mutable region into positional chunks:

```python
result = multiplexed_ism(
    model, seq, n_var_per_run=10, n_samples=500,
    chunk_size=200,  # process 200 positions at a time
)
```

### Dict and multi-output models

```python
# Model returns {"logits": Tensor, "embeddings": Tensor}
result = multiplexed_ism(model, seq, ..., output_key="logits")

# Or use target_fn for custom extraction
result = multiplexed_ism(
    model, seq, ...,
    target_fn=lambda d: d["logits"][:, :2],  # first 2 outputs only
)
```

## Testing

```bash
pytest tests/ -v
```

## License

MIT
