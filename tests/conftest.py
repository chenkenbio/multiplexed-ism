"""Shared fixtures and toy models for testing multiplexed ISM."""

import pytest
import torch
import torch.nn as nn


class SumModel(nn.Module):
    """Output = weighted sum over positions. Linear and position-independent."""

    def __init__(self, n_channels: int = 4) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.arange(n_channels, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, L) -> (B, 1)
        return (x * self.weights[None, :, None]).sum(dim=(1, 2), keepdim=False).unsqueeze(-1)


class LinearModel(nn.Module):
    """Flatten + linear. Each position contributes independently."""

    def __init__(self, seq_len: int, n_outputs: int = 2) -> None:
        super().__init__()
        self.linear = nn.Linear(4 * seq_len, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.reshape(x.shape[0], -1))


class DictModel(nn.Module):
    """Returns a dict output to test output_key / target_fn extraction."""

    def __init__(self, seq_len: int) -> None:
        super().__init__()
        self.linear = nn.Linear(4 * seq_len, 3)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.linear(x.reshape(x.shape[0], -1))
        return {"logits": out, "extra": out * 2}


class ConvModel(nn.Module):
    """Simple 1D conv model for realistic genomic-style testing."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(4, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv(x))
        h = self.pool(h).squeeze(-1)
        return self.fc(h)


@pytest.fixture
def short_seq() -> torch.Tensor:
    """One-hot sequence of length 20."""
    torch.manual_seed(42)
    bases = torch.randint(0, 4, (20,))
    seq = torch.zeros(4, 20)
    seq[bases, torch.arange(20)] = 1.0
    return seq


@pytest.fixture
def medium_seq() -> torch.Tensor:
    """One-hot sequence of length 100."""
    torch.manual_seed(123)
    bases = torch.randint(0, 4, (100,))
    seq = torch.zeros(4, 100)
    seq[bases, torch.arange(100)] = 1.0
    return seq
