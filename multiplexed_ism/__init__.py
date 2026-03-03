"""Multiplexed in silico mutagenesis for PyTorch genomic models."""

__version__ = "0.1.0"

from .api import multiplexed_ism
from .types import MultiplexedISMResult

__all__ = ["multiplexed_ism", "MultiplexedISMResult"]
