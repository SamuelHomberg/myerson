"""A python package to calculate Myerson values from game theory and use them as explanations for graph neural networks."""

from .myerson import MyersonCalculator, MyersonSampler
from .shapley import ShapleyCalculator, ShapleySampler
import logging

log = logging.getLogger(__name__)

try:
    import torch #type: ignore
except ImportError:
    logging.warning("Failed to import torch. Explanations not available.")
try:
    import torch_geometric #type: ignore
    from . import pyg_explain
except ImportError:
    logging.warning("Failed to import torch_geometric. PyG explanations not available.")
try:
    import chemprop #type: ignore
    from . import chemprop_explain
except ImportError:
    logging.warning("Failed to import chemprop. MPNN explanations not available.")

__all__ = [
    "MyersonCalculator",
    "MyersonSampler",
    "ShapleyCalculator",
    "ShapleySampler",
    "pyg_explain",
    "chemprop_explain",
]

__version__ = "0.1.8" # update this