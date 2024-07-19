"""A python package to calculate Myerson values from game theory and use them as explanations for graph neural networks."""

from .myerson import MyersonCalculator, MyersonSampler
import warnings
try:
    import torch
    import torch_geometric
    from .myerson_explain import MyersonExplainer, MyersonSamplingExplainer
    from .myerson_explain import explain
except ImportError:
    warnings.warn("Failed to import torch and/or torch_geometric. Explanations not available.")

__version__ = "0.1.7"