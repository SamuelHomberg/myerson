"""A python package to calculate Myerson values from game theory and use them as explanations for graph neural networks."""

from .myerson import MyersonCalculator, MyersonSampler
try:
    import torch
    import torch_geometric
    from .myerson import MyersonExplainer, MyersonSamplingExplainer
except:
    pass
from .myerson import explain

__version__ = "0.1.7"