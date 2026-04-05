"""A python package to calculate Myerson values from game theory and use them as explanations for graph neural networks."""

from .myerson import MyersonCalculator, MyersonSampler
from .shapley import ShapleyCalculator, ShapleySampler
import warnings
try:
    import torch
    import torch_geometric
    from .explain.myerson_explain import MyersonExplainer, MyersonSamplingExplainer
    from .explain.myerson_explain import MyersonClassExplainer, MyersonSamplingClassExplainer
    from .explain.myerson_explain import explain

    from .explain.shapley_explain import ShapleyExplainer, ShapleySamplingExplainer
    from .explain.shapley_explain import ShapleyClassExplainer, ShapleySamplingClassExplainer

    from .explain.perturbation_explain import PerturbationExplainer, PerturbationClassExplainer
except ImportError:
    warnings.warn("Failed to import torch and/or torch_geometric. Explanations not available.")

__version__ = "0.1.8" # update this