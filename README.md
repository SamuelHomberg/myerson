![Myerson logo banner](docs/source/images/logo_banner_embedded.svg "Myerson")

[![test](https://github.com/kochgroup/myerson/actions/workflows/test.yml/badge.svg)](https://github.com/kochgroup/myerson/actions/workflows/test.yml)
[![Documentation Status](https://readthedocs.org/projects/myerson/badge/?version=latest)](https://myerson.readthedocs.io/en/latest/?badge=latest)

# Calculate Myerson values and explain GNNs

This package implements the Myerson solution concept from cooperative game theory. The Myerson values attribute every player of a game their fair contribution to the games payoff. Myerson values are related to Shapley values but the player cooperation is restricted by a graph.

A graph neural network (GNN) can be treated as a coalition function for a game and the Myerson values can be used as feature attribution explanations to understand a model prediction. This package also implements Methods to explain PyG GNNs with Myerson values.

Calculating the Myerson value scales exponentially with bigger graphs / more players. Therfore, Monte Carlo sampling techniques were implemented to approximate the Myerson values.

## Installation
Install the package with the following command:

```bash
pip install myerson
```

## Examples and Documentation
Example uses can be found [here](https://myerson.readthedocs.io/en/latest/get_started.html#get-started). The full documentation can be found at https://myerson.readthedocs.io/.

## Citation

TBD.