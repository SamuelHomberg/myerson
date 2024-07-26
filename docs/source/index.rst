.. figure:: images/logo_banner_embedded.svg
   :alt: Myerson


===================================
Game Theory and XAI
===================================

Welcome to the documentation for the ``myerson`` package. This package implements
the `Myerson solution concept <https://www.jstor.org/stable/3689511>`_ from
cooperative game theory. The Myerson values attribute
every player of a game their fair contribution to the games payoff. Myerson values are 
related to Shapley values but the player cooperation is restricted by a graph.

A graph neural network (GNN) can be treated as a coalition function for a game
and the Myerson values can be used as feature attribution explanations to understand
a model prediction. This package also implements Methods to explain `PyG <https://pyg.org//>`_ 
GNNs with Myerson values.

Calculating the Myerson value scales exponentially with bigger graphs / more players.
Therfore, Monte Carlo sampling techniques were implemented to approximate the Myerson values.

Installation
============

Install the complete package with PyTorch dependencies using one of the following commands:

.. code-block:: bash

   # pip
   pip install myerson[explain]

   # conda / mamba
   conda install myerson

   # for conda / mamba, you need to manually install pytorch dependencies, for example:
   conda install pytorch torchvision torchaudio cpuonly -c pytorch
   conda install conda install pyg -c pyg

If you are only interested in the game theoretic part you don't need to install PyTorch:

.. code-block:: bash

   # pip 
   pip install myerson

   # conda / mamba
   conda install myerson

Examples
========

For example code have a look on the :ref:`get started` page.


Contents
========
.. toctree::
   :maxdepth: 3

   Home <self>
   Get started <get_started>
   Documentation <documentation>
   GitHub ⧉ <https://github.com/kochgroup/myerson>
   ChemRxiv ⧉ <https://chemrxiv.org/engage/chemrxiv/article-details/668c3885c9c6a5c07aaca81e>
