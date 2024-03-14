.. figure:: images/logo_banner.svg
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

Install the package with the following command:

.. code-block::

   pip install -i https://test.pypi.org/simple/ myerson

Examples
========

For examples have a look in the :ref:`get started` section.


Contents
========
.. toctree::
   :maxdepth: 3

   Home <self>
   Get started <get_started>
   Documentation <documentation>
   GitHub ⧉ <https://github.com/SamuelHomberg/myerson>
   ChemRxiv ⧉ <https://chemrxiv.org/engage/chemrxiv/article-details/6456c89707c3f0293753101d>
