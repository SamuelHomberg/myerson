import math
import networkx as nx
import numpy as np
from itertools import combinations, chain

from typing import Callable 

from tqdm import tqdm
import logging


class ShapleyCalculator():
    r"""Calculates the exact Shapley values. 
        For a game described by a coalition function :math:`v` the Shapley values
        attribute the individual players contribution to the payoff of the game
        (:math:`S`: coalition of players, :math:`N`: grand coalition of all players):

        .. math::

            \text{Sh}_i\,({v}) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! \: (|N| - |S| - 1)!}{|N|!}\big( {v}\,(S \cup \{i\}) - {v}\,(S) \big)

    Args:
        graph (nx.classes.graph.Graph): The players of the game. The graph
            structure (edges) are not used for the shapley value.
        coalition_function (Callable): The coalition for which to calculate
            the payoff of the game. Expects a coalition (tuple of node
            indices) and a graph which contains additional information on
            the players to decide on the payoff. 

        disable_tqdm (bool, optional): Disables progress bar. Defaults to
            True.
    """

    def __init__(self,
                 graph: nx.classes.graph.Graph,
                 coalition_function: Callable,
                 disable_tqdm: bool=True) -> None:
        """Instantiate the class.
        """

        self.disable_tqdm = disable_tqdm
        self.log = logging.getLogger("ShapleyCalculator")
        self.nx_graph = graph
        self.grand_coalition = list(graph.nodes()) # alias: set of players / set of nodes / F
        self.coalition_function = coalition_function

    def calculate_coalitions(self, grand_coalition: list) -> list:
        r"""Calculate all possible coalitions for a set of players / all nodes in
        a graph.

        Args:
            grand_coalition (list): Set of players / grand coalition / atoms in
                graph as a list of tupels.

        Returns:
            list: All :math:`2^N` coalitions.
        """
        self.log.info(f"Calculating number of coalitions.")
        coalitions = [combinations(grand_coalition, len(grand_coalition)-x) for x in \
                      tqdm(range(len(grand_coalition)), desc="Calculate coalitions", disable=self.disable_tqdm)]
        coalitions = list(chain.from_iterable(coalitions)) # chaining removes empty set
        coalitions.append(())
        self.log.info(f"Number of coalitions: {len(coalitions)}")
        return coalitions

    def calculate_worth_of_single_coalition(self,
        coalition: tuple,
        nx_graph: nx.classes.graph.Graph) -> float:
        """Calculate the worth of a singel coalition.

        Args:
            coalition (tuple): Coalition as node indices.
            nx_graph (nx.classes.graph.Graph): Additional information for the
                coalition function, i.e. the entire graph with node parameters. 
                The result of the coalition function should depend only on the
                coalition, however the node parameters might contain necessary 
                information.

        Returns:
            float: Worth, the output of the coalition function for the players in
                coalition.
        """
        return self.coalition_function(coalition, nx_graph)

    def calculate_worth_of_coalitions(self, coalitions: set) -> dict:
        """Calculate the worth of every coalition and map it to its worth. 

        Args:
            coalitions (set): Set of coalitions as tuples of node indices.

        Returns:
            dict: Dictionary mapping each coalition to its worth.
        """
        self.log.info(f"Calculating worth of coalitions.")
        coalitions_to_worth = {}
        for coalition in tqdm(coalitions,
                              desc="Calculating worth of coalitions",
                              disable=self.disable_tqdm):
            worth = self.calculate_worth_of_single_coalition(coalition, self.nx_graph)
            coalitions_to_worth.update({coalition: worth})
        return coalitions_to_worth

    def calculate_worth_of_grand_coalition(self, nx_graph: nx.classes.graph.Graph) -> float:
        """Calculate payoff of the game, i.e. the payoff of all players / the
        grand coalition.

        Args:
            coalition_function (Callable): The coalition function associating a
                coalition with a payoff.
            nx_graph (nx.classes.graph.Graph): The players are nodes in a graph.

        Returns:
            float: Payoff of the game / worth of grand coalition.
        """
        return self.coalition_function(self.grand_coalition, nx_graph)

    def calculate_single_shapley_value(self, node: int, grand_coalition: tuple,
                                  coalitions: list[tuple], coalition_to_worth: dict) -> float:
        """Calculate a single Shapley value.

        Args:
            node (int): Node index for which to calculate the Shapley value.
            grand_coalition (tuple): Set of all players.
            coalitions (list[tuple]): List of all coalitions.
            coalition_to_worth (dict): Mapping of every coalition to its worth.

        Returns:
            float: Shapley value.
        """
        sh = 0
        size_grand_coalition = len(grand_coalition)
        factorial_size_grand_coalition = math.factorial(size_grand_coalition)
        for coalition in [S for S in coalitions if node not in S]:
            size_coalition = len(coalition)
            prefactor = ((math.factorial(size_coalition)
                         *math.factorial(size_grand_coalition-size_coalition-1))
                         /factorial_size_grand_coalition)
            worth_of_coalition = coalition_to_worth[coalition]
            worth_of_coalition_with_node = coalition_to_worth[tuple(sorted(coalition+(node,)))]
            sh += prefactor * (worth_of_coalition_with_node - worth_of_coalition)
        return sh

    def calculate_all_shapley_values(self) -> dict:
        """Calculate the Shapley values for every node / player in the graph.

        Returns:
            dict: Mapping of each node index to the Shapley value.
        """
        self.calculate_all_mappings()
        self.log.info(f"Calculating Shapley values.")
        sh_values = {}
        for node in tqdm(self.grand_coalition, desc="Calculating Shapley values.", disable=self.disable_tqdm):
            sh_val = self.calculate_single_shapley_value(node, self.grand_coalition,
                                                  self.coalitions, self.coalitions_to_worth)
            sh_values.update({node: sh_val})
        log_string = "".join([f"\t{k}: {v}\n" for k, v in sh_values.items()])
        self.log.info(f"Shapley Values:\n{log_string}")
        return sh_values

    def calculate_all_mappings(self) -> None:
        """Calculates all coalitions, and their associated worths as class attributes:

            * `self.coalitions` (list[tuple])
            * `self.coalitions_to_worth` (dict)
        """
        self.coalitions = self.calculate_coalitions(self.grand_coalition)

        self.coalitions_to_worth \
            = self.calculate_worth_of_coalitions(self.coalitions)