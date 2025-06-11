import math
import networkx as nx
import numpy as np
from itertools import combinations, chain

from typing import Callable 

from tqdm import tqdm
import logging


class ShapleyCalculator():
    r"""Calculates the exact Myerson values. 
        For a game described by a coalition function :math:`v` the Myerson values
        attribute the individual players contribution to the payoff of the game. For
        a complete graph (every node connected to every other node) the Myerson
        value is equal to the Shapley value (:math:`S`: coalition of players,
        :math:`N`: grand coalition of all players):

        .. math::

            \text{Sh}_i\,({v}) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! \: (|N| - |S| - 1)!}{|N|!}\big( {v}\,(S \cup \{i\}) - {v}\,(S) \big)

        Else, the additional gain players can obthain through coalition is
        restricted only to players which are connected by edges in the graph. 

    Args:
        graph (nx.classes.graph.Graph): The coalition structure of the game.
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

    # def calculate_graph_restricted_coalitions(self, coalitions: list, 
    #                               nx_graph: nx.classes.graph.Graph) -> tuple[set, dict]:
    #     """Calculate the graph restricted coalitions for each coalition. The 
    #     graph restricted coalitions are tuples of nodes which are connected.

    #     Args:
    #         coalitions (list): All coalitions.
    #         nx_graph (nx.classes.graph.Graph): NetworkX Graph for which to
    #         calculate the Myerson values.

    #     Returns:
    #         tuple[set, dict]: Set of all possible graph restricted coalitions as
    #             a tuple of nodes, dictionary mapping each coalition to its
    #             graph restricted coalitions.
    #     """
    #     self.log.info(f"Calculating number of graph restricted coalitions.")
    #     graph_restricted_coalitions = [self.restrict(coalition, nx_graph) \
    #                                    for coalition in tqdm(coalitions,
    #                                        desc="Calculate graph restricted coalitions",
    #                                        disable=self.disable_tqdm)]
    #     coalitions_to_graph_restricted_coalitions = dict(zip(coalitions, graph_restricted_coalitions))
    #     graph_restricted_coalitions = list(chain.from_iterable(graph_restricted_coalitions))
    #     self.log.info(f"Removing dublicates from {len(graph_restricted_coalitions)} graph restricted coalitions.")
    #     graph_restricted_coalitions = set(x for x in tqdm(graph_restricted_coalitions, desc="Remove duplicates", disable=self.disable_tqdm))
    #     self.log.info(f"Number of graph restricted coalitions: {len(graph_restricted_coalitions)}")
    #     return graph_restricted_coalitions, coalitions_to_graph_restricted_coalitions

    def calculate_worth_of_single_coalition(self, # previously calculate_worth_of_single_graph_restricted_coalition
        coalition: tuple,
        nx_graph: nx.classes.graph.Graph) -> float:
        return self.coalition_function(coalition, nx_graph)

    def calculate_worth_of_coalitions(self, # previously calculate_worth_of_graph_restricted_coalitions
        coalitions: set) -> dict:
        self.log.info(f"Calculating worth of coalitions.")
        coalitions_to_worth = {}
        for coalition in tqdm(coalitions,
                              desc="Calculating worth of coalitions",
                              disable=self.disable_tqdm):
            worth = self.calculate_worth_of_single_coalition(coalition, self.nx_graph)
            coalitions_to_worth.update({coalition: worth})
        return coalitions_to_worth

    # def map_coalition_to_worth(self, coalitions: list[tuple], 
    #                    coalitions_to_graph_restricted_coalitions: dict,
    #                    graph_restricted_coalitions_to_worth: dict) -> dict:
    #     """Map every coalition to its worth.

    #     Args:
    #         coalitions (list): List of all coalitions (2^{num_nodes}). 
    #         coalitions_to_graph_restricted_coalitions (dict): Dictionary mapping
    #             the coalitions to the corresponding graph restricted coalitions.
    #         graph_restricted_coalitions_to_worth (dict): Dictionary mapping the 
    #             graph restricted coalitions to their worth.

    #     Returns:
    #         dict: Dictionary mapping each coalition to its worth.
    #     """
    #     self.log.info(f"Mapping coalitions to worth.")
    #     coalition_to_worth = {}
    #     for coalition in tqdm(coalitions, desc="Mapping coalitions to worth", disable=self.disable_tqdm):
    #         worth = 0.
    #         for graph_restricted_coalition in coalitions_to_graph_restricted_coalitions[coalition]:
    #             worth += graph_restricted_coalitions_to_worth[graph_restricted_coalition]
    #         coalition_to_worth.update({coalition: worth})
    #     return coalition_to_worth

    def calculate_worth_of_grand_coalition(self, nx_graph: nx.classes.graph.Graph) -> float:
        """Calculate payoff of the game, i.e. the payoff of all players / the
        grand coalition.

        Args:
            coalition_function (Callable): The coalition function associating a
                coalition with a payoff.
            nx_graph (nx.classes.graph.Graph): Coalition structure of the game
                as a graph.

        Returns:
            float: Payoff of the game / worth of grand coalition.
        """
        return self.coalition_function(self.grand_coalition, nx_graph)

    # def restrict(self, coalition: tuple, nx_graph: nx.classes.graph.Graph) -> list[tuple]:
    #     """Restricts a graph through a (sub)set of nodes / players. Generate a
    #     list of graph restricted coalitions, i. e. a list of node indices of
    #     connected nodes in the subgraph.

    #     Args:
    #         coalition (tuple): Nodes that remain in the graph.
    #         nx_graph (nx.classes.graph.Graph): Graph from which to generate
    #             subgraphs.

    #     Returns:
    #         list[tuple]: Graph restricted coalitions as tuples of node indices.
    #     """
    #     remove_nodes = set(nx_graph.nodes)-set(coalition)
    #     if remove_nodes == set(nx_graph.nodes):
    #         return [()] # empty_graph 
    #     else:
    #         G_reduced = nx_graph.copy()
    #         G_reduced.remove_nodes_from(remove_nodes)
    #         result = [tuple(comp) for comp in nx.connected_components(G_reduced)]
    #         return result

    # def subgraph_from_coalition(self, graph_restricted_coalition: tuple,
    #                            nx_graph: nx.classes.graph.Graph) -> nx.classes.graph.Graph:
    #     """Generates a subgraph from a graph restricted coalition (a subset of
    #     nodes / players) and a graph.

    #     Args:
    #         graph_restricted_coalition (tuple): Nodes / players which form the
    #             subgraph.
    #         nx_graph (nx.classes.graph.Graph): Subgraph induced in this graph by
    #             the nodes_set.

    #     Returns:
    #         nx.classes.graph.Graph: The new subgraph.
    #     """
    #     if len(graph_restricted_coalition) == 0:
    #         return nx.Graph()
    #     else:
    #         return nx_graph.subgraph(graph_restricted_coalition)

    def calculate_single_shapley_value(self, node: int, grand_coalition: tuple,
                                  coalitions: list[tuple], coalition_to_worth: dict) -> float:
        """Calculate a single Shapley value.

        Args:
            node (int): Node index for which to calculate the Myerson value.
            grand_coalition (tuple): Set of all players.
            coalitions (list[tuple]): List of all coalitions.
            coalition_to_worth (dict): Mapping of every coalition to its worth.

        Returns:
            float: Myerson value.
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
            dict: Mapping of each node index to the Myerson value.
        """
        self.calculate_all_mappings()
        self.log.info(f"Calculating Shapley values.")
        sh_values = {}
        for node in tqdm(self.grand_coalition, desc="Calculating Myerson values.", disable=self.disable_tqdm):
            sh_val = self.calculate_single_shapley_value(node, self.grand_coalition,
                                                  self.coalitions, self.coalitions_to_worth)
            sh_values.update({node: sh_val})
        log_string = "".join([f"\t{k}: {v}\n" for k, v in sh_values.items()])
        self.log.info(f"Shapley Values:\n{log_string}")
        return sh_values

    def calculate_all_mappings(self) -> None:
        """Calculates all coalitions, graph restricted coalitions, and their
        associated worths as class attributes:

            * `self.coalitions` (list[tuple])
            * `self.graph_restricted_coalitions` (set[tuple])
            * `self.coalitions_to_graph_restricted_coalitions` (dict)
            * `self.graph_restricted_coalitions_to_worth` (dict)
            * `self.coalitions_to_worth` (dict)
        """
        self.coalitions = self.calculate_coalitions(self.grand_coalition)

        # self.graph_restricted_coalitions, self.coalitions_to_graph_restricted_coalitions \
        #     = self.calculate_graph_restricted_coalitions(self.coalitions, self.nx_graph)

        # self.graph_restricted_coalitions_to_worth \
        #     = self.calculate_worth_of_graph_restricted_coalitions(self.graph_restricted_coalitions)

        # self.coalitions_to_worth \
        #     = self.map_coalition_to_worth(self.coalitions, 
        #                                   self.coalitions_to_graph_restricted_coalitions,
        #                                   self.graph_restricted_coalitions_to_worth)

        self.coalitions_to_worth \
            = self.calculate_worth_of_coalitions(self.coalitions)