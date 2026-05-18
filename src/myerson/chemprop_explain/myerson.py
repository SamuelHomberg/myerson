import logging
import networkx as nx
import numpy as np

try:
    import chemprop
except ImportError:
    raise ImportError("Failed to import chemprop. MPNN explanations not available.")
import torch
from tqdm import tqdm

from chemprop.models.model import MPNN
from chemprop.data.molgraph import MolGraph
from chemprop.data.collate import BatchMolGraph

from myerson import MyersonCalculator, MyersonSampler
from myerson.chemprop_explain.utils import to_networkx


class MyersonExplainer(MyersonCalculator):
    r"""Explains the prediction of a chemprop MPNN with Myerson values.
        The MPNN is treated as the coalition function of a game and its prediction
        as the payoff of the game. The Myerson values show how much each node of 
        the graph contributed to the final prediction.

    Args:
        molgraph (MolGraph): The chemprop MolGraph instance that is to be explained.
        coalition_function (MPNN): The message passing neural network.
        disable_tqdm (bool, optional): Disables progress bar. Defaults to True.
    """

    def __init__(self, 
                molgraph: MolGraph,
                coalition_function: MPNN,
                disable_tqdm: bool=True) -> None:
        """Instantiate the class.
        """

        self.disable_tqdm = disable_tqdm
        self.log = logging.getLogger("MyersonExplainer")

        self.molgraph = molgraph
        self.coalition_function = coalition_function

        self.nx_graph = to_networkx(molgraph)
        self.grand_coalition = list(self.nx_graph.nodes()) # alias: set of players / set of nodes / F
        cc = nx.number_connected_components(self.nx_graph)
        if cc > 1:
            self.log.warning(f"Your graph has {cc} individual components. The worth"
                        " of the grand coalition and the prediction of a GNN can"
                        " differ.")
            pred = self.calculate_prediction()
            worth = self.calculate_worth_of_grand_coalition()
            self.log.warning(f"Prediction={pred:.4f}, Worth={worth:.4f}")

    def calculate_worth_of_single_graph_restricted_coalition(self,
        graph_restricted_coalition: tuple,
        molgraph: MolGraph) -> float:
        """Calculate the worth of a graph restricted coalition, i. e. a single
        connected component.

        Args:
            graph_restricted_coalition (tuple): Graph restricted coalition as
                node indices.
            molgraph (MolGraph): Graph from which a subgraph
                of the connected components will be extracted according to the
                graph restricted coalition.

        Returns:
            float: Worth, the output of the coalition function for the connected
            subgraph. 
        """
        if graph_restricted_coalition == ():
            return 0.
        subgraph = self.subgraph_from_coalition(graph_restricted_coalition, molgraph)
        subgraph = BatchMolGraph([subgraph])
        subgraph.to(self.coalition_function.device)
        out = self.coalition_function(subgraph)
        return out.cpu().item()

    def calculate_worth_of_graph_restricted_coalitions(self,
        graph_restricted_coalitions: list) -> dict:
        """Calculate the worth of every graph restricted coalition and map it to
        its worth. 

        Args:
            graph_restricted_coalitions (list): Set of connected components as
                tuples of node indices.

        Returns:
            dict: Dictionary mapping each connected component to its worth.
        """
        self.log.info(f"Calculating worth of graph restricted coalitions.")
        graph_restricted_coalitions_to_worth = {}
        for coalition in tqdm(graph_restricted_coalitions,
                            desc="Calculating worth of graph restricted coalitions",
                            disable=self.disable_tqdm):
            worth = self.calculate_worth_of_single_graph_restricted_coalition(coalition,
                                                                            self.molgraph)
            graph_restricted_coalitions_to_worth.update({coalition: worth})
        return graph_restricted_coalitions_to_worth

    def calculate_worth_of_grand_coalition(self) -> float:
        """Calculate payoff of the game, i.e. the model prediction. Note that a
        disconnected graph (> 2 molecules) can lead to differeces between 
        the model prediction and this function. 

        Args:
            coalition_function (Callable): The coalition function associating a
                coalition with a payoff.
            nx_graph (nx.classes.graph.Graph): Coalition structure of the game
                as a graph.

        Returns:
            float: Payoff of the game / worth of grand coalition.
        """
        restricted_grand_coalition = self.restrict(self.grand_coalition, self.nx_graph)
        worth = sum([self.calculate_worth_of_single_graph_restricted_coalition(S, self.molgraph) \
                    for S in restricted_grand_coalition])
        return worth 

    def calculate_prediction(self) -> float:
        """Calculate the prediction of the GNN for the investigated graph. When 
        the graph is disconnected this prediction may differ from the worth 
        of the grand coalition.

        Returns:
            float: Prediction.
        """
        input = BatchMolGraph([self.molgraph])
        input.to(self.coalition_function.device)
        return self.coalition_function(input).cpu().item()

    def subgraph_from_coalition(self, graph_restricted_coalition: tuple, 
                                molgraph: MolGraph) -> MolGraph:
        """Generates a subgraph from a graph restricted coalition (a subset of
        nodes / players) and a graph.

        Args:
            nodes (tuple): Nodes which form the subgraph.
            molgraph (MolGraph): Subgraph induced in this graph by the subset of
                nodes.

        Returns:
            MolGraph: The new subgraph.
        """
        # unsorted nodes can result in the wrong edges
        nodes = sorted(graph_restricted_coalition)
        nodes = np.array(nodes, dtype=np.int32)
        node_mask = np.zeros(molgraph.V.shape[0], dtype=bool)
        node_mask[nodes] = True
        V = molgraph.V[node_mask]

        edge_mask = node_mask[molgraph.edge_index[0]] & node_mask[molgraph.edge_index[1]]
        edge_index = molgraph.edge_index[:, edge_mask]
        # fancy indexing to relabel edge_index, rev_edge_index
        node_idx = np.zeros(node_mask.size, dtype=np.int32)
        node_idx[nodes] = np.arange(node_mask.sum())
        edge_index = node_idx[edge_index]

        E = molgraph.E[edge_mask]

        edge_idx_map = np.full(molgraph.edge_index.shape[1], -1, dtype=np.int32)
        edge_idx_map[edge_mask] = np.arange(edge_mask.sum())
        edge_idx_map
        rev_edge_index_masked = molgraph.rev_edge_index[edge_mask]
        rev_edge_index = edge_idx_map[rev_edge_index_masked]

        subgraph = MolGraph(V=V, E=E, edge_index=edge_index, rev_edge_index=rev_edge_index)
        return subgraph

class MyersonSamplingExplainer(MyersonSampler, MyersonExplainer):
    """A class explaining GNN predictions with approximated Myerson values.

    Args:
        molgraph (MolGraph): The chemprop MolGraph instance that is to be explained.
        coalition_function (MPNN): The message passing neural network.
        seed (None | int, optional): Seed for randomness. Defaults to None.
        number_of_samples (int, optional): Number of sampling steps. Defaults to 1000.
        disable_tqdm (bool, optional): Disables progress bar. Defaults to True.
    """
    def __init__(self,
                molgraph: MolGraph,
                coalition_function: MPNN,
                seed: None | int = None, 
                number_of_samples: int = 1000,
                disable_tqdm: bool=True) -> None:
        """Instantiates the class.
        """
        self.disable_tqdm = disable_tqdm
        self.log = logging.getLogger("MyersonSamplingExplainer")

        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.number_of_samples = number_of_samples

        self.molgraph = molgraph
        self.coalition_function = coalition_function

        self.nx_graph = to_networkx(molgraph)
        self.grand_coalition = list(self.nx_graph.nodes()) # alias: set of players / set of nodes / F
        cc = nx.number_connected_components(self.nx_graph)
        if cc > 1:
            self.log.warning(f"Your graph has {cc} individual components. The worth"
                        " of the grand coalition and the prediction of a GNN can"
                        " differ.")
            pred = self.calculate_prediction()
            worth = self.calculate_worth_of_grand_coalition()
            self.log.warning(f"Prediction={pred:.4f}, Worth={worth:.4f}")


class MyersonClassExplainer(MyersonExplainer):
    r"""Explains the prediction of a graph neural network (GNN) classifier with
        Myerson values.  The GNN is treated as the coalition function of a game
        and its prediction as the payoff of the game. The Myerson values show
        how much each node of the graph contributed to the final prediction.

    Args:
        molgraph (MolGraph): The chemprop MolGraph instance that is to be explained.
        coalition_function (MPNN): The message passing neural network.
        disable_tqdm (bool, optional): Disables progress bar. Defaults to True.
    """

    def __init__(self, 
                molgraph: MolGraph,
                coalition_function: MPNN,
                disable_tqdm: bool=True) -> None:
        """Instantiate the class.
        """

        self.disable_tqdm = disable_tqdm
        self.log = logging.getLogger("MyersonClassExplainer")

        self.molgraph = molgraph
        self.coalition_function = coalition_function

        self.nx_graph = to_networkx(molgraph)
        self.grand_coalition = list(self.nx_graph.nodes()) # alias: set of players / set of nodes / F
        self.pred = self.calculate_prediction()
        cc = nx.number_connected_components(self.nx_graph)
        if cc > 1:
            self.log.warning(f"Your graph has {cc} individual components. The worth"
                        " of the grand coalition and the prediction of a GNN can"
                        " differ.")
            pred = self.calculate_prediction()
            worth = self.calculate_worth_of_grand_coalition()
            self.log.warning(f"Prediction={pred}, Worth={worth}")
    
    def calculate_worth_of_single_graph_restricted_coalition(self, 
        graph_restricted_coalition: tuple,
        molgraph: MolGraph) -> torch.Tensor:
        """Calculate the worth of a graph restricted coalition, i. e. a single
        connected component.

        Args:
            graph_restricted_coalition (tuple): Graph restricted coalition as
                node indices.
            molgraph (MolGraph): Graph from which a subgraph
                of the connected components will be extracted according to the
                graph restricted coalition.

        Returns:
            tensor: Worth, the output of the coalition function for the connected
            subgraph. 
        """
        if graph_restricted_coalition == ():
            return torch.zeros(self.pred.shape)
        subgraph = self.subgraph_from_coalition(graph_restricted_coalition, molgraph)
        subgraph = BatchMolGraph([subgraph])
        subgraph.to(self.coalition_function.device)
        out = self.coalition_function(subgraph)
        
        return out.detach().cpu().squeeze(0)

    def calculate_prediction(self) -> torch.Tensor:
        """Calculate the prediction of the GNN for the investigated graph. When 
        the graph is disconnected this prediction may differ from the worth 
        of the grand coalition.

        Returns:
            torch.Tensor: Prediction.
        """
        input = BatchMolGraph([self.molgraph])
        input.to(self.coalition_function.device)
        return self.coalition_function(input).cpu().squeeze(0)


class MyersonSamplingClassExplainer(MyersonSamplingExplainer, MyersonClassExplainer):
    """A class explaining a GNNs classifier predictions with approximated Myerson values.

    Args:
        molgraph (MolGraph): The chemprop MolGraph instance that is to be explained.
        coalition_function (MPNN): The message passing neural network.
        seed (None | int, optional): Seed for randomness. Defaults to None.
        number_of_samples (int, optional): Number of sampling steps. Defaults to 1000.
        disable_tqdm (bool, optional): Disables progress bar. Defaults to True.
    """
    def __init__(self,
                molgraph: MolGraph,
                coalition_function: MPNN,
                seed: None | int = None, 
                number_of_samples: int = 1000,
                disable_tqdm: bool=True) -> None:
        """Instantiates the class.
        """
        self.disable_tqdm = disable_tqdm
        self.log = logging.getLogger("MyersonSamplingClassExplainer")

        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.number_of_samples = number_of_samples

        self.molgraph = molgraph
        self.coalition_function = coalition_function

        self.nx_graph = to_networkx(molgraph)
        self.grand_coalition = list(self.nx_graph.nodes()) # alias: set of players / set of nodes / F
        self.pred = self.calculate_prediction()
        cc = nx.number_connected_components(self.nx_graph)
        if cc > 1:
            self.log.warning(f"Your graph has {cc} individual components. The worth"
                        " of the grand coalition and the prediction of a GNN can"
                        " differ.")
            pred = self.calculate_prediction()
            worth = self.calculate_worth_of_grand_coalition()
            self.log.warning(f"Prediction={pred}, Worth={worth}")

    def map_coalition_to_worth(self, coalitions: list[tuple], 
                       coalitions_to_graph_restricted_coalitions: dict,
                       graph_restricted_coalitions_to_worth: dict) -> dict:
        """Map every coalition to its worth.

        Args:
            coalitions (list): List of all coalitions (2^{num_nodes}). 
            coalitions_to_graph_restricted_coalitions (dict): Dictionary mapping
                the coalitions to the corresponding graph restricted coalitions.
            graph_restricted_coalitions_to_worth (dict): Dictionary mapping the 
                graph restricted coalitions to their worth.

        Returns:
            dict: Dictionary mapping each coalition to its worth.
        """
        self.log.info(f"Mapping coalitions to worth.")
        coalition_to_worth = {}
        for coalition in tqdm(coalitions, desc="Mapping coalitions to worth", disable=self.disable_tqdm):
            worth = torch.zeros(self.pred.shape)
            for graph_restricted_coalition in coalitions_to_graph_restricted_coalitions[coalition]:
                worth += graph_restricted_coalitions_to_worth[graph_restricted_coalition]
            coalition_to_worth.update({coalition: worth})
        return coalition_to_worth
    
    def sample_all_myerson_values(self) -> np.ndarray:
        """Use Monte Carlo sampling to approximate the Myerson values for every
        node / player in the graph.

        Returns:
            np.ndarray: Sampled Myerson values.
        """
        self.sample_all_mappings()
        pred = self.calculate_prediction()
        nodes_array = np.array(self.grand_coalition)
        my_values = np.zeros((len(nodes_array), pred.shape[0]), dtype=float)
        self.log.info(f"Calculating sampled Myerson values.")
        for permutation in tqdm(self.permutations_without_random_node,
                              disable=self.disable_tqdm,
                              desc="Calculate sampled Myerson values"):
            for node_idx, node in enumerate(nodes_array):

                sampled_permutation_with_current_swapped_in_random_node = permutation.copy()
                sampled_permutation_with_current_swapped_in_random_node \
                    = self._replace_in_array(sampled_permutation_with_current_swapped_in_random_node,
                                             node,
                                             self.random_node)

                worth_with_node = self.coalitions_to_worth[tuple(np.sort(np.append(sampled_permutation_with_current_swapped_in_random_node, node)))]
                worth_without_node = self.coalitions_to_worth[tuple(np.sort(sampled_permutation_with_current_swapped_in_random_node))]
                my_values[node_idx] = (my_values[node_idx] + worth_with_node.numpy().squeeze() - worth_without_node.numpy().squeeze())

        my_values = my_values / self.number_of_samples
        log_string = "".join([f"\t{node}: {val}\n" for node, val in zip(self.grand_coalition, my_values)])
        self.log.info(f"Sampled Myerson Values:\n{log_string}")
        return my_values

def explain(molgraph: MolGraph,
            model: MPNN,
            sample_if_more_nodes_than: int=20,
            verbose: bool=False) -> dict:
    """A function to quickly get started with explaining GNN predictions using Myerson values.

    Args:
        molgraph (MolGraph): The chemprop MolGraph instance.
        model (MPNN): The message passing neural network.
        sample_if_more_nodes_than (int, optional): Barrier for when to start
            sampling instead of exact calculations. Defaults to 20.
        verbose (bool, optional): Whether to log information to the output and
            show progress bars. Defaults to False.

    Returns:
        dict: The (sampled) Myerson values.
    """

    if verbose:
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s - %(levelname)s] %(message)s', force=True)
        disable_tqdm=False
    else:
        disable_tqdm=True

    node_count = molgraph.V.shape[0]
    if node_count > sample_if_more_nodes_than:
        logging.info("Sampling Myerson values.")
        sampler = MyersonSamplingExplainer(molgraph, model, disable_tqdm=disable_tqdm)
        return sampler.sample_all_myerson_values()
    else:
        logging.info("Calculating exact Myerson values.")
        explainer = MyersonExplainer(molgraph, model, disable_tqdm=disable_tqdm)
        return explainer.calculate_all_myerson_values()
