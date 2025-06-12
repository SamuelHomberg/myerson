from myerson import ShapleyCalculator, ShapleySampler

import torch
import torch_geometric
import numpy as np

import networkx as nx
from tqdm import tqdm
import logging


class ShapleyExplainer(ShapleyCalculator):
    r"""Explains the prediction of a graph neural network (GNN) with Myerson values.
        The GNN is treated as the coalition function of a game and its prediction
        as the payoff of the game. The Myerson values show how much each node of 
        the graph contributed to the final prediction.

    Args:
        graph (torch_geometric.data.Data): The data instance that is to be explained.
        coalition_function (torch.nn.Module): The GNN.
        disable_tqdm (bool, optional): Disables progress bar. Defaults to True.
    """

    def __init__(self, 
                graph: torch_geometric.data.Data,
                coalition_function: torch.nn.Module,
                disable_tqdm: bool=True) -> None:
        """Instantiate the class.
        """

        self.disable_tqdm = disable_tqdm
        self.log = logging.getLogger("ShapleyExplainer")

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.log.info(f"using device {self.device}")
        self.pyg_graph = graph.to(self.device)
        self.coalition_function = coalition_function
        self.coalition_function.to(self.device)

        self.nx_graph = torch_geometric.utils.to_networkx(graph, to_undirected=True)
        self.grand_coalition = list(self.nx_graph.nodes()) # alias: set of players / set of nodes / F
        # cc = nx.number_connected_components(self.nx_graph)
        # if cc > 1:
        #     self.log.warn(f"Your graph has {cc} individual components. The worth"
        #                 " of the grand coalition and the prediction of a GNN can"
        #                 " differ.")
        #     pred = self.calculate_prediction()
        #     worth = self.calculate_worth_of_grand_coalition()
        #     self.log.warn(f"Prediction={pred:.4f}, Worth={worth:.4f}")

    def calculate_worth_of_single_coalition(self,
        coalition: tuple,
        pyg_graph: torch_geometric.data.Data) -> float:
        """Calculate the worth of a graph restricted coalition, i. e. a single
        connected component.

        Args:
            graph_restricted_coalition (tuple): Graph restricted coalition as
                node indices.
            pyg_graph (torch_geometric.data.Data): Graph from which a subgraph
                of the connected components will be extracted according to the
                graph restricted coalition.

        Returns:
            float: Worth, the output of the coalition function for the connected
            subgraph. 
        """
        if coalition == ():
            return 0.
        subgraph = self.subgraph_from_coalition(coalition, pyg_graph)
        out = self.coalition_function(subgraph.x, subgraph.edge_index, self._batch_var(subgraph))
        
        return out.cpu().item()

    def calculate_worth_of_coalitions(self,
        coalitions: set) -> dict:
        """Calculate the worth of every graph restricted coalition and map it to
        its worth. 

        Args:
            graph_restricted_coalitions (set): Set of connected components as
                tuples of node indices.

        Returns:
            dict: Dictionary mapping each connected component to its worth.
        """
        self.log.info(f"Calculating worth of coalitions.")
        coalitions_to_worth = {}
        for coalition in tqdm(coalitions,
                            desc="Calculating worth of coalitions",
                            disable=self.disable_tqdm):
            worth = self.calculate_worth_of_single_coalition(coalition, self.pyg_graph)
            coalitions_to_worth.update({coalition: worth})
        return coalitions_to_worth

    def calculate_prediction(self) -> float:
        """Calculate the prediction of the GNN for the investigated graph. When 
        the graph is disconnected this prediction may differ from the worth 
        of the grand coalition.

        Returns:
            float: Prediction.
        """
        return self.coalition_function(self.pyg_graph.x, self.pyg_graph.edge_index,
                                    self._batch_var(self.pyg_graph)).cpu().item()

    def _batch_var(self, pyg_graph: torch_geometric.data.Data) -> torch.tensor:
        """Return a batch argument for single graphs, required for models 
        trained in batches.

        Args:
            pyg_graph (torch_geometric.data.Data): Graph for which to generate
                batch.

        Returns:
            torch.tensor: Batch attribute in the correct dimensions.
        """
        return torch.zeros(pyg_graph.x.shape[0], dtype=int, device=pyg_graph.x.device)

    def subgraph_from_coalition(self, coalition: tuple, 
                                pyg_graph: torch_geometric.data.Data) -> torch_geometric.data.Data:
        """Generates a subgraph from a graph restricted coalition (a subset of
        nodes / players) and a graph.

        Args:
            nodes (tuple): Nodes which form the subgraph.
            pyg_graph (torch_geometric.data.Data): Subgraph induced in this
                graph by the subset of nodes.

        Returns:
            torch_geometric.data.Data: The new subgraph.
        """
        # unsorted nodes can result in the wrong edges
        nodes = sorted(coalition)
        nodes = torch.tensor(nodes, dtype=torch.long, device=pyg_graph.x.device)
        node_mask = torch.zeros(pyg_graph.x.shape[0], dtype=torch.bool, device=pyg_graph.x.device)
        node_mask[nodes] = True
        x = pyg_graph.x[node_mask]

        edge_mask = node_mask[pyg_graph.edge_index[0]] & node_mask[pyg_graph.edge_index[1]]
        edge_index = pyg_graph.edge_index[:, edge_mask]
        # fancy indexing to relabel edge_index
        node_idx = torch.zeros(node_mask.size(0), dtype=torch.long, device=pyg_graph.x.device)
        node_idx[nodes] = torch.arange(node_mask.sum().item(), device=pyg_graph.x.device)
        edge_index = node_idx[edge_index]

        subgraph = torch_geometric.data.Data(x=x, edge_index=edge_index)
        return subgraph

class ShapleySamplingExplainer(ShapleySampler, ShapleyExplainer):
    """A class explaining GNN predictions with approximated Myerson values.

    Args:
        graph (torch_geometric.data.Data): The data instance that is to be explained.
        coalition_function (torch.nn.Module): The GNN.
        seed (None | int, optional): Seed for randomness. Defaults to None.
        number_of_samples (int, optional): Number of sampling steps. Defaults to 1000.
        disable_tqdm (bool, optional): Disables progress bar. Defaults to True.
    """
    def __init__(self,
                graph: torch_geometric.data.Data,
                coalition_function: torch.nn.Module,
                seed: None | int = None, 
                number_of_samples: int = 1000,
                disable_tqdm: bool=True) -> None:
        """Instantiates the class.
        """
        self.disable_tqdm = disable_tqdm
        self.log = logging.getLogger("ShapleySamplingExplainer")

        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.number_of_samples = number_of_samples

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.log.info(f"using device {self.device}")
        self.pyg_graph = graph.to(self.device)
        self.coalition_function = coalition_function
        self.coalition_function.to(self.device)

        self.nx_graph = torch_geometric.utils.to_networkx(graph, to_undirected=True)
        self.grand_coalition = list(self.nx_graph.nodes()) # alias: set of players / set of nodes / F
        # cc = nx.number_connected_components(self.nx_graph)
        # if cc > 1:
        #     self.log.warn(f"Your graph has {cc} individual components. The worth"
        #                 " of the grand coalition and the prediction of a GNN can"
        #                 " differ.")
        #     pred = self.calculate_prediction()
        #     worth = self.calculate_worth_of_grand_coalition()
        #     self.log.warn(f"Prediction={pred:.4f}, Worth={worth:.4f}")

        # # if "myerson.cpp_graph_divide" in sys.modules:
        #     self.fast_restrict_available = True
        # else:
        #     self.fast_restrict_available = False
        # self.set_restrict(self.fast_restrict_available)

class ShapleyClassExplainer(ShapleyExplainer):
    r"""Explains the prediction of a graph neural network (GNN) classifier with
        Myerson values.  The GNN is treated as the coalition function of a game
        and its prediction as the payoff of the game. The Myerson values show
        how much each node of the graph contributed to the final prediction.

    Args:
        graph (torch_geometric.data.Data): The data instance that is to be explained.
        coalition_function (torch.nn.Module): The GNN.
        disable_tqdm (bool, optional): Disables progress bar. Defaults to True.
    """

    def __init__(self, 
                graph: torch_geometric.data.Data,
                coalition_function: torch.nn.Module,
                disable_tqdm: bool=True) -> None:
        """Instantiate the class.
        """

        self.disable_tqdm = disable_tqdm
        self.log = logging.getLogger("ShapleyClassExplainer")

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.log.info(f"using device {self.device}")
        self.pyg_graph = graph.to(self.device)
        self.coalition_function = coalition_function
        self.coalition_function.to(self.device)

        self.nx_graph = torch_geometric.utils.to_networkx(graph, to_undirected=True)
        self.grand_coalition = list(self.nx_graph.nodes()) # alias: set of players / set of nodes / F
        self.pred = self.calculate_prediction()
        # cc = nx.number_connected_components(self.nx_graph)
        # if cc > 1:
        #     self.log.warn(f"Your graph has {cc} individual components. The worth"
        #                 " of the grand coalition and the prediction of a GNN can"
        #                 " differ.")
        #     pred = self.calculate_prediction()
        #     worth = self.calculate_worth_of_grand_coalition()
        #     self.log.warn(f"Prediction={pred}, Worth={worth}")

    def calculate_worth_of_single_coalition(self, 
        coalition: tuple,
        pyg_graph: torch_geometric.data.Data) -> torch.tensor:
        """Calculate the worth of a graph restricted coalition, i. e. a single
        connected component.

        Args:
            graph_restricted_coalition (tuple): Graph restricted coalition as
                node indices.
            pyg_graph (torch_geometric.data.Data): Graph from which a subgraph
                of the connected components will be extracted according to the
                graph restricted coalition.

        Returns:
            tensor: Worth, the output of the coalition function for the connected
            subgraph. 
        """
        if coalition == ():
            return torch.zeros(self.pred.shape)
        subgraph = self.subgraph_from_coalition(coalition, pyg_graph)
        out = self.coalition_function(subgraph.x, subgraph.edge_index, self._batch_var(subgraph))
        
        return out.detach().cpu()

    def calculate_prediction(self) -> torch.tensor:
        """Calculate the prediction of the GNN for the investigated graph. When 
        the graph is disconnected this prediction may differ from the worth 
        of the grand coalition.

        Returns:
            float: Prediction.
        """
        return self.coalition_function(self.pyg_graph.x, self.pyg_graph.edge_index,
                                    self._batch_var(self.pyg_graph)).cpu()


class ShapleySamplingClassExplainer(ShapleySamplingExplainer, ShapleyClassExplainer):
    """A class explaining a GNNs classifier predictions with approximated Myerson values.

    Args:
        graph (torch_geometric.data.Data): The data instance that is to be explained.
        coalition_function (torch.nn.Module): The GNN.
        seed (None | int, optional): Seed for randomness. Defaults to None.
        number_of_samples (int, optional): Number of sampling steps. Defaults to 1000.
        disable_tqdm (bool, optional): Disables progress bar. Defaults to True.
    """
    def __init__(self,
                graph: torch_geometric.data.Data,
                coalition_function: torch.nn.Module,
                seed: None | int = None, 
                number_of_samples: int = 1000,
                disable_tqdm: bool=True) -> None:
        """Instantiates the class.
        """
        self.disable_tqdm = disable_tqdm
        self.log = logging.getLogger("ShapleySamplingClassExplainer")

        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.number_of_samples = number_of_samples

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.log.info(f"using device {self.device}")
        self.pyg_graph = graph.to(self.device)
        self.coalition_function = coalition_function
        self.coalition_function.to(self.device)

        self.nx_graph = torch_geometric.utils.to_networkx(graph, to_undirected=True)
        self.grand_coalition = list(self.nx_graph.nodes()) # alias: set of players / set of nodes / F
        self.pred = self.calculate_prediction()
        # cc = nx.number_connected_components(self.nx_graph)
        # if cc > 1:
        #     self.log.warn(f"Your graph has {cc} individual components. The worth"
        #                 " of the grand coalition and the prediction of a GNN can"
        #                 " differ.")
        #     pred = self.calculate_prediction()
        #     worth = self.calculate_worth_of_grand_coalition()
        #     self.log.warn(f"Prediction={pred}, Worth={worth}")

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
    #         worth = torch.zeros(self.pred.shape)
    #         for graph_restricted_coalition in coalitions_to_graph_restricted_coalitions[coalition]:
    #             worth += graph_restricted_coalitions_to_worth[graph_restricted_coalition]
    #         coalition_to_worth.update({coalition: worth})
    #     return coalition_to_worth
    
    def sample_all_shapley_values(self) -> dict:
        """Use Monte Carlo sampling to approximate the Myerson values for every
        node / player in the graph.

        Returns:
            dict: Mapping of each node index to the sampled Myerson value.
        """
        self.sample_all_mappings()
        pred = self.calculate_prediction()
        nodes_array = np.array(self.grand_coalition)
        sh_values = np.zeros((len(nodes_array), pred.shape[1]), dtype=float)
        self.log.info(f"Calculating sampled Shapley values.")
        for permutation in tqdm(self.permutations_without_random_node,
                              disable=self.disable_tqdm,
                              desc="Calculate sampled Shapley values"):
            for node_idx, node in enumerate(nodes_array):

                sampled_permutation_with_current_swapped_in_random_node = permutation.copy()
                sampled_permutation_with_current_swapped_in_random_node \
                    = self._replace_in_array(sampled_permutation_with_current_swapped_in_random_node,
                                             node,
                                             self.random_node)

                worth_with_node = self.coalitions_to_worth[tuple(np.sort(np.append(sampled_permutation_with_current_swapped_in_random_node, node)))]
                worth_without_node = self.coalitions_to_worth[tuple(np.sort(sampled_permutation_with_current_swapped_in_random_node))]
                sh_values[node_idx] = (sh_values[node_idx] + worth_with_node.numpy().squeeze() - worth_without_node.numpy().squeeze())

        sh_values = sh_values / self.number_of_samples
        sh_values = {i: sh_i for i, sh_i in enumerate(sh_values)}
        log_string = "".join([f"\t{k}: {v}\n" for k, v in sh_values.items()])
        self.log.info(f"Sampled Shapley Values:\n{log_string}")
        return sh_values
