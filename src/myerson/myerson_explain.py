from myerson import MyersonCalculator, MyersonSampler

import torch
import torch_geometric
import numpy as np

import networkx as nx
from tqdm import tqdm
import logging
# try: 
#     from .myerson import fast_restrict
# except:
#     pass


class MyersonExplainer(MyersonCalculator):
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

        Args:
            graph (torch_geometric.data.Data): _description_
            coalition_function (torch.nn.Module): _description_
            disable_tqdm (bool, optional): _description_. Defaults to True.
        """

        self.disable_tqdm = disable_tqdm
        self.log = logging.getLogger("MyersonExplainer")

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.log.info(f"using device {self.device}")
        self.pyg_graph = graph.to(self.device)
        self.coalition_function = coalition_function
        self.coalition_function.to(self.device)

        self.nx_graph = torch_geometric.utils.to_networkx(graph, to_undirected=True)
        self.grand_coalition = list(self.nx_graph.nodes()) # alias: set of players / set of nodes / F
        cc = nx.number_connected_components(self.nx_graph)
        if cc > 1:
            self.log.warn(f"Your graph has {cc} individual components. The worth"
                        " of the grand coalition and the prediction of a GNN can"
                        " differ.")
            pred = self.calculate_prediction()
            worth = self.calculate_worth_of_grand_coalition()
            self.log.warn(f"Prediction={pred:.4f}, Worth={worth:.4f}")

        # if "myerson.fast_restrict" in reversed(sys.modules):
        #     self.fast_restrict_available = True
        # else:
        #     self.fast_restrict_available = False
        # self.set_restrict(self.fast_restrict_available)
    
    # def set_restrict(self, use_fast_restrict: bool) -> None:
    #     """Set wheter to use the fast C++ implementation or the networkX
    #     implementation of `restrict(...)`.

    #     Args:
    #         use_fast_restrict (bool): True or False.
    #     """
    #     if use_fast_restrict:
    #         self._set_variables_for_fast_restrict()
    #         self.log.info("Using fast_restrict() from external C++ library.")
    #         self.restrict = self.fast_restrict
    #     else:
    #         self.log.info("Using python only slow `restrict()` (networkx package).")
    #         self.restrict = super().restrict

    def calculate_worth_of_single_graph_restricted_coalition(self,
        graph_restricted_coalition: tuple,
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
        if graph_restricted_coalition == ():
            return 0.
        subgraph = self.subgraph_from_coalition(graph_restricted_coalition, pyg_graph)
        out = self.coalition_function(subgraph.x, subgraph.edge_index, self._batch_var(subgraph))
        
        return out.cpu().item()

    def calculate_worth_of_graph_restricted_coalitions(self,
        graph_restricted_coalitions: set) -> dict:
        """Calculate the worth of every graph restricted coalition and map it to
        its worth. 

        Args:
            graph_restricted_coalitions (set): Set of connected components as
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
                                                                            self.pyg_graph)
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
        worth = sum([self.calculate_worth_of_single_graph_restricted_coalition(S, self.pyg_graph) \
                    for S in restricted_grand_coalition])
        return worth 

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

    # def fast_restrict(self, coalition: tuple, nx_graph: nx.classes.graph.Graph) -> list[tuple]:
    #     """Restricts a graph through a (sub)set of nodes / players. Generate a
    #     list of graph restricted coalitions, i. e. a list of node indices of
    #     connected nodes in the subgraph. Uses python wrapped C++ code for 
    #     efficiency.

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
    #     component_map = cpp_graph_divide.get_connected_components(self.num_nodes, # type: ignore
    #                                                 self.num_edges,
    #                                                 self.edge_from_ptr,
    #                                                 self.edge_to_ptr,
    #                                                 list(remove_nodes))
    #     connected_subgraph_nodes = {}
    #     seen_component = []
    #     for node, component in enumerate(component_map): 
    #         if (component not in seen_component) and (node not in remove_nodes):
    #             connected_subgraph_nodes.update({component: [node]})
    #             seen_component.append(component)
    #         else:
    #             if node not in remove_nodes:
    #                 connected_subgraph_nodes.update({component: connected_subgraph_nodes[component]+[node]})
    #     return [tuple(connected_subgraph_nodes[key]) for key in connected_subgraph_nodes.keys()]

    def subgraph_from_coalition(self, graph_restricted_coalition: tuple, 
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
        nodes = sorted(graph_restricted_coalition)
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

    # def _set_variables_for_fast_restrict(self) -> None:
    #     """Set class variables for "fast_restrict" function (pointers passed to C++)
    #     """
    #     self.num_nodes = self.pyg_graph.num_nodes
    #     self.edge_from = self.pyg_graph.edge_index.cpu().numpy()[0]
    #     self.edge_to = self.pyg_graph.edge_index.cpu().numpy()[1]
    #     self.edge_from_ptr = self.edge_from.__array_interface__['data'][0]
    #     self.edge_to_ptr = self.edge_to.__array_interface__['data'][0]
    #     self.num_edges = len(self.edge_from)

class MyersonSamplingExplainer(MyersonSampler, MyersonExplainer):
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
        self.log = logging.getLogger("MyersonSamplingExplainer")

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
        cc = nx.number_connected_components(self.nx_graph)
        if cc > 1:
            self.log.warn(f"Your graph has {cc} individual components. The worth"
                        " of the grand coalition and the prediction of a GNN can"
                        " differ.")
            pred = self.calculate_prediction()
            worth = self.calculate_worth_of_grand_coalition()
            self.log.warn(f"Prediction={pred:.4f}, Worth={worth:.4f}")

        # if "myerson.cpp_graph_divide" in sys.modules:
        #     self.fast_restrict_available = True
        # else:
        #     self.fast_restrict_available = False
        # self.set_restrict(self.fast_restrict_available)


def explain(graph: torch_geometric.data.Data,
            model: torch.nn.Module,
            sample_if_more_nodes_than: int=20,
            verbose: bool=False) -> dict:
    """A function to quickly get started with explaining GNN predictions using Myerson values.

    Args:
        graph (torch_geometric.data.Data): The graph.
        model (torch.nn.Module): The graph neural network.
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

    node_count = graph.x.size()[0]
    if node_count > sample_if_more_nodes_than:
        logging.info("Sampling Myerson values.")
        sampler = MyersonSamplingExplainer(graph, model, disable_tqdm=disable_tqdm)
        return sampler.sample_all_myerson_values()
    else:
        logging.info("Calculating exact Myerson values.")
        explainer = MyersonExplainer(graph, model, disable_tqdm=disable_tqdm)
        return explainer.calculate_all_myerson_values()

# TODO: multi class classification