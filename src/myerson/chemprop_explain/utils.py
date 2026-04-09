import networkx as nx 
import numpy as np
import torch

from chemprop.data.molgraph import MolGraph
from chemprop.data.collate import BatchMolGraph

def to_networkx(mg: MolGraph) -> nx.classes.graph.Graph:
    """
    Converts a Chemprop MolGraph into an undirected NetworkX Graph.

    Args:
        mg (MolGraph): The Chemprop MolGraph containing atom (V) and
            bond (edge_index) data.

    Returns:
        nx.Graph: An undirected NetworkX Graph.
    """
    G = nx.Graph()
    for node in range(mg.V.shape[0]):
        G.add_node(node)
    for v, w in mg.edge_index.transpose().tolist():
        G.add_edge(v, w)
    return G

def unbatch(bmg: BatchMolGraph) -> list[MolGraph]:
    """
    Unbatches a BatchMolGraph into a list of individual MolGraph objects.

    This function extracts the individual molecular graphs from a batched
    representation by isolating the nodes and edges for each molecule and
    remapping the global batch-level edge indices back to local atom indices.

    Args:
        bmg (BatchMolGraph): The batched molecular graph to be unbatched.

    Returns:
        list[MolGraph]: A list of individual Chemprop MolGraph objects.
    """
    molgraphs = []
    for mol_idx in range(torch.max(bmg.batch) + 1):
        atom_mask = (bmg.batch == mol_idx)

        V = bmg.V[atom_mask].cpu().numpy()
        edge_mask = atom_mask[bmg.edge_index[0]] & atom_mask[bmg.edge_index[1]]
        E = bmg.E[edge_mask].cpu().numpy()
        if edge_mask.any():
            # Remap atom indices for edge_index
            global_to_local_atom = torch.zeros_like(bmg.batch)
            global_to_local_atom[atom_mask] = torch.arange(atom_mask.sum(), device=bmg.batch.device)
            
            edge_index = global_to_local_atom[bmg.edge_index[:, edge_mask]]
            edge_index = edge_index.cpu().numpy()
            
            # Remap edge indices for rev_edge_index
            global_to_local_edge = torch.zeros(bmg.edge_index.shape[1], dtype=torch.long, device=bmg.batch.device)
            global_to_local_edge[edge_mask] = torch.arange(edge_mask.sum(), device=bmg.batch.device)
            
            rev_edge_index = global_to_local_edge[bmg.rev_edge_index[edge_mask]]
            rev_edge_index = rev_edge_index.cpu().numpy()
        else:
            # Safeguard for single-atom molecules (no edges)
            edge_index = np.empty((2, 0), dtype=np.int64)
            rev_edge_index = np.empty((0,), dtype=np.int64)
        molgraphs.append(MolGraph(V=V, E=E, edge_index=edge_index, rev_edge_index=rev_edge_index))
    return molgraphs