# %%
import math
import pandas as pd
import networkx as nx
from tqdm import tqdm

from rdkit import Chem

from myerson import MyersonCalculator

def rdkit_to_nx(mol):
    """
    Converts an RDKit Molecule object into a NetworkX Graph.
    
    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The RDKit molecule to convert.
    
    Returns:
    nx.Graph: A NetworkX graph with atom and bond attributes.
    """
    G = nx.Graph(mol=mol)
    
    for i, atom in enumerate(mol.GetAtoms()):
        G.add_node(
            atom.GetIdx(),
            numHs=atom.GetTotalNumHs(),
            atomic_num=atom.GetAtomicNum(),
            element=atom.GetSymbol(),
            formal_charge=atom.GetFormalCharge(),
            hybridization=str(atom.GetHybridization()),
            is_aromatic=atom.GetIsAromatic(),
            atom=atom,
        )
        
    for bond in mol.GetBonds():
        G.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond_type=bond.GetBondType(),        # e.g., SINGLE, DOUBLE, AROMATIC
            bond_type_as_double=bond.GetBondTypeAsDouble(),  # e.g., 1.0, 2.0, 1.5
            is_aromatic=bond.GetIsAromatic(),
            bond=bond,
        )
        
    return G

def get_substruct_coalition_func(smarts: str):
    """
    Creates a coalition function that returns 1.0 if the given coalition
    contains all atoms of at least one match for the specified substructure.
    """
    substruct_mol = Chem.MolFromSmarts(smarts)
    if substruct_mol is None:
        raise ValueError(f"Invalid SMARTS string: {smarts}")

    cache_key = f"matches_{smarts}"

    def substruct_coalition_func(coalition, nx_graph) -> float:
        # Cache the matches as sets to avoid recomputing them for every coalition
        if cache_key not in nx_graph.graph:
            mol = nx_graph.graph['mol']
            nx_graph.graph[cache_key] = [set(match) for match in mol.GetSubstructMatches(substruct_mol)]
            
        matches = nx_graph.graph[cache_key]
        coalition_set = set(coalition)
        
        # Check if any of the matches is fully contained within the current coalition
        for match in matches:
            if match.issubset(coalition_set):
                return 1.0
        return 0.0

    return substruct_coalition_func

# %%
failed_smi = []
df = pd.read_csv('/nfs/home/s_homb04/clones/myerson/250k_rndm_zinc_drugs_clean_3.csv')
graphs = []
for smi in tqdm(df['smiles'][:1000]): 
    mol = Chem.MolFromSmiles(smi)
    try:
        G = rdkit_to_nx(mol)
        graphs.append(G)
    except:
        failed_smi.append(smi)

if failed_smi:
    print(f"{len(failed_smi)=}")
    print(failed_smi[:10])

# %% test single example
id = 1
smarts_to_check = "c1cnccc1"
# smarts_to_check = "O" # for id 5
smi = df['smiles'][id]
mol = Chem.MolFromSmiles(smi)
G = rdkit_to_nx(mol)

substruct_func = get_substruct_coalition_func(smarts_to_check)

my_calculator = MyersonCalculator(G, substruct_func)
my_values = my_calculator.calculate_all_myerson_values()

has_substruct = mol.HasSubstructMatch(Chem.MolFromSmarts(smarts_to_check))
print(f"SMARTS: {smarts_to_check} | Has substruct: {has_substruct}")
print(f"{sum(my_values)=:.4f} (Expected={1.0 if has_substruct else 0.0:.4f})")

for my, node in zip(my_values, list(G.nodes(data=True))):
    print(f"{my:>8.4f} Atom {node[0]:>2} ({node[1]['element']})")

"""
SMARTS: c1cnccc1 | Has substruct: True
sum(my_values)=1.0000 (Expected=1.0000)
  0.0000 Atom  0 (C)
  0.0000 Atom  1 (C)
  0.0000 Atom  2 (C)
  0.0000 Atom  3 (C)
  0.0000 Atom  4 (N)
  0.1667 Atom  5 (C)
  0.1667 Atom  6 (C)
  0.1667 Atom  7 (N)
  0.1667 Atom  8 (C)
  0.1667 Atom  9 (C)
  0.0000 Atom 10 (C)
  0.0000 Atom 11 (N)
  0.0000 Atom 12 (N)
  0.0000 Atom 13 (C)
  0.0000 Atom 14 (N)
  0.0000 Atom 15 (C)
  0.1667 Atom 16 (C)
  0.0000 Atom 17 (C)
  0.0000 Atom 18 (C)
  0.0000 Atom 19 (C)
  0.0000 Atom 20 (C)
"""
# %% testing multiple graphs
counter = 0
my_values_dict = {}
smarts_to_check = "C=O" # Example: Carbonyl group
substruct_func = get_substruct_coalition_func(smarts_to_check)
smarts_mol = Chem.MolFromSmarts(smarts_to_check)

for i, G in enumerate(tqdm(graphs)):
    if len(G) > 16:
        continue
        
    mol = G.graph['mol']
    my_calculator = MyersonCalculator(G, substruct_func)
    my_values = my_calculator.calculate_all_myerson_values()
    
    if counter < 10:
        has_substruct = mol.HasSubstructMatch(smarts_mol)
        print(f"Graph {i:>3}: Sum Myerson={sum(my_values):>8.4f} | Has Substructure={has_substruct}")
        
    my_values_dict[i] = my_values
    counter += 1

print(f"{len(my_values_dict)} graphs computed.")

# %%
for k, v in my_values_dict.items():
    mol = graphs[k].graph['mol']
    expected_val = 1.0 if mol.HasSubstructMatch(smarts_mol) else 0.0
    assert math.isclose(sum(v), expected_val, abs_tol=1e-5), f"Graph {k} failed validation"
