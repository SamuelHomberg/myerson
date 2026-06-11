# %%
from myerson import MyersonCalculator, MyersonSampler

try:
    import rdkit
except ImportError:
    raise ImportError("Failed to impor rdkit")
import numpy as np

import networkx as nx
from tqdm import tqdm
import logging


log = logging.getLogger()

# %%
import networkx as nx
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import _CalcCrippenContribs

def crippen_contrib_per_atom(mol, no_assert=False):
    '''Will throw exception when ground truth is calculated incorrectly.'''
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem import Crippen 

    # Calculate exactly using AddHs
    mol_h = Chem.AddHs(mol)
    contribs_h = rdMolDescriptors._CalcCrippenContribs(mol_h)
    
    crippen = [0.0] * mol.GetNumAtoms()
    
    for i, atom in enumerate(mol_h.GetAtoms()):
        contrib = contribs_h[i][0]
        idx = atom.GetIdx()
        if idx < mol.GetNumAtoms():
            crippen[idx] += contrib
        else:
            neighbor = atom.GetNeighbors()[0]
            crippen[neighbor.GetIdx()] += contrib
            
    if no_assert:
        return crippen
    assert len(crippen) == mol.GetNumAtoms(), f"len(crippen)={len(crippen)}, num atoms={mol.GetNumAtoms()}"
    total_crippen = sum(crippen)
    ref_crippen = Crippen.MolLogP(mol)
    if not abs(total_crippen - ref_crippen) <= 1e-4:
        raise ValueError(f"sum(crippen)={total_crippen:.5f}, completeCrippen={ref_crippen:.5f}")
    return crippen

def rdkit_to_nx(mol):
    """
    Converts an RDKit Molecule object into a NetworkX Graph.
    
    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The RDKit molecule to convert.
    
    Returns:
    nx.Graph: A NetworkX graph with atom and bond attributes.
    """
    G = nx.Graph(mol=mol)
    atom_clogp = crippen_contrib_per_atom(mol)
    
    for i, atom in enumerate(mol.GetAtoms()):
        G.add_node(
            atom.GetIdx(),
            clogp=atom_clogp[i],
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

def clogp_coalition_func(coalition, nx_graph) -> float:
    return sum(nx_graph.nodes[node]['clogp'] for node in coalition)


# %% testing
failed_smi = []
import pandas as pd
from tqdm import tqdm
df = pd.read_csv('/nfs/home/s_homb04/clones/myerson/250k_rndm_zinc_drugs_clean_3.csv')
graphs = []
for smi in tqdm(df['smiles']): 
    mol = Chem.MolFromSmiles(smi)
    try:
        G = rdkit_to_nx(mol)
        graphs.append(G)
    except:
        failed_smi.append(smi)

print(f"{len(failed_smi)=}")
print(failed_smi[:10])
# %%    

# %% test single example
id = 5
smi = df['smiles'][id]
mol = Chem.MolFromSmiles(smi)
G = rdkit_to_nx(mol)
my_calculator = MyersonCalculator(G, clogp_coalition_func)
my_values = my_calculator.calculate_all_myerson_values()
print(sum(my_values), Chem.Crippen.MolLogP(Chem.MolFromSmiles(df['smiles'][5])))
print(f"{sum(my_values)=:4f} sum(G.nodes)={sum([G.nodes(data=True)[x]['clogp'] for x in G.nodes()]):.4f}")
recalc_crip = crippen_contrib_per_atom(smi)
for my, node, crip in zip(my_values, list(G.nodes(data=True)), recalc_crip):
    print(f"{my:>8.4f} {node[1]['clogp']:>8.4f} {crip:>8.4f}")

# %% 
failing_smi = "c1ccc2c3ccccc3n(C)c2c1"
failing_smi = "C/C(=N/O)c1ccc2c3ccccc3n(C)c2c1"
crip = crippen_contrib_per_atom(Chem.MolFromSmiles(failing_smi), no_assert=True)
mol = Chem.MolFromSmiles(failing_smi)
true_crip = Chem.Crippen.MolLogP(mol)#, includeHs=False, force=True)
print(f"{sum(crip):.4f}    {true_crip:.4f}")
sum(crip)- true_crip

# for atom, c in zip(Chem.MolFromSmiles(failing_smi).GetAtoms(), crip):
#     print(f"{atom.GetIdx():>3} {atom.GetSymbol():>2} {c:>8.4f}")




# %%
counter = 0
my_values_dict = {}
for i, G in enumerate(tqdm(graphs)):
    if len(G) > 16:
        continue
    my_calculator = MyersonCalculator(G, clogp_coalition_func)
    my_values = my_calculator.calculate_all_myerson_values()
    if counter < 10:
        print(sum(my_values), Chem.Crippen.MolLogP(Chem.MolFromSmiles(df['smiles'][i])))
    my_values_dict[i] = my_values
    counter +=1

print(len(my_values_dict), counter)
# %%
from rdkit.Chem import rdMolDescriptors
import math
for k, v in my_values_dict.items():
    # print(f"{k:>4} {sum(v):>8.4f} {Chem.Crippen.MolLogP(Chem.MolFromSmiles(df['smiles'][k])):>8.4f}")
    foo, bar = sum(v), Chem.Crippen.MolLogP(Chem.MolFromSmiles(df['smiles'][k]))
    assert math.isclose(foo, bar, abs_tol=1e-5), f"Graph {k} failed validation {foo:.5f} {bar:.5f}"
    contribs = crippen_contrib_per_atom(Chem.MolFromSmiles(df['smiles'][k]))
    # print(v); print(np.array(contribs))
    np.testing.assert_allclose(v, np.array(contribs),atol=1e-5)
# %%

np.testing.assert_allclose(my_values_dict[23], np.array(crippen_contrib_per_atom(Chem.MolFromSmiles(df['smiles'][23]))))