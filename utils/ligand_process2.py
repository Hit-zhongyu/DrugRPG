import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from rdkit import Chem
from rdkit import RDConfig
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import ChemicalFeatures, QED
from evaluation.sascorer import compute_sa_score

ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable',
                 'ZnBinder']
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}
BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}
BOND_NAMES = {i: t for i, t in enumerate(BondType.names.keys())}

def parse_sdf_file(path):
    
    supplier = Chem.SDMolSupplier(path, removeHs=False, sanitize=False)
    mol = next(iter(supplier))

    if mol is None:
        raise ValueError("Failed to load molecule from SDF")

    atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    pos = mol.GetConformer().GetPositions()  # numpy array of shape (N_atoms, 3)
    num_atoms = mol.GetNumAtoms()

    bond_dict = {}
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        idx_pair = (a, b) if a < b else (b, a)
        bond_type = BOND_TYPES.get(bond.GetBondType(), 0)
        bond_dict[idx_pair] = bond_type

    row, col, edge_type = [], [], []
    for i in range(num_atoms):
        for j in range(i+1, num_atoms):
            if i == j:
                continue  # 跳过自环
            a, b = (i, j) if i < j else (j, i)
            bond = bond_dict.get((a, b), 0)
            row.append(i)
            col.append(j)
            edge_type.append(bond)

    edge_index = np.array([row, col], dtype=int)
    edge_type = np.array(edge_type, dtype=int)
    
    data = {
        'element': np.array(atoms, dtype=int),  # 原子类型
        'pos': np.array(pos, dtype=np.float32),  # 坐标
        'bond_index': edge_index,  # 边
        'bond_type': edge_type,   # 单键 双键或其他
        'mol': mol,
    }
    return data


if __name__ == '__main__':
    path = '/mnt/rna01/lzy/pocketdiff5/8pkm/8pkm.sdf'
    data = parse_sdf_file(path)
    np.set_printoptions(threshold=np.inf) 
    print(data['bond_type'])


