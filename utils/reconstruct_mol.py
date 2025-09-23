
import numpy as np
import torch
from rdkit import Chem
from rdkit.Geometry import Point3D
from .bond_analyze import  allowed_fc_bonds
from utils.reconstruct_mdm import mol2smiles

from .reconstruct_utils import *
from evaluation.sascorer import compute_sa_score
from rdkit.Chem.Descriptors import qed
from evaluation.score_func import obey_lipinski



bond_list = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
             Chem.rdchem.BondType.AROMATIC]

stability_bonds = {Chem.rdchem.BondType.SINGLE: 1, Chem.rdchem.BondType.DOUBLE: 2, Chem.rdchem.BondType.TRIPLE: 3,
                   Chem.rdchem.BondType.AROMATIC: 1.5}

# Bond lengths from:
# http://www.wiredchemist.com/chemistry/data/bond_energies_lengths.html
# And:
# http://chemistry-reference.com/tables/Bond%20Lengths%20and%20Enthalpies.pdf

# allowed_bonds = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1, 'B': 3, 'Al': 3,
#                  'Si': 4, 'P': [3, 5],
#                  'S': 4, 'Cl': 1, 'As': 3, 'Br': 1, 'I': 1, 'Hg': [1, 2],
#                  'Bi': [3, 5]}
allowed_fc_bonds = {'H': {0: 1, 1: 0, -1: 0},
                    'C': {0: [3, 4], 1: 3, -1: 3},
                    'N': {0: [2, 3], 1: [2, 3, 4], -1: 2},
                    'O': {0: 2, 1: 3, -1: 1},
                    'F': {0: 1, -1: 0},
                    'B': 3, 'Al': 3, 'Si': 4,
                    'P': {0: [3, 5], 1: 4},
                    'S': {0: [2, 6], 1: [2, 3], 2: 4, 3: 5, -1: 3},
                    'Cl': 1, 'As': 3,
                    'Br': {0: 1, 1: 2}, 'I': 1, 'Hg': [1, 2], 'Bi': [3, 5], 'Se': [2, 4, 6]}
# allowed_bonds = {'H': [0, 1], 'C': [3, 4], 'N': [2, 3, 4], 'O': [1, 2, 3], 'F': [0, 1], 'B': 3, 'Al': 3,
#                  'Si': 4, 'P': [3, 4, 5],
#                  'S': [3, 4, 5, 6], 'Cl': 1, 'As': 3, 'Br': [1, 2], 'I': 1, 'Hg': [1, 2],
#                  'Bi': [3, 5], 'Se': [2, 4, 6]}

def valid_smile(mol, ori_smile=None):
    # complete_n, valid_n = 0, 0
    smiles = mol2smiles(mol)
    if smiles is not None:
        try:
            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
        except:
            pass
        if len(mol_frags) == 1:
            complete_n = 1
        largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
        smiles = mol2smiles(largest_mol)
        valid_n = 1
    return smiles, complete_n, valid_n

def check_stability(mol, atom_num):
    nr_bonds = np.zeros(atom_num, dtype='int')
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        # if bond_type in stability_bonds:
        #     order = stability_bonds[bond_type]
        # else:
        #     order = 0
        order = stability_bonds[bond_type]
        nr_bonds[start] += order
        nr_bonds[end] += order

    formal_charges = torch.zeros(atom_num)
     # stability
    nr_stable_bonds = 0
    atom_types_str = [atom.GetSymbol() for atom in mol.GetAtoms()]
    for atom_type_i, nr_bonds_i, fc_i in zip(atom_types_str, nr_bonds, formal_charges):
        fc_i = fc_i.item()
        possible_bonds = allowed_fc_bonds[atom_type_i]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == nr_bonds_i
        elif type(possible_bonds) == dict:
            expected_bonds = possible_bonds[fc_i] if fc_i in possible_bonds.keys() else possible_bonds[0]
            is_stable = expected_bonds == nr_bonds_i if type(expected_bonds) == int else nr_bonds_i in expected_bonds
        else:
            is_stable = nr_bonds_i in possible_bonds
        nr_stable_bonds += int(is_stable)
    
    molecule_stable = nr_stable_bonds == atom_num

    return molecule_stable, nr_stable_bonds

def valid_smile(mol, largest_mol_flag=False):
    smiles = mol2smiles(mol)
    if smiles is not None:
        if largest_mol_flag:
            try:
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
                gmol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                smiles = mol2smiles(gmol)
                return smiles, gmol
                # print("largest generated smile part:", smiles)
            except:
                print('Largest invalid,continue')
    
    return smiles, mol

def build_mol_with_edge(positions, atom_types, edge_types, dataset_info, frag_bond_index=None):
    
    atom_decoder = dataset_info['atom_decoder']
    # convert to RDKit Mol, add atom
    mol = Chem.RWMol()
    atoms = []
    for atom in atom_types:
        a = Chem.Atom(atom_decoder[atom.item()])
        atoms.append(atom_decoder[atom.item()])
        mol.AddAtom(a)
        
    oxygen_num = atoms.count('O')
    nitrogen_num = atoms.count('N')
    if oxygen_num >= len(atoms) / 2 or nitrogen_num >= len(atoms) / 2:
        return None
    
    # add positions to Mol
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(positions[i][0].item(), positions[i][1].item(), positions[i][2].item()))
    mol.AddConformer(conf)

    # add bonds
    edge_index = torch.nonzero(edge_types)
    mask = edge_index[:, 0] < edge_index[:, 1]
    edge_index = edge_index[mask]
    bond_types = edge_types[edge_index[:, 0], edge_index[:, 1]]
    edge_index = edge_index.t()
    edge_index, bond_types = remove_bond(edge_index, positions, bond_types, frag_bond_index)
    edge_index, bond_types = remove_cycles(edge_index, positions, bond_types)
    edge_index, bond_types = adjust_bond_types_by_valence(atoms, edge_index, bond_types, frag_bond_index)
    edge_index = edge_index.t()
    # print(edge_index)
    for i in range(edge_index.size(0)):
        src, dst = edge_index[i]
        order = bond_types[i]
        mol.AddBond(src.item(), dst.item(), bond_list[int(order)])
    return mol


def Remove_ring(positions, atom_types, edge_types, dataset_info):
    atom_decoder = dataset_info['atom_decoder']
    # convert to RDKit Mol, add atom
    mol = Chem.RWMol()
    atoms = []
    for atom in atom_types:
        a = Chem.Atom(atom_decoder[atom.item()])
        atoms.append(atom_decoder[atom.item()])
        mol.AddAtom(a)
    
    # add positions to Mol
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(positions[i][0].item(), positions[i][1].item(), positions[i][2].item()))
    mol.AddConformer(conf)
    # add bonds
    edge_index = torch.nonzero(edge_types)
    mask = edge_index[:, 0] < edge_index[:, 1]
    edge_index = edge_index[mask]
    bond_types = edge_types[edge_index[:, 0], edge_index[:, 1]]
    edge_index = edge_index.t()
    edge_index, bond_types = remove_bond(edge_index, positions, bond_types, threshold=2.0)
    edge_index, bond_types = remove_cycles(edge_index, positions, bond_types)
    edge_index, bond_types = adjust_bond_types_by_valence(atoms, edge_index, bond_types)
    edge_index = edge_index.t()
    
    for i in range(edge_index.size(0)):
        src, dst = edge_index[i]
        order = bond_types[i]
        mol.AddBond(src.item(), dst.item(), bond_list[int(order)])
    
    return mol


def build_mol_with_edge22(positions, atom_types, edge_types, dataset_info, frag_bond_index=None):
    
    atom_decoder = dataset_info['atom_decoder']
    # convert to RDKit Mol, add atom
    mol = Chem.RWMol()
    atoms = []
    for atom in atom_types:
        a = Chem.Atom(atom_decoder[atom.item()])
        atoms.append(atom_decoder[atom.item()])
        mol.AddAtom(a)
    
    # add positions to Mol
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(positions[i][0].item(), positions[i][1].item(), positions[i][2].item()))
    mol.AddConformer(conf)

    edge_index = torch.nonzero(edge_types)
    for i in range(edge_index.size(0)):
        src, dst = edge_index[i]
        if src < dst:
            order = edge_types[src, dst]
            mol.AddBond(src.item(), dst.item(), bond_list[int(order)])
    
    return mol


def check_alert_structures(mol, alert_smarts_list):
    Chem.GetSSSR(mol)
    patterns = [Chem.MolFromSmarts(sma) for sma in alert_smarts_list]
    for p in patterns:
        subs = mol.GetSubstructMatches(p)
        if len(subs) != 0:
            return True
    else:
        return False