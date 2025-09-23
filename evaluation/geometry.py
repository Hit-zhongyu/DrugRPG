import numpy as np
from rdkit import Chem

BOND_SYMBOLS = {
    Chem.rdchem.BondType.SINGLE: '-',
    Chem.rdchem.BondType.DOUBLE: '=',
    Chem.rdchem.BondType.TRIPLE: '#',
    Chem.rdchem.BondType.AROMATIC: ':',
}

def bond_repr(mol, pair):
    i = mol.GetAtomWithIdx(pair[0]).GetSymbol()
    j = mol.GetAtomWithIdx(pair[1]).GetSymbol()
    ij = BOND_SYMBOLS[mol.GetBondBetweenAtoms(pair[0], pair[1]).GetBondType()]
    # Unified (sorted) representation
    return f'{i}{ij}{j}' if i <= j else f'{j}{ij}{i}'

def get_bond_distances(mol, bonds):
    i, j = np.array(bonds).T
    x = mol.GetConformer().GetPositions()
    xi = x[i]
    xj = x[j]
    bond_distances = np.linalg.norm(xi - xj, axis=1)
    return bond_distances

def angle_repr(mol, triplet):
        i = mol.GetAtomWithIdx(triplet[0]).GetSymbol()
        j = mol.GetAtomWithIdx(triplet[1]).GetSymbol()
        k = mol.GetAtomWithIdx(triplet[2]).GetSymbol()
        ij = BOND_SYMBOLS[mol.GetBondBetweenAtoms(triplet[0], triplet[1]).GetBondType()]
        jk = BOND_SYMBOLS[mol.GetBondBetweenAtoms(triplet[1], triplet[2]).GetBondType()]

        # Unified (sorted) representation
        if i < k:
            return f'{i}{ij}{j}{jk}{k}'
        elif i > j:
            return f'{k}{jk}{j}{ij}{i}'
        elif ij <= jk:
            return f'{i}{ij}{j}{jk}{k}'
        else:
            return f'{k}{jk}{j}{ij}{i}'

def get_angle_values(mol, triplets):
    i, j, k = np.array(triplets).T
    x = mol.GetConformer().GetPositions()
    xi = x[i]
    xj = x[j]
    xk = x[k]
    vji = xi - xj
    vjk = xk - xj
    angles = np.arccos((vji * vjk).sum(axis=1) / (np.linalg.norm(vji, axis=1) * np.linalg.norm(vjk, axis=1)))
    return np.degrees(angles)