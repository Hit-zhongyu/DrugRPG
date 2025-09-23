
# from .utils import *

import pandas as pd
import numpy as np
from rdkit import Chem
from evaluation.Reos import REOS
from rdkit.Chem import AllChem
from rdkit.Chem.rdForceFieldHelpers import UFFGetMoleculeForceField

def is_nan(value):
    return value is None or pd.isna(value) or np.isnan(value)

def eval_other(mol, protein_path):

    clash_num_ligand, clash_score_ligand = clash_score(mol)
    is_clash_ligand = 1 - (clash_score_ligand == 0)

    protein = Chem.MolFromPDBFile(str(protein_path), sanitize=False)
    clash_num_pockets, clash_score_pockets = clash_score(mol, protein)
    is_clash_pockets = 1- (clash_score_pockets == 0)

    REOS_results = {}
    reos = REOS()
    for rule_set in reos.get_available_rule_sets():
        reos.set_active_rule_sets([rule_set])
        if rule_set == 'PW':
            reos.drop_rule('furans')
        
        reos_res = reos.process_mol(mol)
        REOS_results[rule_set] = reos_res[0] == 'ok'

    REOS_result = all([bool(value) if not is_nan(value) else False for value in REOS_results.values()])

    # mol_copy = Chem.Mol(mol)
    # mol_copy = Chem.AddHs(mol_copy, addCoords=True)
    # uff = UFFGetMoleculeForceField(mol_copy, confId=-1)
    # energy  = uff.CalcEnergy()

    energy = compute_energy_rdkit(mol)
    # _, lowest_energy = find_lowest_energy_conformer(mol)

    # strain_energy = energy - lowest_energy

    other_results =  {
        'clash_num_ligand': clash_num_ligand,
        'clash_score_ligand': clash_score_ligand,
        'is_clash_ligand': is_clash_ligand,
        'clash_num_pockets': clash_num_pockets,
        'clash_score_pockets': clash_score_pockets,
        'is_clash_pockets': is_clash_pockets,
        'REOS_result': REOS_result,
        'energy': energy,
        # 'strain_energy': strain_energy,
        }
    
    return other_results

# def compute_strain_energy(mol_copy, ref_energy):
#     # mol_copy is assumed to be a copy with a 3D conformer already
#     try:
#         e = compute_energy_rdkit(mol_copy, -1)
#         strain_energy = e - ref_energy
#         return e, strain_energy
#     except:
#         return None, None

def compute_energy_rdkit(mol, conf_id=-1):
    mol_copy = Chem.Mol(mol)
    mol_copy = Chem.AddHs(mol_copy, addCoords=True)
    ff = UFFGetMoleculeForceField(mol_copy, confId=-1)
    return ff.CalcEnergy()

def find_lowest_energy_conformer(mol, num_confs=16):
    mol_copy = Chem.Mol(mol)
    mol_copy = Chem.AddHs(mol_copy)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    conf_ids = AllChem.EmbedMultipleConfs(mol_copy, numConfs=num_confs, params=params)
    energies = []
    for cid in conf_ids:
        try:
            energy = compute_energy_rdkit(mol_copy, cid)
            energies.append((cid, energy))
        except:
            continue
    if not energies:
        raise ValueError("No valid conformer energies calculated.")
    # Find conformer with lowest energy
    cid_min, e_min = min(energies, key=lambda x: x[1])
    return cid_min, e_min

def clash_score(rdmol1, rdmol2=None, tolerance=0.4):
        """
        Computes a clash score as the number of atoms that have at least one
        clash divided by the number of atoms in the molecule.

        INTERMOLECULAR CLASH SCORE
        If rdmol2 is provided, the score is the percentage of atoms in rdmol1
        that have at least one clash with rdmol2.
        We define a clash if two atoms are closer than "margin times the sum of
        their van der Waals radii".

        INTRAMOLECULAR CLASH SCORE
        If rdmol2 is not provided, the score is the percentage of atoms in rdmol1
        that have at least one clash with other atoms in rdmol1.
        In this case, a clash is defined by margin times the atoms' smallest
        covalent radii (among single, double and triple bond radii). This is done
        so that this function is applicable even if no connectivity information is
        available.
        """

        intramolecular = rdmol2 is None
        if intramolecular:
            rdmol2 = rdmol1
            ligand_info = parse_sdf_file(rdmol1)
            ligand_pos = np.array(ligand_info['pos'])
            ligand_intra_mask = (~ligand_info['bond_adj']) ^ np.eye(len(ligand_pos), dtype=bool)

        coord1, radii1 = coord_and_radii(rdmol1, intramolecular=intramolecular)
        coord2, radii2 = coord_and_radii(rdmol2, intramolecular=intramolecular)

        dist = np.sqrt(np.sum((coord1[:, None, :] - coord2[None, :, :]) ** 2, axis=-1))
        if intramolecular:
            np.fill_diagonal(dist, np.inf)

        clashes = dist < (radii1[:, None] + radii2[None, :]) - tolerance
        if intramolecular:
            clashes = clashes * ligand_intra_mask
            clashes_num = int(np.sum(clashes) / 2)
        else:
            clashes_num = int(np.sum(clashes))
        clashes = np.any(clashes, axis=1)
        return clashes_num, np.mean(clashes)
    
def coord_and_radii(rdmol, intramolecular,ignore={'H'}):
    _periodic_table = Chem.GetPeriodicTable()
    _get_radius = _periodic_table.GetRcovalent if intramolecular else _periodic_table.GetRvdw

    coord = rdmol.GetConformer().GetPositions()
    radii = np.array([_get_radius(a.GetSymbol()) for a in rdmol.GetAtoms()])

    mask = np.array([a.GetSymbol() not in ignore for a in rdmol.GetAtoms()])
    coord = coord[mask]
    radii = radii[mask]

    assert coord.shape[0] == radii.shape[0]
    return coord, radii

def parse_sdf_file(input_mol):
    if type(input_mol) == str:
        mol = read_sdf(input_mol)[0]
    else:
        mol = input_mol
    mol_info = {}
    atomic_type = []
    atomic_number = []
    atomic_coords = []
    # Iterate through each atom in the molecule
    for atom in mol.GetAtoms():
        atomic_type.append(atom.GetSymbol())
        atomic_number.append(atom.GetAtomicNum())
        pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
        atomic_coords.append((pos.x, pos.y, pos.z))

    mol_info['atom_name'] = atomic_type
    mol_info['element'] = np.array(atomic_number)
    mol_info['pos'] = np.array(atomic_coords)
    mol_info['bond_adj'] = np.zeros((len(atomic_type), len(atomic_type)), dtype=bool)
    for bond in mol.GetBonds():
        start_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        mol_info['bond_adj'][start_idx, end_idx] = True
        mol_info['bond_adj'][end_idx, start_idx] = True
    return mol_info