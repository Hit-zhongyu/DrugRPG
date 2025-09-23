from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pickle
import glob
import os.path as osp
from rdkit.Chem import rdMolAlign as MA
import copy

def GetBestRMSD(probe, ref):
    probe = Chem.RemoveHs(probe)
    ref = Chem.RemoveHs(ref)
    rmsd = MA.GetBestRMS(probe, ref)
    return rmsd
def read_pkl(pkl_file):
    with open(pkl_file,'rb') as f:
        data = pickle.load(f)
    return data

def get_rdkit_rmsd(mol, n_conf=20, random_seed=42):
    """
    Calculate the alignment of generated mol and rdkit predicted mol
    Return the rmsd (max, min, median) of the `n_conf` rdkit conformers
    """
    copy_mol = copy.deepcopy(mol)
    Chem.SanitizeMol(copy_mol)
    copy_mol = Chem.AddHs(copy_mol)
    mol3d = Chem.AddHs(copy_mol)
    rmsd_list = []
    
    confIds = AllChem.EmbedMultipleConfs(mol3d, n_conf, randomSeed=random_seed)
    for confId in confIds:
        AllChem.UFFOptimizeMolecule(mol3d, confId=confId)
        rmsd = Chem.rdMolAlign.GetBestRMS(copy_mol, mol3d, refId=confId)
        rmsd_list.append(rmsd)

    rmsd_list = np.array(rmsd_list)
    if rmsd_list.size > 0:
        return [np.mean(rmsd_list),np.max(rmsd_list), np.min(rmsd_list), np.median(rmsd_list)]
    else:
        return None, None, None, None
    

    
