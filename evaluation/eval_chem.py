from rdkit.Chem.Descriptors import MolLogP, qed, HeavyAtomMolWt
from .sascorer import *
from .score_func import *
from collections import Counter
from evaluation.utils import *

def eval_chem(mol):
    _, sa_score = compute_sa_score(mol)
    print("Generate SA score:", sa_score)

    qed_score = qed(mol)
    print("Generate QED score:", qed_score)

    logp_score = MolLogP(mol)
    print("Generate logP:", logp_score)

    lipinski_score = obey_lipinski(mol)
    print("Generate Lipinski:", lipinski_score)

    ring_info = mol.GetRingInfo()
    ring_size = Counter([len(r) for r in ring_info.AtomRings()])
    num_atoms = mol.GetNumAtoms()
    print("Generate atom num:", num_atoms)

    fused_num = judge_fused_ring(mol)
    unexpected_num = judge_unexpected_ring(mol) 

    chem_results =  {
        # 'ligand_file': ligand_filename, 
        # 'smile': smile,
        'qed': qed_score,
        'sa': sa_score,
        'logp': logp_score,
        'lipinski': lipinski_score,
        'ring_size': ring_size,
        'num_atoms': num_atoms,
        'fused_ring_num': int(fused_num),
        'unexpected_ring_num': int(unexpected_num),

    }
    return chem_results