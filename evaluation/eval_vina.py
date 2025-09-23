import os
from pathlib import Path

from .docking import *
from .docking_2 import *
from .docking_vina import * 

def eval_vina(mol, ligand_filename, protein_filename, protein_root, out_dir, n):

    receptor_file = protein_filename.replace('.pdb','')+'.pdbqt'
    receptor_file = Path(os.path.join(protein_root,receptor_file))

    index = n%100
    g_vina_score = calculate_qvina2_score(
            receptor_file, mol, out_dir, return_rdmol=False, index=index)[0]
    # print("Generate vina score:", g_vina_score)

    # for vina_score
    vina_task = VinaDockingTask.from_generated_mol(mol, ligand_filename, protein_filename, protein_root=protein_root)
    score_only_results = vina_task.run(mode='score_only', exhaustiveness=16)
    minimize_results = vina_task.run(mode='minimize', exhaustiveness=16)
    docking_results = vina_task.run(mode='dock', exhaustiveness=16, index=index)
    docked_rmsd = docking_results[0]['rmsd']
    print(f"RMSD: {docked_rmsd}")

    num_atoms = mol.GetNumAtoms()

    vina_results = {
        'qvina2': g_vina_score,
        'score_only': score_only_results[0]['affinity'],
        'minimize': minimize_results[0]['affinity'],
        'vina_dock':docking_results[0]['affinity'],
        'LE': g_vina_score/num_atoms,
        'vina_score_LE': round(score_only_results[0]['affinity'] / num_atoms, 4),
        'vina_dock_LE': round(docking_results[0]['affinity'] / num_atoms, 4),
        'vina_mini_LE': round(minimize_results[0]['affinity'] / num_atoms, 4),
        'RMSD':round(float(docked_rmsd), 4),
    }

    return vina_results