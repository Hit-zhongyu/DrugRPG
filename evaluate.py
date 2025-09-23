import argparse
import pickle
import os
import torch

import numpy as np
from rdkit.Chem import AllChem as Chem
from tqdm import tqdm

from statistics import mean
from pathlib import Path

from joblib import Parallel, delayed

from configs.dataset_config import get_dataset_info
from evaluation import *
from collections import defaultdict

from evaluation.eval_chem import eval_chem
from evaluation.eval_geom import eval_geom
from evaluation.eval_vina import eval_vina
from evaluation.eval_other import eval_other
from evaluation.utils import calculate_top
from evaluation.geometry import eval_bond_length, eval_bond_angle



def evaluate(m, n, protein_root='/mnt/rna01/lzy/PMDM/data/crossdocked_pocket10'):
    smile = m['smile']
    protein_filename = m['protein_file']
    ligand_filename = m['ligand_file']
    mol = m['mol']
    # print("ligand_filename",ligand_filename)

    # for other methods
    target = protein_filename.split('/')[0]
    # ligand_filename = os.path.join(protein_root, target, ligand_filename)
    ligand_filename = os.path.join(protein_root, target, ligand_filename.split("/")[1])
    # print(ligand_filename)
    # ligand_filename = os.path.join(protein_root, target, ligand_filename.split("test100/")[1])

    protein_path = Path(os.path.join(protein_root, protein_filename))

    print("Generate smile:", smile)
    g_valid = 0
    try:
        Chem.SanitizeMol(mol)
        g_valid = 1
    except:
        print('mol error')
        return None

    # for chemical proterty
    chem_results = eval_chem(mol)

    # for geometry
    geom_results = eval_geom(mol)

    # for vina
    vina_results = eval_vina(mol, ligand_filename, protein_filename, protein_root, out_dir, n)
    print("Generate vina results: ", vina_results)

    # for other
    other_results = eval_other(mol, protein_path)
    print("other results:", other_results)

    success_flag = 0
    if chem_results['qed'] > 0.25 and chem_results['sa'] > 0.59 and vina_results['vina_dock'] < -8.18:
        success_flag = 1
    

    metrics = {'valid':g_valid, 'success':success_flag, 'chem_results':chem_results, 
               'geom_results': geom_results, 'vina_results': vina_results, 'other_results':other_results}
    result = {
        'mol': mol,
        'smile': smile,
        'protein_file': protein_filename,
        'ligand_file': ligand_filename,
        'metrics': metrics,
        }

    return result


def save_sdf(mol, sdf_dir, gen_file_name):
    writer = Chem.SDWriter(os.path.join(sdf_dir, gen_file_name))
    writer.write(mol, confId=0)
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='crossdock')
    parser.add_argument('--path', type=str, default='')
    args = parser.parse_args()

    dataset_info = get_dataset_info(args.dataset, False)
    path = args.path
    print(os.path.dirname(path))
    # save_mol_result_path = os.path.join(os.path.dirname(path), 'mol_results.pkl')
    # if os.path.exists(save_mol_result_path):
    #     with open(save_mol_result_path, 'rb') as f:
    #         results = pickle.load(f)
    # else:
        # results = {}
    with open(path, 'rb') as f:
        data = pickle.load(f)
        # print(data)
    # exit(0)
    if args.dataset == 'crossdock':
        protein_root ='/mnt/rna01/lzy/PMDM/data/crossdocked_pocket10'

    out_dir = os.path.join(os.path.dirname(path),'ligand')
    os.makedirs(out_dir,exist_ok=True)
    sdf_dir = os.path.dirname(path)
    results_mol = []
    high_affinity = []
    stable = 0
    valid = 0
    smile_list = []
    num_samples = 0
    position_list = []
    atom_num_list = []
    sa_list = []
    qed_list = []
    logP_list = []
    weight_list = []
    Lipinski_list = []
    bond_len_list = []
    bond_angle_list = []
    ring_list = []
    qvina_score_list = []
    vina_score_list = []
    vina_dock_list = []
    vina_LE_list = []
    vina_score_LE_list = []
    vina_dock_LE_list = []
    vina_mini_LE_list = []
    RMSD_list = []
    mini_vina_score_list = []
    mean_vina_score_list = []
    diversity_list = []
    REOS_list = []
    mol_dict = {}
    success_list = []
    idx = 0
    faild = 0
    fused_ring_nums = 0
    unexpected_ring_nums = 0
    # num_atoms = 0
    t_vina_dict = {}
    g_dict = defaultdict(lambda: {'sa': [], 'qed': [], 'vina': [],'atom_num': [],
                                   'lipinski': [], 'logp': [], 'vina_score':[], 
                                   'vina_mini':[], 'vina_dock':[], 'LE':[],
                                    'vina_score_LE':[], 'vina_dock_LE':[],'vina_mini_LE':[]})
    
    clash_dict = defaultdict(list)

    with open('unique_protein_filenames.pkl', 'rb') as f:
        correct_paths = pickle.load(f)
    filename_to_correct_path = {os.path.basename(p): p for p in correct_paths}

    with open('/mnt/rna01/lzy/SBDD/test_vina_{}_dict.pkl'.format(args.dataset), 'rb') as f:
        test_vina_score_list = pickle.load(f)

    for d in tqdm(data):
        mol = d['mol']
        protein_filename = d['protein_file']
        # print(protein_filename)
        # for other methods
        # protein_filename = protein_filename.split("test100/")[1].replace("-", "_")
        # filename = os.path.basename(protein_filename)
        # if filename in filename_to_correct_path:
        #     correct_path = filename_to_correct_path[filename]
        #     d['protein_file'] = correct_path
        # if correct_path not in mol_dict.keys():
        #     mol_dict[correct_path] = []
        # mol_dict[correct_path].append(mol)

        if protein_filename not in mol_dict.keys():
            mol_dict[protein_filename] = []
        mol_dict[protein_filename].append(mol)
    
    results = Parallel(n_jobs=32)(delayed(evaluate)(m,n) for n, m in enumerate(tqdm(data)))
    
    for result in tqdm(results):
        if result is not None:
            results_mol.append(result)
            chem_results = result['metrics']["chem_results"]
            geom_results = result['metrics']["geom_results"]
            vina_results = result['metrics']["vina_results"]
            other_results = result['metrics']["other_results"]
            # bond_length = result['bond_dist']
            success_flag = result['metrics']["success"]
            success_list.append(success_flag)

            g_name = result['ligand_file']
            g_smile = result['smile']
            smile_list.append(g_smile)
            
            g_sa,  g_qed, g_logP, g_Lipinski, g_rings, g_num, g_fused, g_unexpected = chem_results['sa'], chem_results['qed'], chem_results[
                                        'logp'], chem_results['lipinski'], chem_results['ring_size'], chem_results['num_atoms'], chem_results[
                                        'fused_ring_num'], chem_results['unexpected_ring_num']
            sa_list.append(g_sa)
            qed_list.append(g_qed)
            logP_list.append(g_logP)
            Lipinski_list.append(g_Lipinski)
            ring_list.append(g_rings)
            atom_num_list.append(g_num)
            fused_ring_nums += g_fused
            unexpected_ring_nums += g_unexpected

            g_bond_len, g_bond_angle = geom_results['bond_dist'], geom_results['bond_angle']
            bond_len_list += g_bond_len
            bond_angle_list += g_bond_angle

            g_qvina2_score, vina_score, minimize_score, vina_dock, vina_LE, vina_score_LE, vina_dock_LE, vina_mini_LE, rmsd = vina_results['qvina2'], vina_results[
                                        'score_only'], vina_results['minimize'], vina_results['vina_dock'], vina_results['LE'], vina_results[
                                            'vina_score_LE'], vina_results['vina_dock_LE'],vina_results['vina_mini_LE'], vina_results['RMSD']
            qvina_score_list.append(g_qvina2_score)
            vina_score_list.append(vina_score)
            vina_dock_list.append(vina_dock)
            vina_LE_list.append(vina_LE)
            vina_score_LE_list.append(vina_score_LE)
            vina_dock_LE_list.append(vina_dock_LE)
            vina_mini_LE_list.append(vina_mini_LE)
            RMSD_list.append(rmsd)
            mini_vina_score_list.append(minimize_score)
            
            g_clash_num_ligand, g_clash_score_ligand, g_is_clash_ligand, g_clash_num_pockets, g_clash_score_pockets, g_is_clash_pockets, g_REOS_result = other_results[
                                            'clash_num_ligand'], other_results['clash_score_ligand'],other_results['is_clash_ligand'],other_results[
                                            'clash_num_pockets'], other_results['clash_score_pockets'], other_results['is_clash_pockets'],other_results['REOS_result']
            clash_dict['clash_num_ligand'].append(g_clash_num_ligand)
            clash_dict['clash_score_ligand'].append(g_clash_score_ligand)
            clash_dict['is_clash_ligand'].append(g_is_clash_ligand)
            clash_dict['clash_num_pockets'].append(g_clash_num_pockets)
            clash_dict['clash_score_pockets'].append(g_clash_score_pockets)
            clash_dict['is_clash_pockets'].append(g_is_clash_pockets)
            REOS_list.append(g_REOS_result)

            # high_affinity.append(g_h_a)
            valid+=1
            
            g_dict[g_name]['sa'].append(g_sa)
            g_dict[g_name]['qed'].append(g_qed)
            g_dict[g_name]['vina'].append(g_qvina2_score)
            g_dict[g_name]['lipinski'].append(g_Lipinski)
            g_dict[g_name]['logp'].append(g_logP)
            g_dict[g_name]['vina_score'].append(vina_score)
            g_dict[g_name]['vina_dock'].append(vina_dock)
            g_dict[g_name]['vina_mini'].append(minimize_score)
            g_dict[g_name]['LE'].append(vina_LE)
            g_dict[g_name]['vina_score_LE'].append(vina_score_LE)
            g_dict[g_name]['vina_dock_LE'].append(vina_dock_LE)
            g_dict[g_name]['vina_mini_LE'].append(vina_mini_LE)
            g_dict[g_name]['atom_num'].append(g_num)

        else:
            faild += 1
            


            # if g_vina < 0:
            #     vina_score_list.append(g_vina)
            #     mean_vina_score_list.append(m_vina)
            # for key in range(3, 10):
            #     ring_num_dict[key] += g_rings[key]

            # ring_num_dict[10] += g_rings['other']
            # for bond_type, lengths in g_bond_length.items():
            #     bond_length_dict[bond_type].extend(lengths)

            

    num_samples = 500
    # bond_length_profile = get_bond_length_profile(bond_list)
    # bond_js_metrics = eval_bond_length_profile(bond_length_profile)
    # validity_dict = analyze_stability_for_molecules(position_list, atom_type_list, dataset_info)
    # print(validity_dict)
    # print("Final validity:", valid / num_samples)
    print('success rate:', len(success_list) / len(success_list))
    print("Fused_ring_rate:", fused_ring_nums / valid)
    print("Unexpected_ring_rate:", unexpected_ring_nums / valid)
    print("Unique:", len(set(smile_list)) / len(smile_list))
    print('mean sa:%f' % mean(sa_list))
    print('mean qed:%f' % mean(qed_list))
    print('mean logP:%f' % mean(logP_list))
    print('mean Lipinski:%f' % np.mean(Lipinski_list))
    print('mean qvina_score:%f' % mean(qvina_score_list))
    print('mean vina_score:%f' % mean(vina_score_list))
    print('mean vina_dock_score:%f' % mean(vina_dock_list))
    print('mean mini_vina:%f' % mean(mini_vina_score_list))
    print('mean vina_LE:%f' % mean(vina_LE_list))
    print('mean vina_score_LE:%f' % mean(vina_score_LE_list))
    print('mean vina_dock_LE:%f' % mean(vina_dock_LE_list))
    print('mean vina_mini_LE:%f' % mean(vina_mini_LE_list))
    print('mean RMSD:%f' % mean(RMSD_list))
    print('mean REOS:%f' % mean(REOS_list))
    print('high affinity:%d' % np.sum(high_affinity))
    print(f"Invalid ligands: {faild}/{faild + valid} ({faild/(faild + valid):.2%})")

    c_bond_length_profile = eval_bond_length.get_bond_length_profile(bond_len_list)
    c_bond_length_dict = eval_bond_length.eval_bond_length_profile(c_bond_length_profile)
    print("bond_len_js_metrics:", c_bond_length_dict)

    c_bond_angle_profile = eval_bond_angle.get_bond_angle_profile(bond_angle_list)
    c_bond_angle_dict = eval_bond_angle.eval_bond_angle_profile(c_bond_angle_profile)
    print("bond_angle_js_metrics:", c_bond_angle_dict)

    
    for ring_size in range(3, 10):
        n_mol = 0
        n_mol = sum(1 for counter in ring_list if ring_size in counter)
        ratio = n_mol / len(ring_list)
        print(f'ring size: {ring_size} ratio: {ratio:.3f}')
    
    for key, values in clash_dict.items():
        avg = sum(values) / len(values)
        print(f"{key}: mean: {avg:.4f}")


    top_n_values = [1, 3, 5, 10]  # 要计算的 top_n
    print("QED")
    for top_n in top_n_values:
        mean_sa, mean_vina, mean_qed, mean_lipinski, mean_logp, mean_atom_num, mean_vina_score, mean_vina_dock, mean_vina_mini, mean_LE, \
                                        mean_vina_score_LE, mean_vina_dock_LE, mean_vina_mini_LE = calculate_top(g_dict, top_n=top_n, metric="qed")
        print(f"Top QED {top_n} 的均值:  SA: {mean_sa:.4f}, QED: {mean_qed:.4f}, Lipinski: {mean_lipinski:.4f}, LogP: {mean_logp:.4f}, Atom_num: {mean_atom_num:.4f}, "
              f"Vina/LE: {mean_vina:.4f}/{mean_LE:.4f}, Vina_score/LE: {mean_vina_score:.4f}/{mean_vina_score_LE:.4f}, Vina_dock/LE: {mean_vina_dock:.4f}/{mean_vina_dock_LE:.4f}, "
              f"Vina_mini/LE: {mean_vina_mini:.4f}/{mean_vina_mini_LE:.4f}")

    print("SA")
    for top_n in top_n_values:
        mean_sa, mean_vina, mean_qed, mean_lipinski, mean_logp, mean_atom_num, mean_vina_score, mean_vina_dock, mean_vina_mini, mean_LE, \
                                        mean_vina_score_LE, mean_vina_dock_LE, mean_vina_mini_LE = calculate_top(g_dict, top_n=top_n, metric="sa")
        print(f"Top QED {top_n} 的均值:  SA: {mean_sa:.4f}, QED: {mean_qed:.4f}, Lipinski: {mean_lipinski:.4f}, LogP: {mean_logp:.4f}, Atom_num: {mean_atom_num:.4f}, "
              f"Vina/LE: {mean_vina:.4f}/{mean_LE:.4f}, Vina_score/LE: {mean_vina_score:.4f}/{mean_vina_score_LE:.4f}, Vina_dock/LE: {mean_vina_dock:.4f}/{mean_vina_dock_LE:.4f}, "
              f"Vina_mini/LE: {mean_vina_mini:.4f}/{mean_vina_mini_LE:.4f}")
    
    print("vina_dock")
    for top_n in top_n_values:
        mean_sa, mean_vina, mean_qed, mean_lipinski, mean_logp, mean_atom_num, mean_vina_score, mean_vina_dock, mean_vina_mini, mean_LE, \
                                        mean_vina_score_LE, mean_vina_dock_LE, mean_vina_mini_LE = calculate_top(g_dict, top_n=top_n, metric="vina_dock")
        print(f"Top QED {top_n} 的均值:  SA: {mean_sa:.4f}, QED: {mean_qed:.4f}, Lipinski: {mean_lipinski:.4f}, LogP: {mean_logp:.4f}, Atom_num: {mean_atom_num:.4f}, "
              f"Vina/LE: {mean_vina:.4f}/{mean_LE:.4f}, Vina_score/LE: {mean_vina_score:.4f}/{mean_vina_score_LE:.4f}, Vina_dock/LE: {mean_vina_dock:.4f}/{mean_vina_dock_LE:.4f}, "
              f"Vina_mini/LE: {mean_vina_mini:.4f}/{mean_vina_mini_LE:.4f}")


    sa_list = torch.tensor(sa_list)
    qed_list = torch.tensor(qed_list)
    logP_list = torch.tensor(logP_list)
    # weight_list = torch.tensor(weight_list)
    Lipinski_list = torch.tensor(Lipinski_list)
    qvina_score_list = torch.tensor(qvina_score_list)
    vina_score_list = torch.tensor(vina_score_list)
    vina_dock_list = torch.tensor(vina_dock_list)
    mini_vina_score_list = torch.tensor(mini_vina_score_list)
    atom_num_list = torch.tensor(atom_num_list)
    
    metrics_list = {
        'diversity': diversity_list,
        'sa': sa_list,
        'qed': qed_list,
        'logP': logP_list,
        'Lipinski': Lipinski_list,
        'qvina': qvina_score_list,
        'vina': vina_score_list,
        'vina_dock': vina_dock_list,
        'mini_vina': mini_vina_score_list,
        'high_affinity': high_affinity}

    save_mol_result_path = os.path.join(os.path.dirname(path), 'mol_results.pkl')
    with open(save_mol_result_path, 'wb') as f:
        pickle.dump(results_mol, f)
        f.close()

    save_metric_result_path = os.path.join(os.path.dirname(path), 'metric_results.pkl')
    with open(save_metric_result_path, 'wb') as f:
        pickle.dump(metrics_list, f)
        f.close()