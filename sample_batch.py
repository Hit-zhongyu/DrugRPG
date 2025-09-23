import os
import argparse
import torch
import logging
import pickle
import warnings
from rdkit import Chem
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from rdkit.Chem import Draw
from configs.dataset_config import get_dataset_info
from utils.misc import *
from utils.datasets import *
from utils.reconstruct import *
from utils.reconstruct_mdm import make_mol_openbabel
from utils.sample import construct_dataset_pocket
from utils.protein_process import PDBProtein
from utils.ligand_process import parse_sdf_file
from utils.data import ProteinLigandData, torchify_dict
from utils.transforms import *
from model.DrugRPG import DrugRPG
from utils.reconstruct_mol import * 
# from MolDiff.models.bond_predictor import BondPredictor
# from MolDiff2.models.bond_predictor import BondPredictor
from utils.reconstruct_utils import * 

# import utils.visualizer as vis
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
logging.getLogger().setLevel(logging.INFO)
ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable',
                 'ZnBinder']
FOLLOW_BATCH = ['ligand_atom', 'protein_atom']
atomic_numbers_crossdock = torch.LongTensor([6, 7, 8, 9, 15, 16, 17, 34])
atomic_numbers_pocket = torch.LongTensor([6, 7, 8, 9, 15, 16, 17, 34])

def save_sdf_file(mol, sdf_dir, gen_file_name):
    writer = Chem.SDWriter(os.path.join(sdf_dir, gen_file_name))
    writer.write(mol, confId=0)
    writer.close()

def get_data(pdb_path, sdf_path, center=0,):
    center = torch.FloatTensor(center)
    ptable = Chem.GetPeriodicTable()

    pocket_dict = PDBProtein(pdb_path).to_dict_atom()
    residue_dict = PDBProtein(pdb_path).to_dict_residue() 
    if sdf_path is None:
        data = ProteinLigandData.from_protein_ligand_dicts(
            protein_dict = torchify_dict(pocket_dict),
            residue_dict = torchify_dict(residue_dict)
        )
    else:
        
        ligand_data = torchify_dict(parse_sdf_file(sdf_path))
        data = ProteinLigandData.from_protein_ligand_dicts(
            protein_dict=torchify_dict(pocket_dict),
            ligand_dict=torchify_dict(ligand_data),
            residue_dict = torchify_dict(residue_dict))
    return data

def construct_dist_mat(protein_ligand_dist):
    tbl = Chem.GetPeriodicTable()
    ligand_atom_indices = MAP_INDEX_TO_ATOM_TYPE_ONLY

    # ligand_atom_indices = MAP_INDEX_TO_ATOM_TYPE_ONLY

    max_atom_index = max([tbl.GetAtomicNumber(atom) for atom_tuple in protein_ligand_dist for atom in atom_tuple]) + 1
    protein_ligand_dist_mat = np.ones((len(ligand_atom_indices), max_atom_index)) * 2
    ligand_atom_map = {}
    for ligand_atom_idx in ligand_atom_indices:
        ligand_atom = ligand_atom_indices[ligand_atom_idx]
        if ligand_atom not in ligand_atom_map: ligand_atom_map[ligand_atom] = []
        ligand_atom_map[ligand_atom].append(ligand_atom_idx)
    
    for atom1, atom2 in protein_ligand_dist:
        dist = protein_ligand_dist[(atom1, atom2)]
        atom1_num, atom2_num = tbl.GetAtomicNumber(atom1), tbl.GetAtomicNumber(atom2)
        if atom1_num in ligand_atom_map:
            for ligand_atom_idx in ligand_atom_map[atom1_num]:
                protein_ligand_dist_mat[ligand_atom_idx, atom2_num] = dist
        
        if atom2_num in ligand_atom_map:
            for ligand_atom_idx in ligand_atom_map[atom2_num]:
                protein_ligand_dist_mat[ligand_atom_idx, atom1_num] = dist            
    
    return protein_ligand_dist_mat

def generate(args):

    pdb_name = os.path.basename(args.pocket_path)[:4]

    # Load configs
    ckpt = torch.load(args.ckpt, weights_only=False)
    config = ckpt['config']

    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")

    seed_all(args.seed)
    log_dir = os.path.join(os.path.dirname(os.path.dirname(args.save_dir)), 'logging')

    output_dir = get_new_log_dir(log_dir, args.sampling_type, tag="result")
    logging.info("output_dir: %s", output_dir)

    logging.info('Loading {} data...'.format(config.dataset.name))
    atomic_numbers = atomic_numbers_pocket
    dataset_info = get_dataset_info('crossdock_pocket', False)

    transform = Compose([
        FeaturizeProteinAtom(),
        FeaturizeLigandAtom(),
        # FeaturizeProteinResidue(),
        # FeaturizeLigandBond(),
        GetAdj()
    ])

    dataset, subsets = get_dataset(args.data_path, args.split_data_path, args.split, transform=transform,)
    train_set, test_set = subsets['train'], subsets['test']
    test_set_selected = []
    # FOLLOW_BATCH = ['ligand_atom_type','protein_atom_feature_full']
    for i, data in enumerate(test_set):
        if not (args.start_idx <= i < args.end_idx): continue
        test_set_selected.append(data)
    print('Total sample {} proteins, for each generate {} ligands'.format(len(test_set_selected), args.num_samples))
        
    from typing import Literal
    logging.info('Building model...')
    config.model['ouroboros_path'] = 'configs/Ouroboros'
    model = DrugRPG(config.model, device=device).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    save_results = args.save_result
    save_sdf = args.save_sdf
    if save_sdf:
        sdf_dir = os.path.join(os.path.dirname(args.save_dir), 'generate_sdf')
        print('sdf idr:', sdf_dir)
    if save_results:
        if not os.path.exists(sdf_dir):
            os.mkdir(sdf_dir)

    
    batch_size = args.batch_size
    num_points = args.num_atom
    time_list = []
    protein_files = []
    results = []
    valid = 0
    valid_openbabel = 0
    stable = 0

    atom_pred_embeddings = {}

    for i, data in enumerate(tqdm(test_set_selected)):

        num_samples = args.num_samples
        if num_samples == 0:
            raise ValueError("num_samples can not be zero!")

        protein_atom = data['protein_atom'].float()
        protein_pos = data['protein_pos']
        protein_bond_index = data['protein_bond_index']
        # protein_bond_type = data['protein_bond_type']
        protein_num_nodes = data['protein_num_nodes']
    
        ligand_atom = data['ligand_element'].detach()
        ligand_pos = data['ligand_pos'].detach()
        # ligand_feat = data['ligand_atom_feature'].detach()

        protein_files.append(data['protein_filename'])

        f_dir, f_name = os.path.split(data['protein_filename'])
        gen_file_name = f_name.split('.')[0]
        pdb_name = f_name.split('_')[0]
        write_dir = os.path.join(sdf_dir, gen_file_name)
        os.makedirs(write_dir, exist_ok=True)
        print("PDB file name: ", f_name)
        # if protein_atom.shape[0] < 250:
        #     warnings.warn(
        #     f"The protein has too few atoms ({protein_atom.shape[0]}). "
        #     "This may lead to unreliable results. We recommend using proteins with **no fewer than 250 atoms**.",
        #     UserWarning
        #     )
        #     continue

        # rmol = reconstruct_from_generated(ligand_pos, ligand_atom, ligand_feat)
        # r_smile = Chem.MolToSmiles(rmol)
        # print("reference smile: ", r_smile)
        # if rmol.GetNumAtoms() < 14:
        #     warnings.warn(
        #     f"The ligand has too few atoms ({rmol.GetNumAtoms()}). "
        #     "Reference molecule should have no fewer than 14 atoms.",
        #     UserWarning
        #     )
        #     continue
        total_sample = 0


        with torch.no_grad():
            num_points = ligand_atom.size(0)
            t_pocket_start = time.time()

            while num_samples > 0:
                if total_sample > 500:
                    print("Maximum {} number of samples exceeded".format(total_sample))
                    break
                data_list, sample_nums = construct_dataset_pocket(num_samples, batch_size, dataset_info, num_points, num_points, None, None, 
                                            False, protein_atom, protein_pos, protein_bond_index, protein_num_nodes)
                batch = Batch.from_data_list(data_list[0], follow_batch=FOLLOW_BATCH).to(device)

                total_sample += sample_nums
                try:
                    ligand_atom, ligand_pos, ligand_edge, emb_dict = model.ddpm_sample(batch, device)
                    
                    ligand_atom_list = unbatch(ligand_atom, batch['ligand_atom_batch'])
                    ligand_pos_list = unbatch(ligand_pos, batch['ligand_atom_batch'])

                    ligand_num = batch['ligand_num_nodes']
                    for i in range(sample_nums):
                        try:
                            atom_type = ligand_atom_list[i].detach().cpu()
                            pos = ligand_pos_list[i].detach().cpu()
                            edge_type = ligand_edge[i][:ligand_num[i], :ligand_num[i]].detach().cpu()

                            new_element = torch.argmax(atom_type, dim=1)
                            mol = build_mol_with_edge(pos, new_element, edge_type, dataset_info)
                            # mol = make_mol_openbabel(pos, new_element, dataset_info)
                            # openbabel_smile = Chem.MolToSmiles(openbabel_mol)

                            pred_smile = Chem.MolToSmiles(mol)
                            
                            if pred_smile is not None:
                                if pred_smile.count('.') > 1:
                                    raise MolReconsError()
                                smiles, mol = valid_smile(mol, largest_mol_flag=False)
                                if smiles is None or smiles.count('.') > 0 or len(smiles) < 4:
                                    raise MolReconsError()
                                if "." not in smiles:
                                    stable += 1

                                if save_sdf:
                                    gen_file_name = '{}_{}.sdf'.format(pdb_name, str(num_samples))
                                    save_sdf_file(mol, write_dir, gen_file_name)
                                valid += 1
                                num_samples -= 1
                                atom_pred_embeddings[smiles] = emb_dict
                                if save_results:
                                    result = {'atom_type': atom_type.detach().cpu(),
                                            'pos': pos.detach().cpu(),
                                            'smile': smiles,
                                            'protein_file': data['protein_filename'],
                                            'ligand_file': data['ligand_filename'],
                                            'mol': mol,
                                            }
                                    results.append(result)
                                
                                print("generated with check",smiles)
                                print('Successfully generate molecule for {}, remaining {} samples generated'.format(
                                    pdb_name, num_samples))
                                if num_samples == 0:
                                    break
                            else:
                                raise MolReconsError()

                        except(RuntimeError, MolReconsError, TypeError, IndexError,
                                OverflowError):
                            print('Invalid, continue')
                            # traceback.print_exc()
                            # break
                        
                    
                except (FloatingPointError): 
                    logging.warning(
                        'Ignoring, because reconstruction error encountered or retrying with local clipping or vina error.')
                    print('Resample the number of the atoms and regenerate!')
            time_list.append(time.time() - t_pocket_start)
            logging.info('the {} sample takes {:.2f} seconds'.format(num_samples, time.time() - t_pocket_start))
            
    # logging.info('sa score is : {}'.format(np.mean(sa)))
    # logging.info('qed score is : {}'.format(np.mean(qeds)))
    # logging.info('Lp score is : {}'.format(np.mean(LPs)))

    if save_results:
        save_path = os.path.join(output_dir, 'samples_all.pkl')
        logging.info('Saving samples to: %s' % save_path)

        # save_smile_path = os.path.join(output_dir, 'samples_smile.pkl')

        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
            f.close()
        
        emb_save_path = os.path.join(output_dir, 'emb_all.pkl')
        with open(emb_save_path, 'wb') as f:
            pickle.dump(atom_pred_embeddings, f)
            f.close()
        # save_time_path = os.path.join(output_dir, 'time.pkl')
        # logging.info('Saving time to: %s' % save_path)
        # with open(save_time_path, 'wb') as f:
        #     pickle.dump(time_list, f)
        #     f.close()
    
    # save_path = os.path.join('/mnt/rna01/lzy/SBDD_final2', 'unique_protein_filenames.pkl')
    # with open(save_path, 'wb') as f:
    #     pickle.dump(protein_files, f)
    #     f.close()



        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ddim',help='using ddim or ddpm model for sampling')
    parser.add_argument('--pocket_path', type=str, default='/mnt/rna01/lzy/PMDM/data/crossdocked_pocket10/1A1C_MALDO_2_433_0/1m4n_A_rec_1m7y_ppg_lig_tt_min_0_pocket10.pdb')
    parser.add_argument('--sdf_path', type=str, default=None, help='path to the sdf file of reference ligand')
    # parser.add_argument('--config', type=str, default='./configs/train_config.yml')
    parser.add_argument('--num_atom', type=int, default=30)
    parser.add_argument('--data_path', type=str, default='/mnt/rna01/lzy/SBDD/data/crossdocked_pocket10')
    parser.add_argument('--split_data_path', type=str, default='/mnt/rna01/lzy/SBDD/data/split_by_name.pt')
    parser.add_argument('--build_method', type=str, default='reconstruct', help='build or reconstruct')
    parser.add_argument('--protein_ligand_dist', type=str, default='/mnt/rna01/lzy/Protein_base2/protein_ligand_dist.txt')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--cuda', type=str, default=True)
    parser.add_argument('--split', type=bool, default=True)
    parser.add_argument('--ckpt', type=str, help='path for loading the checkpoint')
    parser.add_argument('--bond_ckpt', type=str, help='path for loading the checkpoint')
    parser.add_argument('--save_sdf', type=bool, default=True)
    parser.add_argument('--save_result', type=bool, default=True)
    parser.add_argument('--save_dir', type=str, default='./sample_batch')
    parser.add_argument('--num_samples', type=int, default=20, help='generate sample number')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--check_rings', type=bool, default=True)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--clip', type=float, default=1000.0)
    parser.add_argument('--n_steps', type=int, default=1000,
                        help='sampling num steps; for DSM framework, this means num steps for each noise scale')
    
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=20)

    # Parameters for DDPM
    parser.add_argument('--sampling_type', type=str, default='generalized',
                        help='generalized, ddpm_noisy, ld: sampling method for DDIM, DDPM or Langevin Dynamics')
    parser.add_argument('--eta', type=float, default=0,
                        help='weight for DDIM and DDPM: 0->DDIM, 1->DDPM')
    args = parser.parse_args()

    generate(args)














