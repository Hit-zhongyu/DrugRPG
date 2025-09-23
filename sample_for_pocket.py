import os
import argparse
import torch
import logging
import pickle
import warnings
from rdkit import Chem
from torch_geometric.data import Batch
from Bio import BiopythonWarning
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Selection import unfold_entities
from torch_geometric.transforms import Compose
from configs.dataset_config import get_dataset_info
from utils.misc import *
from utils.datasets import *
from utils.reconstruct import *
from utils.sample import construct_dataset_pocket
from utils.protein_process import PDBProtein
from utils.ligand_process import parse_sdf_file
from utils.data import ProteinLigandData, torchify_dict
from utils.transforms import *
from model.DrugRPG import DrugRPG
from utils.reconstruct_mol import * 
from utils.protein_process import *
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

def pdb_to_pocket_data(pdb_path, center=0, bbox_size=0):
    center = torch.FloatTensor(center)
    warnings.simplefilter('ignore', BiopythonWarning)
    ptable = Chem.GetPeriodicTable()
    parser = PDBParser()
    model = parser.get_structure(None, pdb_path)[0]

    protein_dict = EasyDict({
        'element': [],
        'pos': [],
        'is_backbone': [],
        'atom_to_aa_type': [],
    })
    for atom in unfold_entities(model, 'A'):
        res = atom.get_parent()
        resname = res.get_resname()
        if resname == 'MSE': resname = 'MET'
        if resname not in AA_NAME_NUMBER: continue   # Ignore water, heteros, and non-standard residues.

        element_symb = atom.element.capitalize()
        if element_symb == 'H': continue
        x, y, z = atom.get_coord()
        pos = torch.FloatTensor([x, y, z])
        # if (pos - center).abs().max() > (bbox_size / 2): 
        #     continue

        protein_dict['element'].append( ptable.GetAtomicNumber(element_symb))
        protein_dict['pos'].append(pos)
        protein_dict['is_backbone'].append(atom.get_name() in ['N', 'CA', 'C', 'O'])
        protein_dict['atom_to_aa_type'].append(AA_NAME_NUMBER[resname])
        
    # if len(protein_dict['element']) == 0:
    #     raise ValueError('No atoms found in the bounding box (center=%r, size=%f).' % (center, bbox_size))

    protein_dict['element'] = torch.LongTensor(protein_dict['element'])
    protein_dict['pos'] = torch.stack(protein_dict['pos'], dim=0)
    protein_dict['is_backbone'] = torch.BoolTensor(protein_dict['is_backbone'])
    protein_dict['atom_to_aa_type'] = torch.LongTensor(protein_dict['atom_to_aa_type'])

    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict = protein_dict,
        ligand_dict = {
            'element': torch.empty([0,], dtype=torch.long),
            'pos': torch.empty([0, 3], dtype=torch.float),
            'atom_feature': torch.empty([0, 8], dtype=torch.float),
            'bond_index': torch.empty([2, 0], dtype=torch.long),
            'bond_type': torch.empty([0,], dtype=torch.long),
        }
    )
    return data

def generate(args):

    pdb_name = os.path.basename(args.pdb_path)[:4]
    protein_filename = os.path.basename(args.pdb_path)

    # Load configs
    ckpt = torch.load(args.ckpt, weights_only=False)
    config = ckpt['config']

    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")

    seed_all(args.seed)
    log_dir = os.path.join(os.path.dirname(os.path.dirname(args.save_dir)), 'custom_pdb')

    output_dir = get_new_log_dir(log_dir, args.sampling_type, tag="result")
    logging.info("output_dir: %s", output_dir)

    logging.info('Loading {} data...'.format(config.dataset.name))
    dataset_info = get_dataset_info('crossdock_pocket', False)

    transform = Compose([
        FeaturizeProteinAtom(),
        FeaturizeLigandAtom(),
        GetAdj()
    ])

    data = pdb_to_pocket_data(args.pdb_path)
    data = transform(data)

    logging.info('Building model...')
    # logging.info(f'Config Model: {config.model}')
    config.model['ouroboros_path'] = 'configs/Ouroboros'
    model = DrugRPG(config.model, device=device).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    save_results = args.save_result
    save_sdf = args.save_sdf
    if save_sdf:
        sdf_dir = os.path.join(os.path.dirname(args.save_dir), 'generate_pocket')
        print('sdf idr:', sdf_dir)
    if save_results:
        if not os.path.exists(sdf_dir):
            os.mkdir(sdf_dir)

    
    _, f_name = os.path.split(protein_filename)
    gen_file_name = f_name.split('.')[0]
    write_dir = os.path.join(sdf_dir, gen_file_name)
    os.makedirs(write_dir, exist_ok=True)
    
    batch_size = args.batch_size
    num_samples = args.num_samples
    num_atom = args.num_atom
    results = []
    valid = 0
    stable = 0

    protein_atom = data['protein_atom'].float()
    protein_atom = data['protein_atom'].float()
    protein_pos = data['protein_pos']
    protein_bond_index = data['protein_bond_index']
    protein_num_nodes = data['protein_num_nodes']

    data_list, sample_nums = construct_dataset_pocket(num_samples, batch_size, dataset_info, num_atom, num_atom, None, None, 
                                            False, protein_atom, protein_pos, protein_bond_index, protein_num_nodes)


    for i, datas in enumerate(tqdm(data_list)):
        batch = Batch.from_data_list(datas, follow_batch=FOLLOW_BATCH).to(device)

        with torch.no_grad():
            while num_samples > 0:
                try:
                    ligand_atom, ligand_pos, ligand_edge = model.ddpm_sample(batch, device)
                    
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

                                if save_results:
                                    result = {'atom_type': atom_type.detach().cpu(),
                                            'pos': pos.detach().cpu(),
                                            'smile': smiles,
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
            

    if save_results:
        save_path = os.path.join(output_dir, 'samples_all.pkl')
        logging.info('Saving samples to: %s' % save_path)

        # save_smile_path = os.path.join(output_dir, 'samples_smile.pkl')

        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
            f.close()

        # save_time_path = os.path.join(output_dir, 'time.pkl')
        # logging.info('Saving time to: %s' % save_path)
        # with open(save_time_path, 'wb') as f:
        #     pickle.dump(time_list, f)
        #     f.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ddim',help='using ddim or ddpm model for sampling')
    parser.add_argument('--pdb_path', type=str, default='/mnt/rna01/lzy/SBDD_final2/data/IL11_ligand_pocket10.pdb')
    parser.add_argument('--sdf_path', type=str, default=None, help='path to the sdf file of reference ligand')
    parser.add_argument('--num_atom', type=int, default=30)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--cuda', type=str, default=True)
    parser.add_argument('--split', type=bool, default=True)
    parser.add_argument('--ckpt', type=str, help='path for loading the checkpoint')
    parser.add_argument('--save_sdf', type=bool, default=True)
    parser.add_argument('--save_result', type=bool, default=True)
    parser.add_argument('--save_dir', type=str, default='./sample_batch')
    parser.add_argument('--num_samples', type=int, default=1000, help='generate sample number')
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--check_rings', type=bool, default=True)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--tag', type=str, default='')

    # Parameters for DDPM
    parser.add_argument('--sampling_type', type=str, default='Pokcet')
    args = parser.parse_args()

    generate(args)














