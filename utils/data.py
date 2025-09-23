import copy
import torch
import numpy as np
from torch_geometric.data import Data, Batch
# from torch_geometric.loader import DataLoader
# from torch.utils.data import Dataset

FOLLOW_BATCH = ['protein_element', 'ligand_context_element', 'pos_real', 'pos_fake']

def torchify_dict(data):
    output = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            output[k] = torch.from_numpy(v)
        else:
            output[k] = v
    return output

class ProteinLigandData(Data):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_protein_ligand_dicts(protein_dict=None, ligand_dict=None, residue_dict=None, **kwargs):
        instance = ProteinLigandData(**kwargs)

        if protein_dict is not None:
            for key, item in protein_dict.items():
                # print(key)
                instance['protein_' + key] = item     # 获取protein feature

        if residue_dict is not None:
            for key, item in residue_dict.items():
                instance['residue_' + key] = item

        if ligand_dict is not None:
            for key, item in ligand_dict.items():
                instance['ligand_' + key] = item      # 获取ligand feature
        
        # instance['ligand_nbh_list'] = {i.item(): [j.item() for k, j in enumerate(instance.ligand_bond_index[1]) if
        #                                           instance.ligand_bond_index[0, k].item() == i] for i in
        #                                instance.ligand_bond_index[0]}
        # print(instance)
        # exit(0)
        return instance

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'ligand_bond_index':
            if 'ligand_element' in self.keys():
                return self['ligand_element'].size(0)
            
            return self['ligand_atom'].size(0)
        if key == 'protein_bond_index':
            if 'protein_element' in self.keys():
                return self['protein_element'].size(0)
            return self['protein_atom'].size(0)
        if key == 'pocket_bond_index':
            if 'pocket_element' in self.keys():
                return self['pocket_element'].size(0)
            return self['pocket_atom_type'].size(0)
        elif key == 'ligand_context_bond_index':
            return self['ligand_context_element'].size(0)
        else:
            return super().__inc__(key, value)


def batch_from_data_list(data_list):
    return Batch.from_data_list(data_list, follow_batch=['ligand_element', 'protein_element'])


def torchify_dict(data):
    output = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            output[k] = torch.from_numpy(v)
        else:
            output[k] = v
    return output

def get_node_mask(node_num, pad_len, dtype=int):
    node_mask = torch.zeros(pad_len, dtype=dtype)
    node_mask[:node_num] = 1.
    return node_mask.unsqueeze(0)

def padding_data(x, pad_len):
    if x.dim() == 1:
        x_len = x.size(0)
        if x_len < pad_len:
            padding = x.new_zeros(pad_len - x_len, dtype=x.dtype) 
            x = torch.cat([x, padding]) 
        return x.unsqueeze(0)  

    elif x.dim() == 2:
        x_len, x_dim = x.size()
        if x_len < pad_len:
            new_x = x.new_zeros([pad_len, x_dim], dtype=x.dtype)  
            new_x[:x_len, :] = x 
            x = new_x
        return x.unsqueeze(0)  

    else:
        raise ValueError("Unsupported tensor dimension: x must be either 1D or 2D.")

def pad_edge(x, pad_len):

    x_len, _, x_dim = x.size()
    if x_len < pad_len:
        new_x = x.new_zeros([pad_len, pad_len, x_dim])
        new_x[:x_len, :x_len, :] = x
        x = new_x
    return x.unsqueeze(0)



def collate_mols(mol_dicts):
    data_batch = {}
    batch_size = len(mol_dicts)
    # ProteinLigandData(protein_element=[401], protein_pos=[401, 3], protein_is_backbone=[401], protein_atom_name=[401], 
    # protein_atom_to_aa_type=[401], residue_amino_acid=[46], residue_center_of_mass=[46, 3], residue_pos_CA=[46, 3], residue_len=[46], 
    # residue_attr=[46, 6], ligand_element=[18], ligand_pos=[18, 3], ligand_bond_index=[2, 38], ligand_bond_type=[38], 
    # ligand_center_of_mass=[3], ligand_atom_feature=[18, 8], protein_filename='HYES_HUMAN_230_551_0/4ocz_A_rec_4y2u_49r_lig_tt_min_0_pocket10.pdb',
    #  ligand_filename='HYES_HUMAN_230_551_0/4ocz_A_rec_4y2u_49r_lig_tt_min_0.sdf', id=14738, protein_atom=[401, 31], ligand_atom=[18, 8],
    #  ligand_atom_full=[18, 16], residue_atom=[46, 26], ligand_edge_feature=[18, 18, 2])
    for key in ['protein_pos', 'protein_element', 'protein_is_backbone', 'residue_amino_acid', 'residue_center_of_mass', 'residue_pos_CA', 'protein_atom_to_aa_type', 'residue_attr', 'ligand_element', 'ligand_bond_type']:
        data_batch[key] = torch.cat([mol_dict[key] for mol_dict in mol_dicts], dim=0)
    
    ligand_atom_num = torch.tensor([len(mol_dict['ligand_element']) for mol_dict in mol_dicts])
    protein_atom_num = torch.tensor([len(mol_dict['protein_element']) for mol_dict in mol_dicts])
    residue_num = torch.tensor([len(mol_dict['residue_amino_acid']) for mol_dict in mol_dicts])

    data_batch['ligand_atom_num'] = ligand_atom_num
    data_batch['protein_atom_num'] = protein_atom_num
    data_batch['residue_num'] = residue_num

    # for padding data
    data_batch['ligand_atom'] = torch.cat([padding_data(mol_dict['ligand_atom'], max(ligand_atom_num)) for mol_dict in mol_dicts])
    data_batch['protein_atom'] = torch.cat([padding_data(mol_dict['protein_atom'], max(protein_atom_num)) for mol_dict in mol_dicts])
    data_batch['residue_atom'] = torch.cat([padding_data(mol_dict['residue_atom'], max(residue_num)) for mol_dict in mol_dicts])

    data_batch['ligand_pos'] = torch.cat([padding_data(mol_dict['ligand_pos'], max(ligand_atom_num)) for mol_dict in mol_dicts])
    data_batch['protein_pos'] = torch.cat([padding_data(mol_dict['protein_pos'], max(protein_atom_num)) for mol_dict in mol_dicts])
    data_batch['residue_pos'] = torch.cat([padding_data(mol_dict['residue_center_of_mass'], max(residue_num)) for mol_dict in mol_dicts])

    data_batch['ligand_atom_mask'] =  torch.cat([get_node_mask(i, max(ligand_atom_num)) for i in ligand_atom_num])
    data_batch['protein_atom_mask'] =  torch.cat([get_node_mask(i, max(protein_atom_num)) for i in protein_atom_num])
    data_batch['residue_atom_mask'] =  torch.cat([get_node_mask(i, max(residue_num)) for i in residue_num])


    # for edge
    data_batch['ligand_edge_feature'] =  torch.cat([pad_edge(mol_dict['ligand_edge_feature'], max(ligand_atom_num) + max(protein_atom_num)) for mol_dict in mol_dicts])
    ligand_edge_mask = data_batch['ligand_atom_mask'].unsqueeze(1) * data_batch['ligand_atom_mask'].unsqueeze(2)
    ligand_diag_mask = ~torch.eye(ligand_edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    ligand_edge_mask *= ligand_diag_mask
    data_batch['ligand_edge_mask'] = ligand_edge_mask.reshape(-1, 1)

    # protein_edge_mask = data_batch['protein_atom_mask'].unsqueeze(1) * data_batch['protein_atom_mask'].unsqueeze(2)
    # protein_diag_mask = ~torch.eye(protein_edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    # protein_edge_mask *= protein_diag_mask
    # data_batch['protein_edge_mask'] = protein_edge_mask.reshape(-1, 1)
    # print( protein_edge_mask.reshape(-1, 1))


    data_batch['protein_filename'] = [mol_dict['protein_filename'] for mol_dict in mol_dicts]
    data_batch['ligand_filename'] = [mol_dict['ligand_filename'] for mol_dict in mol_dicts]


    return data_batch


    edge_num = torch.tensor([len(mol_dict['ligand_bond_type']) for mol_dict in mol_dicts])
    ligand_atom_num = torch.tensor([len(mol_dict['ligand_element']) for mol_dict in mol_dicts])
    data_batch['edge_batch'] = torch.repeat_interleave(torch.arange(batch_size), edge_num)
    data_batch['ligand_batch'] = torch.repeat_interleave(torch.arange(batch_size), ligand_atom_num)
    data_batch['ligand_bond_index'] = torch.cat([mol_dict['ligand_bond_index'] for mol_dict in mol_dicts], dim=1)
    # unsqueeze dim0
    for key in ['xn_pos', 'yn_pos', 'ligand_torsion_xy_index', 'y_pos']:
        cat_list = [mol_dict[key].unsqueeze(0) for mol_dict in mol_dicts if len(mol_dict[key]) > 0]
        if len(cat_list) > 0:
            data_batch[key] = torch.cat(cat_list, dim=0)
        else:
            data_batch[key] = torch.tensor([])
    # follow batch
    for key in ['protein_element', 'ligand_context_element', 'current_atoms', 'amino_acid', 'cand_mols']:
        repeats = torch.tensor([len(mol_dict[key]) for mol_dict in mol_dicts])
        data_batch[key + '_batch'] = torch.repeat_interleave(torch.arange(batch_size), repeats)
    for key in ['ligand_element_torsion']:
        repeats = torch.tensor([len(mol_dict[key]) for mol_dict in mol_dicts if len(mol_dict[key]) > 0])
        if len(repeats) > 0:
            data_batch[key + '_batch'] = torch.repeat_interleave(torch.arange(len(repeats)), repeats)
        else:
            data_batch[key + '_batch'] = torch.tensor([])

    # distance matrix prediction
    p_idx, q_idx = torch.cartesian_prod(torch.arange(4), torch.arange(2)).chunk(2, dim=-1)
    p_idx, q_idx = p_idx.squeeze(-1), q_idx.squeeze(-1)
    protein_offsets = torch.cumsum(data_batch['protein_element_batch'].bincount(), dim=0)
    ligand_offsets = torch.cumsum(data_batch['ligand_context_element_batch'].bincount(), dim=0)
    protein_offsets, ligand_offsets = torch.cat([torch.tensor([0]), protein_offsets]), torch.cat([torch.tensor([0]), ligand_offsets])
    ligand_idx, protein_idx = [], []
    for i, mol_dict in enumerate(mol_dicts):
        if len(mol_dict['true_dm']) > 0:
            protein_idx.append(mol_dict['dm_protein_idx'][p_idx] + protein_offsets[i])
            ligand_idx.append(mol_dict['dm_ligand_idx'][q_idx] + ligand_offsets[i])
    if len(ligand_idx) > 0:
        data_batch['dm_ligand_idx'], data_batch['dm_protein_idx'] = torch.cat(ligand_idx), torch.cat(protein_idx)

    # structure refinement (alpha carbon - ligand atom)
    sr_ligand_idx, sr_protein_idx = [], []
    for i, mol_dict in enumerate(mol_dicts):
        if len(mol_dict['true_dm']) > 0:
            ligand_atom_index = torch.arange(len(mol_dict['ligand_context_pos']))
            p_idx, q_idx = torch.cartesian_prod(torch.arange(len(mol_dict['ligand_context_pos'])), torch.arange(len(mol_dict['protein_alpha_carbon_index']))).chunk(2, dim=-1)
            p_idx, q_idx = p_idx.squeeze(-1), q_idx.squeeze(-1)
            sr_ligand_idx.append(ligand_atom_index[p_idx] + ligand_offsets[i])
            sr_protein_idx.append(mol_dict['protein_alpha_carbon_index'][q_idx] + protein_offsets[i])
    if len(ligand_idx) > 0:
        data_batch['sr_ligand_idx'], data_batch['sr_protein_idx'] = torch.cat(sr_ligand_idx).long(), torch.cat(sr_protein_idx).long()

    # structure refinement (ligand atom - ligand atom)
    sr_ligand_idx0, sr_ligand_idx1 = [], []
    for i, mol_dict in enumerate(mol_dicts):
        if len(mol_dict['true_dm']) > 0:
            ligand_atom_index = torch.arange(len(mol_dict['ligand_context_pos']))
            p_idx, q_idx = torch.cartesian_prod(torch.arange(len(mol_dict['ligand_context_pos'])), torch.arange(len(mol_dict['ligand_context_pos']))).chunk(2, dim=-1)
            p_idx, q_idx = p_idx.squeeze(-1), q_idx.squeeze(-1)
            sr_ligand_idx0.append(ligand_atom_index[p_idx] + ligand_offsets[i])
            sr_ligand_idx1.append(ligand_atom_index[q_idx] + ligand_offsets[i])
    if len(ligand_idx) > 0:
        data_batch['sr_ligand_idx0'], data_batch['sr_ligand_idx1'] = torch.cat(sr_ligand_idx0).long(), torch.cat(sr_ligand_idx1).long()
    # index
    if len(data_batch['y_pos']) > 0:
        repeats = torch.tensor([len(mol_dict['ligand_element_torsion']) for mol_dict in mol_dicts if len(mol_dict['ligand_element_torsion']) > 0])
        offsets = torch.cat([torch.tensor([0]), torch.cumsum(repeats, dim=0)])[:-1]
        data_batch['ligand_torsion_xy_index'] += offsets.unsqueeze(1)

    offsets1 = torch.cat([torch.tensor([0]), torch.cumsum(data_batch['num_atoms'], dim=0)])[:-1]
    data_batch['current_atoms'] += torch.repeat_interleave(offsets1, data_batch['current_atoms_batch'].bincount())
    # cand mols: torch geometric Data
    cand_mol_list = []
    for data in mol_dicts:
        if len(data['cand_labels']) > 0:
            cand_mol_list.extend(data['cand_mols'])
    if len(cand_mol_list) > 0:
        data_batch['cand_mols'] = Batch.from_data_list(cand_mol_list)
    return data_batch


def collate_mols_simple(mol_dicts):
    data_batch = {}
    batch_size = len(mol_dicts)
    for key in ['protein_pos', 'protein_atom_feature', 'ligand_pos', 'ligand_atom_feature_full', 'vina']:
        data_batch[key] = torch.cat([mol_dict[key] for mol_dict in mol_dicts], dim=0).float()

    # follow batch
    for key in ['protein_element', 'ligand_element']:
        repeats = torch.tensor([len(mol_dict[key]) for mol_dict in mol_dicts])
        data_batch[key + '_batch'] = torch.repeat_interleave(torch.arange(batch_size), repeats)

    return data_batch