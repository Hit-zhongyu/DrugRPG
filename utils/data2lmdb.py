import os
import torch
import lmdb
import pickle
import numpy as np


from tqdm.auto import tqdm
from ligand_process import parse_sdf_file
from protein_process import PDBProtein
from torch_geometric.data import Data

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
    def from_protein_ligand_dicts(protein_dict=None, ligand_dict=None, **kwargs):
        instance = ProteinLigandData(**kwargs)

        if protein_dict is not None:
            for key, item in protein_dict.items():
                instance['protein_' + key] = item     # 获取protein feature

        if ligand_dict is not None:
            for key, item in ligand_dict.items():
                instance['ligand_' + key] = item      # 获取ligand feature
        
        # instance['ligand_nbh_list'] = {i.item(): [j.item() for k, j in enumerate(instance.ligand_bond_index[1]) if
        #                                           instance.ligand_bond_index[0, k].item() == i] for i in
        #                                instance.ligand_bond_index[0]}
        return instance

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'ligand_bond_index':
            if 'ligand_element' in self.keys():
                return self['ligand_element'].size(0)
            return self['ligand_atom_feature'].size(0)
        if key == 'protein_bond_index':
            if 'protein_element' in self.keys():
                return self['protein_element'].size(0)
            return self['protein_atom_feature_full'].size(0)
        if key == 'pocket_bond_index':
            if 'pocket_element' in self.keys():
                return self['pocket_element'].size(0)
            return self['pocket_atom_type'].size(0)
        elif key == 'ligand_context_bond_index':
            return self['ligand_context_element'].size(0)
        else:
            return super().__inc__(key, value)

def data2lmdb(path, lmdb_path):
    index_path = os.path.join(path, 'index_test.pkl')
    # lmdb_path = os.path.join(os.path.dirname(path), 'dataset/' + 'data_processed.lmdb')
    db = lmdb.open(
            lmdb_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
    with open(index_path, 'rb') as f:
        index = pickle.load(f)
    # print(index)
    
    num_skipped = 0
    with db.begin(write=True, buffers=True) as txn:
        for i, (pocket_fn, ligand_fn, _, _) in enumerate(tqdm(index)):
            if pocket_fn is None: continue
            try:
                # print(os.path.join(path, ligand_fn))
                ligand_dict = parse_sdf_file(os.path.join(path, ligand_fn))
                # print(ligand_dict)
                ligand_pos = ligand_dict['pos']
                #   pocket_dict = PDBProtein(os.path.join(self.raw_path, pocket_fn)).to_dict_atom()
                pocket_dict = PDBProtein(os.path.join(path, pocket_fn)).select_residue(ligand_pos, 8.0)   # 筛选出与ligand可能有作用的残基
                # print(ligand_dict)
                # print(pocket_dict)
                # exit(0)
                
                data = ProteinLigandData.from_protein_ligand_dicts(
                        protein_dict=torchify_dict(pocket_dict),   # 看到这里，  protein看完了   明天继续看ligand
                        ligand_dict=torchify_dict(ligand_dict),
                )
                # print(data)
                data.protein_filename = pocket_fn
                data.ligand_filename = ligand_fn
                txn.put(
                    key=str(i).encode(),
                    value=pickle.dumps(data)
                )
            except:
                num_skipped += 1
                print('Skipping (%d) %s' % (num_skipped, ligand_fn,))
                continue
    db.close()






if __name__ == '__main__':
    import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument('path', type=str)
    # args = parser.parse_args()
    path = '/home/user/ydliu/dif_condition/data'
    dataset = data2lmdb(path)
    # data = read_data(path)