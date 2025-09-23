import os
import pickle
# import sys
import lmdb
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from ..data import ProteinLigandData, torchify_dict
from ..protein_process import PDBProtein
from ..ligand_process2 import parse_sdf_file

import traceback

class PocketLigandPairDataset(Dataset):

    def __init__(self, raw_path, transform=None):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')

        self.file_path = self.raw_path
        self.index_path = os.path.join(self.raw_path, 'index.pkl')  # crossdock 'crossdock_cutoff/'+
        # self.processed_path = os.path.join(os.path.dirname(self.raw_path) , 'crossdocked_pocket10/'+  os.path.basename(
            # self.raw_path) + '_processed_dis10_sa&seq.lmdb')
        # self.name2id_path = os.path.join(os.path.dirname(self.raw_path), 'crossdocked_pocket10/' + 
                                            #  os.path.basename(self.raw_path) + '_processed_dis10_sa.pt')
        self.processed_path = os.path.join(os.path.dirname(self.raw_path) , 'crossdocked_pocket10/'+  os.path.basename(
            self.raw_path) + '_processed_ouroboros2.lmdb')
        self.name2id_path = os.path.join(os.path.dirname(self.raw_path), 'crossdocked_pocket10/' + 
                                             os.path.basename(self.raw_path) + '_processed_ouroboros2.pt')

        # self.name2id_path = os.path.join(os.path.dirname(self.raw_path), os.path.basename(self.raw_path) + '_name2id.pt')
        self.transform = transform
        self.db = None
        self.keys = None

        if not os.path.exists(self.processed_path):
            self._process()
            self._precompute_name2id()
        if not os.path.exists(self.name2id_path):
            self._precompute_name2id()
        self.name2id = torch.load(self.name2id_path, weights_only=True)
        # print(self.name2id)
        # exit(0)

    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None

    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)
        # all_coords_zeros = []
        # all_coords_pocker = []
        # all_coords_protein = []

        num_skipped = 0
        invalid_num = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, (pocket_fn, ligand_fn, _, rmsd_str) in enumerate(tqdm(index)):
                if pocket_fn is None: continue
                import traceback
                try:
                    ligand_dict = parse_sdf_file(os.path.join(self.raw_path, ligand_fn))
                    ligand_pos = ligand_dict['pos']

                    pocket_dict = PDBProtein(os.path.join(self.raw_path, pocket_fn)).to_dict_atom(ligand_pos, distance=10)
                    # residue_dict = PDBProtein(os.path.join(self.raw_path, pocket_fn)).to_dict_residue()

                    # protein_pos = pocket_dict['pos']
                    # if protein_pos.size != 0 and ligand_pos.size != 0:
                    #     protein_pos = protein_pos - np.mean(ligand_pos, axis=0)
                    #     ligand_pos = ligand_pos - np.mean(ligand_pos, axis=0)

                        # all_coords_protein.append(protein_pos)
                        # all_coords_protein.append(ligand_pos)

                    data = ProteinLigandData.from_protein_ligand_dicts(
                        protein_dict=torchify_dict(pocket_dict), 
                        # residue_dict=torchify_dict(residue_dict),
                        ligand_dict=torchify_dict(ligand_dict),
                    )   # data 实际上是获取pocket 以及ligand信息 以及 邻居信息

                    data.protein_filename = pocket_fn
                    data.ligand_filename = ligand_fn
                    # assert data.protein_pos.size(0) > 0
                    txn.put(
                        key=str(i).encode(),
                        value=pickle.dumps(data)
                    )
                except:
                    num_skipped += 1
                    print('Skipping (%d) %s' % (num_skipped, ligand_fn,))
                    traceback.print_exc()
                    break
                    # continue

        # all_coords_protein_ = np.vstack(all_coords_protein)
        # std_dev_protein = np.std(all_coords_protein_, axis=0)
        # print("Standard Deviations for protein coordinates:", std_dev_protein)

        # overall_std_protein = np.std(all_coords_protein_.ravel())
        # print("Standard Deviations for protein coordinates:", overall_std_protein)

        db.close()


    def _precompute_name2id(self):
        name2id = {}
        for i in tqdm(range(self.__len__()), 'Indexing'):
            # if i<63340:
            #     continue
            try:
                data = self.__getitem__(i)
                
            except AssertionError as e:
                print(i, e)
                continue
            name = (data['protein_filename'], data['ligand_filename'])
            
            name2id[name] = i
        # print(name2id)
        # print(self.name2id_path)
        torch.save(name2id, self.name2id_path)

    def __len__(self):
        if self.db is None:
            self._connect_db()
        # print(len(self.keys))
        return len(self.keys)

    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        # print(idx)
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data.id = idx

        assert data.protein_pos.size(0) > 0
        if self.transform is not None:
            data = self.transform(data)
        
        # exit(0)
        return data


if __name__ == '__main__':
    import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument('path', type=str)
    # args = parser.parse_args()
    path = '/home/user/ydliu/dif_condition/data/'
    dataset = PocketLigandPairDataset(path)