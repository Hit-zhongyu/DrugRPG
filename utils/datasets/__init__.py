import torch
import os
from torch.utils.data import Subset

from .pl import PocketLigandPairDataset
from torch_geometric.transforms import Compose as TorchCompose
import random

def get_dataset(data_path, split_data_path, split, *args, **kwargs):

    root = data_path
    # name = config.name

    dataset = PocketLigandPairDataset(root, *args, **kwargs)
    if split:  # 划分数据
        split_by_name = torch.load(split_data_path, weights_only=True)
        
        split = {
            k: [dataset.name2id[n] for n in names if n in dataset.name2id]
            for k, names in split_by_name.items()
        }

        subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
        return dataset, subsets
    else:
        return dataset

# class Compose(TorchCompose):
#     def __call__(self, data, new_data=None):
#         for t in self.transforms:
#             data, new_data = t(data, new_data)
#         return data, new_data

# def collate_fn(batch):
#     # print(batch[0].keys())
#     # batch = {prop: batch_stack([mol[prop] for mol in batch])
#     #          for prop in batch[0].keys()if prop not in ['protein_filename', 'ligand_filename']}
#     batch = {prop: batch_stack([mol[prop] for mol in batch])
#              for prop in batch[0].keys()}
#     # batched = {}
#     # pos = ['ligand_pos','protein_pos','residue_pos']

#     # for prop in batch[0].keys():
#     #     prop_values = []
#     #     for mol in batch:
#     #         prop_values.append(mol[prop])
#     #     if prop in pos:
#     #         stacked_prop = batch_stack(prop_values, padding_value=1e10)
#     #     else:
#     #         stacked_prop = batch_stack(prop_values, padding_value=0)
#     #     batched[prop] = stacked_prop

#     return batch

# class GeomDrugsDataLoader(DataLoader):
#     def __init__(self, dataset, batch_size, shuffle=False, sampler=None, drop_last=False):

#         super().__init__(dataset, batch_size, sampler=sampler, shuffle=shuffle,
#                             collate_fn=collate_fn, drop_last=drop_last)

# def batch_stack(props, padding_value=0):
#     """
#     Stack a list of torch.tensors so they are padded to the size of the
#     largest tensor along each axis.

#     Parameters
#     ----------
#     props : list of Pytorch Tensors
#         Pytorch tensors to stack

#     Returns
#     -------
#     props : Pytorch tensor
#         Stacked pytorch tensor.

#     Notes
#     -----
#     TODO : Review whether the behavior when elements are not tensors is safe.
#     """
#     if not torch.is_tensor(props[0]):
#         return torch.tensor(props)
#     elif props[0].dim() == 0:
#         return torch.stack(props)
#     else:
#         return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=padding_value)
