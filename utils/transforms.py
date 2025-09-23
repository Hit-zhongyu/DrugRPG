import sys
sys.path.append("..")
import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from torch_geometric.nn import radius_graph

from torch_geometric.nn.pool import knn_graph
from torch_geometric.utils import subgraph
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.data import Data, Batch
from torch_scatter import scatter_add
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem

from .data import ProteinLigandData
from .ligand_process import ATOM_FAMILIES

from .misc import get_adj_matrix
from torch_geometric.nn import radius_graph

class FeaturizeProteinAtom(object):

    def __init__(self):
        super().__init__()
        # self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 16, 34])    # H, C, N, O, S, Se
        # self.atomic_numbers = torch.LongTensor([6, 7, 8, 16, 34])  # H, C, N, O, S, Se
        self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 9, 15, 16, 17, 34, 119])
        self.max_num_aa = 20

    @property
    def feature_dim(self):
        return self.atomic_numbers.size(0) + self.max_num_aa + 1


    def __call__(self, data: ProteinLigandData):

        element = data['protein_element'].view(-1, 1) == self.atomic_numbers.view(1, -1) 
        amino_acid = F.one_hot(data['protein_atom_to_aa_type'], num_classes=self.max_num_aa)
        is_backbone = data['protein_is_backbone'].view(-1, 1).long().contiguous()
        x = torch.cat([element, amino_acid, is_backbone], dim=-1)
        data['protein_atom'] = x 

        data['protein_num_nodes'] = len(data['protein_element'])
        return data

# class FeaturizeProteinResidue(object):

#     def __init__(self):
#         super().__init__()
#         self.max_num_aa = 20

#     def __call__(self, data: ProteinLigandData):
#         # element = self.atom2onehot(data['residue_atoms_type'])
#         amino_acid = F.one_hot(data['residue_amino_acid'], num_classes=self.max_num_aa)
#         # amino_acid = amino_acid /  self.max_num_aa
#         attr = data['residue_attr'] 
#         x = torch.cat([amino_acid, attr], dim=-1)
#         data['residue_atom'] = x 
#         data['residue_num_nodes'] = len(data['residue_amino_acid'])
#         return data

class FeaturizeLigandAtom(object):

    def __init__(self):
        super().__init__()
        # self.atomic_numbers = torch.LongTensor([1,6,7,8,9,15,16,17])  # H C N O F P S Cl
        self.atomic_numbers = torch.LongTensor([6, 7, 8, 9, 15, 16, 17, 34])  #H C N O F P S Cl

    @property
    def num_properties(self):
        return len(ATOM_FAMILIES)

    @property
    def feature_dim(self):
        return self.atomic_numbers.size(0) + len(ATOM_FAMILIES)

    def __call__(self, data: ProteinLigandData):
        element = data['ligand_element'].view(-1, 1) == self.atomic_numbers.view(1, -1)  # (N_atoms, N_elements)
        # element = element / len(self.atomic_numbers)
        # atom_feat = data['ligand_atom_feature'] 
        # x = torch.cat([element,atom_feat], dim=-1)
        data['ligand_atom'] = element
        # data['ligand_atom_full'] = x
        data['ligand_num_nodes'] = len(data['ligand_element'])

        ligand_num_nodes = data['ligand_num_nodes']
        protein_num_nodes = data['protein_num_nodes']
        data['num_nodes'] = ligand_num_nodes + protein_num_nodes
        # data['mol'] = data['ligand_mol']
        return data

class FeaturizeLigandBond(object):
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
    
    def __call__(self, data: ProteinLigandData):
        # print(type(data['ligand_bond_type']))
        # exit(0)
        # ligand_bond = torch.tensor(data['ligand_bond_type'])
        # data['ligand_bond_type'] = F.one_hot(ligand_bond, num_classes=self.num_classes)

        data['ligand_bond_type'] = F.one_hot(data['ligand_bond_type'], num_classes=self.num_classes)
        return data 

class GetAdj(object):
    def __init__(self, cutoff=3.5, protein_cutoff=4, num_classes=7):
        super().__init__()
        self.cutoff = cutoff
        self.protein_cutoff = protein_cutoff
        self.num_classes = num_classes

    def __call__(self, data: ProteinLigandData):

        ligand_num_nodes = data['ligand_num_nodes']
        protein_num_nodes = data['protein_num_nodes']
        data['num_nodes'] = ligand_num_nodes + protein_num_nodes
        # ligand_pos = data['ligand_pos']
        # ligand_index = data['ligand_bond_index']
        # ligand_bond_type = data['ligand_bond_type']

        # ligand_mark = torch.ones((ligand_num_nodes,))
        # protein_mark = torch.zeros((protein_num_nodes,))
        # mask = torch.concat([ligand_mark, protein_mark], dim=0).bool()

        protein_pos = data['protein_pos']
        protein_index = knn_graph(protein_pos, k=16, batch=None, loop=False)
        protein_index = protein_index[[1, 0], :]
        # protein_bond_type = torch.ones(protein_index.size(1), dtype=torch.long) + 4
        # protein_bond_type = protein_bond_type / 6.
        
        data['protein_bond_index'] = protein_index
        # data['protein_bond_type'] = protein_bond_type 

        # pos = torch.concat([ligand_pos, protein_pos], dim=0)
        # index = radius_graph(pos, self.cutoff, batch=None, loop=False)
        # index = index[[1, 0], :]
        # ligand_protein_mask = (mask[index[0]] & ~mask[index[1]]) | (~mask[index[0]] & mask[index[1]])
        # ligand_protein_index = index[:, ligand_protein_mask]
        # ligand_protein_bond_type =  torch.ones(ligand_protein_index.size(1), dtype=torch.long) + 5
        # # ligand_protein_bond_type = ligand_protein_bond_type / 6.

        # edge_index = torch.cat([ligand_index, ligand_protein_index, protein_index + ligand_num_nodes], dim=-1)
        # bond_type = torch.cat([ligand_bond_type, ligand_protein_bond_type, protein_bond_type], dim=0)
        
        # data['edge_index'] = edge_index
        # data['bond_type'] = bond_type / 6. 

        return data


# This is for edge embbedding use one-hot
# class GetAdj(object):
#     def __init__(self, cutoff=3.5, protein_cutoff=2, num_classes=7):
#         super().__init__()
#         self.cutoff = cutoff
#         self.protein_cutoff = protein_cutoff
#         self.num_classes = num_classes

#     def __call__(self, data: ProteinLigandData):

#         ligand_num_nodes = data['ligand_num_nodes']
#         protein_num_nodes = data['protein_num_nodes']
#         data['num_nodes'] = ligand_num_nodes + protein_num_nodes
#         ligand_pos = data['ligand_pos']
#         ligand_index = data['ligand_bond_index']
#         ligand_bond_type = data['ligand_bond_type']

#         ligand_mark = torch.ones((ligand_num_nodes,))
#         protein_mark = torch.zeros((protein_num_nodes,))
#         mask = torch.concat([ligand_mark, protein_mark], dim=0).bool()

#         protein_pos = data['protein_pos']
#         protein_index = radius_graph(protein_pos, self.protein_cutoff, batch=None, loop=False)
#         protein_index = protein_index[[1, 0], :]
#         protein_bond_type = torch.ones(protein_index.size(1), dtype=torch.long) + 4
#         protein_bond_type = protein_bond_type
        
#         data['protein_bond_index'] = protein_index
#         data['protein_bond_type'] = protein_bond_type 

#         pos = torch.concat([ligand_pos, protein_pos], dim=0)
#         index = radius_graph(pos, self.cutoff, batch=None, loop=False)
#         index = index[[1, 0], :]
#         ligand_protein_mask = (mask[index[0]] & ~mask[index[1]]) | (~mask[index[0]] & mask[index[1]])
#         ligand_protein_index = index[:, ligand_protein_mask]
#         ligand_protein_bond_type =  torch.ones(ligand_protein_index.size(1), dtype=torch.long) + 5
#         ligand_protein_bond_type = ligand_protein_bond_type

#         edge_index = torch.cat([ligand_index, ligand_protein_index, protein_index + ligand_num_nodes], dim=-1)
#         bond_type = torch.cat([ligand_bond_type, ligand_protein_bond_type, protein_bond_type], dim=0)
        
#         data['edge_index'] = edge_index
#         data['bond_type'] = bond_type  #F.one_hot(bond_type, num_classes=self.num_classes)

#         return data


# class FeaturizeLigandBond(object):

#     def __init__(self, include_aromatic=False):
#         super().__init__()
#         self.include_aromatic = include_aromatic

#     def __call__(self, data: ProteinLigandData):

#         data['ligand_bond_feature'] = F.one_hot((data['ligand_bond_type'] - 1) % 3, num_classes=3)  # (1,2,3) to (0,1,2)-onehot
        
#         return data


class CountNodesPerGraph(object):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data):
        data.ligand_num_nodes_per_graph = torch.LongTensor([data.ligand_element.size(0)])
        data.ligand_nodes = torch.LongTensor([data.ligand_element.size(0)])
        data.protein_atoms_nodes = torch.LongTensor([data.protein_element.size(0)])
        # data.residue_amino_acid =  torch.tensor(data.residue_amino_acid, dtype=int)
        data.protein_residue_nodes = torch.LongTensor([data.residue_amino_acid.size(0)])
        # print(data.ligand_element.size(0))
        return data
    

# class GetAdj(object):
#     def __init__(self, cutoff=None, only_prot=False) -> None:
#         super().__init__()
#         self.cutoff = cutoff
#         self.only_prot = only_prot

#     def __call__(self, data):
#         '''
#         full connected edges or radius edges
#         '''
#         ligand_n_particles = data.ligand_nodes
#         if not self.only_prot:
#             if self.cutoff is None:
#                 ligand_adj = get_adj_matrix(ligand_n_particles)
#             else:
#                 ligand_adj = radius_graph(data.ligand_pos, self.cutoff, batch=None, loop=False)
#             data.ligand_bond_index = ligand_adj
#             ligand_bond_type = torch.ones(ligand_adj.size(1), dtype=int) * 2
#             data.ligand_bond_type = ligand_bond_type
        
#         data.ligand_edge_index = data.ligand_bond_index
#         data.ligand_edge_type = data.ligand_bond_type

#         # protein_n_particles = data.protein_pos.size(0)
#         # # protein_adj = get_adj_matrix(protein_n_particles)
#         # data.protein_pos = torch.tensor(data.protein_pos, dtype=torch.float)
#         protein_adj = radius_graph(data.protein_pos, 2.5, batch=None, loop=False)
#         protein_bond_type = torch.ones(protein_adj.size(1), dtype=int) * 4  # define the protien edge type as 2
#         data.protein_bond_index = protein_adj
#         data.protein_bond_type = protein_bond_type

#         # data.residue_pos = torch.tensor(data.residue_pos_CA, dtype=torch.float)
#         residue_adj = radius_graph(data.residue_pos_CA, 6, batch=None, loop=False)
#         residue_bond_type = torch.ones(residue_adj.size(1), dtype=int) * 4 
#         data.residue_bond_index = residue_adj
#         data.residue_bond_type = residue_bond_type

#         return data