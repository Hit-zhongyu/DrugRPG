import numpy as np
import torch
from torch.distributions.categorical import Categorical
from torch_geometric.data import Data
from tqdm import tqdm

from .data import ProteinLigandData
from .misc import get_adj_matrix
from torch.utils.data import Dataset
from utils.datasets import *

from .sample_atom_num import *
# n_nodes= {5: 3393, 6: 4848, 4: 9970, 2: 13832, 3: 9482,
#             8: 150, 1: 13364, 7: 53, 9: 48, 
#                 10: 26, 12: 25}
# n_nodes= {5: 193930, 6: 4848, 4: 39700, 2: 13832, 3: 9482,
#              1: 13364, 7: 53}
n_nodes= {3: 1, 4: 1,5:1,6:1}

class DistributionNodes:
    def __init__(self, histogram):
        self.n_nodes = []
        prob = []
        self.keys = {}
        for i, nodes in enumerate(histogram):
            self.n_nodes.append(nodes)
            self.keys[nodes] = i
            prob.append(histogram[nodes])
        self.n_nodes = torch.tensor(self.n_nodes)
        prob = np.array(prob)
        prob = prob / np.sum(prob)

        self.prob = torch.from_numpy(prob).float()

        entropy = torch.sum(self.prob * torch.log(self.prob + 1e-30))
        print("Entropy of n_nodes: H[N]", entropy.item())

        self.m = Categorical(torch.tensor(prob))

    def sample(self, n_samples=1):
        idx = self.m.sample((n_samples,))
        return self.n_nodes[idx]

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1

        idcs = [self.keys[i.item()] for i in batch_n_nodes]
        idcs = torch.tensor(idcs).to(batch_n_nodes.device)

        log_p = torch.log(self.prob + 1e-30)

        log_p = log_p.to(batch_n_nodes.device)

        log_probs = log_p[idcs]

        return log_probs

class Data_loaders(Dataset):
    def __init__(self, atom_dim, n_particles_list=None, node_num=0, ligand_data=None, protein_data=None):
        if n_particles_list is None:
            self.n_particles_list = [node_num]
        else:
            self.n_particles_list = n_particles_list  # 每个样本的原子数列表 
        self.atom_dim = atom_dim
        self.protein_data = protein_data
        self.ligand_data = ligand_data

    def __len__(self):
        return len(self.n_particles_list) 

    def __getitem__(self, idx):
        n_particles = self.n_particles_list[idx]
        n_particles = max(20, min(n_particles, 25))
        if self.ligand_data is None:
            atom_type = torch.randn(n_particles, self.atom_dim)
            pos = torch.randn(n_particles, 3)
            num_node = torch.tensor([n_particles])
            mask = torch.ones(n_particles)
        # else:
        #     atom_type = self.ligand_data[0]
        #     pos = self.ligand_data[1]
        #     mask =self.ligand_data[2]
        #     num_node = len(atom_type)
            
        if self.protein_data is not None:
            return {
                'ligand_atom': self.ligand_data['ligand_atom'],
                'ligand_pos': self.ligand_data['ligand_pos'],
                'ligand_num_node': len(self.ligand_data['ligand_atom']),
                'ligand_atom_mask': self.ligand_data['ligand_pad_mask'],
                'protein_atom':self.protein_data['protein_atom'],
                'protein_pos':self.protein_data['protein_pos'],
                'protein_atom_mask':self.protein_data['protein_atom_mask'],
                'residue_atom':self.protein_data['residue_atom'],
                'residue_pos':self.protein_data['residue_pos'],
                'residue_atom_mask':self.protein_data['residue_atom_mask'],
            }
        else:
            raise Exception('Missing protein_data')
    
def construct_dataset_pocket(num_sample, batch_size, dataset_info, num_points, num_for_pdb=None, start_linker=None, ligand_data=None, 
                              scaffold=False, *protein_information):

    nodes_dist = DistributionNodes(dataset_info['n_nodes'])
    num_node_frag = 0
    if start_linker is not None:
        print('linker atom number:', len(start_linker['element']))
        num_node_frag = len(start_linker['element'])
        linker_atom_num = start_linker['linker_atom_num']
    
    atom_dim = len(dataset_info['atom_decoder'])

    data_list = []

    protein_atom, protein_pos, protein_bond_index, protein_num_nodes = protein_information

    pocket_size = get_space_size(protein_pos.detach().cpu().numpy())
    # print(pocket_size)

    if batch_size > num_sample:
        batch_size = num_sample

    for n in tqdm(range(int(num_sample // batch_size))):
        if batch_size < 20:
            batch_size = 10
        # num_atoms_sample = [sample_atom_num(pocket_size).astype(int) for _ in range(batch_size)]
        num_atoms_sample = nodes_dist.sample(batch_size).tolist()
        # nodesxsample = sample_atom_num(pocket_size, batch_size).tolist()
        # num_atoms_sample = [min(35, x) for x in num_atoms_sample]
        # num_atoms_sample = [max(16, x) for x in num_atoms_sample]
        # num_atoms_sample = [min(35, max(12, x)) for x in num_atoms_sample]
        num_atoms_sample = [min(35, max(15, x)) for x in num_atoms_sample]
        # num_atoms_sample = [min(35, x) for x in num_atoms_sample]

        datas = []
        if ligand_data is None:
            for i, n_particles in enumerate(num_atoms_sample):
                # ligand_atom = torch.randn(n_particles, atom_dim)
                if start_linker is not None:
                    # num_node_linker = n_particles - num_node_frag
                    # if num_node_linker < 1 or linker_atom_num < 1:
                    #     num_node_linker = random.randint(4, 8)
                    #     n_particles = num_node_frag + num_node_linker
                    if scaffold:
                        num_node_linker = random.randint(5, 15)
                        n_particles = num_node_frag + num_node_linker
                    else:
                        num_node_linker = n_particles - num_node_frag
                else:
                    num_node_linker = n_particles - num_node_frag
                num_node_linker = torch.tensor(num_node_linker)
                n_particles = torch.tensor(n_particles)
                ligand_atom = torch.zeros(num_node_linker, atom_dim)
                ligand_pos = torch.zeros(num_node_linker, 3)
                ligand_edge_index = get_adj_matrix(n_particles)
                # ligand_pos = ligand_pos.normal_()
                if start_linker is not None:
                    ligand_atom = torch.cat([start_linker['linker_atom_type'], ligand_atom])
                    frag_mask = torch.cat(
                        [torch.ones(num_node_frag, dtype=torch.long), torch.zeros(num_node_linker, dtype=torch.long)])
                    ligand_pos = torch.cat([start_linker['pos'], ligand_pos])
                    # frag_edge_mask = start_linker['frag_edge_mask']
                    edge_bond_mask = torch.zeros((n_particles, n_particles), dtype=torch.long)
                    edge_bond_mask[:num_node_frag, :num_node_frag] = 1
                    # edge_bond_mask = edge_bond_mask.flatten()  
                    upper_mask = torch.triu(torch.ones_like(edge_bond_mask, dtype=torch.bool), diagonal=1) 
                    edge_bond_mask = edge_bond_mask[upper_mask] 
                # num_node = torch.tensor(num_node_linker + num_node_frag)
                # print(num_node)
                # ligand_edge_type = torch.randn(n_particles, n_particles, 6)
                if  start_linker is not None:
                    # data = ProteinLigandData(ligand_atom=ligand_atom, ligand_pos=ligand_pos, num_nodes=n_particles, ligand_bond_index=ligand_edge_index, 
                                        #   ligand_num_nodes=n_particles, protein_atom=protein_atom, protein_pos=protein_pos, protein_bond_index=protein_bond_index,
                                        #   protein_num_nodes=protein_num_nodes, frag_mask= frag_mask, anchor = start_linker['anchor_local'],
                                        #  )
                    data = ProteinLigandData(ligand_atom=ligand_atom, ligand_pos=ligand_pos, num_nodes=n_particles, ligand_bond_index=ligand_edge_index, 
                                          ligand_num_nodes=n_particles, protein_atom=protein_atom, protein_pos=protein_pos, protein_bond_index=protein_bond_index,
                                          protein_num_nodes=protein_num_nodes, frag_mask=frag_mask, edge_bond_mask=edge_bond_mask, frag_bond=start_linker['frag_bond'] 
                                         )
                else:
                    data = ProteinLigandData(ligand_atom=ligand_atom, ligand_pos=ligand_pos, num_nodes=n_particles, ligand_bond_index=ligand_edge_index, 
                                          ligand_num_nodes=n_particles, protein_atom=protein_atom, protein_pos=protein_pos, protein_bond_index=protein_bond_index,
                                          protein_num_nodes=protein_num_nodes
                                         )
                datas.append(data)
        else:
            ligand_atom, ligand_pos = ligand_data['ligand_atom'], ligand_data['ligand_pos']
            num_node = torch.tensor([ligand_atom.size(0)])
            ligand_edge_index = get_adj_matrix(num_node)
            # ligand_edge_type = torch.randn(n_particles, n_particles, 6)
            data = ProteinLigandData(ligand_atom=ligand_atom, ligand_pos=ligand_pos, num_nodes=n_particles, ligand_bond_index=ligand_edge_index, 
                                          ligand_num_nodes=n_particles, protein_atom=protein_atom, protein_pos=protein_pos, protein_bond_index=protein_bond_index, 
                                           protein_num_nodes=protein_num_nodes
                                         )
            datas.extend([data for i in range(batch_size)])
            # datas.extend([data for _ in range(batch_size)])
    data_list.append(datas)
    return data_list, batch_size

