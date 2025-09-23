import sys
import numpy as np
from rdkit import RDConfig
import os
import torch
import torch.nn as nn

# from .utils import MLP
# from drug_diffusion2.protein.Protein_feature import Res3DGraphModel, Atom3DGraphModel
from protein_model.EGNN import EGNN

from torch_geometric.nn import radius_graph


class ProteinEncoder(nn.Module):
    def __init__(self, atom_in_dim=31, hidden_dim=64, atom_out_dim=64, 
                ):
        super(ProteinEncoder, self).__init__()
        

        self.atom_emb = EGNN(input_dim=atom_in_dim, hidden_dim=hidden_dim, out_dim=atom_out_dim, device='cpu', act_fn=nn.SiLU(), n_layers=2, 
                 attention=True,  tanh=False, coords_range=15, norm_constant=1, inv_sublayers=3,
                 sin_embedding=False, normalization_factor=1, aggregation_method='sum')
        # self.res_MLP = MLP(in_dim=256, out_dim=out_dim, num_layers=3)
        
    def forward(self, protein_atom, protein_pos, protein_edge_index=None, protein_bond_type=None):
        
        # edge_index = radius_graph(protein_pos, 2, batch=protein_batch, loop=False)
        # edge_index = edge_index[[1, 0], :]
        # protein_bond_type =  torch.zeros(edge_index.size(1), dtype=torch.long, device=protein_pos.device) + 5
        atom, pos  = self.atom_emb(protein_atom, protein_pos, protein_edge_index, protein_bond_type)

        return atom, pos 
        # return protein_atom_init.view(bs, n, -1), protein_pos_init.view(bs, n, -1)
        # residue_batch = torch.arange(r_bs).repeat_interleave(r_n).to(device)

        # residue_pos = residue_pos.view(r_bs*r_n, -1)
        # residue_edge_index = radius_graph(residue_pos, self.residue_cutoff, batch=residue_batch, loop=False)
        # residue_edge_length =torch.norm(residue_pos[residue_edge_index[0]] - residue_pos[residue_edge_index[1]], dim=1)
        # residue_edge_attr = self._rbf(residue_edge_length)
        
        # atom_index = self._process_index(atom_index_temp, residue_edge_index, residue_edge_length)
        
        # atom_edge_attr = self.distance_expansion(edge_length)
        
        # atom_feat = self.atom_emb(protein_atom, edge_index, edge_length)  # (N, 256)

        # residue_atom = residue_atom.view(r_bs*r_n, -1).clone() * residue_atom_mask
        # residue_pos = residue_pos.view(r_bs*r_n, -1).clone() * residue_atom_mask
        # res_feat = self.res_emb(residue_atom, residue_edge_index, residue_edge_length)   # (N, 256)
        
        # atom_feat = self.atom_MLP(atom_feat)
        # res_feat = self.res_MLP(res_feat)
        # atom_feat = atom_feat * protein_atom_mask

        # atom_feat = atom_feat.view(p_bs, p_n, -1)
        # # res_feat = res_feat.view(r_bs, r_n, -1)

        # # return res_feat, atom_feat
        # return atom_feat

    
    # @staticmethod
    # def _process_index(atom_index, residue_edge_index, protein_residue_len):
    #     device = protein_residue_len.device
    #     indeces = np.cumsum(protein_residue_len.cpu()) 
    #     index_s = []
    #     index_e = []
    #     start_indices = torch.cat((torch.tensor([0]), indeces[:-1]))
    #     end_indices = indeces - 1
        
    #     for start, end in zip(start_indices, end_indices):
    #         mask = (atom_index[1] >= start) & (atom_index[1] <= end)
    #         start_points = atom_index[0][mask]
    #         end_points = atom_index[1][mask]
            
    #         # 确保起始点也在同一个氨基酸中
    #         sub_mask = (start_points >= start) & (start_points <= end)
    #         index_e.append(start_points[sub_mask])
    #         index_s.append(end_points[sub_mask])

    #     temp_s = np.zeros_like(residue_edge_index[1].cpu())
    #     temp_e = np.zeros_like(residue_edge_index[0].cpu())
    #     for i in range(len(start_indices)):
    #         temp_s[residue_edge_index[1].cpu() == i] = start_indices[i] + 1
    #     for i in range(len(residue_edge_index[0])):
    #         temp_e[i] = start_indices[residue_edge_index[0][i].cpu()] + 1
    #     # residue_index_new =torch.stack([torch.from_numpy(temp_e), torch.from_numpy(temp_s)], dim=0)
    #     index_s.append(torch.tensor(temp_s).to(device))
    #     index_e.append(torch.tensor(temp_e).to(device))
        
    #     index_s = torch.cat(index_s, dim=0)
    #     index_e = torch.cat(index_e, dim=0)

    #     atom_index_new = torch.stack([index_e, index_s], dim=0)
    #     # print(len(atom_index_new[1]))

    #     return atom_index_new


    # def _rbf(self, D):
    #     D_min, D_max, D_count = 0., 20., self.num_rbf
    #     #D_mu = torch.linspace(D_min, D_max, D_count).cuda()
    #     D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
    #     D_mu = D_mu.view([1,1,1,-1]).contiguous()
    #     D_sigma = (D_max - D_min) / D_count
    #     D_expand = torch.unsqueeze(D, -1)
    #     RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        
    #     return RBF
    
    # def 

