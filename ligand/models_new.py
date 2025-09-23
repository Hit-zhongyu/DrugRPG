import torch
import torch.nn as nn
import numpy as np

from .gvp import GVPModel
from ..egnn.egnns import EGNN
from torch_geometric.nn import radius_graph
from .utils import remove_mean, remove_mean_with_mask
from ..egnn.egnn_new import EGNN_Sparse_Network
# from ..egnn.egnn import EGNN_Sparse_Network
# from ..model.common import MultiLayerPerceptron
import torch.nn.init as init

class EGNN_encoder(nn.Module):
    def __init__(self, in_node_nf,  out_node_nf, hidden_nf=64,
                  n_layers=3, cutoff=2.5
                 ):
        
        super().__init__()

        self.cutoff = cutoff
        self.egnns = EGNN_Sparse_Network(n_layers=n_layers,
            feats_input_dim=in_node_nf,
            feats_dim=hidden_nf,
            m_dim=hidden_nf,
            soft_edge=True,
            norm_coors=True)
        self.atom_emb = nn.Linear(in_node_nf, hidden_nf)
        # init.xavier_uniform_(self.atom_emb.weight)
        self.atom_mlp = nn.Linear(in_node_nf, hidden_nf)
        self.h_out = nn.Linear(hidden_nf, hidden_nf)

        # self.h_out = nn.Sequential(
        #     nn.Linear(hidden_nf, hidden_nf *2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_nf * 2, hidden_nf)
        #     )
        # self.edge_mlp = MultiLayerPerceptron(1, [hidden_nf // 1, hidden_nf], activation='relu')

    def forward(self, ligand_atom, ligand_pos, ligand_pad_mask=None):  
        
        device = ligand_atom.device
        bs, n, dim = ligand_atom.shape

        ligand_pad_mask = ligand_pad_mask.bool()
        ligand_batch = torch.arange(bs).repeat_interleave(n).to(device)
        
        ligand_atom = ligand_atom.view(bs*n, -1)
        ligand_pos = ligand_pos.view(bs*n, -1)
        ligand_pad_mask = ligand_pad_mask.view(-1)

        # ligand_atom = ligand_atom * ligand_pad_mask
        # ligand_pos = ligand_pos * ligand_pad_mask
        atom = ligand_atom[ligand_pad_mask]
        pos = ligand_pos[ligand_pad_mask]
        batch_index = ligand_batch[ligand_pad_mask]

        atom = self.atom_emb(atom)
        edge_index = radius_graph(pos, self.cutoff, batch=batch_index, loop=False)

        h_out, x_out = self.egnns(h=atom, x=pos, edge_index=edge_index, batch=batch_index, edge_attr=None, linker_mask=None)
        # print(x_out)
        ligand_pos[ligand_pad_mask] = x_out.clone()
        x = ligand_pos.view(bs, n, -1)
        
        if torch.any(torch.isnan(x)):
            print('Warning: detected nan, resetting EGNN output to zero.')
            x = torch.zeros_like(x)
    
        h_out = self.h_out(h_out)
        ligand_atom = self.atom_mlp(ligand_atom)
        ligand_atom[ligand_pad_mask] = h_out.clone()
        h = ligand_atom.view(bs, n, -1)

        return h, x


class EGNN_Decoder(nn.Module):
    def __init__(self, in_node_nf,  out_node_nf, hidden_nf=64,  
                 n_layers=4,  cutoff=2.5
                 ):
        
        super().__init__()
        self.cutoff = cutoff
        self.egnns = EGNN_Sparse_Network( n_layers=n_layers,
            feats_input_dim=in_node_nf,
            feats_dim=hidden_nf,
            m_dim=hidden_nf,
            soft_edge=True,
            norm_coors=True)
        
        # self.edge_mlp = MultiLayerPerceptron(1, [hidden_nf // 1, hidden_nf], activation='relu')
        self.atom_emb = nn.Linear(in_node_nf, hidden_nf)
        self.atom_out = nn.Linear(in_node_nf, out_node_nf)
        self.h_out = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf *2),
            nn.SiLU(),
            nn.Linear(hidden_nf * 2, out_node_nf)
            )
        # self.h_out = nn.Linear(hidden_nf, out_node_nf)

    def forward(self, ligand_atom, ligand_pos, ligand_pad_mask=None):  
        device = ligand_atom.device
        bs, n, dim = ligand_atom.shape

        ligand_pad_mask = ligand_pad_mask.bool()
        ligand_batch = torch.arange(bs).repeat_interleave(n).to(device)
        
        ligand_atom = ligand_atom.view(bs*n, -1)
        ligand_pos = ligand_pos.view(bs*n, -1)
        ligand_pad_mask = ligand_pad_mask.view(-1)

        atom = ligand_atom[ligand_pad_mask]
        pos = ligand_pos[ligand_pad_mask]
        batch_index = ligand_batch[ligand_pad_mask]
        atom = self.atom_emb(atom)
        # ligand_atom_mask = ligand_atom_mask.view(bs*n, 1).bool()
        # ligand_pad_mask = ligand_pad_mask.bool()
        edge_index = radius_graph(pos, self.cutoff, batch=batch_index, loop=False)
        # distance = (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)
        # edge_attr = self.edge_mlp(distance.unsqueeze(-1))

        h_out, x_out = self.egnns(h=atom, x=pos, edge_index=edge_index, batch=batch_index, edge_attr=None, linker_mask=None)
        ligand_pos[ligand_pad_mask] = x_out.clone()
        x = ligand_pos.view(bs, n, -1)

        if torch.any(torch.isnan(x)):
            print('Warning: detected nan, resetting EGNN output to zero.')
            x = torch.zeros_like(x)
        ligand_atom = self.atom_out(ligand_atom)
        h_out = self.h_out(h_out)
        ligand_atom[ligand_pad_mask] = h_out.clone()
        h = ligand_atom.view(bs, n, -1)

        return h, x
