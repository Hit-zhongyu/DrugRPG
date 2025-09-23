import torch
import torch.nn as nn
import numpy as np

from .gvp import GVPModel
from ..egnn.egnns import EGNN
from torch_geometric.nn import radius_graph
from .utils import remove_mean, remove_mean_with_mask

class EGNN_encoder(nn.Module):
    def __init__(self, in_node_nf,  out_node_nf, n_dims, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 tanh=False, norm_constant=0, cutoff=2.5, coords_range=15, coor_norm=100,
                 inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum',
                 ):
        
        super().__init__()

        self.cutoff = cutoff
        self.norm_constant = norm_constant
        self.coor_norm = coor_norm
        self.egnn = EGNN(in_node_nf=in_node_nf, out_node_nf=hidden_nf, 
                hidden_nf=hidden_nf, device=device, act_fn=act_fn,coords_range=coords_range,
                n_layers=n_layers, attention=attention, tanh=tanh, norm_constant=norm_constant,
                inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method
            )
        
        # self.x_var = EquivariantUpdate(hidden_nf=hidden_nf, normalization_factor=normalization_factor, 
        #                                aggregation_method=aggregation_method, edges_in_d=1, 
        #                                act_fn=nn.SiLU(), tanh=tanh, coords_range=coords_range)
        
        self.h_mu = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf * 2),
            nn.SiLU(),
            nn.Linear(hidden_nf * 2, out_node_nf),
        )
            # nn.SiLU(),
        #   nn.Linear(hidden_nf, out_node_nf))

        # self.h_var = nn.Sequential(
        #     nn.Linear(hidden_nf, hidden_nf * 2),
        #     nn.SiLU(),
        #     nn.Linear(hidden_nf * 2, out_node_nf),
        # )
            # nn.SiLU(),
            # nn.Linear(hidden_nf, out_node_nf))


    def forward(self, ligand_atom, ligand_pos, ligand_pad_mask=None):  
        
        device = ligand_atom.device
        bs, n, dim = ligand_atom.shape

        ligand_batch = torch.arange(bs).repeat_interleave(n).to(device)
        
        ligand_atom = ligand_atom.view(bs*n, -1)
        ligand_pos = ligand_pos.view(bs*n, -1)
        
        # ligand_atom_mask = ligand_atom_mask.view(bs*n, 1).bool()
        ligand_pad_mask = ligand_pad_mask.view(bs*n, 1).bool()

        edge_index = radius_graph(ligand_pos, self.cutoff, batch=ligand_batch, loop=False)
        mask_edge = (ligand_pad_mask[edge_index[0]] & ligand_pad_mask[edge_index[1]]).squeeze()
        edge_index = edge_index[:, mask_edge]
        
        # ligand_pos = torch.tanh(ligand_pos/self.coor_norm)
        # ligand_atom_mask = ligand_atom_mask.float()
        ligand_pad_mask = ligand_pad_mask.float()
        ligand_atom = ligand_atom.clone() * ligand_pad_mask
        ligand_pos = ligand_pos.clone() * ligand_pad_mask

        h_final, x_final = self.egnn(ligand_atom, ligand_pos, edge_index, ligand_pad_mask)

        # x_final = self.coor_norm * torch.arctanh(x_final)
        x_final = x_final * ligand_pad_mask
        x_final = x_final.view(bs, n, -1)

        if torch.any(torch.isnan(x_final)):
            print('Warning: detected nan, resetting EGNN output to zero.')
            x_final = torch.zeros_like(x_final)

        # h_mu, h_var = h_final.chunk(2, dim=-1)
        h_mu = self.h_mu(h_final)
        # h_var = self.h_var(h_final)

        h_mu = h_mu.view(bs, n, -1)
        # h_var = h_var.view(bs, n, -1)

        if torch.any(torch.isnan(h_mu)):
            print('Warning: detected nan, resetting var to zero.')
            h_mu = torch.zeros_like(h_mu)

        # return mu, var, x_final
        # return h_mu, h_var, x_mu, x_var   
        return h_mu, h_mu, x_final


class EGNN_Decoder(nn.Module):
    def __init__(self, in_node_nf,  out_node_nf, n_dims, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 tanh=False, norm_constant=0, cutoff=2.5, coor_norm=100,
                 inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum'
                 ):
        
        super().__init__()
        self.cutoff = cutoff
        self.coor_norm = coor_norm
        self.egnn = EGNN(
                in_node_nf=in_node_nf, out_node_nf=hidden_nf, 
                hidden_nf=hidden_nf, device=device, act_fn=act_fn,
                n_layers=n_layers, attention=attention, tanh=tanh, norm_constant=norm_constant,
                inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method
            )
        
        self.h_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, out_node_nf))
        

    def forward(self, ligand_atom, ligand_pos, ligand_pad_mask=None):  
        device = ligand_atom.device
        bs, n, dim = ligand_atom.shape 

        ligand_batch = torch.arange(bs).repeat_interleave(n).to(device)
        ligand_atom = ligand_atom.view(bs*n, -1)
        ligand_pos = ligand_pos.view(bs*n, -1)
        ligand_pad_mask = ligand_pad_mask.view(bs*n, 1).bool()

        edge_index = radius_graph(ligand_pos, self.cutoff, batch=ligand_batch, loop=False)
        mask_edge = (ligand_pad_mask[edge_index[0]] & ligand_pad_mask[edge_index[1]]).squeeze()
        edge_index = edge_index[:, mask_edge]

        # ligand_decoder_mask = ligand_decoder_mask.float()
        ligand_pad_mask = ligand_pad_mask.float()
        ligand_atom = ligand_atom.clone() * ligand_pad_mask
        ligand_pos = ligand_pos.clone() * ligand_pad_mask
        # edge_feature = self._build_edge_feature(x, edges_index, edges_type)
        h_final, x_final = self.egnn(ligand_atom, ligand_pos, edge_index, ligand_pad_mask)

        x_final = x_final * ligand_pad_mask
        x_final = x_final.view(bs, n, -1)

        if torch.any(torch.isnan(x_final)):
            print('Warning: detected nan, resetting EGNN output to zero.')
            x_final = torch.zeros_like(x_final)

        h_final = self.h_mlp(h_final)

        if ligand_pad_mask is not None:
            h_final = h_final * ligand_pad_mask
        
        h_final = h_final.view(bs, n, -1)

        return h_final, x_final
