import torch.nn.functional as F
import torch
import numpy as np
from torch import nn
from torch import Tensor
from typing import Optional
from torch_geometric.typing import Adj, OptTensor
from torch_scatter import scatter_softmax, scatter_sum
from .utils import *


class Atom_Intra_Layer(nn.Module):
    def __init__(self, x_channels, out_channels, heads, dropout, edge_dim):
        super(Atom_Intra_Layer, self).__init__()

        self.x_channels = x_channels
        self.in_channels = in_channels = x_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout

        self.lin_norm = nn.LayerNorm(in_channels * 2 + edge_dim)
        self.lin_query = nn.Linear(in_channels, heads * out_channels)
                    
        self.lin_key = nn.Sequential(
                        nn.Linear(in_channels * 2 + edge_dim, in_channels),
                        nn.SiLU(),
                        nn.Linear(in_channels, heads * out_channels)
                    )
        self.lin_value = nn.Sequential(
                        nn.Linear(in_channels * 2 + edge_dim, in_channels),
                        nn.SiLU(),
                        nn.Linear(in_channels, heads * out_channels)
                    )

    def forward(self, x: OptTensor,  edge_index: Adj, edge_attr: OptTensor = None) -> Tensor:

        H, C = self.heads, self.out_channels

        hi, hj = x[edge_index[0]], x[edge_index[1]]
        x_feat = self.lin_norm(torch.cat([edge_attr, hi, hj], dim=-1))
        query = self.lin_query(x).view(-1, H, C)
        key = self.lin_key(x_feat).view(-1, H, C)
        value = self.lin_value(x_feat).view(-1, H, C)

        alpha = scatter_softmax((query[edge_index[0]] * key / np.sqrt(key.shape[-1])).sum(-1), edge_index[0], dim=0, dim_size=x.size(0))
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        m = alpha.unsqueeze(-1) * value
        out_x = scatter_sum(m, edge_index[0], dim=0, dim_size=x.size(0))  # (N, heads, H_per_head)
        out_x = out_x.view(-1, self.heads * self.out_channels)

        return out_x

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class Ligand_Intra_Atom_Layer(nn.Module):
    def __init__(self, node_dim, edge_dim, time_dim, num_heads, act=nn.SiLU(), dropout=0.0):
        super().__init__()

        self.act = act
        self.dropout = nn.Dropout(dropout)
        
        self.atom_emb = nn.Linear(node_dim + time_dim, node_dim)
        self.edge_emb = nn.Linear(edge_dim + edge_dim, edge_dim)
        self.edge_norm = nn.LayerNorm(edge_dim + edge_dim)
        self.norm_node = nn.LayerNorm(node_dim + time_dim)
        
        # message passing layer
        self.atom_mpnn = Atom_Intra_Layer(node_dim, node_dim // num_heads, num_heads, edge_dim=edge_dim, dropout=dropout)

        self.dist_layer = GaussianSmearing(stop=15, num_gaussians=edge_dim)

    def forward(self, pos, h, edge_attr, edge_index, node_time_emb=None):
        # obtain distance feature
        # distance = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=-1)
        row, col = edge_index
        distance = (pos[row] - pos[col]).pow(2).sum(dim=-1).sqrt().unsqueeze(-1)
        distance = self.dist_layer(distance)
        edge_attr = self.edge_emb(self.edge_norm(torch.cat([edge_attr, distance], dim=-1)))
        
        h = self.atom_emb(self.norm_node(torch.cat([h, node_time_emb], dim=-1)))
        h_node = self.atom_mpnn(h, edge_index, edge_attr)

        return h_node

class Ligand_Intra_Bond_Layer(nn.Module):
    """The version for involving the edge feature. Multiply Msg. Without FFN and norm."""

    _alpha: OptTensor

    def __init__(self, hidden_dim, edge_dim,  time_dim):
        super(Ligand_Intra_Bond_Layer, self).__init__()

        self.node_linear = nn.Linear(hidden_dim, hidden_dim)
        self.bond_linear = nn.Linear(edge_dim, hidden_dim)

        self.bond_linear_left = nn.Linear(edge_dim, hidden_dim)
        self.bond_linear_right = nn.Linear(edge_dim, hidden_dim)

        self.node_linear_left = nn.Linear(hidden_dim, hidden_dim)
        self.node_linear_right = nn.Linear(hidden_dim, hidden_dim)

        self.norm_edge = nn.LayerNorm(edge_dim, elementwise_affine=False, eps=1e-6)

        self.bond_linear_left_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.bond_linear_right_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.layer_norm = nn.LayerNorm(hidden_dim * 3)
        self.bond_out = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, edge_dim),
        )


        # self.edge_emb = nn.Linear(edge_dim + edge_dim + time_dim, edge_dim)
        # self.edge_norm = nn.LayerNorm(edge_dim + edge_dim + time_dim)
        self.edge_emb = nn.Linear(edge_dim + time_dim, edge_dim)
        self.edge_norm = nn.LayerNorm(edge_dim + time_dim)
        self.dist_layer = GaussianSmearing(stop=15, num_gaussians=edge_dim)

    def forward(self, h, edge_index, edge_attr, edge_time_emb) -> Tensor:

        # distance = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=-1)
        # distance = self.dist_layer(distance)
        # edge_attr = self.edge_emb(self.edge_norm(torch.cat([edge_attr, distance, edge_time_emb], dim=-1)))
        edge_attr = self.edge_emb(self.edge_norm(torch.cat([edge_attr, edge_time_emb], dim=-1)))

        x_i = self.node_linear_left(h[edge_index[0]])
        x_j = self.node_linear_right(h[edge_index[1]])

        edge_i = self.bond_linear_left(edge_attr)
        edge_j = self.bond_linear_right(edge_attr)

        inter_i = self.bond_linear_left_mlp(torch.cat([edge_i, x_i], dim=-1))
        inter_j = self.bond_linear_right_mlp(torch.cat([edge_j, x_j], dim=-1))

        inter_i = scatter_sum(inter_i, edge_index[0], dim=0, dim_size=h.shape[0])
        inter_i = inter_i[edge_index[0]]

        inter_j = scatter_sum(inter_j, edge_index[1], dim=0, dim_size=h.shape[0])
        inter_j = inter_j[edge_index[1]]

        h_node = self.node_linear(h)

        h_bond = torch.cat([inter_i+inter_j, h_node[edge_index[0]]+h_node[edge_index[1]], self.bond_linear(edge_attr)], dim=-1)
        h_bond = self.layer_norm(h_bond)
        h_bond = self.bond_out(h_bond)

        return h_bond
