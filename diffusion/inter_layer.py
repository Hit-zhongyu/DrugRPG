import torch.nn.functional as F
import torch
import numpy as np
from torch import nn
from torch import Tensor
from typing import Optional
from torch_geometric.typing import Adj, OptTensor
from torch_scatter import scatter_softmax, scatter_sum
from .utils import *

class Atom_Inter_Layer(nn.Module):
    def __init__(self, x_channels, out_channels, heads, dropout, edge_dim):
        super(Atom_Inter_Layer, self).__init__()

        self.x_channels = x_channels
        self.in_channels = in_channels = x_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout

        self.lin_norm = nn.LayerNorm(in_channels * 2 + edge_dim)

        self.lin_query = nn.Sequential(
                        nn.Linear(in_channels, in_channels * 2),
                        nn.LayerNorm(in_channels * 2),
                        nn.SiLU(),
                        nn.Linear(in_channels * 2, heads * out_channels)
                    )
                    
        self.lin_key = nn.Sequential(
                        nn.Linear(in_channels * 2 + edge_dim, in_channels),
                        nn.LayerNorm(in_channels),
                        nn.SiLU(),
                        nn.Linear(in_channels, heads * out_channels)
                    )
        self.lin_value = nn.Sequential(
                        nn.Linear(in_channels * 2 + edge_dim, in_channels * 2),
                        nn.LayerNorm(in_channels * 2),
                        nn.SiLU(),
                        nn.Linear(in_channels * 2, heads * out_channels)
                    )
        
        self.ln_out = nn.Sequential(
                        nn.Linear(in_channels, in_channels * 2),
                        nn.SiLU(),
                        nn.Linear(in_channels * 2, in_channels)
                    )

    def forward(self, x, edge_index, edge_attr):

        H, C = self.heads, self.out_channels

        hi, hj = x[edge_index[0]], x[edge_index[1]]
        x_feat = self.lin_norm(torch.cat([edge_attr, hi, hj], dim=-1))
        query = self.lin_query(x).view(-1, H, C)
        key = self.lin_key(x_feat).view(-1, H, C)
        value = self.lin_value(x_feat).view(-1, H, C)

        alpha = scatter_softmax((query[edge_index[1]] * key / np.sqrt(key.shape[-1])).sum(-1), edge_index[1], dim=0, dim_size=x.size(0))
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        m = alpha.unsqueeze(-1) * value
        out_x = scatter_sum(m, edge_index[1], dim=0, dim_size=x.size(0))
        out_x = out_x.view(-1, self.heads * self.out_channels)

        out_x = self.ln_out(out_x)

        return out_x

class Ligand_Inter_Atom_Layer(nn.Module):

    def __init__(self, node_dim, edge_dim, dist_dim, time_dim, num_heads, act=nn.SiLU(), dropout=0.0):
        super().__init__()
        self.act = act
        self.atom_emb = nn.Linear(node_dim + time_dim, node_dim)
        self.edge_emb = nn.Linear(edge_dim + dist_dim, edge_dim)
        self.norm_node = nn.LayerNorm(node_dim + time_dim)
        self.atom_mpnn = Atom_Inter_Layer(node_dim, node_dim // num_heads, num_heads, dropout=dropout, edge_dim=edge_dim)
        # self.node_time_mlp = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(time_dim, node_dim * 2)
        # )
        self.dist_layer = GaussianSmearing(stop=15, num_gaussians=dist_dim)

    def forward(self, pos, h, edge_attr, edge_index, node_time_emb):
        # obtain distance feature
        # distance = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], p=2, dim=-1, keepdim=True)
        row, col = edge_index
        distance = (pos[row] - pos[col]).pow(2).sum(dim=-1).sqrt().unsqueeze(-1)
        distance = self.dist_layer(distance)
        edge_attr = self.edge_emb(torch.cat([distance, edge_attr], dim=-1))

        # node_shift_msa, node_scale_msa = self.node_time_mlp(node_time_emb).chunk(2, dim=1)

        # h = modulate(self.norm1_node(h), node_shift_msa, node_scale_msa)
        h = self.atom_emb(self.norm_node(torch.cat([h, node_time_emb], dim=-1)))
        h_node = self.atom_mpnn(h, edge_index, edge_attr)

        return h_node
    