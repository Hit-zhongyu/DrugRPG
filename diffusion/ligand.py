import torch
import torch.nn as nn

from .utils import *
from .layers import * 
from .equivariance import CondEquiUpdate

class Block_Ligand(nn.Module):
    """Equivariant block based on graph relational transformer layer, without extra heads."""

    def __init__(self, node_dim, edge_dim, time_dim, num_heads, mlp_ratio=2, act=nn.SiLU(), dropout=0.0):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.act = act
        dist_dim = edge_dim

        self.edge_emb = nn.Linear(edge_dim + dist_dim, edge_dim)

        # message passing layer
        self.attn_mpnn = Trans_Layer(node_dim, node_dim // num_heads, num_heads,
                                     edge_dim=edge_dim)
        
        # Normalization for MPNN
        self.norm1_node = nn.LayerNorm(node_dim, elementwise_affine=False, eps=1e-6)
        self.norm1_edge = nn.LayerNorm(edge_dim, elementwise_affine=False, eps=1e-6)

        # Feed forward block -> node.
        self.ff_linear1 = nn.Linear(node_dim, node_dim * mlp_ratio)
        self.ff_linear2 = nn.Linear(node_dim * mlp_ratio, node_dim)
        self.norm2_node = nn.LayerNorm(node_dim, elementwise_affine=False, eps=1e-6)

        self.dist_layer = GaussianSmearing(stop=15, num_gaussians=dist_dim)

        self.node_time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, node_dim)
        )

    def _ff_block_node(self, x):
        x = self.dropout(self.act(self.ff_linear1(x)))
        return self.dropout(self.ff_linear2(x))

    def forward(self, pos, h, edge_attr, edge_index, node_time_emb, edge_time_emb):
        """
        Params:
            pos: [B*N, 3]
            h: [B*N, hid_dim]
            edge_attr: [N_edge, edge_hid_dim]
            edge_index: [2, N_edge]
            node_mask: [B*N, 1]
            extra_heads: [N_edge, extra_heads]
        """
        h_in_node = h

        # obtain distance feature
        distance = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], p=2, dim=-1, keepdim=True)
        distance = self.dist_layer(distance)
        edge_attr = self.edge_emb(torch.cat([edge_attr, distance], dim=-1))

        h = h + self.node_time_mlp(node_time_emb)
        h = self.norm1_node(h)
        edge_attr = self.norm1_edge(edge_attr)
        
        # apply transformer-based message passing, update node features and edge features (FFN + norm)
        h_node = self.attn_mpnn(h, edge_index, edge_attr)
        h_node = h_in_node + h_node 
        h_out = self.norm2_node(h_node) + self._ff_block_node(h_node)

        return h_out