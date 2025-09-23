import torch
import torch.nn as nn

from .utils import *
from .layers import * 
from torch_scatter import scatter
# from .equivariance import CondEquiUpdate


class Trans_Atom_Inter_Layer(MessagePassing):
    """The version for involving the edge feature. Multiply Msg. Without FFN and norm."""

    _alpha: OptTensor

    def __init__(self, x_channels: int, out_channels: int,
                 heads: int = 1, dropout: float = 0., edge_dim: Optional[int] = None,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(Trans_Atom_Inter_Layer, self).__init__(node_dim=0, **kwargs)

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

    def forward(self, x: OptTensor, edge_index: Adj, edge_attr: OptTensor = None) -> Tensor:

        H, C = self.heads, self.out_channels

        hi, hj = x[edge_index[0]], x[edge_index[1]]
        x_feat = self.lin_norm(torch.cat([edge_attr, hi, hj], dim=-1))
        query = self.lin_query(x).view(-1, H, C)
        key = self.lin_key(x_feat).view(-1, H, C)
        value = self.lin_value(x_feat).view(-1, H, C)

        alpha = scatter_softmax((query[edge_index[1]] * key / np.sqrt(key.shape[-1])).sum(-1), edge_index[1], dim=0, dim_size=x.size(0))
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        m = alpha.unsqueeze(-1) * value
        out_x = scatter_sum(m, edge_index[1], dim=0, dim_size=x.size(0))  # (N, heads, H_per_head)
        out_x = out_x.view(-1, self.heads * self.out_channels)

        out_x = self.ln_out(out_x)

        return out_x

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

class Equivariant_Inter(nn.Module):
    """Equivariant block based on graph relational transformer layer, without extra heads."""

    def __init__(self, node_dim, edge_dim, dist_dim, time_dim, num_heads, mlp_ratio=2, act=nn.SiLU(), dropout=0.0):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.act = act

        # self.h_emb = nn.Linear(edge_dim + node_dim, node_dim)
        self.edge_emb = nn.Linear(edge_dim * 2 + dist_dim, edge_dim)

        self.norm1_node = nn.LayerNorm(node_dim, elementwise_affine=False, eps=1e-6)
        self.norm1_edge = nn.LayerNorm(edge_dim, elementwise_affine=False, eps=1e-6)
        
        self.atom_mpnn = Trans_Atom_Inter_Layer(node_dim, node_dim // num_heads, num_heads,edge_dim=edge_dim)
        # self.update_pos = CondEquiUpdate(node_dim, edge_dim, dist_dim, time_dim)

        self.node_time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, node_dim * 2)
        )
        self.edge_time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, edge_dim * 2)
        )

        self.dist_layer = GaussianSmearing(stop=15, num_gaussians=dist_dim)

        # Feed forward block -> node.
        self.ff_linear1 = nn.Linear(node_dim, node_dim * mlp_ratio)
        self.ff_linear2 = nn.Linear(node_dim * mlp_ratio, node_dim)
        self.norm2_node = nn.LayerNorm(node_dim, elementwise_affine=False, eps=1e-6)
    
    def _ff_block_node(self, x):
        x = self.dropout(self.act(self.ff_linear1(x)))
        return self.dropout(self.ff_linear2(x))

    def forward(self, pos, h, edge_attr, clash_feat, edge_index, node_time_emb, edge_time_emb):

        h_in = h
        # obtain distance feature
        distance = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], p=2, dim=-1, keepdim=True)
        distance = self.dist_layer(distance)
        edge_attr = self.edge_emb(torch.cat([clash_feat, distance, edge_attr], dim=-1))

        node_shift_msa, node_scale_msa = self.node_time_mlp(node_time_emb).chunk(2, dim=1)
        edge_shift_msa, edge_scale_msa = self.edge_time_mlp(edge_time_emb).chunk(2, dim=1)

        h = modulate(self.norm1_node(h), node_shift_msa, node_scale_msa)
        edge_attr = modulate(self.norm1_edge(edge_attr), edge_shift_msa, edge_scale_msa)

        h_node = self.atom_mpnn(h, edge_index, edge_attr)
        # pos =  self.update_pos(h_node, pos, edge_index, edge_attr, distance, clash_feat, edge_time_emb)

        h_node = h_in + self._ff_block_node(h_node)

        return h_node, pos