import torch.nn.functional as F
import torch
import math
import numpy as np
from torch import nn
from torch import Tensor
from torch.nn import Linear
from typing import Tuple, Optional
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter_softmax, scatter_sum

from .utils import * 

class FourierEmbedding(nn.Module):

    def __init__(self, c: int, seed: int = 42) -> None:
        """
        Args:
            c (int): embedding dim.
        """
        super(FourierEmbedding, self).__init__()
        self.c = c
        self.seed = seed
        generator = torch.Generator()
        generator.manual_seed(seed)
        w_value = torch.randn(size=(c,), generator=generator)
        self.w = nn.Parameter(w_value, requires_grad=False)
        b_value = torch.randn(size=(c,), generator=generator)
        self.b = nn.Parameter(b_value, requires_grad=False)

    def forward(self, t_hat_noise_level: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t_hat_noise_level (torch.Tensor): the noise level
                [..., N_sample]

        Returns:
            torch.Tensor: the output fourier embedding
                [..., N_sample, c]
        """
        return torch.cos(
            input=2 * torch.pi * (t_hat_noise_level.unsqueeze(dim=-1) * self.w + self.b)
        )

class LearnedSinusodialposEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb
    https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = x.unsqueeze(-1)
        freqs = x * self.weights.unsqueeze(0) * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered

@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

class GaussianLayer(nn.Module):
    """Gaussian basis function layer for 3D distance features, with time embedding condition"""
    def __init__(self, K, time_dim):
        super().__init__()
        self.K = K - 1
        self.means = nn.Embedding(1, self.K)
        self.stds = nn.Embedding(1, self.K)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 2)
        )
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)

    def forward(self, x, time_emb=None):
        if time_emb is not None:
            scale, shift = self.time_mlp(time_emb).chunk(2, dim=1)
            x = x * (scale + 1) + shift
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return torch.cat([x, gaussian(x, mean, std).type_as(self.means.weight)], dim=-1)

class Trans_Layer(MessagePassing):
    """The version for involving the edge feature. Multiply Msg. Without FFN and norm."""

    _alpha: OptTensor

    def __init__(self, x_channels: int, out_channels: int,
                 heads: int = 1, dropout: float = 0., edge_dim: Optional[int] = None,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(Trans_Layer, self).__init__(node_dim=0, **kwargs)

        self.x_channels = x_channels
        self.in_channels = in_channels = x_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim

        self.lin_key = Linear(in_channels, heads * out_channels, bias=bias)
        self.lin_query = Linear(in_channels, heads * out_channels, bias=bias)
        self.lin_value = Linear(in_channels, heads * out_channels, bias=bias)

        self.lin_edge0 = Linear(edge_dim, heads * out_channels, bias=False)
        self.lin_edge1 = Linear(edge_dim, heads * out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_edge0.reset_parameters()
        self.lin_edge1.reset_parameters()

    def forward(self, x: OptTensor,
                edge_index: Adj,
                edge_attr: OptTensor = None
                ) -> Tensor:
        """"""

        H, C = self.heads, self.out_channels

        x_feat = x
        query = self.lin_query(x_feat).view(-1, H, C)
        key = self.lin_key(x_feat).view(-1, H, C)
        value = self.lin_value(x_feat).view(-1, H, C)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out_x = self.propagate(edge_index, query=query, key=key, value=value, edge_attr=edge_attr, size=None)

        out_x = out_x.view(-1, self.heads * self.out_channels)

        return out_x

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tuple[Tensor, Tensor]:

        edge_attn = self.lin_edge0(edge_attr).view(-1, self.heads, self.out_channels)
        edge_attn = torch.tanh(edge_attn)
        alpha = (query_i * key_j * edge_attn).sum(dim=-1) / math.sqrt(self.out_channels)

        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # node feature message
        msg = value_j
        msg = msg * torch.tanh(self.lin_edge1(edge_attr).view(-1, self.heads, self.out_channels))
        msg = msg * alpha.view(-1, self.heads, 1)

        return msg

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class Trans_Layer_Inter(MessagePassing):
    """The version for involving the edge feature. Multiply Msg. Without FFN and norm."""

    _alpha: OptTensor

    def __init__(self, x_channels: int, out_channels: int,
                 heads: int = 1, dropout: float = 0., edge_dim: Optional[int] = None,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(Trans_Layer_Inter, self).__init__(node_dim=0, flow='target_to_source', **kwargs)

        self.x_channels = x_channels
        self.in_channels = in_channels = x_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim

        self.lin_key = Linear(in_channels, heads * out_channels, bias=bias)
        self.lin_query = Linear(in_channels, heads * out_channels, bias=bias)
        self.lin_value = Linear(in_channels, heads * out_channels, bias=bias)

        self.lin_edge0 = Linear(edge_dim, heads * out_channels, bias=False)
        self.lin_edge1 = Linear(edge_dim, heads * out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_edge0.reset_parameters()
        self.lin_edge1.reset_parameters()

    def forward(self, x: OptTensor, p:OptTensor,
                edge_index: Adj,
                edge_attr: OptTensor = None
                ) -> Tensor:
        """"""

        H, C = self.heads, self.out_channels

        x_feat = x
        p_feat = p
        query = self.lin_query(x_feat).view(-1, H, C)
        key = self.lin_key(p_feat).view(-1, H, C)
        value = self.lin_value(p_feat).view(-1, H, C)
        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out_x = self.propagate(edge_index, query=query, key=key, value=value, edge_attr=edge_attr, size=(x.size(0), p.size(0)))
        out_x = out_x.view(-1, self.heads * self.out_channels)
        return out_x

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tuple[Tensor, Tensor]:
        edge_attn = self.lin_edge0(edge_attr).view(-1, self.heads, self.out_channels)
        edge_attn = torch.tanh(edge_attn)
        alpha = (query_i * key_j * edge_attn).sum(dim=-1) / math.sqrt(self.out_channels)

        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # node feature message
        msg = value_j
        msg = msg * torch.tanh(self.lin_edge1(edge_attr).view(-1, self.heads, self.out_channels))
        msg = msg * alpha.view(-1, self.heads, 1)

        return msg

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class Trans_Atom_Layer(MessagePassing):
    """The version for involving the edge feature. Multiply Msg. Without FFN and norm."""

    _alpha: OptTensor

    def __init__(self, x_channels: int, out_channels: int,
                 heads: int = 1, dropout: float = 0., edge_dim: Optional[int] = None,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(Trans_Atom_Layer, self).__init__(node_dim=0, **kwargs)

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

class Trans_Bond_Layer(MessagePassing):
    """The version for involving the edge feature. Multiply Msg. Without FFN and norm."""

    _alpha: OptTensor

    def __init__(self, hidden_dim: int, out_channels: int, edge_dim: Optional[int] = None,  time_dim: Optional[int] = None, dropout: float = 0., **kwargs):
        super(Trans_Bond_Layer, self).__init__(node_dim=0, **kwargs)

        self.out_channels = out_channels

        self.node_linear = nn.Linear(hidden_dim, hidden_dim)
        self.bond_linear = nn.Linear(edge_dim, hidden_dim)

        self.bond_linear_left = nn.Linear(edge_dim, hidden_dim)
        self.bond_linear_right = nn.Linear(edge_dim, hidden_dim)

        self.node_linear_left = nn.Linear(hidden_dim, hidden_dim)
        self.node_linear_right = nn.Linear(hidden_dim, hidden_dim)

        self.norm_edge = nn.LayerNorm(edge_dim, elementwise_affine=False, eps=1e-6)

        self.bond_linear_left_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.left_gate = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.bond_linear_right_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.right_gate = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.layer_norm = nn.LayerNorm(hidden_dim * 3)
        self.bond_out = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, edge_dim),
        )

        self.edge_time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, edge_dim * 2)
        )

    def forward(self, x: OptTensor,  edge_index: Adj, edge_attr: OptTensor = None, edge_time_emb: OptTensor = None) -> Tensor:

        edge_shift_msa, edge_scale_msa = self.edge_time_mlp(edge_time_emb).chunk(2, dim=1)
        edge_attr = modulate(self.norm_edge(edge_attr), edge_shift_msa, edge_scale_msa)

        x_i = self.node_linear_left(x[edge_index[0]])
        x_j = self.node_linear_right(x[edge_index[1]])

        edge_i = self.bond_linear_left(edge_attr)
        edge_j = self.bond_linear_right(edge_attr)

        inter_i = self.bond_linear_left_mlp(edge_i * x_i)
        inter_j = self.bond_linear_right_mlp(edge_j * x_j)

        gate_i = self.left_gate(torch.cat([edge_attr, x[edge_index[0]]], dim=-1))
        inter_i = inter_i * torch.sigmoid(gate_i)
        inter_i = scatter_sum(inter_i, edge_index[0], dim=0, dim_size=x.shape[0])
        inter_i = inter_i[edge_index[0]]

        gate_j = self.right_gate(torch.cat([edge_attr, x[edge_index[1]]], dim=-1))
        inter_j = inter_j * torch.sigmoid(gate_j)
        inter_j = scatter_sum(inter_j, edge_index[1], dim=0, dim_size=x.shape[0])
        inter_j = inter_j[edge_index[1]]

        h_node = self.node_linear(x)

        h_bond = torch.cat([inter_i+inter_j, h_node[edge_index[0]]+h_node[edge_index[1]], self.bond_linear(edge_attr)], dim=-1)
        h_bond = self.layer_norm(h_bond)
        h_bond = self.bond_out(h_bond)

        return h_bond

# class CondEquiUpdate(nn.Module):
#     """Update atom coordinates equivariantly, use time emb condition."""

#     def __init__(self, hidden_dim, edge_dim, dist_dim, time_dim, num_heads):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#         self.num_heads = num_heads
#         # self.coord_norm = CoorsNorm(scale_init=1e-2)
#         # self.time_mlp = nn.Sequential(
#         #     nn.SiLU(),
#         #     nn.Linear(time_dim, hidden_dim * 2)
#         # )
#         input_ch = hidden_dim * 2 + edge_dim + dist_dim
#         self.input_lin = nn.Linear(input_ch, hidden_dim)
#         self.ln = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)

#         self.lin_key = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim * 2),
#             nn.SiLU(),
#             nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
#         )
#         self.lin_query = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim * 2),
#             nn.SiLU(),
#             nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
#         )
#         self.lin_value = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim * 2),
#             nn.SiLU(),
#             nn.Linear(hidden_dim * 2, num_heads, bias=False)
#         )

#         self.weight = nn.Sequential(nn.Linear(dist_dim, 1), nn.Sigmoid())

#     def forward(self, h, pos, edge_index, edge_attr, dist, time_emb=None):
#         row, col = edge_index
#         h_input = torch.cat([h[row], h[col], edge_attr, dist], dim=1)
#         coord_diff = pos[row] - pos[col]
#         coord_diff = torch.norm(coord_diff, p=2, dim=-1, keepdim=True)
#         inv = self.ln(self.input_lin(h_input))

#         k = self.lin_key(inv).view(-1, self.num_heads, self.hidden_dim // self.num_heads)
#         q = self.lin_query(h).view(-1, self.num_heads, self.hidden_dim // self.num_heads)
#         # v = self.lin_value(inv).view(-1, 16, self.hidden_dim // 16)
#         v = self.lin_value(inv)
#         v = v * self.weight(dist)
#         v = v.unsqueeze(-1) * coord_diff.unsqueeze(1)
#         alpha = scatter_softmax((q[edge_index[0]] * k / np.sqrt(k.shape[-1])).sum(-1), edge_index[0], dim=0, dim_size=h.shape[0])
#         m = alpha.unsqueeze(-1) * v
#         agg = scatter_sum(m, edge_index[0], dim=0, dim_size=h.shape[0])
#         pos = pos + agg.mean(1)
#         return pos

class Trans_Layer_test2(MessagePassing):
    """The version for involving the edge feature. Multiply Msg. Without FFN and norm."""

    _alpha: OptTensor

    def __init__(self, x_channels: int, out_channels: int,
                 heads: int = 1, dropout: float = 0., edge_dim: Optional[int] = None,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(Trans_Layer_test2, self).__init__(node_dim=0, **kwargs)

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

    def forward(self, x: OptTensor, p:OptTensor,
                edge_index: Adj,
                edge_attr: OptTensor = None
                ) -> Tensor:

        H, C = self.heads, self.out_channels
        hi, hj = x[edge_index[0]], p[edge_index[1]]
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