import torch
import torch.nn as nn
import torch_geometric

from math import pi as PI
from torch_geometric.nn import MessagePassing
from torch.nn import Module, Sequential, ModuleList, Linear, Conv1d

from ..model.common import ShiftedSoftplus, GaussianSmearing


class Res3DGraphModel(nn.Module):
    def __init__(self, vocab=26, node_dim=32, res_emb=64, residue_cutoff=6, edge_dim=64, heads=8, gcn=[64, 128, 256, 256]):
        super(Res3DGraphModel, self).__init__()
        # self.node_emb = nn.Embedding(vocab, node_dim)
        # self.res_layernorm = nn.LayerNorm(node_dim+6)
        # self.res_emb = nn.Linear(vocab,  res_emb)
        self.node_emb = nn.Linear(vocab, res_emb)
        # self.res_layernorm = nn.LayerNorm(res_emb)

        # self.edge_layernorm = nn.LayerNorm(edge_in_dim)
        # self.edge_emb = nn.Linear(edge_in_dim, edge_dim)
        self.edge_emb = GaussianSmearing(stop=residue_cutoff, num_gaussians=edge_dim)
        layers = []
        for i in range(len(gcn) - 1):
            layers.append((
                torch_geometric.nn.TransformerConv(
                    gcn[i], gcn[i + 1], edge_dim=edge_dim, heads=heads, dropout=0.1),
                'x, edge_index, edge_attr -> x'
            ))
            layers.append(nn.LeakyReLU())

        self.gcn = torch_geometric.nn.Sequential(
            'x, edge_index, edge_attr', layers)        
        # self.pool = torch_geometric.nn.global_mean_pool

    def forward(self, residue_atom, edge_index, edge_length):

        node_emb = self.node_emb(residue_atom)
        # res = torch.cat([node_emb, residue_attr], dim=-1)
        # res = self.res_layernorm(res)
        # res_emb = self.res_layernorm(node_emb)

        edge_emb = self.edge_emb(edge_length)
        
        res_feat = self.gcn(node_emb, edge_index, edge_emb)
       
        # res_feat = torch_geometric.nn.global_mean_pool(res_feat, batch)
        return res_feat

class CFConv(MessagePassing):

    def __init__(self, in_channels, out_channels, num_filters, edge_channels, cutoff=10.0, smooth=False):
        super().__init__(aggr='add')
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn_W = Sequential(
            Linear(edge_channels, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )  # Network for generating filter weights
        self.cutoff = cutoff
        self.smooth = smooth

    def forward(self, x, edge_index, edge_length, edge_attr):
        W = self.nn_W(edge_attr)

        if self.smooth:
            C = 0.5 * (torch.cos(edge_length * PI / self.cutoff) + 1.0)
            C = C * (edge_length <= self.cutoff) * (edge_length >= 0.0)  # Modification: cutoff
        else:
            C = (edge_length <= self.cutoff).float()
        # if self.cutoff is not None:
        #     C = 0.5 * (torch.cos(edge_length * PI / self.cutoff) + 1.0)
        #     C = C * (edge_length <= self.cutoff) * (edge_length >= 0.0)     # Modification: cutoff
        W = W * C.view(-1, 1).contiguous()
        W = W.contiguous()
        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x  

class InteractionBlock(Module):

    def __init__(self, hidden_channels, out_channels, num_gaussians, num_filters, cutoff, smooth=False):
        super(InteractionBlock, self).__init__()
        self.conv = CFConv(in_channels=hidden_channels, out_channels=out_channels, 
                           num_filters=num_filters, edge_channels=num_gaussians, 
                           cutoff=cutoff, smooth=smooth)   # 消息传递
        self.act = ShiftedSoftplus()   # 平衡函数
        self.lin = Linear(out_channels, out_channels)

    def forward(self, x, edge_index, edge_length, edge_attr):
        x = self.conv(x, edge_index, edge_length, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x

class Atom3DGraphModel(nn.Module):
    def __init__(self, atom_in_dim=39, node_dim=64, edeg_dim=64, cutoff=2.5, num_filters=256, out_channels=64, atom_cutoff=10, num_interactions=6):
        super(Atom3DGraphModel, self).__init__()
        self.node_emb = nn.Linear(atom_in_dim, node_dim)
        self.edge_emb = GaussianSmearing(stop=cutoff, num_gaussians=edeg_dim)

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels=node_dim, out_channels=out_channels, num_gaussians=edeg_dim,
                                     num_filters=num_filters,cutoff=atom_cutoff, smooth=True)
            self.interactions.append(block)

    def forward(self, atom, atom_index, edge_length):
        # atom = (batch.protein_atom_feature).to(dtype=torch.float32)
        atom = atom.to(dtype=torch.float32)
        edge_attr = self.edge_emb(edge_length)
        # print(atom)
        h = self.node_emb(atom) 
        for interaction in self.interactions:
            h = h + interaction(h, atom_index, edge_length, edge_attr)
        return h











