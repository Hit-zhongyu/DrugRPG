import torch
import torch.nn as nn
import numpy as np
from torch_scatter import scatter
from .utils import *
import torch.nn.functional as F
from torch_scatter import scatter_sum

class Inter_pos_Update(nn.Module):
    """Update atom coordinates equivariantly, use time emb condition."""

    def __init__(self, hidden_dim, edge_dim, dist_dim, time_dim):
        super().__init__()

        self.node_left_lin = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.node_right_lin = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.edge_lin = nn.Linear(edge_dim + dist_dim, hidden_dim)
        self.node_lin = nn.Linear(hidden_dim, hidden_dim)

        self.inter_feat_mlp = nn.Sequential(
            nn.Linear(hidden_dim + time_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, 1)
        )

        self.scale_net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )

        self.dist_layer = GaussianSmearing(stop=15, num_gaussians=dist_dim)

    def forward(self, pos, h, edge_index, edge_attr, time):
        row, col = edge_index

        relative_vec = pos[row] - pos[col]
        # distance = torch.norm(relative_vec, p=2, dim=-1)
        distance = relative_vec.pow(2).sum(dim=-1).sqrt()
        dist_emb = self.dist_layer(distance)
        dis_uns = distance.unsqueeze(-1)

        direction = F.normalize(relative_vec, p=2, dim=-1)

        h_left = self.node_left_lin(h[row])
        h_right = self.node_right_lin(h[col])

        edge_attr = self.edge_lin(torch.cat([edge_attr, dist_emb],dim=-1))
        node_feat = self.node_lin(h_left * h_right)
        inter_feat = self.inter_feat_mlp(torch.cat([edge_attr * node_feat, time], dim=-1))

        force_edge = inter_feat / (dis_uns + 1.) * direction
        delta_pos = scatter_sum(force_edge, row, dim=0, dim_size=h.shape[0])

        delta_pos = delta_pos * self.scale_net(torch.cat([h, torch.norm(delta_pos, dim=-1, keepdim=True)],dim=-1))

        return delta_pos

class Intra_pos_Update(nn.Module):
    """Update atom coordinates equivariantly, use time emb condition."""

    def __init__(self, hidden_dim, edge_dim, dist_dim, time_dim):
        super().__init__()

        self.node_left_lin = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.node_right_lin = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.edge_lin = nn.Linear(edge_dim + dist_dim, hidden_dim)
        self.node_lin = nn.Linear(hidden_dim, hidden_dim)

        self.inter_feat_mlp = nn.Sequential(
            nn.Linear(hidden_dim + time_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, 1)
        )

        self.scale_net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )

        self.dist_layer = GaussianSmearing(stop=15, num_gaussians=dist_dim)

    def forward(self, pos, h, edge_index, edge_attr, time):
        row, col = edge_index

        relative_vec = pos[row] - pos[col]
        # distance = torch.norm(relative_vec, p=2, dim=-1)
        distance = relative_vec.pow(2).sum(dim=-1).sqrt()
        dist_emb = self.dist_layer(distance)
        dis_uns = distance.unsqueeze(-1)
        
        direction = F.normalize(relative_vec, p=2, dim=-1)

        h_left = self.node_left_lin(h[row])
        h_right = self.node_right_lin(h[col])

        edge_attr = self.edge_lin(torch.cat([edge_attr, dist_emb],dim=-1))
        node_feat = self.node_lin(h_left * h_right)
        inter_feat = self.inter_feat_mlp(torch.cat([edge_attr * node_feat, time], dim=-1))

        # gate = self.gate(torch.cat([edge_attr, node_feat, time], dim=-1))
        # inter_feat = inter_feat * torch.sigmoid(gate)

        force_edge = inter_feat / (dis_uns + 1.) * direction
        delta_pos = scatter_sum(force_edge, row, dim=0, dim_size=h.shape[0])
        delta_pos = delta_pos * self.scale_net(torch.cat([h, torch.norm(delta_pos, dim=-1, keepdim=True)],dim=-1))

        return delta_pos