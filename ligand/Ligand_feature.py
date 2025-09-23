import torch
import torch.nn as nn
import torch_geometric
from .gvp import GVP, GVPConvLayer, LayerNorm




class GVPModel(nn.Module):
    def __init__(self, config, 
                 node_in_dim=[15, 1], node_h_dim=[128, 64],
                 edge_in_dim=[19, 1], edge_h_dim=[128, 64],
                 num_rbf=16, num_layers=3, dropout=0.1, vec_out=0):
        """
        Parameters
        ----------
        node_in_dim : list of int
            Input dimension of drug node features (si, vi).
            Scalar node feartures have shape (N, si).
            Vector node features have shape (N, vi, 3).
        node_h_dims : list of int
            Hidden dimension of drug node features (so, vo).
            Scalar node feartures have shape (N, so).
            Vector node features have shape (N, vo, 3).
        """
        super(GVPModel, self).__init__()

        self.num_rbf = num_rbf
        self.node_emb = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )

        self.edge_emb = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )

        self.layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=dropout)
            for _ in range(num_layers))

        self.out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (node_h_dim[0], vec_out)))

    def forward(self, ligand_atom, ligand_pos, ligand_bond_feature, ligand_bond_index):
        
        # ligand_batch = batch.ligand_element_batch

        # ligand_node_v = batch.ligand_pos.unsqueeze(1)
        # ligand_node_s = torch.as_tensor(batch.ligand_atom_feature_full, dtype=torch.float32)  # dim=15
        ligand_node_v = ligand_pos.unsqueeze(1)
        ligand_node_s = torch.as_tensor(ligand_atom, dtype=torch.float32)  # dim=15
        h_node = (ligand_node_s, ligand_node_v)

        ligand_bond_feature = ligand_bond_feature
        bond_index = ligand_bond_index
        ligand_edge_v, ligand_edge_s = self._build_edge_feature(ligand_node_v, bond_index, ligand_bond_feature)
        h_edge = (ligand_edge_s, ligand_edge_v)
        
        ligand_node = self.node_emb(h_node)
        ligand_edge = self.edge_emb(h_edge)

        for layer in self.layers:
            ligand_node = layer(ligand_node, bond_index, ligand_edge)
        out = self.out(ligand_node)
        # per-graph mean
        # out = torch_geometric.nn.global_add_pool(out, ligand_batch)
        ligand_s, ligand_v = out
        ligand_v = ligand_v.squeeze(1)
        out = (ligand_s, ligand_v)
        return out
    
    def _build_edge_feature(self, pos, edge_index, bond_type):
        edge_vector = pos[edge_index[0]] - pos[edge_index[1]]
        edge_s = self._rbf(edge_vector.norm(dim=-1))
        edge_v = self._normalize(edge_vector)

        edge_s, edge_v = map(torch.nan_to_num, (edge_s, edge_v))
        edge_s = torch.cat([bond_type, edge_s.squeeze()], dim=-1)

        return edge_v, edge_s
    

    def _rbf(self, D):
        D_min, D_max, D_count = 0., 4.5, self.num_rbf
        #D_mu = torch.linspace(D_min, D_max, D_count).cuda()
        D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
        D_mu = D_mu.view([1,1,1,-1]).contiguous()
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)

        return RBF
    
    def _normalize(self, tensor, dim=-1):
        '''
        Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
        '''
        return torch.nan_to_num(
            torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))