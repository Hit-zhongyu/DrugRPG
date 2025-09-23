import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_softmax, scatter_sum, scatter_mean, scatter
from .utils import *
from .layers import * 

from torch_geometric.nn import radius, knn
# from .inter import Trans_Atom_Inter_Layer
from .inter_layer import Ligand_Inter_Atom_Layer
from .intra_layer import Ligand_Intra_Atom_Layer, Ligand_Intra_Bond_Layer
from .equivariance import Inter_pos_Update, Intra_pos_Update

class DiffusionTransformer(nn.Module):
    def __init__(self, h_input_dim, hidden_dim, edge_input_dim, edge_hidden_dim, dist_dim, n_layers, p_n_layers,
                  n_heads, p_input_dim, cutoff=3, dropout=0.2):
        super().__init__()
        
        self.cutoff = cutoff
        self.n_layers = n_layers
        self.p_n_layers = p_n_layers

        time_dim = edge_hidden_dim

        self.ligand_atom_emb = nn.Linear(h_input_dim, hidden_dim)
        self.protein_atom_emb = nn.Linear(p_input_dim, hidden_dim)
        self.ligand_edge_emb = nn.Linear(edge_input_dim, edge_hidden_dim)
        self.edge_type_emb = nn.Linear(4, edge_hidden_dim)
        self.lin_node = nn.Linear(hidden_dim, hidden_dim)
        # self.lin_edge = nn.LayerNorm(edge_hidden_dim)

        for i in range(n_layers):
            self.add_module("e_block_inter_atom_%d" % i, Ligand_Inter_Atom_Layer(hidden_dim, edge_hidden_dim, dist_dim, time_dim, 
                                                               n_heads, dropout=dropout,))

            self.add_module("e_block_intra_atom_%d" % i, Ligand_Intra_Atom_Layer(hidden_dim, edge_hidden_dim, time_dim,
                                                                n_heads, dropout=dropout))
            
            self.add_module("e_block_intra_bond_%d" % i, Ligand_Intra_Bond_Layer(hidden_dim, edge_hidden_dim, time_dim))

            self.add_module("e_block_inter_pos_%d" % i, Inter_pos_Update(hidden_dim, edge_hidden_dim, dist_dim, time_dim))
            self.add_module("e_block_intra_pos_%d" % i, Intra_pos_Update(hidden_dim, edge_hidden_dim, dist_dim, time_dim))

        self.node_pred_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, h_input_dim)
        )
        self.node_pred_norm = nn.LayerNorm(hidden_dim)

        self.edge_pred_mlp = nn.Sequential(
            nn.Linear(edge_hidden_dim, edge_hidden_dim // 2),
            nn.LayerNorm(edge_hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(edge_hidden_dim // 2, edge_input_dim)
        )
        self.edge_pred_norm = nn.LayerNorm(edge_hidden_dim)

        self.time_mlp = FourierEmbedding(time_dim)

        # self.project_mlp = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim * 2),
        #     nn.SiLU(),
        #     nn.Linear(hidden_dim * 2, hidden_dim * 4),
        #     nn.SiLU(),
        #     nn.Linear(hidden_dim * 4, 2048)
        # )

    def bulid_context(self, protein_atom, ligand_atom, protein_pos, ligand_pos, protein_batch, ligand_batch
                      ,linker_mask=None, linker_edge_index=None):

        batch_ctx = torch.cat([protein_batch, ligand_batch], dim=0)
        sort_idx = torch.sort(batch_ctx, stable=True).indices

        ligand_mask = torch.cat([
                torch.zeros([protein_batch.size(0)], device=protein_batch.device).bool(),
                torch.ones([ligand_batch.size(0)], device=ligand_batch.device).bool(),
            ], dim=0)[sort_idx]
        
        if linker_mask is not None:
            linker_mask = torch.cat([
                torch.zeros([protein_pos.size(0)],device=protein_batch.device).bool(),
                linker_mask.bool(),
            ], dim=0)[sort_idx]

        batch_ctx = batch_ctx[sort_idx]
        h_ctx = torch.cat([protein_atom, ligand_atom], dim=0)[sort_idx]  # (N_protein+N_ligand, H)
        pos_ctx = torch.cat([protein_pos, ligand_pos], dim=0)[sort_idx]  # (N_protein+N_ligand, 3)

        ligand_index_in_ctx = find_index_after_sorting(
                        len(h_ctx), len(protein_atom), len(ligand_atom), sort_idx, ligand_batch.device)
        return h_ctx, pos_ctx, batch_ctx, ligand_mask, linker_mask, ligand_index_in_ctx

    def build_edge_index(self, ligand_pos, protein_pos, ligand_batch, protein_batch, ligand_edge_index, protein_edge_index):

        inter_edge_index = knn(y=ligand_pos, x=protein_pos, k=24,
                                    batch_x=protein_batch, batch_y=ligand_batch)
        

        edge_index = merge_batched_graph_indices(ligand_batch, protein_batch, ligand_edge_index, protein_edge_index, inter_edge_index)

        return edge_index
    
    def build_edge_type(self, edge_index, ligand_mask):
        """
        Args:
            edge_index: (2, E)
            mask_ligand: (N, )
            decomp_group_idx: (N, )
        """
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = ligand_mask[src] == 1
        n_dst = ligand_mask[dst] == 1
        edge_type[n_src & n_dst] = 0
        edge_type[n_src & ~n_dst] = 1
        edge_type[~n_src & n_dst] = 2
        edge_type[~n_src & ~n_dst] = 3

        edge_type = F.one_hot(edge_type, num_classes=4)
        return edge_type.float()
    
    def forward(self, batch, h, x, edge_feat, time,  ligand_edge_index, ligand_batch, protein_batch, protein_pos_norm,
                linker_mask=None):

        device = h.device 
        n = edge_feat.shape[0] // 2
        protein_atom = batch['protein_atom'].to(device).float()
        protein_edge_index = batch['protein_bond_index'].to(device)
        
        ligand_atom = h
        ligand_pos = x
        protein_atom = protein_atom
        protein_pos = protein_pos_norm

        ligand_atom = self.ligand_atom_emb(ligand_atom)
        ligand_edge_attr = self.ligand_edge_emb(edge_feat)
        protein_atom = self.protein_atom_emb(protein_atom)

        time_emb = self.time_mlp(time)

        h_ctx, pos_ctx, batch_ctx, ligand_mask, linker_mask, l_index_in_ctx  = self.bulid_context(protein_atom, ligand_atom, 
                                                            protein_pos, ligand_pos, protein_batch, ligand_batch, linker_mask)
        
        edge_index = self.build_edge_index(ligand_pos, protein_pos, ligand_batch, protein_batch, ligand_edge_index, protein_edge_index)
        edge_type = self.build_edge_type(edge_index, ligand_mask)
        edge_type_emb = self.edge_type_emb(edge_type)

        bond_index_in_all = l_index_in_ctx[ligand_edge_index]

        node_time_emb = time_emb[batch_ctx]
        edge_time_emb = time_emb[batch_ctx[edge_index[0]]]

        edge_time_emb2 = time_emb[batch_ctx[bond_index_in_all[0]]]

        if linker_mask is not None:
            ligand_mask_ = ligand_mask
            if not frag:
                ligand_mask = linker_mask
            
        else:
            ligand_mask_ = ligand_mask
        # torch.set_printoptions(profile="full")
        # print(ligand_mask)
        for i in range(self.n_layers):

            intra_atom_update = self._modules['e_block_intra_atom_%d' % i](pos_ctx, h_ctx, ligand_edge_attr, bond_index_in_all, node_time_emb)
            h_ctx = h_ctx + intra_atom_update
            inter_atom_update = self._modules['e_block_inter_atom_%d' % i](pos_ctx, h_ctx, edge_type_emb, edge_index, node_time_emb)
            h_ctx = h_ctx + inter_atom_update

            intra_pos_update = self._modules['e_block_intra_pos_%d' % i](pos_ctx, h_ctx, bond_index_in_all, ligand_edge_attr, edge_time_emb2)
            pos_ctx = pos_ctx + intra_pos_update * ligand_mask[:, None]
            inter_pos_update = self._modules['e_block_inter_pos_%d' % i](pos_ctx, h_ctx, edge_index, edge_type_emb, edge_time_emb)
            pos_ctx = pos_ctx + inter_pos_update * ligand_mask[:, None]

            edge_attr_update = self._modules['e_block_intra_bond_%d' % i](h_ctx, bond_index_in_all, ligand_edge_attr, edge_time_emb2)
            ligand_edge_attr = ligand_edge_attr + edge_attr_update

            # if linker_edge_feat is not None:
            #     h_ctx_ = h_ctx[ligand_mask_]
            #     # pos_ctx_ = pos_ctx[ligand_mask_]
            #     edge_attr_update_ = self._modules['e_block_intra_bond_%d' % i](h_ctx_, linker_edge_index, linker_edge_attr, edge_time_emb3)
            #     linker_edge_attr = linker_edge_attr + edge_attr_update_
            
            if i+1 == 4:
                zs = h_ctx[ligand_mask_] 
                # zs = self.project_mlp(h_ctx[ligand_mask_])
            

        ligand_atom = h_ctx[ligand_mask_]
        ligand_atom = self.node_pred_norm(ligand_atom)
        atom_pred = self.node_pred_mlp(ligand_atom)
        
        # if linker_edge_feat is not None:
        #     ligand_edge_attr = self.edge_pred_norm(linker_edge_attr)
        #     n = linker_edge_attr.shape[0] // 2
        #     ligand_edge_attr = ligand_edge_attr[:n,:] + ligand_edge_attr[n:,:]
        # else:
        ligand_edge_attr = self.edge_pred_norm(ligand_edge_attr)
        ligand_edge_attr = ligand_edge_attr[:n,:] + ligand_edge_attr[n:,:]
        edge_pred = self.edge_pred_mlp(ligand_edge_attr)

        pos_pred = pos_ctx[ligand_mask_]

        # return atom_pred, pos_pred, edge_pred
        return atom_pred, pos_pred, edge_pred, zs



