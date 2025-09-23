import torch
from torch import nn
import numpy as np

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class CoorsNorm(nn.Module):
    def __init__(self, eps=1e-8, scale_init=1.):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors):
        norm = coors.norm(dim=-1, keepdim=True)
        normed_coors = coors / norm.clamp(min=self.eps)
        return normed_coors * self.scale

def coord2dist(x, edge_index):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)
    return radial

def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff/(norm + norm_constant)
    return radial, coord_diff

def outer_product(*vectors):
    for index, vector in enumerate(vectors):
        if index == 0:
            out = vector.unsqueeze(-1)
        else:
            out = out * vector.unsqueeze(1)
            out = out.view(out.shape[0], -1).unsqueeze(-1)
    return out.squeeze()


def coord2diff_adj(x, edge_index, spatial_th=2.):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)
    with torch.no_grad():
        adj_spatial = radial.clone()
        adj_spatial[adj_spatial <= spatial_th] = 1.
        adj_spatial[adj_spatial > spatial_th] = 0.
    return radial, adj_spatial

def clash_pos_modify(protein_pos, ligand_pos, clash_threshold=2.0, k=3, max_iter=5):
    ligand_pos = ligand_pos.clone()
    
    for _ in range(max_iter):
        dists = torch.cdist(ligand_pos, protein_pos, p=2)
        
        topk_dists, topk_idxs = torch.topk(dists, k=k, dim=1, largest=False)
        
        topk_dists, topk_idxs = torch.topk(dists, k=k, dim=1, largest=False)
        close_mask = (topk_dists < clash_threshold).any(dim=1)

        if not close_mask.any():
            break
        close_idxs = close_mask.nonzero(as_tuple=False).squeeze(1)
        topk_dists_clash = topk_dists[close_idxs]
        topk_idxs_clash = topk_idxs[close_idxs]

        threshold_dist = torch.clamp(clash_threshold - topk_dists_clash, min=0)
        push_dist = threshold_dist.mean(dim=1)

        nearest_protein_pos = protein_pos[topk_idxs_clash]  # [N, k, 3]
        mean_protein_pos = nearest_protein_pos.mean(dim=1)

        direction = ligand_pos[close_idxs] - mean_protein_pos
        norm = direction.norm(dim=1, keepdim=True).clamp(min=1e-8)
        unit_direction = direction / norm

        push_dist = push_dist + 0.2 * torch.rand_like(push_dist)
        ligand_pos.index_add_(0, close_idxs, push_dist.unsqueeze(1) * unit_direction - ligand_pos[close_idxs] + ligand_pos[close_idxs])

    return ligand_pos

def find_index_after_sorting(size_all, size_p, size_l, sort_idx, device):
    # find protein/ligand index in ctx
    ligand_index_in_ctx = torch.zeros(size_all, device=device)
    ligand_index_in_ctx[size_p:size_p + size_l] = torch.arange(1, size_l + 1, device=device)
    ligand_index_in_ctx = torch.sort(ligand_index_in_ctx[sort_idx], stable=True).indices[-size_l:]
    ligand_index_in_ctx = ligand_index_in_ctx.to(device)

    # protein_index_in_ctx = torch.zeros(size_all, device=device)
    # protein_index_in_ctx[:size_p] = torch.arange(1, size_p + 1, device=device)
    # protein_index_in_ctx = torch.sort(protein_index_in_ctx[sort_idx], stable=True).indices[-size_p:]
    # protein_index_in_ctx = protein_index_in_ctx.to(device)
    return ligand_index_in_ctx

def merge_batched_graph_indices(ligand_batch, protein_batch, ligand_edge_index, protein_edge_index, inter_edge_index):
    
    device = protein_batch.device
    
    def get_counts(batch):
        changes = torch.cat([
            torch.tensor([True], device=device),
            batch[1:] != batch[:-1],
            torch.tensor([True], device=device)
        ])
        indices = torch.where(changes)[0]
        return indices[1:] - indices[:-1]
    
    prot_counts = get_counts(protein_batch)
    lig_counts = get_counts(ligand_batch)
    
    batch_sizes = prot_counts + lig_counts
    prot_offset = torch.cat([torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(batch_sizes, dim=0)[:-1]])
    lig_offset = prot_offset + prot_counts
    
    prot_local = torch.arange(len(protein_batch), device=device) - torch.repeat_interleave(
        torch.cat([torch.tensor([0], device=device), prot_counts.cumsum(dim=0)[:-1]]),
        prot_counts
    )
    
    lig_local = torch.arange(len(ligand_batch), device=device) - torch.repeat_interleave(
        torch.cat([torch.tensor([0], device=device), lig_counts.cumsum(dim=0)[:-1]]),
        lig_counts
    )
    
    prot_global = prot_offset[protein_batch] + prot_local
    lig_global = lig_offset[ligand_batch] + lig_local
    
    p_src = prot_global[protein_edge_index[0]]
    p_dst = prot_global[protein_edge_index[1]]
    # ligand-ligand
    l_src = lig_global[ligand_edge_index[0]]
    l_dst = lig_global[ligand_edge_index[1]]
    # ligand->protein
    i_src = lig_global[inter_edge_index[0]]
    i_dst = prot_global[inter_edge_index[1]]
    
    combined = torch.cat([
        torch.stack([p_src, p_dst], dim=0), # protein-protein
        torch.stack([i_dst, i_src], dim=0), # protein<-ligand
        torch.stack([i_src, i_dst], dim=0), # ligand->protein
        torch.stack([l_src, l_dst], dim=0), # ligand-ligand
    ], dim=1)
    
    return combined

class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=10.0, num_gaussians=50, type_='exp'):
        super().__init__()
        self.start = start
        self.stop = stop
        if type_ == 'exp':
            offset = torch.exp(torch.linspace(start=np.log(start+1), end=np.log(stop+1), steps=num_gaussians)) - 1
        elif type_ == 'linear':
            offset = torch.linspace(start=start, end=stop, steps=num_gaussians)
        else:
            raise NotImplementedError('type_ must be either exp or linear')
        diff = torch.diff(offset)
        diff = torch.cat([diff[:1], diff])
        coeff = -0.5 / (diff**2)
        self.register_buffer('coeff', coeff)
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.clamp_min(self.start)
        dist = dist.clamp_max(self.stop)
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))
    
