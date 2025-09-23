
import torch
from torch_scatter import scatter_min
from torch_cluster import knn_graph

def G_fn(protein_coords, x, sigma):
    # protein_coords: (n,3) , x: (m,3), output: (m,)
    e = torch.exp(-torch.sum((protein_coords.view(1, -1, 3) - x.view(-1, 1, 3)) ** 2, dim=2) / float(sigma))  # (m, n)
    return -sigma * torch.log(1e-3 + e.sum(dim=1))


def compute_body_intersection_loss(protein_coords, ligand_coords, sigma, surface_ct):
    loss = torch.mean(torch.clamp(surface_ct - G_fn(protein_coords, ligand_coords, sigma), min=0))
    return loss


def compute_batch_clash_loss(protein_pos, pred_ligand_pos, batch_protein, batch_ligand, sigma=25, surface_ct=10):
    loss_clash = torch.tensor(0., device=protein_pos.device)
    num_graphs = batch_ligand.max().item() + 1
    for i in range(num_graphs):
        p_pos = protein_pos[batch_protein == i]
        l_pos = pred_ligand_pos[batch_ligand == i]
        loss_clash += compute_body_intersection_loss(p_pos, l_pos, sigma=sigma, surface_ct=surface_ct)
    return loss_clash


@torch.jit.script
def compute_armsca_prox_loss(pos: torch.Tensor, edge_index: torch.Tensor, min_d: float = 1.2, 
                            max_d: float = 1.8) -> torch.Tensor:
    bond_length = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], p=2, dim=-1)
    loss = torch.mean(
        torch.relu(min_d - bond_length) +  # 距离太小的惩罚
        torch.relu(bond_length - max_d)    # 距离太大的惩罚
    )
    return loss

@torch.jit.script
def compute_loss_single_graph(pos, edge_flat: torch.Tensor, min_d: float, max_d: float) -> torch.Tensor:
    n_nodes = pos.size(0)
    indices = torch.tril_indices(n_nodes, n_nodes, offset=-1, device=pos.device)
    row = indices[0]
    col = indices[1]
    valid_mask = edge_flat > 0
    if valid_mask.sum() == 0:
        return torch.tensor(0., device=pos.device)
    selected_edge_index = torch.stack((row[valid_mask], col[valid_mask]), dim=0)
    loss = compute_armsca_prox_loss(pos, selected_edge_index, min_d, max_d)
    return loss

def compute_batch_armsca_prox_loss(pred_ligand_pos, pred_ligand_edge, ligand_batch, edge_batch, min_d=1.2, max_d=1.8):
    
    device=pred_ligand_pos.device
    batch_losses = torch.tensor(0., device=device)
    num_graphs = ligand_batch.max().item() + 1
    n_valid = 0

    # for i in range(num_graphs):
    #     pos_mask = (ligand_batch == i)
    #     edge_mask = (edge_batch == i)

    #     pos = pred_ligand_pos[pos_mask]
    #     edge_flat = pred_ligand_edge[edge_mask]

    #     n_nodes = pos.shape[0]
    #     edge_index = []
    #     idx = 0
    #     for i_idx in range(1, n_nodes):      # i > j
    #         for j_idx in range(i_idx):       # j < i
    #             if edge_flat[idx] > 0:       # 如果有边
    #                 edge_index.append([i_idx, j_idx])
    #             idx += 1
    #     edge_index = torch.tensor(edge_index, device=pos.device).t()
    #     loss = compute_armsca_prox_loss(pos, edge_index, min_d=min_d, max_d=max_d)
    
    #     batch_losses += loss
    #     n_valid += 1

    for i in range(num_graphs):
        # 选择当前图中的节点与边
        pos_mask = (ligand_batch == i)
        edge_mask = (edge_batch == i)

        pos = pred_ligand_pos[pos_mask]

        edge_flat = pred_ligand_edge[edge_mask]
        loss = compute_loss_single_graph(pos, edge_flat, min_d, max_d)

        batch_losses += loss
        n_valid += 1

    return batch_losses / num_graphs, n_valid


