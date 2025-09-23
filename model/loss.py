import torch
from .utils import *
from torch_scatter import scatter_mean

def rmsd_squared_loss(ori_pos, pred_pos, ligand_batch):

    ori_pos = ori_pos - scatter_mean(ori_pos, ligand_batch, dim=0)[ligand_batch]
    pred_pos = pred_pos - scatter_mean(pred_pos, ligand_batch, dim=0)[ligand_batch]

    align_pos_0 = torch.zeros_like(pred_pos) 
    for batch in ligand_batch.unique():
        indices = (ligand_batch == batch)
        coord_pred = pred_pos[indices]
        coord_tar = ori_pos[indices]
        rotations = kabsch(coord_pred, coord_tar)
        align_pos = torch.einsum("ij,bj->bi", rotations, coord_tar)
        align_pos_0[indices] = align_pos

    rmsd_squared_loss = torch.sum((pred_pos - align_pos_0) ** 2, dim=-1)
    return rmsd_squared_loss

def kabsch(coord_pred, coord_tar):
    A = coord_pred.transpose(0, 1) @ coord_tar  # [3, 3]
    U, S, Vt = torch.linalg.svd(A)
    corr_mat = torch.diag(torch.tensor([1, 1, torch.sign(torch.det(A))], device=coord_pred.device))  # [3, 3]
    rotation = (U @ corr_mat) @ Vt
    return rotation

def REPA_loss(z, zs):
    z = torch.nn.functional.normalize(z, dim=-1)
    zs = torch.nn.functional.normalize(zs, dim=-1)
    j = torch.sum(z * zs, dim=-1) 
    repa_loss = (1 - j)
    return repa_loss

def bond_length_loss(pos_pred, ligand_pos_norm, edge_index, ligand_batch, loss_weight, valid):
    edge_batch_ = build_edge_batch(ligand_batch, edge_index)
    bond_length_ori = torch.norm(ligand_pos_norm[edge_index[0]] - ligand_pos_norm[edge_index[1]], dim=-1)
    bond_length_pred = torch.norm(pos_pred[edge_index[0]] - pos_pred[edge_index[1]], dim=-1)
    bond_loss = scatter_mean(((bond_length_pred - bond_length_ori) ** 2), edge_batch_, dim=0)
    if not valid:
       bond_loss = bond_loss * loss_weight    
    bond_loss = torch.mean(bond_loss)
    # bond_loss = torch.zeros_like(loss_pos)
    return bond_loss

def compute_lj_energy_loss(pred_pos, protein_pos, ligand_batch, protein_batch, num_graphs, weights, valid,
                         N=6, dm_min=0.5, k=16):
        # total_energy = 0.0
        energy_list = []

        for batch_idx in range(num_graphs):
            ligand_coor = pred_pos[ligand_batch == batch_idx]
            protein_coor = protein_pos[protein_batch == batch_idx]

            if ligand_coor.size(0) == 0 or protein_coor.size(0) == 0:
                continue

            # 计算成对距离
            p1_repeat = ligand_coor.unsqueeze(1).repeat(1, protein_coor.size(0), 1)
            p2_repeat = protein_coor.unsqueeze(0).repeat(ligand_coor.size(0), 1, 1)
            dm = torch.sqrt(torch.pow(p1_repeat - p2_repeat, 2).sum(-1) + 1e-10)

            _, topk_indices = torch.topk(dm, k, dim=1, largest=False, sorted=False)
            dm = torch.gather(dm, 1, topk_indices)

            # 避免距离过小
            replace_vec = torch.ones_like(dm) * 1e10
            dm = torch.where(dm < dm_min, replace_vec, dm)

            # 获取配体原子类型及其范德华半径
            # ligand_recon_atom_type = log_sample_categorical(log_ligand_v_recon[batch_ligand == batch_idx])
            # pred_log_atom_type = get_atomic_number_from_index(ligand_recon_atom_type, mode="add_aromatic")
            # ligand_vdw_radii = torch.tensor([
            #     VDWRADII[atom_idx] for atom_idx in pred_log_atom_type
            # ], device=ligand_coor.device, dtype=torch.float)

            # 广播 vdW 半径为矩阵
            # dm_0 = ligand_vdw_radii.unsqueeze(1).repeat(1, protein_coor.size(0))

            # 计算 LJ 势能（标准形式）
            # vdw_term1 = torch.pow(dm_0 / dm, 2 * N)
            # vdw_term2 = -2 * torch.pow(dm_0 / dm, N)
            vdw_term1 = 1 * torch.pow(3 / dm, 2 * N)
            vdw_term2 = -2 * torch.pow(3 / dm, N)
            energy = vdw_term1 + vdw_term2
            energy = energy.clamp(max=100)
            if not valid:
               energy = energy * weights[batch_idx]
            # total_energy += energy.mean()
            energy_list.append(energy.mean())


            # 计算梯度作为损失
            # der = torch.autograd.grad(energy.sum(), dm, retain_graph=True, create_graph=True)[0]
            # total_energy += der.sum()
            if len(energy_list) == 0:
                return torch.tensor(0.0, device=pred_pos.device)

        return torch.stack(energy_list).abs().mean()


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


# def lj_potential(protein_coords, ligand_coords, sigma=1.0, eps_rep=1.0, eps_att=0.1, r_cutoff=6.0):
#     diff = ligand_coords[:, None, :] - protein_coords[None, :, :]  # (m, n, 3)
#     r2 = torch.sum(diff**2, dim=-1) + 1e-12  # Avoid divide-by-zero

#     mask = r2 < (r_cutoff ** 2)

#     inv_r6 = (sigma**2 / r2) ** 3
#     repulsion = eps_rep * (inv_r6 ** 2)      # 1/r^12
#     attraction = eps_att * inv_r6            # 1/r^6

#     lj = repulsion - attraction
#     lj = lj * mask 
#     return lj.sum()

# def compute_batch_lj_loss(protein_pos, ligand_pos, batch_protein, batch_ligand, eps_rep=1.0, eps_att=0.1,
#                     sigma=2.5, r_cutoff=6.0):
#     # loss_lj = torch.tensor(0., device=protein_pos.device)
#     loss_lj_list = []
#     num_graphs = batch_ligand.max().item() + 1
#     for i in range(num_graphs):
#         p_pos = protein_pos[batch_protein == i]
#         l_pos = ligand_pos[batch_ligand == i]
#         loss_lj = lj_potential(p_pos, l_pos, sigma=sigma, eps_rep=eps_rep, eps_att=eps_att, r_cutoff=r_cutoff)
#         # loss_lj += lj_potential(p_pos, l_pos, sigma=sigma, eps_rep=eps_rep, eps_att=eps_att, r_cutoff=r_cutoff)
#         loss_lj_list.append(loss_lj)
#     # return loss_lj
#     return torch.stack(loss_lj_list).mean()

def lj_potential_knn(protein_coords, ligand_coords, sigma=1.0, eps_rep=1.0, eps_att=0.1, k=16):
    """
    Compute LJ potential using only top-k closest protein atoms for each ligand atom.
    """
    diff = ligand_coords[:, None, :] - protein_coords[None, :, :]  # (m, n, 3)
    r2 = torch.sum(diff ** 2, dim=-1) + 1e-12  # (m, n)

    # 获取前 k 小的距离索引 (不取 sqrt 也没关系)
    _, topk_indices = torch.topk(r2, k, dim=1, largest=False, sorted=False)

    # 选取对应距离
    r2_k = torch.gather(r2, 1, topk_indices)

    inv_r6 = (sigma ** 2 / r2_k) ** 3
    repulsion = eps_rep * (inv_r6 ** 2)      # 1/r^12
    attraction = eps_att * inv_r6            # 1/r^6
    lj = repulsion - attraction

    return lj.sum()

def compute_batch_lj_loss(protein_pos, ligand_pos, batch_protein, batch_ligand,
                          eps_rep=1.0, eps_att=0.1, sigma=2.5, k=16):
    loss_list = []
    num_graphs = batch_ligand.max().item() + 1
    for i in range(num_graphs):
        p_pos = protein_pos[batch_protein == i]
        l_pos = ligand_pos[batch_ligand == i]
        loss = lj_potential_knn(p_pos, l_pos, sigma=sigma, eps_rep=eps_rep, eps_att=eps_att, k=k)
        loss_list.append(loss)
    if len(loss_list) == 0:
        return torch.tensor(0.0, device=protein_pos.device)
    return torch.stack(loss_list).mean()
    # return torch.stack(loss_list)


def compute_clash_loss(pos_pred, ligand_batch, protein_pos, protein_batch, threshold=2, p=1):
    device = pos_pred.device
    clash_losses = []
    clash_penalty = []

    num_batches = ligand_batch.max().item() + 1

    for i in range(num_batches):
        ligand_mask = (ligand_batch == i)
        protein_mask = (protein_batch == i)

        if ligand_mask.sum() == 0 or protein_mask.sum() == 0:
            continue 

        ligand_coords = pos_pred[ligand_mask] 
        protein_coords = protein_pos[protein_mask] 

        # for atom
        dists = torch.cdist(ligand_coords, protein_coords, p=2)
        min_dists, _ = dists.min(dim=1)

        dist = threshold - min_dists
        clash_loss = torch.relu(dist * 5).pow(p).mean()
        clash_losses.append(clash_loss)

        penalty = torch.ones_like(min_dists, device=device)
        conflict_amount = torch.clamp(dist, min=0) * 5
        penalty += conflict_amount 

        clash_penalty.append(penalty)

    if len(clash_losses) == 0:
        return torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)

    return torch.stack(clash_losses), torch.cat(clash_penalty)


def compute_center_loss(pred_ligand_pos, linker_centers):
    loss = torch.norm(pred_ligand_pos - linker_centers, p=2, dim=-1)
    return loss

def compute_armsca_prox_loss(arm_pos, sca_pos, arm_index, min_d=1.2, max_d=1.9):
    pairwise_dist = torch.norm(arm_pos.unsqueeze(1) - sca_pos.unsqueeze(0), p=2, dim=-1)
    min_dist_all, _ = scatter_min(pairwise_dist, arm_index, dim=0)
    min_dist, min_dist_sca_idx = min_dist_all.min(-1)
    # loss_armsca_prox += torch.mean(pairwise_dist)
    # 1.2 < min dist < 1.9
    loss = torch.mean(torch.clamp(min_d - min_dist, min=0) + torch.clamp(min_dist - max_d, min=0))
    return loss


def compute_batch_armsca_prox_loss(pred_ligand_pos, batch_ligand, linker_armsca_index, min_d=1.2, max_d=1.8):
    batch_losses = torch.tensor(0., device=pred_ligand_pos.device)
    num_graphs = batch_ligand.max().item() + 1
    n_valid = 0
    for i in range(num_graphs):
        pos = pred_ligand_pos[batch_ligand == i]
        mask = linker_armsca_index[batch_ligand == i]  # -1: scaffold, 1: arms
        arm_mask = (mask == 1)
        sca_mask = (mask == -1)
        arm_pos = pos[arm_mask]
        sca_pos = pos[sca_mask]
        if len(arm_pos) > 0 and len(sca_pos) > 0:
            loss = compute_armsca_prox_loss(arm_pos, sca_pos, mask[arm_mask].long(), min_d=min_d, max_d=max_d)
            batch_losses += loss
            n_valid += 1
    return batch_losses / num_graphs, n_valid

def compute_bond_angle_loss(n0_dst, pred_pos, gt_pos, bond_index, batch_bond):
    src, dst = bond_index

    # Calculate ground truth angles
    gt_pos_ji = gt_pos[src] - gt_pos[dst]
    gt_pos_j0 = gt_pos[n0_dst] - gt_pos[dst]

    gt_angle = compute_bond_angle(gt_pos_ji, gt_pos_j0)

    # Calculate predicted angles
    pred_pos_ji = pred_pos[src] - pred_pos[dst]
    pred_pos_j0 = pred_pos[n0_dst] - pred_pos[dst]

    pred_angle = compute_bond_angle(pred_pos_ji, pred_pos_j0)
    
    angle_mse = scatter_mean(((gt_angle - pred_angle) ** 2), batch_bond, dim=0)
    return angle_mse

def compute_torsion_angle_loss(n0_src, n1_src, n0_dst, n1_dst, pred_pos, gt_pos, bond_index, batch_bond, torsion_type='one'):
    src, dst = bond_index

    gt_pos_ji =  gt_pos[dst] - gt_pos[src]
    pred_pos_ji = pred_pos[dst] - pred_pos[src]
    """
    # Calculate ground truth angles
    gt_pos_ji = gt_pos[src] - gt_pos[dst]
    gt_pos_j0 = gt_pos[n0_dst] - gt_pos[dst]
    gt_pos_j1 = gt_pos[n1_dst] - gt_pos[dst]
    gt_torsion_angle = compute_torsion_angle(gt_pos_ji, gt_pos_j0, gt_pos_j1)

    # Calculate predicted angles
    pred_pos_ji = pred_pos[src] - pred_pos[dst]
    pred_pos_j0 = pred_pos[n0_dst] - pred_pos[dst]
    pred_pos_j1 = pred_pos[n1_dst] - pred_pos[dst]
    pred_torsion_angle = compute_torsion_angle(pred_pos_ji, pred_pos_j0, pred_pos_j1)
    #print(gt_torsion_angle)
    #print(pred_torsion_angle)
    #gt_torsion_angle - - pred_torsion_angle
    torsion_angle_mse = scatter_mean(((gt_torsion_angle - pred_torsion_angle) ** 2), batch_bond, dim=0)
    #print(torsion_angle_mse)
    """
    # Calculate ground truth angles
    src_ref_mask = n0_src == dst
    src_ref = torch.clone(n0_src)
    src_ref[src_ref_mask] = n1_src[src_ref_mask]
    
    dst_ref_mask = n0_dst == src
    dst_ref = torch.clone(n0_dst)
    dst_ref[dst_ref_mask] = n1_dst[dst_ref_mask]
    
    gt_pos_src_ref = gt_pos[src_ref] - gt_pos[dst]
    gt_pos_dst_ref = gt_pos[dst_ref] - gt_pos[dst]
    gt_torsion_angle1 = compute_torsion_angle(gt_pos_ji, gt_pos_src_ref, gt_pos_dst_ref)

    pred_pos_src_ref = pred_pos[src_ref] - pred_pos[dst]
    pred_pos_dst_ref = pred_pos[dst_ref] - pred_pos[dst]
    pred_torsion_angle1 = compute_torsion_angle(pred_pos_ji, pred_pos_src_ref, pred_pos_dst_ref)
    
    angle_diff_1 = (gt_torsion_angle1 - pred_torsion_angle1)
    angle_diff_2 = 2 * torch.pi - (gt_torsion_angle1 - pred_torsion_angle1)
    angle_diff, _ = torch.min(torch.abs(torch.stack([angle_diff_1, angle_diff_2], dim=1)), dim=1)
    torsion_angle_mse = scatter_mean((angle_diff ** 2), batch_bond, dim=0)
    
    return torsion_angle_mse
