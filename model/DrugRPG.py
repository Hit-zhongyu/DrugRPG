import sys

import numpy as np
from rdkit import RDConfig
import os
import torch
import math
import torch.nn as nn
from torch.nn import functional as F
from tqdm.auto import tqdm
from torch_scatter import scatter_mean, scatter_add
from .protein import ProteinEncoder
from diffusion.noise_schedule import NoiseScheduleVP
from .utils import *
# from diffusion.model_3dim import DiffusionTransformer2
# from diffusion.model import DiffusionTransformer
from diffusion.model_frag import DiffusionTransformer
import pdb
from .transition import * 
from diffusion.guidance import *
from .loss import * 
from .ouroboros import GeminiMol

class DrugRPG(nn.Module):
    def __init__(self, config, device='cpu',):
        super(DrugRPG, self).__init__()
        self.timesteps = config.timesteps
        self.deive = device
        self.std = config.ligand_std
        self.ligand_atom_dim = config.ligand_atom_dim 
        self.ligand_edge_dim = config.edge_input_dim
        self.ouroboros_path = config.ouroboros_path
        
        self.ligand_dynamics = DiffusionTransformer(h_input_dim=config.ligand_atom_dim, hidden_dim=config.ligand_hidden_dim, edge_input_dim=config.edge_input_dim, 
                                                     edge_hidden_dim=config.edge_hidden_dim, dist_dim=config.dist_dim, n_layers=config.n_layers, p_n_layers=config.p_n_layers,
                                                     n_heads=config.n_heads, p_input_dim=config.protein_atom_dim)
        self.ouroboros = GeminiMol(model_path=self.ouroboros_path, cache=False, threads=32)
        for param in self.ouroboros.parameters():
            param.requires_grad = False
        self.project_mlp = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.SiLU(),
            nn.Linear(1024, 256),
            nn.SiLU(),
            nn.Linear(256, config.ligand_hidden_dim)
        )

        pos_betas = get_beta_schedule(
            num_timesteps=self.timesteps,
            **config.diff_pos
        )
        self.pos_transition = ContigousTransition(pos_betas)
        
        node_betas = get_beta_schedule(
            num_timesteps=self.timesteps,
            **config.diff_atom
        )
        self.atom_transition = AtomTransition(node_betas, self.ligand_atom_dim)

        edge_betas = get_beta_schedule(
            num_timesteps=self.timesteps,
            **config.diff_bond
        )
        # self.edge_transition = CategoricalTransition(edge_betas, self.ligand_edge_dim)
        self.edge_transition = GeneralCategoricalTransition(edge_betas, self.ligand_edge_dim)
    def forward(self, batch, device, valid=False):

        protein_pos = batch['protein_pos'].to(device)
        protein_batch = batch['protein_element_batch'].to(device)

        ligand_atom = batch['ligand_atom'].to(device).float()
        ligand_atom = ligand_atom.argmax(-1)
        ligand_pos = batch['ligand_pos'].to(device)
        ligand_batch = batch['ligand_element_batch'].to(device)
        ligand_edge_index = batch['ligand_bond_index'].to(device)
        ligand_bond_type = batch['ligand_bond_type'].to(device)
        ligand_bond_type =  ligand_bond_type.argmax(-1)
        ligand_mol = batch['ligand_mol']
        edge_batch = build_edge_batch(ligand_batch, ligand_edge_index)

        if valid:
            torch_seed = torch.Generator(device=device).manual_seed(1)
        else:
            torch_seed = None
            protein_pos = protein_pos + torch.randn(protein_pos.shape, device=device) * 0.1
        protein_pos_norm = (protein_pos - scatter_mean(protein_pos, protein_batch, dim=0)[protein_batch])
        ligand_pos_norm = (ligand_pos - scatter_mean(protein_pos, protein_batch, dim=0)[ligand_batch])

        N = ligand_batch[-1] + 1
        time_step = torch.randint(
            0, self.timesteps, size=(N // 2 + 1,), device=device, generator=torch_seed)
        time_step = torch.cat(
            [time_step, self.timesteps - time_step - 1], dim=0)[:N]

        pos_pert = self.pos_transition.add_noise(ligand_pos_norm, time_step, ligand_batch, torch_seed)
        atom_pert, log_atom_t, log_atom_0  = self.atom_transition.add_noise(ligand_atom, time_step, ligand_batch)
        edge_pert, log_edge_t, log_edge_0 = self.edge_transition.add_noise(ligand_bond_type, time_step, edge_batch)

        edge_index = torch.cat([ligand_edge_index, ligand_edge_index.flip(0)], dim=1)
        edge_type = torch.cat([edge_pert, edge_pert], dim=0)
        
        noise_level = time_step / self.timesteps
        atom_pred, pos_pred, edge_bond_pred, zs = self.ligand_dynamics(batch, atom_pert.float(), pos_pert, edge_type.float(), noise_level, 
                                 edge_index, ligand_batch, protein_batch, protein_pos_norm)
        
        loss_weight = get_timestep_weight(time_step)
        log_atom_recon = F.log_softmax(atom_pred, dim=-1)
        log_atom_post_true = self.atom_transition.q_v_posterior(log_atom_0, log_atom_t, time_step, ligand_batch)
        log_atom_post_pred = self.atom_transition.q_v_posterior(log_atom_recon, log_atom_t, time_step, ligand_batch)
        kl_atom = self.atom_transition.compute_v_Lt(log_atom_post_true, log_atom_post_pred, log_atom_0, t=time_step, batch=ligand_batch)
        if not valid:
           kl_atom = kl_atom * loss_weight[ligand_batch]
        loss_atom = torch.mean(kl_atom) * 200

        log_edge_recon = F.log_softmax(edge_bond_pred, dim=-1)
        log_edge_post_true = self.edge_transition.q_v_posterior(log_edge_0, log_edge_t, time_step, edge_batch)
        log_edge_post_pred = self.edge_transition.q_v_posterior(log_edge_recon, log_edge_t, time_step, edge_batch)
        kl_edge = self.edge_transition.compute_v_Lt(log_edge_post_true, log_edge_post_pred, log_edge_0, t=time_step, batch=edge_batch)
        if not valid:
           kl_edge = kl_edge * loss_weight[edge_batch]
        loss_bond = torch.mean(kl_edge) * 200
        
        loss_pos = F.mse_loss(pos_pred, ligand_pos_norm, reduction='none').sum(dim=-1)
        loss_pos = scatter_mean(loss_pos, ligand_batch, dim=0)
        if not valid:
           loss_pos = loss_pos * loss_weight
        loss_pos = torch.mean(loss_pos)

        loss_bond_len = bond_length_loss(pos_pred, ligand_pos_norm, ligand_edge_index, ligand_batch, loss_weight, valid)

        loss_energy = compute_lj_energy_loss(pos_pred, protein_pos, ligand_batch, protein_batch, N, loss_weight, valid)

        with torch.no_grad():
            ouroboros_emb = self.ouroboros.encode(ligand_mol)
            ouroboros_atom_emb = ouroboros_emb.ndata['atom_type']
            
        ouroboros_embs = self.project_mlp(ouroboros_atom_emb)        
        repa_loss = REPA_loss(ouroboros_embs, zs)
        # repa_loss = REPA_loss(ouroboros_atom_emb, zs)
        repa_loss = scatter_mean(repa_loss, ligand_batch, dim=0)
        # if not valid:
        #    repa_loss = repa_loss * loss_weight
        repa_loss = torch.mean(repa_loss)

        extra_loss = 0.1 * loss_bond_len + 0.1 * loss_energy + 0.1 * repa_loss
        # extra_loss = 0.1 * loss_bond_len + 0.1 * repa_loss
        losses = 1 * loss_pos + 1 * loss_atom + 1 * loss_bond + extra_loss
        
        return losses, loss_atom, loss_pos, loss_bond, extra_loss, atom_pred, edge_bond_pred
        
        # src, dst = edge_index     
        # n0, n1 = find_closest_points(bond_length_ori, edge_index)
        
        # n0_dst = torch.index_select(n0, 0, dst)
        # n1_dst = torch.index_select(n1, 0, dst)

        # loss_bond_angle = self.compute_bond_angle_loss(n0_dst, pos_pred, ligand_pos_norm, edge_index, edge_batch_)
        # loss_bond_angle = torch.mean(loss_bond_angle)

        # n0_src = torch.index_select(n0, 0, src)
        # n1_src = torch.index_select(n1, 0, src)
        # loss_torsion_angle = self.compute_torsion_angle_loss(n0_src, n1_src, n0_dst, n1_dst, pos_pred, ligand_pos_norm, edge_index, 
        #                                                 edge_batch_)
        # loss_torsion_angle = torch.mean(loss_torsion_angle)
        # bond_loss = torch.zeros_like(loss_atom)

        # loss_bond = torch.zeros_like(loss_atom)
        # losses = 1 * loss_pos + 1 * loss_atom + 1 * loss_bond + 0.5 * bond_loss + 0.5 * clash_penalty
        # extra_loss = bond_loss + clash_penalty
        # losses = 1 * loss_pos + 1 * loss_atom
        # extra_loss = bond_loss + clash_penalty

        # losses = 1 * loss_pos + 1 * loss_atom + 1 * loss_bond + 0.1 * bond_loss + 0.1 * loss_bond_angle
        # extra_loss = bond_loss + loss_bond_angle

        

        # losses = 1 * loss_pos + 1 * loss_atom + 1.5 * loss_bond 
        # extra_loss = bond_loss

        
        

    def ddpm_sample(self, batch, device):

        protein_pos = batch['protein_pos'].to(device)
        protein_batch = batch['protein_atom_batch'].to(device)

        ligand_atom = batch['ligand_atom'].to(device).float()
        ligand_pos = batch['ligand_pos'].to(device)
        ligand_batch = batch['ligand_atom_batch'].to(device)
        ligand_edge_index = batch['ligand_bond_index'].to(device)
        # print(batch['ligand_num_nodes'])
        ligand_num_nodes = batch['ligand_num_nodes'].to(device)
        num_nodes = max(ligand_num_nodes)
        
        _, atom_init, log_atom_type = self.atom_transition.sample_init(ligand_atom.shape[0])
        pos_init = self.pos_transition.sample_init([ligand_atom.shape[0], 3])
        _, edge_init, log_edge_type = self.edge_transition.sample_init(ligand_edge_index.shape[1])

        ligand_atom = atom_init
        ligand_pos = pos_init

        N = ligand_batch[-1] + 1

        protein_mean = scatter_mean(protein_pos, protein_batch, dim=0)
        ligand_pos = ligand_pos + protein_mean[ligand_batch]

        protein_pos_norm = protein_pos - protein_mean[protein_batch]
        ligand_pos_norm = ligand_pos - protein_mean[ligand_batch]

        edge_batch = build_edge_batch(ligand_batch, ligand_edge_index)
        edge_index = torch.cat([ligand_edge_index, ligand_edge_index.flip(0)], dim=1)
        ligand_edge_feat = torch.cat([edge_init, edge_init], dim=0)

        time_seq = list(reversed(range(0, self.timesteps)))
        torch.set_printoptions(threshold=torch.inf)

        save_timesteps = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
        atom_pred_embeddings = {}

        with torch.no_grad():
            for i in tqdm(time_seq, desc='sampling', total=len(time_seq)):

                t = torch.full(size=(N,), fill_value=i, dtype=torch.long, device=protein_pos.device)
                noise_level = t / self.timesteps

                atom_pred, pos_pred, edge_pred, zs = self.ligand_dynamics(batch, ligand_atom.float(), ligand_pos_norm, ligand_edge_feat.float(), noise_level, 
                                                                edge_index, ligand_batch, protein_batch, protein_pos_norm)

                if i in save_timesteps:
                    atom_pred_embeddings[i] = zs.clone().detach()

                pos_prev = self.pos_transition.get_prev_from_recon(x_t=ligand_pos_norm, x_recon=pos_pred, t=t, batch=ligand_batch)
                
                log_atom_recon = F.log_softmax(atom_pred, dim=-1)
                log_atom_type = self.atom_transition.q_v_posterior(log_atom_recon, log_atom_type, t, ligand_batch)
                atom_type_prev = log_sample_categorical(log_atom_type)
                atom_prev = self.atom_transition.onehot_encode(atom_type_prev)
                
                # halfedge types
                log_edge_recon = F.log_softmax(edge_pred, dim=-1)
                log_edge_type = self.edge_transition.q_v_posterior(log_edge_recon, log_edge_type, t, edge_batch)
                edge_type_prev = log_sample_categorical(log_edge_type)
                edge_prev = self.edge_transition.onehot_encode(edge_type_prev)

                # 测试3，7 效果最好
                # with torch.enable_grad():
                #     xt = ligand_pos_norm.requires_grad_(True) 
                #     energy = compute_batch_clash_loss(
                #          protein_pos_norm, xt, protein_batch, ligand_batch,
                #          sigma=3, surface_ct=6)
                #     energy_grad = torch.autograd.grad(energy, xt)[0]
                #     pos_prev -= energy_grad
                
                ligand_pos_norm = pos_prev
                ligand_atom = atom_prev
                # ligand_atom = self.atom_transition.onehot_encode(atom_type_prev)
                # edge_prev = self.edge_transition.onehot_encode(edge_type_prev)
                ligand_edge_feat = torch.cat([edge_prev,edge_prev], dim=0)

                # if use_pocket_data != None:
                #     ligand_pos_norm = pocket_guidance(use_pocket_data, ligand_pos_norm, ligand_atom.argmax(dim=-1))
                    
                
            # protein_final = scatter_mean(protein_pos_norm, protein_batch, dim=0)
            torch.set_printoptions(profile="full")
            ligand_atom = atom_pred
            atom = F.one_hot(torch.argmax(ligand_atom, dim=-1), self.ligand_atom_dim)
            pos = ligand_pos_norm + protein_mean[ligand_batch]

            ligand_edge_feat = torch.cat([edge_pred, edge_pred], dim=0)
            edge_type = torch.argmax(ligand_edge_feat, dim=-1)
            edge_matrix = torch.zeros((N, num_nodes, num_nodes), device=device)
            for i in ligand_batch.unique():
                mask = (ligand_batch[edge_index[0]] == i) & (ligand_batch[edge_index[1]] == i)
                edges = edge_index[:, mask]
                edges = edges - min(edges[0])
                edge_matrix[i, edges[0], edges[1]] = edge_type[mask].float()

            return atom, pos, edge_matrix, atom_pred_embeddings
    

    def fragment_sample(self, batch, device):
        protein_pos = batch['protein_pos'].to(device)
        protein_batch = batch['protein_atom_batch'].to(device)
        ligand_atom = batch['ligand_atom'].to(device).float()
        ligand_pos = batch['ligand_pos'].to(device)
        ligand_batch = batch['ligand_atom_batch'].to(device)
        ligand_edge_index = batch['ligand_bond_index'].to(device)
        num_nodes = batch['ligand_num_nodes'].to(device)
        max_num_nodes = max(num_nodes)
        frag_mask = batch['frag_mask'].type(torch.bool)
        frag_bond = batch['frag_bond'].to(device)
        edge_bond_mask = batch['edge_bond_mask'].bool().to(device)
        
        linker_mask = ~frag_mask 
        frag_batch = ligand_batch[frag_mask]
        linker_batch = ligand_batch[linker_mask]
    
        # ligand_pos = ligand_pos_init
        # linker_pos = ligand_pos_init[linker_mask, :]

        _, atom_init, log_atom_type = self.atom_transition.sample_init(linker_mask.sum())
        pos_init = self.pos_transition.sample_init([linker_mask.sum(), 3])
        _, edge_init, log_edge_type = self.edge_transition.sample_init(ligand_edge_index.shape[1])
        # ligand_pos[linker_mask, :] = pos_init
        frag_bond_one_hot = F.one_hot(frag_bond.long(), num_classes=5).to(edge_init.dtype)
        edge_init[edge_bond_mask] = frag_bond_one_hot
        # exit(0)

        protein_mean = scatter_mean(protein_pos, protein_batch, dim=0)
        # ligand_pos = ligand_pos + protein_mean[ligand_batch]

        protein_pos_norm = protein_pos - protein_mean[protein_batch]
        ligand_pos_norm = ligand_pos - protein_mean[ligand_batch]
        
        frag_pos_init = ligand_pos_norm[frag_mask, :]
        frag_atom_init = ligand_atom[frag_mask, :]

        frag_pos_init_mean = scatter_mean(frag_pos_init, frag_batch, dim=0)

        # ligand_pos_norm[linker_mask] = pos_init + frag_pos_init_mean[linker_batch]
        ligand_pos_norm[linker_mask] = pos_init
        ligand_atom[linker_mask] = atom_init

        
        N = ligand_batch[-1] + 1

        edge_batch = build_edge_batch(ligand_batch, ligand_edge_index)
        edge_index = torch.cat([ligand_edge_index, ligand_edge_index.flip(0)], dim=1)
        ligand_edge_feat = torch.cat([edge_init, edge_init], dim=0)

        torch_seed = torch.Generator(device=device).manual_seed(2024)
        time_seq = list(reversed(range(0, self.timesteps)))
        with torch.no_grad():
            for i in tqdm(time_seq, desc='sampling', total=len(time_seq)):

                t = torch.full(size=(N,), fill_value=i, dtype=torch.long, device=protein_pos.device)
                noise_level = t / self.timesteps

                # frag_pos_pert = self.pos_transition.add_noise(frag_pos_init, t, frag_batch, torch_seed)
                # ligand_pos_norm[frag_mask] = frag_pos_pert
                ligand_pos_norm[frag_mask] = frag_pos_init
                
                # frag_atom_pert, _, _ = self.atom_transition.add_noise(frag_atom_init.argmax(-1), t, frag_batch)
                # ligand_atom[frag_mask] = frag_atom_pert
                ligand_atom[frag_mask] = frag_atom_init

                atom_pred, pos_pred, edge_pred, _ = self.ligand_dynamics(batch, ligand_atom.float(), ligand_pos_norm, ligand_edge_feat.float(), noise_level, 
                                                                edge_index, ligand_batch, protein_batch, protein_pos_norm)

                pos_prev = self.pos_transition.get_prev_from_recon(x_t=ligand_pos_norm, x_recon=pos_pred, t=t, batch=ligand_batch)
                
                linker_atom_pred = atom_pred[linker_mask]
                log_atom_recon = F.log_softmax(linker_atom_pred, dim=-1)
                log_atom_type = self.atom_transition.q_v_posterior(log_atom_recon, log_atom_type, t, linker_batch)
                atom_type_prev = log_sample_categorical(log_atom_type)
                linker_atom_prev = self.atom_transition.onehot_encode(atom_type_prev)
                
                # halfedge types
                log_edge_recon = F.log_softmax(edge_pred, dim=-1)
                log_edge_type = self.edge_transition.q_v_posterior(log_edge_recon, log_edge_type, t, edge_batch)
                edge_type_prev = log_sample_categorical(log_edge_type)
                edge_prev = self.edge_transition.onehot_encode(edge_type_prev)

                with torch.enable_grad():
                    xt = ligand_pos_norm.requires_grad_(True) 
                    energy = compute_batch_clash_loss(
                         protein_pos_norm, xt, protein_batch, ligand_batch,
                         sigma=2, surface_ct=6)
                    energy_grad = torch.autograd.grad(energy, xt)[0]
                    pos_prev -= energy_grad

                ligand_pos_norm = pos_prev
                ligand_atom[linker_mask] = linker_atom_prev
                edge_prev[edge_bond_mask] = frag_bond_one_hot
                ligand_edge_feat = torch.cat([edge_prev,edge_prev], dim=0)

                ligand_pos_norm[frag_mask] = frag_pos_init
                ligand_atom[frag_mask] = frag_atom_init
        
        torch.set_printoptions(profile="full")
        atom = F.one_hot(torch.argmax(ligand_atom, dim=-1), self.ligand_atom_dim)

        pos = ligand_pos_norm + protein_mean[ligand_batch]
        edge_type = torch.argmax(ligand_edge_feat, dim=-1)
        
        edge_matrix = torch.zeros((N, max_num_nodes, max_num_nodes), device=device)

        for i in ligand_batch.unique():
            mask = (ligand_batch[edge_index[0]] == i) & (ligand_batch[edge_index[1]] == i)
            edges = edge_index[:, mask]
            edges = edges - min(edges[0])
            edge_matrix[i, edges[0], edges[1]] = edge_type[mask].float()
        
        return atom, pos, edge_matrix
    
    def linker_sample(self, batch, frag_edge_type, device):
        protein_pos = batch['protein_pos'].to(device)
        protein_batch = batch['protein_atom_batch'].to(device)

        ligand_atom = batch['ligand_atom'].to(device).float()
        ligand_pos = batch['ligand_pos'].to(device)
        ligand_batch = batch['ligand_atom_batch'].to(device)
        ligand_edge_index = batch['ligand_bond_index'].to(device)
        anchor = batch['anchor'].to(device)
        num_nodes = batch['ligand_num_nodes'].to(device)
        max_num_nodes = max(num_nodes)
        
        N = ligand_batch[-1] + 1
        
        
        frag_mask = batch['frag_mask'].type(torch.bool)
        
        linker_mask = ~frag_mask 
        frag_batch = ligand_batch[frag_mask]
        linker_batch = ligand_batch[linker_mask]

        src_frag = frag_mask[ligand_edge_index[0]] 
        dst_frag = frag_mask[ligand_edge_index[1]]
        frag_edge_mask = src_frag & dst_frag
        # frag_edge_index = ligand_edge_index[:, frag_edge_mask]
        frag_edge_type_batch  = frag_edge_type.repeat(N).to(device)
        frag_edge_type_one_hot = F.one_hot(frag_edge_type_batch, self.ligand_edge_dim).float()

        linker_anchor_mask = linker_mask.clone()
        linker_anchor_mask[anchor] = True
        src_linker = linker_anchor_mask[ligand_edge_index[0]] 
        dst_linker = linker_anchor_mask[ligand_edge_index[1]]
        linker_edge_mask = src_linker & dst_linker
        linker_edge_index = ligand_edge_index[:, linker_edge_mask] 

        linker_armsca_index = torch.zeros_like(linker_mask, dtype=torch.int, device=device)
        linker_armsca_index[anchor] = -1
        linker_armsca_index[linker_mask] = 1
        # print(linker_armsca_index)

        _, atom_init, log_atom_type = self.atom_transition.sample_init(linker_mask.sum())
        pos_init = self.pos_transition.sample_init([linker_mask.sum(), 3])
        _, edge_init, log_edge_type = self.edge_transition.sample_init(linker_edge_index.shape[1])

        ligand_edge_attr = torch.zeros((ligand_edge_index.shape[1], self.ligand_edge_dim), device=device)
        assert frag_edge_mask.sum() == frag_edge_type_one_hot.shape[0]
        ligand_edge_attr[frag_edge_mask] = frag_edge_type_one_hot
        ligand_edge_attr[linker_edge_mask] = edge_init

        # ligand_pos[linker_mask, :] = pos_init

        protein_mean = scatter_mean(protein_pos, protein_batch, dim=0)
        # ligand_pos = ligand_pos + protein_mean[ligand_batch]

        protein_pos_norm = protein_pos - protein_mean[protein_batch]
        ligand_pos_norm = ligand_pos - protein_mean[ligand_batch]

        frag_pos_init = ligand_pos_norm[frag_mask, :]
        frag_atom_init = ligand_atom[frag_mask, :]

        frag_center = scatter_mean(frag_pos_init, frag_batch, dim=0)

        pos_init = pos_init * 0.05 + frag_center[linker_batch]
        ligand_pos_norm[linker_mask] = pos_init
        ligand_atom[linker_mask] = atom_init

        edge_batch = build_edge_batch(ligand_batch, linker_edge_index)
        edge_index = torch.cat([ligand_edge_index, ligand_edge_index.flip(0)], dim=1)
        ligand_edge_feat = torch.cat([ligand_edge_attr, ligand_edge_attr], dim=0)
        linker_edge_index_ = torch.cat([linker_edge_index, linker_edge_index.flip(0)], dim=1)
        linker_edge_feat_ = torch.cat([edge_init, edge_init], dim=0)

        torch_seed = torch.Generator(device=device).manual_seed(1)
        time_seq = list(reversed(range(0, self.timesteps)))
        with torch.no_grad():
            for i in tqdm(time_seq, desc='sampling', total=len(time_seq)):

                t = torch.full(size=(N,), fill_value=i, dtype=torch.long, device=protein_pos.device)
                noise_level = t / self.timesteps

                # frag_pos_pert = self.pos_transition.add_noise(frag_pos_init, t, frag_batch, torch_seed)
                # ligand_pos_norm[frag_mask] = frag_pos_pert

                # _, frag_atom_pert, _ = self.atom_transition.add_noise(frag_atom_init.argmax(-1), t, frag_batch)
                # ligand_atom[frag_mask] = frag_atom_pert

                atom_pred, pos_pred, edge_pred = self.ligand_dynamics(batch, ligand_atom.float(), ligand_pos_norm, ligand_edge_feat.float(), noise_level, 
                                                                edge_index, ligand_batch, protein_batch, protein_pos_norm, linker_mask, 
                                                                linker_edge_index_, linker_edge_feat_.float())

                pos_prev = self.pos_transition.get_prev_from_recon(x_t=ligand_pos_norm, x_recon=pos_pred, t=t, batch=ligand_batch)
                
                linker_atom_pred = atom_pred[linker_mask]
                log_atom_recon = F.log_softmax(linker_atom_pred, dim=-1)
                log_atom_type = self.atom_transition.q_v_posterior(log_atom_recon, log_atom_type, t, linker_batch)
                atom_type_prev = log_sample_categorical(log_atom_type)
                linker_atom_prev = self.atom_transition.onehot_encode(atom_type_prev)
                
                # halfedge types
                log_edge_recon = F.log_softmax(edge_pred, dim=-1)
                log_edge_type = self.edge_transition.q_v_posterior(log_edge_recon, log_edge_type, t, edge_batch)
                edge_type_prev = log_sample_categorical(log_edge_type)
                edge_prev = self.edge_transition.onehot_encode(edge_type_prev)

                with torch.enable_grad():
                    xt = ligand_pos_norm.requires_grad_(True) 
                    energy_all = 0
                    energy_center = compute_center_loss(
                         xt, frag_center[ligand_batch])
                    energy_center = energy_center[linker_mask].mean()
                    energy_center = torch.autograd.grad(energy_center, xt)[0]
                    energy_all += energy_center * 0.01
                    # pos_prev -= energy_grad * 0.05 * linker_mask.unsqueeze(-1).float()

                    energy_sca, n_valid = compute_batch_armsca_prox_loss(
                                xt, ligand_batch, linker_armsca_index,
                                )
                    if n_valid > 0:
                        energy_sca = torch.autograd.grad(energy_sca, xt)[0]
                        energy_all += energy_sca * 0.1

                    pos_prev -= energy_all * linker_mask.unsqueeze(-1).float()

                ligand_pos_norm = pos_prev
                ligand_atom[linker_mask] = linker_atom_prev
                # ligand_edge_feat = torch.cat([edge_prev,edge_prev], dim=0)
                ligand_edge_feat_ = ligand_edge_feat[:edge_index.shape[1] // 2,:]
                ligand_edge_feat_[frag_edge_mask] = frag_edge_type_one_hot
                ligand_edge_feat_[linker_edge_mask] = edge_prev
                ligand_edge_feat = torch.cat([ligand_edge_feat_,ligand_edge_feat_], dim=0)

                linker_edge_feat_ = torch.cat([edge_prev,edge_prev], dim=0)

                ligand_pos_norm[frag_mask] = frag_pos_init
                ligand_atom[frag_mask] = frag_atom_init
        
        torch.set_printoptions(profile="full")
        atom = F.one_hot(torch.argmax(ligand_atom, dim=-1), self.ligand_atom_dim)

        pos = ligand_pos_norm + protein_mean[ligand_batch]

        edge_type = torch.argmax(ligand_edge_feat, dim=-1)
        edge_matrix = torch.zeros((N, max_num_nodes, max_num_nodes), device=device)

        for i in ligand_batch.unique():
            mask = (ligand_batch[edge_index[0]] == i) & (ligand_batch[edge_index[1]] == i)
            edges = edge_index[:, mask]
            edges = edges - min(edges[0])
            edge_matrix[i, edges[0], edges[1]] = edge_type[mask].float()
        
        return atom, pos, edge_matrix
    
    def emb(self, mol):
        with torch.no_grad():
            ouroboros_emb = self.ouroboros.encode(mol)
            ouroboros_atom_emb = ouroboros_emb.ndata['atom_type']
            
        ouroboros_embs = self.project_mlp(ouroboros_atom_emb)  

        return ouroboros_embs

def pocket_guidance(use_pocket_data, pred_ligand_pos, pred_ligand_v, k=3):
    """
    apply additional point cloud shape guidance
    """
    pocket_atom_pos, pocket_atom_elems, kdtree, protein_ligand_mat = use_pocket_data
    pred_ligand_pos_numpy = np.array(pred_ligand_pos.cpu())
    pred_ligand_v_numpy = np.array(pred_ligand_v.cpu())
    dists, k_point_idxs = kdtree.query(pred_ligand_pos_numpy, k=k)
    
    closeatom_idxs = set()
    closeatom_dists = np.zeros((pred_ligand_pos.shape[0]))
    for i in range(k):
        close_point_idxs = k_point_idxs[:, i]
        close_protein_elems = pocket_atom_elems[close_point_idxs]
        close_protein_ligand_threshold = protein_ligand_mat[pred_ligand_v_numpy, close_protein_elems]
        closeatom_idx = np.where(dists[:, i] < close_protein_ligand_threshold)[0]
        threshold_dist = close_protein_ligand_threshold[closeatom_idx] - dists[closeatom_idx, i]
        try:
            closeatom_dists[closeatom_idx] = np.max([closeatom_dists[closeatom_idx], threshold_dist], axis=0)
        except:
            pdb.set_trace()
        closeatom_idxs.update(closeatom_idx.tolist())
    closeatom_idxs = np.array(list(closeatom_idxs))

    if len(closeatom_idxs) == 0: return pred_ligand_pos
    try:
        closeatom_dists = closeatom_dists[closeatom_idxs]
    except:
        pdb.set_trace()
    closeatom_points = pred_ligand_pos_numpy[closeatom_idxs, :]
    closeatom_point_idxs = k_point_idxs[closeatom_idxs, :]
    #print("step %d: %d close atoms" % (step, len(closeatom_idxs)))
    changed_closeatom_idxs = []
    changed_closeatom_points = []
    
    j = 0
    while len(closeatom_idxs) > 0 and j < 5:
        # change outmesh_points
        closeatom_nearest_points = np.mean(pocket_atom_pos.cpu().numpy()[closeatom_point_idxs, :], axis=1)
        
        distance_dir = closeatom_points - closeatom_nearest_points
        distance_val = np.sqrt(np.sum(distance_dir ** 2, axis=1)).reshape(-1, 1)
        unit_dir = distance_dir / distance_val
        distance_scalar = closeatom_dists + np.random.random(len(closeatom_dists)) * 0.2
    
        new_closeatom_points = closeatom_points + distance_scalar.reshape(-1, 1) * unit_dir
        
        dists, k_point_idxs = kdtree.query(new_closeatom_points, k=k)
        
        faratom_idxs = np.zeros_like(k_point_idxs)
        closeatom_dists = np.zeros((len(k_point_idxs)))
        for i in range(k):
            far_point_idxs = k_point_idxs[:, i]
            far_protein_elems = pocket_atom_elems[far_point_idxs]
            far_protein_ligand_threshold = protein_ligand_mat[pred_ligand_v_numpy[closeatom_idxs], far_protein_elems]
            far_idxs = np.where(dists[:, i] > far_protein_ligand_threshold)[0]
            faratom_idxs[far_idxs, i] = 1
            close_idxs = np.where(dists[:, i] < far_protein_ligand_threshold)[0]
            close_dist = far_protein_ligand_threshold[close_idxs] - dists[close_idxs, i]
            closeatom_dists[close_idxs] = np.max([closeatom_dists[close_idxs], close_dist], axis=0)

        far_atom = (np.sum(faratom_idxs, axis=1) == 3)
        changed_closeatom_idxs.extend(closeatom_idxs[far_atom])
        changed_closeatom_points.extend(new_closeatom_points[far_atom, :])

        closeatom_idxs = closeatom_idxs[~far_atom]
        closeatom_dists = closeatom_dists[~far_atom]
        closeatom_points = new_closeatom_points[~far_atom]
        closeatom_point_idxs = k_point_idxs[~far_atom, :]
        j += 1

    #print("still step %d: %d close atoms" % (step, len(closeatom_idxs)))
    if j == 5:
        changed_closeatom_idxs.extend(closeatom_idxs)
        changed_closeatom_points.extend(closeatom_points)
    #print("step %d: %d close atoms" % (step, len(closeatom_idxs)))
    changed_closeatom_idxs = torch.LongTensor(np.array(changed_closeatom_idxs))
    if len(changed_closeatom_idxs) > 0:
        pred_ligand_pos[changed_closeatom_idxs, :] = torch.FloatTensor(np.array(changed_closeatom_points)).cuda()
    return pred_ligand_pos

def pocket_guidance_whole_molecule(use_pocket_data, pred_ligand_pos, pred_ligand_v, k=3, max_iter=5):
    pocket_atom_pos, pocket_atom_elems, kdtree, protein_ligand_mat = use_pocket_data
    pred_ligand_pos_numpy = np.array(pred_ligand_pos.cpu())
    pred_ligand_v_numpy = np.array(pred_ligand_v.cpu())

    if hasattr(pocket_atom_pos, 'cpu'):
        pocket_atom_pos_np = pocket_atom_pos.cpu().numpy()
    else:
        pocket_atom_pos_np = np.array(pocket_atom_pos)
    
    for iter in range(max_iter):
        dists, k_point_idxs = kdtree.query(pred_ligand_pos_numpy, k=k)
        
        total_translation = np.zeros(3)
        total_weight = 0.0
        
        for i in range(pred_ligand_pos_numpy.shape[0]):
            for j in range(k):
                neighbor_idx = k_point_idxs[i, j]
                dist = dists[i, j]
                pocket_elem = pocket_atom_elems[neighbor_idx]
                threshold = protein_ligand_mat[pred_ligand_v_numpy[i], pocket_elem]
                if dist < threshold:
                    pocket_point = pocket_atom_pos_np[neighbor_idx]
                    diff = pred_ligand_pos_numpy[i] - pocket_point
                    norm = np.linalg.norm(diff)
                    if norm > 1e-6:
                        unit_dir = diff / norm
                        weight = threshold - dist
                        total_translation += weight * unit_dir
                        total_weight += weight
        
        if total_weight == 0:
            break
        overall_translation = total_translation / total_weight
        if np.linalg.norm(overall_translation) < 1e-6:
            break
        pred_ligand_pos_numpy = pred_ligand_pos_numpy + overall_translation

    new_pred_ligand_pos = torch.FloatTensor(pred_ligand_pos_numpy).to(pred_ligand_pos.device)
    return new_pred_ligand_pos

    


def radius_based_constraints(pred_ligand_pos, pred_ligand_v, ligand_batch, k=2, max_iter=5, step_size=0.1, radius_range=(0.8, 1.1)):

    device = pred_ligand_pos.device
    pred_ligand_pos = pred_ligand_pos.detach().cpu().numpy()
    pred_ligand_v = pred_ligand_v.detach().cpu().numpy()
    batch_np = ligand_batch.detach().cpu().numpy()

    radius_dict = {0: 0.77, 1: 0.75, 2: 0.73, 3: 0.71, 4: 1.10, 5: 1.02, 6: 0.99, 7: 0.99}
    min_scale, max_scale = radius_range

    # 按batch处理
    for batch_idx in np.unique(batch_np):
        batch_mask = (batch_np == batch_idx)
        batch_pos = pred_ligand_pos[batch_mask]
        batch_types = pred_ligand_v[batch_mask]
        n_atoms = len(batch_pos)
        
        dists = np.linalg.norm(batch_pos[:,None] - batch_pos, axis=-1)
        np.fill_diagonal(dists, np.inf)  # 排除自身
        
        # 每个原子取最近的k个邻居
        neighbors = np.argpartition(dists, k, axis=1)[:,:k]
        constraint_pairs = set()
        for i in range(n_atoms):
            for j in neighbors[i]:
                if (j,i) not in constraint_pairs:
                    constraint_pairs.add((i,j))
        
        # 迭代调整
        for _ in range(max_iter):
            modified = False
            new_pos = batch_pos.copy()
            
            for i,j in constraint_pairs:
                r_sum = radius_dict[batch_types[i]] + radius_dict[batch_types[j]] 
                min_len = r_sum * min_scale
                max_len = r_sum * max_scale
                
                vec = batch_pos[i] - batch_pos[j]
                current_len = np.linalg.norm(vec)
                
                if current_len < min_len:
                    direction = vec / (current_len + 1e-6)
                    adjustment = (min_len - current_len) * step_size
                    new_pos[i] += direction * adjustment * 0.5
                    new_pos[j] -= direction * adjustment * 0.5
                    modified = True
                elif current_len > max_len:
                    direction = vec / (current_len + 1e-6)
                    adjustment = (current_len - max_len) * step_size
                    new_pos[i] -= direction * adjustment * 0.5
                    new_pos[j] += direction * adjustment * 0.5
                    modified = True
            
            if modified:
                batch_pos = new_pos
            else:
                break
        
        # 写回结果
        pred_ligand_pos[batch_mask] = batch_pos

    return torch.FloatTensor(pred_ligand_pos).to(device)


