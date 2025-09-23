import numpy as np
import torch.nn as nn
import torch
from torch.nn import functional as F
from math import pi as PI
# from drug_diffusion.ligand.models_new import EGNN_encoder, EGNN_Decoder
from .utils import norm_pos,GaussianSmearing, ShiftedSoftplus
from torch_geometric.nn import MessagePassing, radius_graph

class CFConv(MessagePassing):

    def __init__(self, in_channels, out_channels, num_filters, edge_channels, cutoff=10.0, smooth=False):
        super().__init__(aggr='add')
        self.lin1 = nn.Linear(in_channels, num_filters, bias=False)
        self.lin2 = nn.Linear(num_filters, out_channels)
        self.nn = nn.Sequential(
            nn.Linear(edge_channels, num_filters),
            ShiftedSoftplus(),
            nn.Linear(num_filters, num_filters),
        )  # Network for generating filter weights
        self.cutoff = cutoff
        self.smooth = smooth

    def forward(self, x, edge_index, edge_length, edge_attr):
        W = self.nn(edge_attr)

        if self.smooth:
            C = 0.5 * (torch.cos(edge_length * PI / self.cutoff) + 1.0)
            C = C * (edge_length <= self.cutoff) * (edge_length >= 0.0)  # Modification: cutoff
        else:
            C = (edge_length <= self.cutoff).float()
        # if self.cutoff is not None:
        #     C = 0.5 * (torch.cos(edge_length * PI / self.cutoff) + 1.0)
        #     C = C * (edge_length <= self.cutoff) * (edge_length >= 0.0)     # Modification: cutoff
        W = W * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j, W):
        return x_j * W
    
class InteractionBlock(nn.Module):

    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff, smooth=False):
        super(InteractionBlock, self).__init__()
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters, num_gaussians, cutoff, smooth)
        self.act = ShiftedSoftplus()
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        # self.bn = BatchNorm(hidden_channels)
        # self.gn = GraphNorm(hidden_channels)

    def forward(self, x, edge_index, edge_length, edge_attr):
        x = self.conv(x, edge_index, edge_length, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        # x = self.bn(x)
        # x = self.gn(x)
        return x
    
class SchNetEncoder(nn.Module):

    def __init__(self, hidden_channels=128, num_filters=128,
                 num_interactions=6, edge_channels=64, cutoff=10.0, input_dim=27):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.input_dim = input_dim
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels)
        self.cutoff = cutoff
        self.emblin = nn.Linear(self.input_dim, hidden_channels)

        self.interactions = nn.ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, edge_channels,
                                     num_filters, cutoff, smooth=True)
            self.interactions.append(block)

    @property
    def out_channels(self):
        return self.hidden_channels

    def forward(self, atom, pos, batch):
        edge_index = radius_graph(pos, self.cutoff, batch=batch, loop=False)
        edge_length = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
        edge_attr = self.distance_expansion(edge_length)
        h = self.emblin(atom)
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_length, edge_attr)
        return h
    
class SchNetDecoder(nn.Module):

    def __init__(self, hidden_channels=128, num_filters=128,
                 num_interactions=6, edge_channels=64, cutoff=10.0, input_dim=27):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.input_dim = input_dim
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels)
        self.cutoff = cutoff
        self.emblin = nn.Linear(self.input_dim, hidden_channels)

        self.interactions = nn.ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, edge_channels,
                                     num_filters, cutoff, smooth=True)
            self.interactions.append(block)

    @property
    def out_channels(self):
        return self.hidden_channels

    def forward(self, atom, pos, batch):
        edge_index = radius_graph(pos, self.cutoff, batch=batch, loop=False)
        edge_length = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
        edge_attr = self.distance_expansion(edge_length)
        h = self.emblin(atom)
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_length, edge_attr)
        return h

class AutoEncoder(nn.Module):
    def __init__(
        self,
        atom_dim = 8,
        timesteps = 1000,
        in_node_nf=16,
        hidden_nf=128,
        out_node_nf=4,
        n_layers=6,
        cutoff=6
    ):
        super().__init__()

        self.atom_dim = atom_dim
        self.timesteps = timesteps

        self.encoders = SchNetEncoder( hidden_channels=hidden_nf,
            num_filters=hidden_nf,
            num_interactions=n_layers,
            edge_channels=hidden_nf,
            cutoff=cutoff,  # 10
            input_dim=in_node_nf)
        
        self.decoders = SchNetDecoder(hidden_channels=hidden_nf,
            num_filters=hidden_nf,
            num_interactions=n_layers,
            edge_channels=hidden_nf,
            cutoff=cutoff,  # 10
            input_dim=out_node_nf)
        
        self.fc1_m = nn.Linear(hidden_nf, 64)
        self.fc2_m = nn.Linear(64, 32)
        self.fc3_m = nn.Linear(32, out_node_nf)

        self.fc1_v = nn.Linear(hidden_nf, 64)
        self.fc2_v = nn.Linear(64, 32)
        self.fc3_v = nn.Linear(32, out_node_nf)

        self.out1 = nn.Linear(hidden_nf, 64)
        self.out2 = nn.Linear(64, in_node_nf)

        self.ratio_x = 0.5

    def sample(self, h, sample):

        m = F.relu(self.fc1_m(h))
        m = F.relu(self.fc2_m(m))
        m = self.fc3_m(m)
        v = F.relu(self.fc1_v(h))
        v = F.relu(self.fc2_v(v))
        v = self.fc3_v(v)
        # return h, H_kl_loss
        std = torch.exp(v * 0.5)
        z = torch.randn_like(v)
        ctx = m + std * z if sample else m

        kl_loss = 0.5 * torch.sum(torch.exp(v) + m ** 2 - 1. - v)

        return ctx, kl_loss

    def encoder(self, ligand_atom, ligand_pos, ligand_pad_mask, sample=False):
        
        bs, n, _ = ligand_atom.shape
        device = ligand_atom.device
        ligand_pad_mask = ligand_pad_mask if ligand_pad_mask.dim() == 3 else ligand_pad_mask.unsqueeze(-1)
        mask = ligand_pad_mask.clone()
        ligand_batch = torch.arange(bs).repeat_interleave(n).to(device)

        ligand_atom = ligand_atom * ligand_pad_mask
        ligand_pos = ligand_pos * ligand_pad_mask

        ligand_atom = ligand_atom.view(bs*n, -1)
        ligand_pos = ligand_pos.view(bs*n, -1)
        mask = mask.view(-1).bool()
        batch = ligand_batch[mask]

        atom = ligand_atom[mask]
        pos = ligand_pos[mask]
        h = self.encoders(atom, pos, batch)
        latant_h, kl_loss = self.sample(h, sample)
        
        ligand_atom_new = ligand_atom[:, :latant_h.shape[-1]]
        ligand_atom_new[mask] = latant_h

        latant_h = ligand_atom_new.view(bs, n , -1)

        return latant_h, kl_loss
    
    def decoder(self, ligand_atom, ligand_pos, ligand_pad_mask):

        bs, n, _ = ligand_atom.shape
        device = ligand_atom.device
        ligand_pad_mask = ligand_pad_mask if ligand_pad_mask.dim() == 3 else ligand_pad_mask.unsqueeze(-1)
        mask = ligand_pad_mask.clone()
        ligand_batch = torch.arange(bs).repeat_interleave(n).to(device)

        ligand_atom = ligand_atom * ligand_pad_mask
        ligand_pos = ligand_pos * ligand_pad_mask

        ligand_atom = ligand_atom.view(bs*n, -1)
        ligand_pos = ligand_pos.view(bs*n, -1)
        mask = mask.view(-1).bool()
        batch = ligand_batch[mask]

        atom = ligand_atom[mask]
        pos = ligand_pos[mask]
        final_h = self.decoders(atom, pos, batch)
        final_h = F.relu(self.out1(final_h))
        final_h = self.out2(final_h)

        ligand_atom_new = ligand_atom[:,:1].repeat(1, final_h.shape[-1]) 
        ligand_atom_new[mask] = final_h
        final_h = ligand_atom_new.view(bs, n , -1)

        return final_h

    
    def forward(self, batch, device, sample=False, valid=False):

        ligand_atom = batch['ligand_atom'].to(device)
        ligand_pos = batch['ligand_pos'].to(device)
        ligand_pad_mask = batch['ligand_pad_mask'].unsqueeze(-1).to(device) 
        mask = batch['ligand_pad_mask']

        latent_h, kl_loss = self.encoder(ligand_atom, ligand_pos, ligand_pad_mask, sample=sample)
        recon_h = self.decoder(latent_h, ligand_pos, ligand_pad_mask)

        bs, n, _ = ligand_atom.shape
        ligand_atom = ligand_atom.view(bs*n, -1)
        recon_h = recon_h.view(bs*n, -1)
        mask = mask.view(-1).bool()

        atom = ligand_atom[mask]
        recon_h = recon_h[mask]

        recon_atom = recon_h[:, :self.atom_dim]
        ori_atom = atom[:, :self.atom_dim]

        recon_feat = recon_h[:, self.atom_dim:]
        atom_feat = atom[:, self.atom_dim:]

        # recon_feat = recon_h[:,10:]
        # ligand_atom_feat = ligand_atom[:,10:]
        recon_loss_h = F.cross_entropy(recon_atom, ori_atom.argmax(dim=-1))
        # recon_loss_f = F.cross_entropy(recon_feat, atom_feat.argmax(dim=-1))
        recon_loss_f = F.binary_cross_entropy_with_logits(recon_feat, atom_feat, reduction='mean')
        
        # recon_loss_feat = F.cross_entropy(recon_feat[mask], ligand_atom_feat[mask].argmax(dim=-1))
        
        # atom_feat = ligand_atom[:,10:]
        # recon_feat = recon_h[:, 10:]
        # recon_loss_feat = F.cross_entropy(recon_feat[mask], atom_feat[mask].argmax(dim=-1))
        

        # loss = (1 - self.ratio_x) * (self.kl_weight_h * kl_loss_h + recon_loss_h) + \
        #             self.ratio_x * (self.kl_weight_x * kl_loss_x +recon_loss_x) 
        loss = recon_loss_h + 0.25 * kl_loss + recon_loss_f
        
        # kl_loss_x = torch.zeros_like(recon_loss_x)
        
        accuracy = 0
        accuracy_f = 0
        accuracy_both = 0

        if valid:
            recon_atom = torch.argmax(recon_atom, dim=1)
            ligand_atoms = torch.argmax(ori_atom, dim=1)
            correct_predictions = (recon_atom == ligand_atoms)
            accuracy = correct_predictions.sum().item() / correct_predictions.size(0)

            recon_f = (torch.sigmoid(recon_feat) > 0.5)
            # ligand_feat = torch.argmax(atom_feat, dim=1)
            correct_pred_f = (recon_f == atom_feat).all(dim=-1).float()
            accuracy_f = correct_pred_f.sum().item() / correct_pred_f.size(0)

            correct_both_predictions = correct_predictions.bool() & correct_pred_f.bool()
            accuracy_both = correct_both_predictions.sum().item() / correct_both_predictions.size(0)

        return loss, kl_loss, recon_loss_h, accuracy, accuracy_f, accuracy_both


