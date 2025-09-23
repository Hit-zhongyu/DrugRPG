import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch_scatter import scatter_min

def to_torch_const(x):
    x = torch.from_numpy(x).float()
    x = nn.Parameter(x, requires_grad=False)
    return x

def log_1_min_a(a):
    return np.log(1 - np.exp(a) + 1e-40)


## --- probabily ---

# categorical diffusion related
def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


def extract(coef, t, batch, ndim=2):
    out = coef[t][batch]
    # warning: test wrong code!
    # out = coef[batch]
    # return out.view(-1, *((1,) * (len(out_shape) - 1)))
    if ndim == 1:
        return out
    elif ndim == 2:
        return out.unsqueeze(-1)
    elif ndim == 3:
        return out.unsqueeze(-1).unsqueeze(-1)
    else:
        raise NotImplementedError('ndim > 3')

def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

def log_sample_categorical2(logits):
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    sample_index = (gumbel_noise + logits)
    # sample_onehot = F.one_hot(sample, self.num_classes)
    # log_sample = index_to_log_onehot(sample, self.num_classes)
    return sample_index

def log_sample_categorical(logits):
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    sample_index = (gumbel_noise + logits).argmax(dim=-1)
    # sample_onehot = F.one_hot(sample, self.num_classes)
    # log_sample = index_to_log_onehot(sample, self.num_classes)
    return sample_index

def categorical_kl(log_prob1, log_prob2):
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=-1)
    return kl

def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=-1)


# ----- beta  schedule -----

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    alphas = np.clip(alphas, a_min=0.001, a_max=1.)
    alphas = np.sqrt(alphas)
    # betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return alphas


def advance_schedule(timesteps, scale_start, scale_end, width, return_alphas_bar=False):
    k = width
    A0 = scale_end
    A1 = scale_start

    a = (A0-A1)/(sigmoid(-k) - sigmoid(k))
    b = 0.5 * (A0 + A1 - a)

    x = np.linspace(-1, 1, timesteps)
    y = a * sigmoid(- k * x) + b
    # print(y)
    
    alphas_cumprod = y 
    alphas = np.zeros_like(alphas_cumprod)
    alphas[0] = alphas_cumprod[0]
    alphas[1:] = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas
    betas = np.clip(betas, 0, 1)
    if not return_alphas_bar:
        return betas
    else:
        return betas, alphas_cumprod

def segment_schedule(timesteps, time_segment, segment_diff):
    assert np.sum(time_segment) == timesteps
    alphas_cumprod = []
    for i in range(len(time_segment)):
        time_this = time_segment[i] + 1
        params = segment_diff[i]
        _, alphas_this = advance_schedule(time_this, **params, return_alphas_bar=True)
        alphas_cumprod.extend(alphas_this[1:])
    alphas_cumprod = np.array(alphas_cumprod)
    
    alphas = np.zeros_like(alphas_cumprod)
    alphas[0] = alphas_cumprod[0]
    alphas[1:] = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas
    betas = np.clip(betas, 0, 1)
    return betas

def sigmoid(x):
    return 1 / (np.exp(-x) + 1)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])

    alphas = np.clip(alphas, a_min=0.001, a_max=1.)

    # Use sqrt of this, so the alpha in our paper is the alpha_sqrt from the
    # Gaussian diffusion in Ho et al.
    alphas = np.sqrt(alphas)
    return alphas

def get_beta_schedule(beta_schedule, num_timesteps, **kwargs):
    
    if beta_schedule == "linear":
        betas = np.linspace(
            kwargs['scale_start'], kwargs['scale_end'], num_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = kwargs['scale_end'] * np.ones(num_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_timesteps, 1, num_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        s = dict.get(kwargs, 's', 6)
        betas = np.linspace(-s, s, num_timesteps)
        betas = sigmoid(betas) * (kwargs['scale_end'] - kwargs['scale_start']) + kwargs['scale_start']
    elif beta_schedule == "cosine":
        s = dict.get(kwargs, 's', 0.01)
        betas = cosine_beta_schedule(num_timesteps, s=s)
    elif beta_schedule == "advance":
        scale_start = dict.get(kwargs, 'scale_start', 0.999)
        scale_end = dict.get(kwargs, 'scale_end', 0.001)
        width = dict.get(kwargs, 'width', 2)
        betas = advance_schedule(num_timesteps, scale_start, scale_end, width)
    elif beta_schedule == "segment":
        betas = segment_schedule(num_timesteps, kwargs['time_segment'], kwargs['segment_diff'])
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_timesteps,)
    return betas



def build_edge_batch(ligand_batch, edge_index):
    edge_batch = torch.zeros(edge_index.shape[1], dtype=torch.long, device=edge_index.device) 
    for i in ligand_batch.unique():
        mask = (ligand_batch[edge_index[0]] == i) & (ligand_batch[edge_index[1]] == i)
        edge_batch[mask] = i
    return edge_batch


def find_closest_points(gt_dist, bond_index, cutoff=6.0):
    src, dst = bond_index
    # find the closest point to dst
    _, argmin0 = scatter_min(gt_dist, dst)
    n0 = src[argmin0]
    add = torch.zeros_like(gt_dist).to(gt_dist.device)
    add[argmin0] = cutoff
    dist1 = gt_dist + add

    # find the second closest point to dst
    _, argmin1 = scatter_min(dist1, dst)
    n1 = src[argmin1]

    assert n0.shape[0] == n1.shape[0] == torch.unique(src).shape[0]
    
    return n0, n1

def compute_bond_angle(pos_ji, pos_jk):
    a = (pos_ji * pos_jk).sum(dim=-1)
    b = torch.cross(pos_ji, pos_jk, dim=-1).norm(dim=-1)
    angle = torch.atan2(b, a)
    return angle

def compute_torsion_angle(pos_ji, pos_1, pos_2, eps=1e-7):
    plane1 = torch.cross(pos_ji, pos_1, dim=-1)
    plane2 = torch.cross(pos_ji, pos_2, dim=-1)

    dist_ji = pos_ji.norm(dim=-1) + eps
    a = (plane1 * plane2).sum(dim=-1) + eps
    b = (torch.cross(plane1, plane2, dim=-1) * pos_ji).sum(dim=-1) / dist_ji

    torsion = torch.atan2(b, a)
    #torsion[torsion<=0] += 2 * math.pi
    return torsion

def get_timestep_weight(t, max_timesteps=1000, max_weight=3.0, min_weight=0.75):

    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float32)
    
    # 将时间步归一化到[0,1]
    t_norm = t / max_timesteps
    
    weight = min_weight + (max_weight - min_weight) * 0.5 * (1 + torch.cos(np.pi * t_norm))
        
    weight = torch.clamp(weight, min_weight, max_weight)
    
    return weight
