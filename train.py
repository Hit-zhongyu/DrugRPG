import os
import torch
import argparse
import yaml
import time
import logging
import shutil

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.utils.tensorboard
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from glob import glob
from torch.optim import AdamW,Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR
from torch.nn.utils import clip_grad_norm_
from utils.transforms import (FeaturizeLigandAtom, FeaturizeProteinAtom,
                                FeaturizeLigandBond, GetAdj)

from utils.datasets import *
from easydict import EasyDict
from model.DrugRPG import DrugRPG
from utils.misc import *
from model.ema import ExponentialMovingAverage
from utils.utils import get_optimizer,  get_scheduler
from torch.utils.data import ConcatDataset
# os.environ['CUDA_VISIBLE_DEVICES'] = "4"

import torch.profiler

torch.set_printoptions(threshold=1000, edgeitems=300, linewidth=200, sci_mode=False)
# logger = get_logger("train")
logging.getLogger().setLevel(logging.INFO)

def ignore_pyc(dir, files):
    return [f for f in files if f.endswith('.pyc') or f == '__pycache__']

def train(config, iter, state, model, train_loader, optimizer, local_rank, writer, ema):

    sum_loss, sum_n = 0, 0
    sum_loss_atom, sum_loss_pos, sum_loss_edge, sum_loss_bond = 0, 0, 0, 0
    model.train()

    with tqdm(total=len(train_loader), desc='Training') as pbar:
        for i, batch in enumerate(train_loader): 
            
            optimizer.zero_grad()
            loss, loss_atom, loss_pos, loss_edge, loss_bond, _, _ = model(batch, local_rank)
            loss = loss.mean()
            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), config.optimizer.grad_clip)
            optimizer.step()
            # grad_norm = optimization_manager(optimizer, model.parameters(), state, config.optimizer)
            ema.update(model.parameters())
            state['step'] += 1

            sum_loss += loss.item()
            sum_n += 1
            sum_loss_atom += loss_atom.mean().item()
            sum_loss_pos += loss_pos.mean().item()
            sum_loss_edge += loss_edge.mean().item()
            sum_loss_bond += loss_bond.mean().item()

            pbar.set_postfix({'loss':'%.2f'%(loss.item())})
            pbar.update(1)
    avg_loss = sum_loss / sum_n
    avg_loss_atom = sum_loss_atom / sum_n
    avg_loss_pos = sum_loss_pos / sum_n
    avg_loss_edge = sum_loss_edge / sum_n
    avg_loss_bond = sum_loss_bond / sum_n

    if dist.get_rank() == 0:
        logging.info('[Train] Epoch %04d | Loss %.2f | Loss(Atom) %.2f | Loss(Pos) %.2f | Loss(Edge) %.2f | Loss(Bond) %.2f | LR %.6f' % (
                    iter, avg_loss, avg_loss_atom, avg_loss_pos, avg_loss_edge, avg_loss_bond, optimizer.param_groups[0]['lr'],
                ))
        writer.add_scalar('train/loss', avg_loss, iter)
        writer.add_scalar('train/loss_atom', avg_loss_atom, iter)
        writer.add_scalar('train/loss_pos', avg_loss_pos, iter)
        writer.add_scalar('train/loss_edge_index', avg_loss_edge, iter)
        writer.add_scalar('train/loss_bond', avg_loss_bond, iter)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], iter)
        writer.add_scalar('train/grad_norm', grad_norm, iter)
        writer.flush()

def valid(iter, model, val_loader, local_rank, scheduler, writer):
    sum_loss, sum_n = 0, 0
    sum_loss_atom, sum_loss_pos, sum_loss_edge, sum_loss_bond = 0, 0, 0, 0
    all_pred_atom, all_true_atom = [], []
    all_pred_bond, all_true_bond = [], []

    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(tqdm(val_loader, desc='Validation')):
            loss, loss_atom, loss_pos, loss_edge, loss_bond, atom_pred, bond_pred = model(batch, local_rank, valid=True)

            sum_loss += loss.mean().item()
            sum_n += 1
            sum_loss_atom += loss_atom.mean().item()
            sum_loss_pos += loss_pos.mean().item()
            sum_loss_edge += loss_edge.mean().item()
            sum_loss_bond += loss_bond.mean().item()
            all_pred_atom.append(atom_pred.argmax(-1).detach().cpu().numpy())
            all_true_atom.append(batch['ligand_atom'].long().argmax(-1).detach().cpu().numpy())
            all_pred_bond.append(bond_pred.argmax(-1).long().detach().cpu().numpy())
            all_true_bond.append(batch['ligand_bond_type'].argmax(-1).detach().cpu().numpy())

    avg_loss = sum_loss / sum_n
    avg_loss_atom = sum_loss_atom / sum_n
    avg_loss_pos = sum_loss_pos / sum_n
    avg_loss_edge = sum_loss_edge / sum_n
    avg_loss_bond = sum_loss_bond / sum_n
    acc_atom = (np.concatenate(all_pred_atom, axis=0) == np.concatenate(all_true_atom, axis=0)).astype(np.float32).mean()
    acc_bond = (np.concatenate(all_pred_bond, axis=0) == np.concatenate(all_true_bond, axis=0)).astype(np.float32).mean()
    scheduler.step(avg_loss)
    if dist.get_rank() == 0:
        logging.info('[Validate] Iter %04d | Loss %.3f | Loss(Atom) %.3f| Loss(Pos) %.3f| Loss(Edge) %.3f| Loss(Bond) %.3f '
                            '|Acc(Atom) %.3f |Acc(Bond) %.3f ' % (
                iter, avg_loss, avg_loss_atom, avg_loss_pos, avg_loss_edge, avg_loss_bond,acc_atom,acc_bond
            ))
        
        writer.add_scalar('val/loss', avg_loss, iter)
        writer.add_scalar('val/atom', avg_loss_atom, iter)
        writer.add_scalar('val/pos', avg_loss_pos, iter)
        writer.add_scalar('val/edge', avg_loss_edge, iter)
        writer.add_scalar('val/bond', avg_loss_bond, iter)
        writer.flush()
    return avg_loss


def main(args):

    with open(args.config, 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    # 主函数
    gradnorm_queue = Queue()
    gradnorm_queue.add(3000)  # Add large value that will be flushed.
    torch.backends.cudnn.benchmark = True
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # set device
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    seed_all(config.train.seed)

    resume = os.path.isdir(args.config)
    if resume:
        print('Resume!')
        config_path = glob(os.path.join(args.config, '*.yml'))[0]
        config_path = './configs/pdbind_epoch.yml' # fintune in PDBind dataset
        # config_path = './configs/crossdock_epoch.yml'
        resume_from = args.config
        log_dir = get_new_log_dir(config.logs.logdir, prefix="crossdock", tag='resume')
        os.symlink(os.path.realpath(resume_from), os.path.join(log_dir, os.path.basename(resume_from.rstrip("/"))))
    else:
        config_path = args.config
        log_dir = get_new_log_dir(config.logs.logdir, prefix="crossdock")
        if not os.path.exists(os.path.join(log_dir, 'models')):
            shutil.copytree('./model', os.path.join(log_dir, 'models'),dirs_exist_ok=True, ignore=ignore_pyc)
            shutil.copytree('./diffusion', os.path.join(log_dir, 'diffusion'),dirs_exist_ok=True, ignore=ignore_pyc)

    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    writer = torch.utils.tensorboard.SummaryWriter(config.logs.logdir)

    if dist.get_rank() == 0:
        logging.info(args)
        logging.info('Loading %s datasets...' % (config.dataset.data_path))
        
    shutil.copyfile(config_path, os.path.join(log_dir, os.path.basename(config_path)))
    transform = Compose([   
        FeaturizeProteinAtom(),
        FeaturizeLigandAtom(),
        # FeaturizeProteinResidue(),
        FeaturizeLigandBond(),
        GetAdj()
    ])
    # torch.set_num_threads(1)

    kwargs = {'num_workers': config.train.num_workers, 'pin_memory': True} if args.cuda else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    dataset, subsets = get_dataset(config.dataset.data_path, config.dataset.split_data_path, config.dataset.split, transform=transform)
    train_set, val_set = subsets['train'], subsets['test']

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, shuffle=False)

    follow_batch = ['protein_element', 'ligand_element', 'residue_amino_acid']
    collate_exclude_keys = ['ligand_nbh_list']
    train_loader = DataLoader(
        train_set, 
        batch_size = config.train.batch_size, 
        shuffle = False,
        sampler = train_sampler,
        follow_batch = follow_batch,
        exclude_keys = collate_exclude_keys,
        # num_workers=config.train.num_workers,
        **kwargs
    )

    val_loader = DataLoader(
        val_set, 
        config.train.batch_size, 
        shuffle=False,
        sampler = val_sampler,
        follow_batch=follow_batch,
        exclude_keys = collate_exclude_keys,
        # num_workers=config.train.num_workers,
        **kwargs
    )

    if dist.get_rank() == 0:
        logging.info('Buliding model...')

    model = DrugRPG(config.model, device=local_rank).to(local_rank)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)

    optimizer = get_optimizer(config.optimizer, model.parameters())
    scheduler = get_scheduler(config.scheduler, optimizer)
    state = dict(optimizer=optimizer, model=model, ema=ema, scheduler=scheduler, step=0)
    # scheduler = get_constant_schedule_with_warmup(
    #     optimizer=optimizer, num_warmup_steps=args.warmup_steps
    # )
    total_params = sum(p.numel() for p in model.parameters())
    logging.info("Model total patams is %d" % total_params)

    start_iter = 1
    train_epoch = config.train.train_epoch
    best_val_loss = float('inf')
    best_epoch = 0
    

    # Resume from checkpoint
    if resume:
        resume_from = args.ckpt
        ckpt =  torch.load(resume_from, weights_only=True)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        # step.load_state_dict(ckpt['step'])
        model.eval()
        # config.train.max_iters = start_iter + 500
        start_iter += 1

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    for iters in range(start_iter, train_epoch):
        start_time = time.time()
        train(config, iters, state, model, train_loader, optimizer, local_rank, writer, ema)
        end_time = (time.time() - start_time)
        
        if dist.get_rank() == 0:
            print('each iteration requires {:.2f} s'.format(end_time))
            avg_val_loss = valid(iters, model, val_loader, local_rank, scheduler, writer)
            if iters % config.train.per_save ==0:
                if avg_val_loss<best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = iters
                ckpt_path = os.path.join(ckpt_dir, '%d.pt' % iters)
                torch.save({
                    'config': config,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': iters,
                    'ema': ema.state_dict(),
                    'avg_val_loss': avg_val_loss,
                }, ckpt_path)

                print('Best epoch is: {}, loss is: {:.2f}. Successfully saved the model!'.format(best_epoch, best_val_loss))
    dist.destroy_process_group()
    
    
if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/train_config.yml')
    parser.add_argument('--cuda', type=bool, default=True)

    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument("--local_rank", default=-1, type=int)
    
    args = parser.parse_args()
    mp.set_start_method('spawn', force=True)
    
    main(args)