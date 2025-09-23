import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from dgl import batch, unbatch
from dgllife.utils import (atom_type_one_hot, atom_formal_charge, atom_hybridization_one_hot, atom_chiral_tag_one_hot, atom_is_in_ring, 
                            atom_is_aromatic, ConcatFeaturizer, BaseAtomFeaturizer, CanonicalBondFeaturizer, smiles_to_bigraph, mol_to_bigraph)
from dgl.nn.pytorch.glob import GlobalAttentionPooling
from dgl.nn import SetTransformerEncoder
from dgllife.model.gnn.wln import WLN
# from .modules import (
#     activation_dict
# )
import concurrent.futures
from tqdm import tqdm
import os
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib
import matplotlib.cm as cm

activation_dict = {
    'GELU': nn.GELU(),
    'ReLU': nn.ReLU(),
    'LeakyReLU': nn.LeakyReLU(),
    'Tanh': nn.Tanh(),
    'SiLU': nn.SiLU(),
    'ELU': nn.ELU(),
    'SELU': nn.SELU(),
    'CELU': nn.CELU(),
    'Softplus': nn.Softplus(),
    'Softsign': nn.Softsign(),
    'Hardshrink': nn.Hardshrink(),
    'Hardtanh': nn.Hardtanh(),
    'Hardsigmoid': nn.Hardsigmoid(),
    'LogSigmoid': nn.LogSigmoid(),
    'Softshrink': nn.Softshrink(),
    'PReLU': nn.PReLU(),
    'Softmin': nn.Softmin(dim=-1),
    'Softmax': nn.Softmax(dim=-1),
    'Softmax2d': nn.Softmax2d(),
    'LogSoftmax': nn.LogSoftmax(dim=-1),
    'Sigmoid': nn.Sigmoid(),
    'Identity': nn.Identity(),
    'Tanhshrink': nn.Tanhshrink(),
    'RReLU': nn.RReLU(),
    'Hardswish': nn.Hardswish(),   
    'Mish': nn.Mish(),
}

class MolecularEncoder(nn.Module):
    '''
    MolecularEncoder

    This is a graph encoder model consisting of a graph neural network from 
    DGL and the MLP architecture in pytorch, which builds an understanding 
    of a compound's conformational space by supervised learning of CSS data.

    '''
    def __init__(
        self,
        num_features = 1024,
        num_layers = 4,
        cache = False,
        graph_library = {},
        threads = 40
    ):
        super().__init__()
        
        ## set up featurizer
        self.atom_featurizer = BaseAtomFeaturizer({
            'atom_type':ConcatFeaturizer(
                [
                    atom_type_one_hot, 
                    atom_hybridization_one_hot, 
                    atom_formal_charge, 
                    atom_chiral_tag_one_hot, 
                    atom_is_in_ring, 
                    atom_is_aromatic
        ])}) 
        self.bond_featurizer = CanonicalBondFeaturizer(
            bond_data_field='bond_type'
        )
        # init the OBEncoder
        self.OBEncoder = WLN(
            self.atom_featurizer.feat_size(feat_name='atom_type'), 
            self.bond_featurizer.feat_size(feat_name='bond_type'), 
            n_layers = num_layers, 
            node_out_feats = num_features
        )
        self.OBEncoder.to('cuda')
        self.cache = cache
        self.graph_library = graph_library
        self.threads = threads

    def update_graph(self, smiles):
        self.graph_library[smiles] = mol_to_bigraph(
            smiles, 
            node_featurizer=self.atom_featurizer, 
            edge_featurizer=self.bond_featurizer
        )

    def make_graph(self, smiles):
        if self.cache:
            if smiles not in self.graph_library.keys():
                    
                self.graph_library[smiles] = mol_to_bigraph(
                    smiles, 
                    node_featurizer=self.atom_featurizer, 
                    edge_featurizer=self.bond_featurizer,
                    canonical_atom_order=False,
                )
            return self.graph_library[smiles]
        else:
            if smiles in self.graph_library.keys():
                return self.graph_library[smiles]
            else:
                return mol_to_bigraph(
                    smiles, 
                    node_featurizer=self.atom_featurizer,
                    edge_featurizer=self.bond_featurizer,
                    canonical_atom_order=False,
                )

    def update_library(self, smiles_list):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.threads) as executor:
            futures = []
            for smiles in smiles_list:
                futures.append(executor.submit(self.update_graph, smiles))
            for _ in concurrent.futures.as_completed(futures):
                pass

    def mol2graph(self, mol_list):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.threads) as executor:
            graphs = list(executor.map(self.make_graph, mol_list))
        input_tensor = batch(graphs).to('cuda')
        return input_tensor

    def forward(self, mol_list):
        mol_graph = self.mol2graph(mol_list)
            
        mol_graph.ndata['atom_type'] = self.OBEncoder(
            mol_graph, 
            mol_graph.ndata['atom_type'], 
            mol_graph.edata['bond_type']
        )
        return mol_graph


class GeminiMol(nn.Module):
    def __init__(
        self,
        model_path = None,
        batch_size = 64,
        encoder_params = {
            "num_layers": 4,
            "encoding_features": 2048,
        },
        pooling_params = {
            "projector": "LeakyReLU",
        },
        cache = False,
        threads = 40
    ):
        # basic setting
        torch.set_float32_matmul_precision('high')
        super(GeminiMol, self).__init__()
        ## load parameters
        self.batch_size = batch_size 
        self.model_path = model_path
        self.model_name = model_path
        if model_path is not None and os.path.exists(f'{model_path}/MolEncoder.pt'):
            self.encoder_params = json.load(open(f'{model_path}/encoder_config.json', 'r'))
            ## create MolecularEncoder
            self.Encoder = MolecularEncoder(
                num_layers = self.encoder_params['num_layers'],
                num_features = self.encoder_params['encoding_features'],
                cache = cache,
                threads = threads
            )
            self.Encoder.load_state_dict(torch.load(f'{model_path}/MolEncoder.pt', weights_only=True))
        else:
            self.encoder_params = encoder_params
            os.makedirs(model_path, exist_ok = True)
            with open(f'{model_path}/encoder_config.json', 'w', encoding='utf-8') as f:
                json.dump(encoder_params, f, ensure_ascii=False, indent=4)
            ## create MolecularEncoder
            self.Encoder = MolecularEncoder(
                num_layers = encoder_params['num_layers'],
                num_features = encoder_params['encoding_features'],
                cache = cache,
                threads = threads
            )
        # if os.path.exists(f'{model_path}/Pooling.pt'):
        #     self.pooling_params = json.load(open(f'{model_path}/pooling_config.json', 'r'))
        #     self.pooling = MolecularPooling(
        #         num_features = self.encoder_params['encoding_features'],
        #         projector = self.pooling_params['projector'],
        #     )
        #     self.pooling.load_state_dict(torch.load(f'{model_path}/Pooling.pt', weights_only=True))
        # else:
        #     self.pooling_params = pooling_params
        #     with open(f'{model_path}/pooling_config.json', 'w', encoding='utf-8') as f:
        #         json.dump(pooling_params, f, ensure_ascii=False, indent=4)
        #     self.pooling = MolecularPooling(
        #         num_features = encoder_params['encoding_features'],
        #         projector = pooling_params['projector'],
        #     )

    def encode(self, mol_list):
        # Encode all sentences using the encoder 
        mol_graph = self.Encoder(mol_list)
        
        # features = self.pooling(mol_graph)
        return mol_graph