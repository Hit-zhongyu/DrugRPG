import os
import torch
import torch.nn as nn
import numpy as np









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
        self.graph_library[smiles] = smiles_to_bigraph(
            smiles, 
            node_featurizer=self.atom_featurizer, 
            edge_featurizer=self.bond_featurizer
        )

    def make_graph(self, smiles):
        if self.cache:
            if smiles not in self.graph_library.keys():
                self.graph_library[smiles] = smiles_to_bigraph(
                    smiles, 
                    node_featurizer=self.atom_featurizer, 
                    edge_featurizer=self.bond_featurizer
                )
            return self.graph_library[smiles]
        else:
            if smiles in self.graph_library.keys():
                return self.graph_library[smiles]
            else:
                return smiles_to_bigraph(
                    smiles, 
                    node_featurizer=self.atom_featurizer,
                    edge_featurizer=self.bond_featurizer
                )

    def update_library(self, smiles_list):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.threads) as executor:
            futures = []
            for smiles in smiles_list:
                futures.append(executor.submit(self.update_graph, smiles))
            for _ in concurrent.futures.as_completed(futures):
                pass

    def smiles2graph(self, smiles_list):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.threads) as executor:
            graphs = list(executor.map(self.make_graph, smiles_list))
        input_tensor = batch(graphs).to('cuda')
        return input_tensor

    def forward(self, smiles_list):
        mol_graph = self.smiles2graph(smiles_list)
        mol_graph.ndata['atom_type'] = self.OBEncoder(
            mol_graph, 
            mol_graph.ndata['atom_type'], 
            mol_graph.edata['bond_type']
        )
        return mol_graph

class MolecularPooling(nn.Module):
    def __init__(
        self,
        num_features = 512,
        projector = 'LeakyReLU',
    ):
        super(MolecularPooling, self).__init__()
        self.projector = projector
        activation = 'ReLU' if projector == 'SeT' else projector
        gate_nn = nn.Sequential(
            nn.Linear(num_features, num_features * 3),
            nn.BatchNorm1d(num_features * 3),
            activation_dict[activation],
            nn.Linear(num_features * 3, 1024),
            nn.BatchNorm1d(1024),
            activation_dict[activation],
            nn.Linear(1024, 128),
            activation_dict[activation],
            nn.Linear(128, 128),
            activation_dict[activation],
            nn.Linear(128, 128),
            activation_dict[activation],
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        if projector == 'SeT':
            self.attention = SetTransformerEncoder(
                d_model = num_features,
                n_heads = 32,
                d_head = 256,
                d_ff = num_features * 4,
                n_layers = 2,
                block_type = 'sab',
                m = None,
                dropouth = 0.3,
                dropouta = 0.3
            )
            self.attention.to('cuda')
            self.readout = GlobalAttentionPooling(
                gate_nn = torch.compile(gate_nn),
                # Adding feat_nn reduces the representation learning capability, /
                # with a significant performance degradation on the zero-shot task
            )
        else:
            # init the readout and output layers
            self.readout = GlobalAttentionPooling(
                gate_nn = torch.compile(gate_nn),
                # Adding feat_nn reduces the representation learning capability, /
                # with a significant performance degradation on the zero-shot task
            )
        self.readout.cuda()
    
    def forward(self, mol_graph, get_atom_weights = False):
        if self.projector == 'SeT':
            mol_graph.ndata['atom_type'] = self.attention(
                mol_graph, 
                mol_graph.ndata['atom_type']
            )
        encoding, atom_weights = self.readout(
            mol_graph, 
            mol_graph.ndata['atom_type'], 
            get_attention = True
        )
        if get_atom_weights:
            return encoding, atom_weights
        else:
            return encoding
