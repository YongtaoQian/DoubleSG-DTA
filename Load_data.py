'''
Author: Yongtao Qian
Time: 2023-2-5
model: DoubleSG-DTA
'''

import pandas as pd
import numpy as np
import os
import json, pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import *


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index


seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000
def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x

seq_drug = "(.02468@BDFHLNPRTVZ/bdfhlnrt#%)+-/13579=ACEGIKMOSUWY[]acegimosuy\\"
seq_dict_drug = {v: (i + 1) for i, v in enumerate(seq_drug)}
seq_dict_drug_len = len(seq_dict)
max_seq_drug_len = 100
def seq_drug(smile):
    x = np.zeros(max_seq_drug_len)
    for i, ch in enumerate(smile[:max_seq_drug_len]):
        x[i] = seq_dict_drug[ch]
    return x



'''
The drug sequences in the dataset are first converted into the corresponding drug maps, and the drug sequences, drug maps, protein sequences and true affinity values are combined into the same pytorch dataset
'''
compound_iso_smiles = []
for dt_name in ['Davis','KIBA','bindingdb']:
    opts = ['train', 'test']
    for opt in opts:
        df = pd.read_csv('autodl-tmp/DoubleSG-DTA/data/' + dt_name + '_' + opt + '.csv')
        compound_iso_smiles += list(df['compound_iso_smiles'])
compound_iso_smiles = set(compound_iso_smiles)
smile_graph = {}
for smile in compound_iso_smiles:
    g = smile_to_graph(smile)
    smile_graph[smile] = g

    
    
datasets = ['Davis','KIBA','bindingdb']
# convert to PyTorch data format
for dataset in datasets:
    processed_data_file_train = 'autodl-tmp/DoubleSG-DTA/data/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'autodl-tmp/DoubleSG-DTA/data/processed/' + dataset + '_test.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        
        df = pd.read_csv('autodl-tmp/DoubleSG-DTA/data/' + dataset + '_train.csv')
        train_drugs, train_prots, train_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(
            df['affinity'])
        XT = [seq_cat(t) for t in train_prots]
        Xd = [seq_drug(t) for t in train_drugs]
        train_drugs, train_prots, train_Y, train_smiles = np.asarray(train_drugs), np.asarray(XT), np.asarray(train_Y), np.asarray(Xd)
        
        
        
        df = pd.read_csv('autodl-tmp/DoubleSG-DTA/data/' + dataset + '_test.csv')
        test_drugs, test_prots, test_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(
            df['affinity'])
        XT = [seq_cat(t) for t in test_prots]
        Xd = [seq_drug(t) for t in test_drugs]
        test_drugs, test_prots, test_Y, test_smiles = np.asarray(test_drugs), np.asarray(XT), np.asarray(test_Y), np.asarray(Xd)

        # make data PyTorch Geometric ready
        print('preparing ', dataset + '_train.pt in pytorch format!')
        train_data = TestbedDataset(root='autodl-tmp/DoubleSG-DTA/data', dataset=dataset + '_train', xd=train_drugs, xt=train_prots, y=train_Y, xs = train_smiles,
                                    smile_graph=smile_graph)
        print('preparing ', dataset + '_test.pt in pytorch format!')
        test_data = TestbedDataset(root='autodl-tmp/DoubleSG-DTA/data', dataset=dataset + '_test', xd=test_drugs, xt=test_prots, y=test_Y, xs = test_smiles,
                                   smile_graph=smile_graph)
        print(processed_data_file_train, ' and ', processed_data_file_test, ' have been created')
    else:
        print(processed_data_file_train, ' and ', processed_data_file_test, ' are already created')
