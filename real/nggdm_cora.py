import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys
import pickle

import tqdm

import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric import transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import FacebookPagePage, Planetoid, TUDataset, QM9, ZINC
from torch_geometric.utils import negative_sampling, to_networkx
from torch_geometric.nn import VGAE, GCNConv

from sklearn.manifold import TSNE

from sklearn.mixture import GaussianMixture
from scipy.special import logsumexp, loggamma

import math

import networkx as nx

sys.path.append('repos/gmcd')
sys.path.append('repos/gmcd/src')
sys.path.append('repos/gmcd/src/datasets')
sys.path.append('repos/gmcd/src/experiment')
sys.path.append('repos/gmcd/src/optimizer')
sys.path.append('repos/gmcd/src/mutils')

sys.path.append('../repos/gmcd/src/model')
from artransformer_diff import ArTransformerDiffusion  # src/model
from gaussian_diff import GaussianDiffusion # src/model
from linear_transformer import DenoisingTransformer # src/model

sys.path.append('repos/DruM/DruM_2D')
sys.path.append('repos/DruM/DruM_2D/parsers')
sys.path.append('repos/DruM/DruM_2D/utils')
from config import get_config, print_cfg
from data_loader import dataloader, graphs_to_dataloader, graphs_to_tensor, init_features
from loader import load_data, load_batch

sys.path.append('../util')
from cython_normterm_discrete import create_fun_with_mem
from vgae import run_vgae
from gmm import GMMModelSelection
from gmm_based_diffmodel import my_start_training
from novelty_generation import MirrorDescentExponentialGradientOptimizer, NewClusterAllocatorMean, NewClusterAllocatorCovariance, VonNeumannOptimizer, kl_divergence_gmm, determine_new_cluster_by_novlety_condition, determine_weights_by_reliablity_condition

outdir = './output/planar'
if not os.path.exists(outdir):
    os.makedirs(outdir)

norm_multinom = create_fun_with_mem()

def train_val_test_split(data, val_ratio: float = 0.15,
                             test_ratio: float = 0.15):
    rnd = torch.rand(len(data.x))
    train_mask = [False if (x > val_ratio + test_ratio) else True for x in rnd]
    val_mask = [False if (val_ratio + test_ratio >= x) and (x > test_ratio) else True for x in rnd]
    test_mask = [False if (test_ratio >= x) else True for x in rnd]
    return torch.tensor(train_mask), torch.tensor(val_mask), torch.tensor(test_mask)

def main(out_channels, M):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load data
    name = 'Cora'
    dataset = Planetoid(root='./data', name=name)
    data = dataset[0]

    transform = T.Compose([
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          split_labels=True, add_negative_train_samples=False)
    ])
    dataset = Planetoid(root='./data', name=name, transform=transform)
    train_data, val_data, test_data = dataset[0]

    train_x_tensor, val_x_tensor, test_x_tensor = train_data.x, val_data.x, test_data.x
    train_edge_index_tensor, val_edge_index_tensor, test_edge_index_tensor = train_data.edge_index, val_data.edge_index, test_data.edge_index
    
    # Step1. Training Step
    # 1) Node Embedding
    in_channels = dataset.num_features
    out_channels = out_channels
    model_vgae = run_vgae(train_x_tensor, val_x_tensor, test_x_tensor, 
                          in_channels, out_channels, device)

    mu_train, logstd_train = model_vgae(train_x_tensor, train_edge_index_tensor)
    mu_val, logstd_val = model_vgae(val_x_tensor, val_edge_index_tensor)
    mu_test, logstd_test = model_vgae(test_x_tensor, test_edge_index_tensor)

    # datasets sampled from estimated mu and logstd
    z_train = mu_train + torch.randn(mu_train.shape).to(device) * torch.exp(logstd_train)
    z_val = mu_val + torch.randn(mu_val.shape).to(device) * torch.exp(logstd_val)
    z_test = mu_test + torch.randn(mu_test.shape).to(device) * torch.exp(logstd_test)

    # 2) Clustering with GMM
    gmm_model_selection = GMMModelSelection(K_min=2, K_max=20, random_state=0, mode='GMM_DNML')
    gmm_model_selection.fit(z_train.cpu().detach().numpy())

    categ_train = gmm_model_selection.predict(z_train.cpu().detach().numpy())
    categ_val = gmm_model_selection.predict(z_val.cpu().detach().numpy())
    categ_test = gmm_model_selection.predict(z_test.cpu().detach().numpy())
    n_cluster = len(gmm_model_selection.model_best_.means_)

    # 3) GMM-based Diffusion Model
    runconfig = CoraRunConfig(dataset=name, S=16, K=n_cluster)

    z_train_2d = z_train.reshape(z_train.shape[0] * z_train.shape[1], z_train.shape[2])
    z_val_2d = z_val.reshape(z_val.shape[0] * z_val.shape[1], z_val.shape[2])
    z_test_2d = z_test.reshape(z_test.shape[0] * z_test.shape[1], z_test.shape[2])

    model_gmmdiff = my_start_training(runconfig, 
                           z_train_2d, z_val_2d, z_test_2d, 
                           categ_train, categ_val, categ_test, 
                           dataset_name=name, 
                           return_result=True)

    # Step2. Novelty Generation Step
    # 1) Novelty Condition
    alloc_mean, alloc_cov = determine_new_cluster_by_novlety_condition(gmm_model_selection)

    # 2) Reliability Condition
    weights = determine_weights_by_reliablity_condition(gmm, alloc_mean, alloc_cov)

    # 3) Generate New Data Samples in Latent Space
    z_new = torch.distributions.MultivariateNormal(
        torch.Tensor(alloc_mean.mu_.detach().numpy()), 
        torch.Tensor(alloc_cov.covariances_.detach().numpy())).sample([M]).to(device)

    z_sample_new = np.vstack([z_train[i_sample, :, :].cpu().detach().numpy(), z_new.cpu().detach().numpy()])

    z_sample_new_reversed = model_gmmdiff.predict(z_sample_new)
    
    edge_tuple_orig = train_adjs_tensor[i_sample].cpu().numpy().nonzero()
    edge_list_orig = np.hstack((edge_tuple_orig[0].reshape(-1, 1), edge_tuple_orig[1].reshape(-1, 1)))
    G_orig = nx.Graph(directed=True)
    G_orig.add_nodes_from(np.arange(z_train.shape[0]))
    G_orig.add_edges_from(edge_list_orig)

    norm = torch.linalg.norm(torch.tensor(z_sample_new_reversed), axis=1)
    probs_edge_new = 1.0/(1.0 + torch.exp(-(z_sample_new_reversed @ z_sample_new_reversed.T) / (norm.reshape(-1, 1) * norm.reshape(1, -1))))

    estimated_edge_new = (probs_edge_new >= 0.7).nonzero()

    ok = ((estimated_edge_new[:, 0] >= z_train.shape[0]) | (estimated_edge_new[:, 1] >= z_train.shape[0])) & (estimated_edge_new[:, 0] != estimated_edge_new[:, 1])
    estimated_edge_new = estimated_edge_new[ok, :]

    G_new = nx.Graph(directed=True)
    G_new.add_nodes_from(np.arange(len(G_orig.nodes)+N_added))
    G_new.add_edges_from(np.vstack((np.array(list(G_orig.edges)), estimated_edge_new)))

if __name__ == 'main':
    main(out_channels=16, M=100)