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

sys.path.append('../repos/gmcd')
sys.path.append('../repos/gmcd/src')
sys.path.append('../repos/gmcd/src/datasets')
sys.path.append('../repos/gmcd/src/experiment')
sys.path.append('../repos/gmcd/src/optimizer')
sys.path.append('../repos/gmcd/src/mutils')

sys.path.append('../repos/gmcd/src/model')
from artransformer_diff import ArTransformerDiffusion  # src/model
from gaussian_diff import GaussianDiffusion # src/model
from linear_transformer import DenoisingTransformer # src/model

sys.path.append('../repos/DruM/DruM_2D')
sys.path.append('../repos/DruM/DruM_2D/parsers')
sys.path.append('../repos/DruM/DruM_2D/utils')
from config import get_config, print_cfg
from data_loader import dataloader, graphs_to_dataloader, graphs_to_tensor, init_features
from loader import load_data, load_batch

sys.path.append('../util')
from cython_normterm_discrete import create_fun_with_mem
from vgae import run_vgae
from gmm import GMMModelSelection
from gmm_based_diffmodel import my_start_training
from novelty_generation import MirrorDescentExponentialGradientOptimizer, NewClusterAllocatorMean, NewClusterAllocatorCovariance, VonNeumannOptimizer, kl_divergence_gmm, determine_new_cluster_by_novlety_condition, determine_weights_by_reliablity_condition

sys.path.append('../config')
from config_synthetic import SBMRunConfig

outdir = './output/planar'
if not os.path.exists(outdir):
    os.makedirs(outdir)

norm_multinom = create_fun_with_mem()

def main(out_channels, M):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = get_config('planar', 123)

    # load data
    train_loader, val_loader, test_loader, feat_dim = dataloader(config)
    with open('data/sbm.pkl', 'rb') as f:
        train_graphs, val_graphs, test_graphs = pickle.load(f)
        
    train_adjs_tensor = graphs_to_tensor(train_graphs, config.data.max_node_num)
    train_x_tensor, train_feat_dim = init_features(config.data, train_adjs_tensor) 

    val_adjs_tensor = graphs_to_tensor(val_graphs, config.data.max_node_num)
    val_x_tensor, val_feat_dim = init_features(config.data, val_adjs_tensor)

    test_adjs_tensor = graphs_to_tensor(test_graphs, config.data.max_node_num)
    test_x_tensor, test_feat_dim = init_features(config.data, test_adjs_tensor) 

    # Step1. Training Step
    # 1) Node Embedding
    in_channels = len(config['data']['feat']['type'])
    out_channels = out_channels
    model_vgae = run_vgae(train_x_tensor, val_x_tensor, test_x_tensor, 
                          in_channels, out_channels, device)

    mu_train = torch.stack([model_vgae(train_x_tensor[i, :, :].to(device), 
                                   train_adjs_tensor[i, :, :].nonzero().t().contiguous().to(device))[0] 
                                   for i in range(train_x_tensor.shape[0])], axis=0)
    logstd_train = torch.stack([model_vgae(train_x_tensor[i, :, :].to(device), 
                                           train_adjs_tensor[i, :, :].nonzero().t().contiguous().to(device))[1] 
                                       for i in range(train_x_tensor.shape[0])], axis=0)
    mu_val = torch.stack([model_vgae(val_x_tensor[i, :, :].to(device), 
                                     val_adjs_tensor[i, :, :].nonzero().t().contiguous().to(device))[0]
                         for i in range(val_x_tensor.shape[0])], axis=0)
    logstd_val = torch.stack([model_vgae(val_x_tensor[i, :, :].to(device), 
                                     val_adjs_tensor[i, :, :].nonzero().t().contiguous().to(device))[1]
                         for i in range(val_x_tensor.shape[0])], axis=0)
    mu_test = torch.stack([model_vgae(test_x_tensor[i, :, :].to(device), 
                                  test_adjs_tensor[i, :, :].nonzero().t().contiguous().to(device))[0] 
                          for i in range(test_x_tensor.shape[0])], axis=0)
    logstd_test = torch.stack([model_vgae(test_x_tensor[i, :, :].to(device), 
                                      test_adjs_tensor[i, :, :].nonzero().t().contiguous().to(device))[1] 
                               for i in range(test_x_tensor.shape[0])], axis=0)

    # datasets sampled from estimated mu and logstd
    z_train = mu_train + torch.randn(mu_train.shape).to(device) * torch.exp(logstd_train)
    z_val = mu_val + torch.randn(mu_val.shape).to(device) * torch.exp(logstd_val)
    z_test = mu_test + torch.randn(mu_test.shape).to(device) * torch.exp(logstd_test)

    # 2) Clustering with GMM
    z_train_list = []
    categ_train_list = []
    n_cluster_train_list = []
    gmm_model_selection_train_list = []
    for i in range(num_graphs_train):
        node_feature = train_x_tensor[i, :, :].to(device)
        edge_index = train_adjs_tensor[i].nonzero().t().contiguous().to(device)
        #print(node_feature.shape, edge_index.shape)
        num_nodes_train = node_feature.shape[0]
        
        z = model_vgae.encode(node_feature, edge_index)
        z_train_list.append(z)
    
        gmm_model_selection = GMMModelSelection(K_min=2, K_max=20, random_state=0, mode='GMM_DNML')
        gmm_model_selection.fit(z.cpu().detach().numpy())
        gmm_model_selection_train_list.append(gmm_model_selection)
    
        categ = gmm_model_selection.predict(z.cpu().detach().numpy())
        categ_train_list.append(categ)
    
        n_cluster = len(gmm_model_selection.model_best_.means_)
        n_cluster_train_list.append(n_cluster)


    z_val_list = []
    categ_val_list = []
    n_cluster_val_list = []
    gmm_model_selection_val_list = []
    for i in range(num_graphs_val):
        node_feature = val_x_tensor[i, :, :].to(device)
        edge_index = val_adjs_tensor[i].nonzero().t().contiguous().to(device)
        #print(node_feature.shape, edge_index.shape)
        num_nodes_val = node_feature.shape[0]
        
        z = model_vgae.encode(node_feature, edge_index)
        z_val_list.append(z)
    
        gmm_model_selection = GMMModelSelection(K_min=2, K_max=20, random_state=0, mode='GMM_DNML')
        gmm_model_selection.fit(z.cpu().detach().numpy())
        gmm_model_selection_val_list.append(gmm_model_selection)

        categ = gmm_model_selection.predict(z.cpu().detach().numpy())
        categ_val_list.append(categ)
    
        n_cluster = len(gmm_model_selection.model_best_.means_)
        n_cluster_val_list.append(n_cluster)
    
    z_test_list = []
    categ_test_list = []
    n_cluster_test_list = []
    gmm_model_selection_test_list = []
    for i in range(num_graphs_test):
        node_feature = test_x_tensor[i, :, :].to(device)
        edge_index = test_adjs_tensor[i].nonzero().t().contiguous().to(device)
        #print(node_feature.shape, edge_index.shape)
        num_nodes_test = node_feature.shape[0]
        
        z = model_vgae.encode(node_feature, edge_index)
        z_test_list.append(z)
    
        gmm_model_selection = GMMModelSelection(K_min=2, K_max=20, random_state=0, mode='GMM_DNML')
        gmm_model_selection.fit(z.cpu().detach().numpy())
        gmm_model_selection_test_list.append(gmm_model_selection)
        
        categ = gmm_model_selection.predict(z.cpu().detach().numpy())
        categ_test_list.append(categ)
    
        n_cluster = len(gmm_model_selection.model_best_.means_)
        n_cluster_test_list.append(n_cluster)

    # 3) GMM-based Diffusion Model
    runconfig = SBMRunConfig(dataset='sbm', S=16, K=2)

    categ_train = torch.tensor(categ_train_list)[np.array(n_cluster_train_list) == 2, :]
    categ_val = torch.tensor(categ_val_list)[np.array(n_cluster_val_list) == 2, :]
    categ_test = torch.tensor(categ_test_list)[np.array(n_cluster_test_list) == 2, :]

    z_train = torch.stack(z_train_list)[np.array([np.unique(c).shape[0] for c in categ_train_list])==2, :, :]
    z_val = torch.stack(z_val_list)[np.array([np.unique(c).shape[0] for c in categ_val_list])==2, :, :]
    z_test = torch.stack(z_test_list)[np.array([np.unique(c).shape[0] for c in categ_test_list])==2, :, :]

    z_train_2d = z_train.reshape(z_train.shape[0] * z_train.shape[1], z_train.shape[2])
    z_val_2d = z_val.reshape(z_val.shape[0] * z_val.shape[1], z_val.shape[2])
    z_test_2d = z_test.reshape(z_test.shape[0] * z_test.shape[1], z_test.shape[2])

    model_gmmdiff = my_start_training(runconfig, 
                           z_train_2d, z_val_2d, z_test_2d, 
                           categ_train, categ_val, categ_test, 
                           dataset_name='planar', 
                           return_result=True)

    # Step2. Novelty Generation Step
    for i_sample, gmm in enumerate(gmm_model_selection_test_list):
        # 1) Novelty Condition
        alloc_mean, alloc_cov = determine_new_cluster_by_novlety_condition(gmm)

        # 2) Reliability Condition
        weights = determine_weights_by_reliablity_condition(gmm, alloc_mean, alloc_cov)

        # 3) Generate New Data Samples in Latent Space
        z_new = torch.distributions.MultivariateNormal(
            torch.Tensor(alloc_mean.mu_.detach().numpy()), 
            torch.Tensor(alloc_cov.covariances_.detach().numpy())).sample([M]).to(device)

        z_sample_new = np.vstack([z_test[i_sample, :, :].cpu().detach().numpy(), z_new.cpu().detach().numpy()])

        z_sample_new_reversed = model_gmmdiff.predict(z_sample_new)
        
        edge_tuple_orig = test_adjs_tensor[i_sample].cpu().numpy().nonzero()
        edge_list_orig = np.hstack((edge_tuple_orig[0].reshape(-1, 1), edge_tuple_orig[1].reshape(-1, 1)))
        G_orig = nx.Graph(directed=True)
        G_orig.add_nodes_from(np.arange(6))
        G_orig.add_edges_from(edge_list_orig)

        norm = torch.linalg.norm(torch.tensor(z_sample_new_reversed), axis=1)
        probs_edge_new = 1.0/(1.0 + torch.exp(-(z_sample_new_reversed @ z_sample_new_reversed.T) / (norm.reshape(-1, 1) * norm.reshape(1, -1))))

        estimated_edge_new = (probs_edge_new >= 0.7).nonzero()

        ok = ((estimated_edge_new[:, 0] >= 64) | (estimated_edge_new[:, 1] >= 64)) & (estimated_edge_new[:, 0] != estimated_edge_new[:, 1])
        estimated_edge_new = estimated_edge_new[ok, :]

        G_new = nx.Graph(directed=True)
        G_new.add_nodes_from(np.arange(len(G_orig.nodes) + M))
        G_new.add_edges_from(np.vstack((np.array(list(G_orig.edges)), estimated_edge_new)))

if __name__ == 'main':
    main(out_channels=4, M=10)