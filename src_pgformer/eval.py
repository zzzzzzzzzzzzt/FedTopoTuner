# eval.py
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from struct import pack
from dataset import *
from utils import my_dict, load_or_generate_node_features, load_or_generate_adj_matrix

# read_fvecs 和 write_fvecs 函数已移到 utils.py，此处可删除
target = ['recall', 'qps', 'distcomp']

@torch.no_grad()
def evaluate(model, val, args, device):
    loss_recall = 0
    loss_qps = 0
    loss_distcomp = 0
    dataset_cnt = 0
    
    bases = val['dataset'].unique()
    pca = PCA(n_components=args.input)
    
    for dataset in bases:
        print(f"Evaluating dataset: {dataset}")
        now_mb = val[val.dataset == dataset]
        
        if len(now_mb) < 1:
            print(f"Skipping empty dataset: {dataset}")
            continue
        
        dataset_cnt += 1
        
        target0 = now_mb[target[0]]
        target1 = now_mb[target[1]]
        target2 = now_mb[target[2]]
        
        now = now_mb.drop(['dataset'] + target, axis=1)
        now = torch.from_numpy(now.values).float().to(device)
        
        target0 = torch.from_numpy(target0.values).float().to(device)
        target1 = torch.from_numpy(target1.values).float().to(device)
        target2 = torch.from_numpy(target2.values).float().to(device)

        feature_path = f"{args.path}/dataset/data/{dataset}/{dataset}_base_{args.input}.fvecs"
        if feature_path not in my_dict:
            """print(f"Node features not in cache, generating for: {feature_path}")"""
            node_feat = load_or_generate_node_features(dataset, args.path, args.input, pca, device)
        else:
            """print(f"Loading node features from cache: {feature_path}")"""
            node_feat = my_dict[feature_path]
        
        adj_path = f"{args.path}/dataset/data/{dataset}/{dataset}_{args.knn_num}_adj_matrix.txt"
        if adj_path not in my_dict:
            """print(f"Adjacency matrix not in cache, generating for: {adj_path}")"""
            adj = load_or_generate_adj_matrix(dataset, args.path, args.knn_num, device, args.directed)
        else:
            """print(f"Loading adjacency matrix from cache: {adj_path}")"""
            adj = my_dict[adj_path]
        
        out = model(node_feat, adj, now)
        
        loss0 = abs(out[:, 0] - target0)
        loss0 = torch.mean(loss0)
        loss1 = abs(out[:, 1] - target1)
        loss1 = torch.mean(loss1)
        loss2 = abs(out[:, 2] - target2)
        loss2 = torch.mean(loss2)
        
        loss_recall += loss0
        loss_qps += loss1
        loss_distcomp += loss2
    
    if dataset_cnt == 0:
        print("Warning: No valid datasets processed in evaluation")
        return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
    
    loss_recall /= dataset_cnt
    loss_qps /= dataset_cnt
    loss_distcomp /= dataset_cnt
    
    print(f"Evaluation results - Recall: {loss_recall:.4f}, QPS: {loss_qps:.4f}, Distcomp: {loss_distcomp:.4f}")
    return loss_recall, loss_qps, loss_distcomp