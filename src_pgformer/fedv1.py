import os
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import faiss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from struct import pack
import warnings
from copy import deepcopy

from dataset import *
from parse import parse_method, parser_add_main_args
from eval import evaluate
from model_test import evaluate_test

from utils import my_dict, load_or_generate_node_features, load_or_generate_adj_matrix
from logger import TrainingLogger

warnings.filterwarnings('ignore')

def fix_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def preprocess_data(df, min_values, max_values, columns_to_normalize, scaler=None):
    """Preprocess DataFrame: normalize specified columns and apply one-hot encoding."""
    # One-hot encoding for graph_type
    df = pd.get_dummies(df, columns=['graph_type'], prefix='graph_type')

    # Normalize specified columns
    for col in columns_to_normalize:
        df[col] = (df[col] - min_values[col]) / (max_values[col] - min_values[col])

    df = df.fillna(1)
    return df


def combined_loss(pred, tar, alpha=0):
    """Combined L1 and Smooth L1 loss."""
    l1_loss = nn.L1Loss()
    smooth_l1_loss = nn.SmoothL1Loss()
    return alpha * l1_loss(pred, tar) + (1 - alpha) * smooth_l1_loss(pred, tar)

def split_dataset_equally(df, num_clients):
    """将数据集均分到指定数量的客户端"""
    # 随机打乱数据
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 计算每个客户端的数据量
    total_samples = len(df_shuffled)
    samples_per_client = total_samples // num_clients
    
    client_datasets = []
    for i in range(num_clients):
        start_idx = i * samples_per_client
        if i == num_clients - 1:  # 最后一个客户端获得剩余的所有数据
            end_idx = total_samples
        else:
            end_idx = (i + 1) * samples_per_client
        
        client_data = df_shuffled.iloc[start_idx:end_idx]
        client_datasets.append(client_data)
    
    return client_datasets

def aggregate_models(global_model, client_models, client_sizes):
    """使用 FedAvg 算法聚合客户端模型"""
    total_size = sum(client_sizes)
    global_state = global_model.state_dict()
    for key in global_state.keys():
        global_state[key] = torch.zeros_like(global_state[key])
        for i, client_model in enumerate(client_models):
            global_state[key] += client_model.state_dict()[key] * (client_sizes[i] / total_size)
    global_model.load_state_dict(global_state)
    return global_model

def train_client(model, client_data, args, device, optimizer, criterion, target, weights, pca):
    """在客户端本地数据上训练模型"""
    model.train()
    total_loss, cnt = 0, 0
    client_pbar = tqdm(total=len(rand_train(client_data, args.batch_size)), desc='客户端训练', unit='batch')
    
    for iter in rand_train(client_data, args.batch_size):
        t_start = time.time()
        bases = iter.dataset.unique()
        train_loss = 0
        dataset_cnt = 0

        optimizer.zero_grad()
        for dataset in bases:
            if int(dataset.split('_')[-1]) > 500000:
                continue

            now_mb = iter[iter.dataset == dataset]
            if len(now_mb) < 1:
                continue

            dataset_cnt += 1
            targets = [torch.from_numpy(now_mb[t].values).float().to(device) for t in target]
            inputs = now_mb.drop(['dataset'] + target, axis=1)
            inputs = torch.from_numpy(inputs.values).float().to(device)

            node_feat = load_or_generate_node_features(dataset, args.path, args.input, pca, device)
            adj = load_or_generate_adj_matrix(dataset, args.path, args.knn_num, device, args.directed)

            out = model(node_feat, adj, inputs)
            loss = sum(w * criterion(out[:, i], targets[i]) for i, w in enumerate(weights))
            loss.backward()
            train_loss += loss.detach().item()

        optimizer.step()
        client_pbar.update(1)
        train_loss /= dataset_cnt
        total_loss += train_loss
        cnt += 1
        client_pbar.set_postfix({'批次时间': f'{time.time() - t_start:.2f}s', '损失': f'{train_loss:.4f}'})
    
    client_pbar.close()
    return total_loss / cnt if cnt > 0 else 0

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='联邦学习训练流程 - 均分数据版本')
    parser_add_main_args(parser)
    parser.add_argument('--comm_rounds', type=int, default=50, help='通信轮次')
    parser.add_argument('--client_fraction', type=float, default=1.0, help='每轮训练的客户端比例')
    parser.add_argument('--num_clients', type=int, default=5, choices=[5, 12], help='客户端数量：5或12')
    args = parser.parse_args()
    print(args)

    # Set random seed
    fix_seed(args.seed)

    # Device configuration
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # TensorBoard setup
    log_dir = f"{args.path}/model/summary_pic{args.id}"
    writer = SummaryWriter(log_dir)
    logger = TrainingLogger(training_id=args.id)
    logger.log_parameters(args)
    print(f"TensorBoard log directory: {log_dir}")
    print(f"训练日志目录: {logger.log_dir}")

    # Load and preprocess data
    mb = read_dataset(args.data_dir)
    test = read_dataset(args.test_dir)

    # Filter data
    mb = mb[mb['size'] <= 50000]
    mb = mb[~(mb['dataset'].str.startswith("mnist", na=False) & (mb['size'] > 8000))]

    # Compute min/max for normalization
    columns_to_normalize = ['size', 'dim', 'K', 'L', 'ITER', 'S', 'R', 'L_refine', 'R_refine', 'C', 'k', 'qps', 'distcomp']
    min_values = {col: mb[col].min() for col in columns_to_normalize}
    max_values = {col: mb[col].max() for col in columns_to_normalize}
    max_values = {col: max_values[col] - min_values[col] for col in columns_to_normalize}

    # Preprocess train and test data
    mb = preprocess_data(mb, min_values, max_values, columns_to_normalize)
    train, val = train_test_split(mb, test_size=0.25, shuffle=True)
    test = preprocess_data(test, min_values, max_values, columns_to_normalize)

    print(f"训练集大小: {train.shape}")
    print(f"验证集大小: {val.shape}")
    print(f"测试集大小: {test.shape}")

    # 分割数据到客户端（均分）
    client_datasets = split_dataset_equally(train, args.num_clients)
    print(f"数据已均分到 {args.num_clients} 个客户端")
    for i, client_data in enumerate(client_datasets):
        print(f"客户端 {i}: {len(client_data)} 个样本")

    # 初始化全局模型
    pca = PCA(n_components=args.input)
    d_para = mb.shape[1] - 4
    global_model, suffix = parse_method(args, args.input, args.output, d_para, device)
    criterion = combined_loss

    print(f"全局模型: {global_model}")
    print(f"后缀: {suffix}")

    # 联邦学习训练循环
    target = ['recall', 'qps', 'distcomp']
    weights = [3, 3, 8]  # al0, al1, al2
    best_val_metrics = {'recall': float('inf'), 'qps': float('inf'), 'distcomp': float('inf')}
    best_test_metrics = {'recall': float('inf'), 'qps': float('inf'), 'distcomp': float('inf')}

    for round in range(args.comm_rounds):
        print(f"\n通信轮次 {round + 1}/{args.comm_rounds}")
        selected_clients = random.sample(range(args.num_clients), 
                                        max(1, int(args.client_fraction * args.num_clients)))
        client_models = []
        client_sizes = []

        # 客户端本地训练
        for client_id in selected_clients:
            client_data = client_datasets[client_id]
            client_size = len(client_data)
            if client_size == 0:
                continue

            # 为每个客户端创建模型副本
            client_model = deepcopy(global_model).to(device)
            optimizer = torch.optim.Adam(client_model.parameters(), 
                                       weight_decay=args.weight_decay, lr=args.lr)

            # 训练客户端模型
            loss = train_client(client_model, client_data, args, device, optimizer, 
                              criterion, target, weights, pca)
            logger.log_training_info(f"客户端 {client_id} 训练损失: {loss:.4f}")

            client_models.append(client_model)
            client_sizes.append(client_size)

        # 聚合客户端模型
        if client_models:
            global_model = aggregate_models(global_model, client_models, client_sizes)
            logger.log_training_info("全局模型已更新")

        # 评估全局模型
        global_model.eval()
        val_losses = evaluate(global_model, val, args, device)
        test_losses = evaluate_test(global_model, test, args, device, pca)
        writer.add_scalars("target", {
            'recall_val': val_losses[0], 'recall_test': test_losses[0],
            'qps_val': val_losses[1], 'qps_test': test_losses[1],
            'distcomp_val': val_losses[2], 'distcomp_test': test_losses[2]
        }, round)

        # 保存最佳模型
        # 修复：使用验证集而非测试集来决定是否保存模型，避免数据泄露
        if val_losses[0] < best_val_metrics['recall']:
            best_val_metrics = dict(zip(['recall', 'qps', 'distcomp'], val_losses))
            best_test_metrics = dict(zip(['recall', 'qps', 'distcomp'], test_losses))
            print('全局模型更新')
            logger.log_training_info('全局模型更新 - 发现更好的性能')
            if args.save_model:
                save_path = f"{args.model_dir}/summary_pic{args.id}/{args.method}_{args.id}.pkl"
                torch.save(global_model.state_dict(), save_path)
                logger.log_training_info(f'模型已保存到: {save_path}')

        # 记录当前轮次的指标
        metrics = {
            'val_recall': best_val_metrics['recall'],
            'test_recall': best_test_metrics['recall'],
            'val_qps': best_val_metrics['qps'],
            'test_qps': best_test_metrics['qps'],
            'val_distcomp': best_val_metrics['distcomp'],
            'test_distcomp': best_test_metrics['distcomp']
        }
        logger.log_metrics(round + 1, metrics)
        
        print(f"验证集 recall 损失: {metrics['val_recall']:.4f}")
        print(f"测试集 recall 损失: {metrics['test_recall']:.4f}")
        print(f"验证集 qps 损失: {metrics['val_qps']:.4f}")
        print(f"测试集 qps 损失: {metrics['test_qps']:.4f}")
        print(f"验证集 distcomp 损失: {metrics['val_distcomp']:.4f}")
        print(f"测试集 distcomp 损失: {metrics['test_distcomp']:.4f}")

    writer.close()

if __name__ == "__main__":
    main()