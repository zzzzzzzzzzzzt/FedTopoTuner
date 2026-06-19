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
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def preprocess_data(df, min_values, max_values, columns_to_normalize, scaler=None):
    """预处理数据：归一化指定列并进行独热编码"""
    df = pd.get_dummies(df, columns=['graph_type'], prefix='graph_type')
    for col in columns_to_normalize:
        df[col] = (df[col] - min_values[col]) / (max_values[col] - min_values[col])
    df = df.fillna(1)
    return df

def combined_loss(pred, tar, alpha=0):
    """结合 L1 和 Smooth L1 损失"""
    l1_loss = nn.L1Loss()
    smooth_l1_loss = nn.SmoothL1Loss()
    return alpha * l1_loss(pred, tar) + (1 - alpha) * smooth_l1_loss(pred, tar)

def split_dataset_v1(df):
    """将数据集分割到12个客户端，每个客户端一个数据集类型"""
    client_datasets = []
    dataset_prefixes = ['sift1M', 'deep1M', 'gist1M', 'crawl1M', 'synthetic1M', 'uqv1M',
                       'Histogram', 'Moments', 'Texture', 'audio', 'mnist6W', 'msong10W']
    for prefix in dataset_prefixes:
        client_data = df[df['dataset'].str.startswith(prefix)]
        client_datasets.append(client_data)
    return client_datasets

def split_dataset_v2(df):
    """将数据集按前缀分组分配到5个客户端"""
    client_datasets = []
    # 客户端1：高维向量数据集
    client_1 = df[df['dataset'].str.startswith(('sift1M', 'deep1M', 'gist1M'))]
    client_datasets.append(client_1)
    
    # 客户端2：文本和合成数据集
    client_2 = df[df['dataset'].str.startswith(('crawl1M', 'synthetic1M', 'uqv1M'))]
    client_datasets.append(client_2)
    
    # 客户端3：图像特征数据集
    client_3 = df[df['dataset'].str.startswith(('Histogram', 'Moments', 'Texture'))]
    client_datasets.append(client_3)
    
    # 客户端4：音频数据集
    client_4 = df[df['dataset'].str.startswith('audio')]
    client_datasets.append(client_4)
    
    # 客户端5：其他数据集
    client_5 = df[df['dataset'].str.startswith(('mnist6W', 'msong10W'))]
    client_datasets.append(client_5)
    
    return client_datasets

def split_dataset(df, num_clients):
    """根据指定的版本选择数据集分割方式"""
    if num_clients == 12:
        return split_dataset_v1(df)
    else:
        return split_dataset_v2(df)

def compute_aggregation_weights(client_models, client_sizes, client_ids, aggregation_strategy='fedavg', val_data=None, args=None, device=None, pca=None):
    """
    计算客户端模型的聚合权重

    Args:
        client_models: 客户端模型列表
        client_sizes: 客户端数据大小列表
        client_ids: 参与训练的客户端ID列表
        aggregation_strategy: 聚合策略 ('fedavg', 'custom', 'performance_based')
        val_data: 验证数据集 (仅在performance_based策略时需要) - 修复：改用验证集而非测试集
        args: 参数对象 (仅在performance_based策略时需要)
        device: 设备 (仅在performance_based策略时需要)
        pca: PCA对象 (仅在performance_based策略时需要)

    Returns:
        weights: 归一化的聚合权重列表
    """
    if aggregation_strategy == 'fedavg':
        # 传统FedAvg：基于数据大小的权重
        total_size = sum(client_sizes)
        weights = [size / total_size for size in client_sizes]
    
    elif aggregation_strategy == 'fedSGD':
        # FedSGD: 所有客户端权重相等
        num_clients = len(client_models)
        weights = [1.0 / num_clients] * num_clients

    elif aggregation_strategy == 'custom':
        # 自定义权重策略：为不同类型的客户端分配不同权重
        # 这里可以根据客户端ID或数据类型来分配权重
        weights = []
        for client_id in client_ids:
            if client_id in [0, 1, 2]:  # 高维向量数据集客户端权重更高
                weights.append(1.5)
            elif client_id in [3]:  # 音频数据集权重中等
                weights.append(1.0)
            else:  # 其他客户端权重较低
                weights.append(0.8)
        
        # 归一化权重
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
    
    elif aggregation_strategy == 'performance_based':
        # 基于性能的权重策略：根据各客户端模型在验证数据集上的性能决定权重
        # 修复：使用验证集而非测试集，避免数据泄露
        if val_data is None or args is None or device is None or pca is None:
            print("警告：performance_based策略需要验证数据集和相关参数，回退到均匀权重")
            num_clients = len(client_models)
            weights = [1.0 / num_clients] * num_clients
        else:
            print("正在计算基于性能的聚合权重...")
            client_performances = []

            # 评估每个客户端模型在验证数据集上的性能
            for i, client_model in enumerate(client_models):
                client_model.eval()
                with torch.no_grad():
                    # 使用evaluate函数评估客户端模型在验证集上的性能
                    from eval import evaluate
                    val_losses = evaluate(client_model, val_data, args, device)

                    # 计算综合性能分数（损失越小，性能越好）
                    # 使用加权平均，权重为[3, 3, 8]，对应recall, qps, distcomp
                    performance_score = (3 * val_losses[0] + 3 * val_losses[1] + 8 * val_losses[2]) / 14
                    client_performances.append(performance_score.item())
                    print(f"客户端 {client_ids[i]} 性能分数（验证集）: {performance_score.item():.4f}")

            # 将性能分数转换为权重（性能越好，权重越高）
            # 使用倒数来转换：性能分数越小（越好），权重越大
            inverse_performances = [1.0 / (perf + 1e-8) for perf in client_performances]  # 加小常数避免除零

            # 归一化权重
            total_weight = sum(inverse_performances)
            weights = [w / total_weight for w in inverse_performances]

            print("基于性能的权重分配:")
            for i, (client_id, weight, perf) in enumerate(zip(client_ids, weights, client_performances)):
                print(f"  客户端 {client_id}: 权重={weight:.4f}, 性能分数（验证集）={perf:.4f}")
    
    else:
        raise ValueError(f"未知的聚合策略: {aggregation_strategy}")
    
    return weights

def aggregate_models_v3(global_model, client_models, client_sizes, client_ids, aggregation_strategy='fedavg', val_data=None, args=None, device=None, pca=None):
    """
    使用可配置的聚合策略聚合客户端模型

    Args:
        global_model: 全局模型
        client_models: 客户端模型列表
        client_sizes: 客户端数据大小列表
        client_ids: 参与训练的客户端ID列表
        aggregation_strategy: 聚合策略
        val_data: 验证数据集 (仅在performance_based策略时需要) - 修复：改用验证集
        args: 参数对象 (仅在performance_based策略时需要)
        device: 设备 (仅在performance_based策略时需要)
        pca: PCA对象 (仅在performance_based策略时需要)

    Returns:
        global_model: 更新后的全局模型
    """
    # 计算聚合权重
    weights = compute_aggregation_weights(client_models, client_sizes, client_ids, aggregation_strategy, val_data, args, device, pca)
    
    # 执行模型聚合
    global_state = global_model.state_dict()
    for key in global_state.keys():
        global_state[key] = torch.zeros_like(global_state[key])
        for i, client_model in enumerate(client_models):
            global_state[key] += client_model.state_dict()[key] * weights[i]
    
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
    parser = argparse.ArgumentParser(description='联邦学习训练流程 v3')
    parser_add_main_args(parser)
    parser.add_argument('--comm_rounds', type=int, default=20, help='通信轮次')
    parser.add_argument('--client_fraction', type=float, default=0.5, help='每轮训练的客户端比例')
    parser.add_argument('--split_version', type=int, default=2, help='数据集划分版本：1表示12个客户端，2表示5个客户端')
    parser.add_argument('--aggregation_strategy', type=str, default='fedavg', 
                       choices=['fedavg', 'custom', 'performance_based', 'fedSGD'], 
                       help='聚合策略：fedavg(传统), custom(自定义权重), performance_based(基于性能), fedSGD(所有客户端权重相等)')
    args = parser.parse_args()
    args.num_clients = 12 if args.split_version == 1 else 5
    print(args)

    fix_seed(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    log_dir = f"{args.path}/model/summary_pic{args.id}"
    writer = SummaryWriter(log_dir)
    logger = TrainingLogger(training_id=args.id)
    logger.log_parameters(args)
    print(f"TensorBoard 日志目录: {log_dir}")
    print(f"训练日志目录: {logger.log_dir}")
    print(f"聚合策略: {args.aggregation_strategy}")

    # 加载和预处理数据
    mb = read_dataset(args.data_dir)
    test = read_dataset(args.test_dir)
    mb = mb[mb['size'] <= 50000]
    mb = mb[~(mb['dataset'].str.startswith("mnist", na=False) & (mb['size'] > 8000))]

    columns_to_normalize = ['size', 'dim', 'K', 'L', 'ITER', 'S', 'R', 'L_refine', 'R_refine', 'C', 'k', 'qps', 'distcomp']
    min_values = {col: mb[col].min() for col in columns_to_normalize}
    max_values = {col: mb[col].max() for col in columns_to_normalize}
    max_values = {col: max_values[col] - min_values[col] for col in columns_to_normalize}

    mb = preprocess_data(mb, min_values, max_values, columns_to_normalize)
    train, val = train_test_split(mb, test_size=0.25, shuffle=True)
    test = preprocess_data(test, min_values, max_values, columns_to_normalize)

    print(f"训练集大小: {train.shape}")
    print(f"验证集大小: {val.shape}")
    print(f"测试集大小: {test.shape}")

    # 分割数据到客户端
    client_datasets = split_dataset(train, args.num_clients)
    print(f"数据已分割到 {args.num_clients} 个客户端")

    # 初始化全局模型
    pca = PCA(n_components=args.input)
    d_para = mb.shape[1] - 4
    global_model, suffix = parse_method(args, args.input, args.output, d_para, device)
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
        client_ids = []

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
            criterion = combined_loss

            # 训练客户端模型
            loss = train_client(client_model, client_data, args, device, optimizer, 
                              criterion, target, weights, pca)
            logger.log_training_info(f"客户端 {client_id} 训练损失: {loss:.4f}")

            client_models.append(client_model)
            client_sizes.append(client_size)
            client_ids.append(client_id)

        # 聚合客户端模型（使用新的聚合策略）
        # 修复：使用验证集而非测试集计算权重，避免数据泄露
        if client_models:
            global_model = aggregate_models_v3(global_model, client_models, client_sizes,
                                             client_ids, args.aggregation_strategy,
                                             val_data=val, args=args, device=device, pca=pca)
            logger.log_training_info(f"全局模型已更新 (策略: {args.aggregation_strategy})")

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
            logger.log_training_info('全局模型更新 - 发现更好的性能')
            if args.save_model:
                save_path = f"{args.model_dir}/summary_pic{args.id}/{args.method}_{args.id}.pkl"
                torch.save(global_model.state_dict(), save_path)
                logger.log_training_info(f'模型已保存到: {save_path}')

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