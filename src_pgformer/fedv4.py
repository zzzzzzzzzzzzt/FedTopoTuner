import os
import time
import argparse
import random
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import faiss
from itertools import combinations
from math import comb
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
        df[col] = (df[col] - min_values[col]) / max_values[col]
    df = df.fillna(1)
    return df

def _rankdata(a):
    a = np.asarray(a)
    n = len(a)
    if n == 0:
        return np.array([], dtype=float)
    sorter = np.argsort(a)
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(n)
    a_sorted = a[sorter]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i + 1
        while j < n and a_sorted[j] == a_sorted[i]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[i:j] = rank
        i = j
    return ranks[inv]

def pearson_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return 0.0
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.linalg.norm(xm) * np.linalg.norm(ym)
    if denom == 0.0:
        return 0.0
    return float((xm @ ym) / denom)

def spearman_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return 0.0
    rx = _rankdata(x)
    ry = _rankdata(y)
    return pearson_corr(rx, ry)

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

def split_dataset_v3(df):
    """将数据集分割到N个客户端，每个客户端一个数据集实例（最细粒度划分）"""
    client_datasets = []
    # 获取所有唯一的数据集实例
    unique_datasets = sorted(df['dataset'].unique())

    # 为每个数据集实例创建一个客户端
    for dataset_name in unique_datasets:
        client_data = df[df['dataset'] == dataset_name]
        client_datasets.append(client_data)

    return client_datasets

def split_dataset_by_modality(df):
    client_datasets = []
    image_prefixes = ('sift1M', 'deep1M', 'gist1M', 'Histogram', 'Moments', 'Texture', 'mnist6W')
    text_prefixes = ('crawl1M', 'uqv1M', 'synthetic1M')
    audio_prefixes = ('audio', 'msong10W')

    image_data = df[df['dataset'].str.startswith(image_prefixes)]
    text_data = df[df['dataset'].str.startswith(text_prefixes)]
    audio_data = df[df['dataset'].str.startswith(audio_prefixes)]

    client_datasets.append(image_data)
    client_datasets.append(text_data)
    client_datasets.append(audio_data)

    return client_datasets

def split_dataset(df, num_clients):
    """根据指定的版本选择数据集分割方式"""
    if num_clients == 12:
        return split_dataset_v1(df)
    elif num_clients == 5:
        return split_dataset_v2(df)
    elif num_clients == 3:
        return split_dataset_by_modality(df)
    else:
        # 默认使用v3版本：每个数据集实例一个客户端
        return split_dataset_v3(df)

def _leverage_score_sampling(num_clients, budget):
    """按杠杆分数 tau_S = 1 / C(K, |S|) 做互补配对采样（算法 3 第 4-10 行）。

    每个采到的联盟 S 与其补集 S̄ 成对出现（|S|=s，|S̄|=K-s），不放回采样；
    返回 (coalitions, realized)，其中 coalitions 为 ``(subset_tuple, size)`` 列表，
    realized[s] 记录实际采到的大小为 s 的联盟数量（用于逆概率加权）。
    """
    interior_budget = max(1, budget - 2)  # 预留空集与全集两个联盟

    def total_entries(c):
        return sum(min(comb(num_clients, s), 2.0 * c) for s in range(1, num_clients))

    # 二分求规模参数 c，使得 interior_budget = sum_s min(C(K, s), 2c)。
    lo, hi = 0.0, 1.0
    while total_entries(hi) < interior_budget:
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if total_entries(mid) < interior_budget:
            lo = mid
        else:
            hi = mid
    c = hi

    coalitions = []
    realized = [0] * (num_clients + 1)
    # 只枚举 s <= K/2：规模 s 与 K-s 通过补集同时覆盖，避免重复采样。
    for s in range(1, num_clients // 2 + 1):
        full = comb(num_clients, s)
        s_comp = num_clients - s
        if full <= 2.0 * c + 1e-9:
            subsets = list(combinations(range(num_clients), s))
        else:
            k_s = max(1, min(int(round(2.0 * c)), full))
            subsets = _sample_subsets(num_clients, s, k_s)
        for sub in subsets:
            coalitions.append((sub, s))
            realized[s] += 1
            if s_comp != s:  # 自补集规模（K 为偶数且 s=K/2）不再重复添加补集
                comp = tuple(i for i in range(num_clients) if i not in sub)
                coalitions.append((comp, s_comp))
                realized[s_comp] += 1
    return coalitions, realized


def _sample_subsets(num_clients, s, k):
    """不放回地采样 k 个互不相同的大小为 s 的子集。"""
    total = comb(num_clients, s)
    if k >= total:
        return list(combinations(range(num_clients), s))
    seen = set()
    out = []
    while len(out) < k:
        sub = tuple(sorted(random.sample(range(num_clients), s)))
        if sub not in seen:
            seen.add(sub)
            out.append(sub)
    return out


def _solve_shapley_wls(coalitions, values, realized, num_clients, v_empty, v_grand):
    """从采样联盟的效用中，通过带效率投影的加权最小二乘恢复 Shapley 向量（算法 3 第 18-23 行）。"""
    delta = (v_grand - v_empty) / num_clients

    rows = []
    for (sub, s), val in zip(coalitions, values):
        if s <= 0 or s >= num_clients:
            continue
        weight = 1.0 / (s * (num_clients - s) * max(realized[s], 1))
        rows.append((sub, s, val, weight))

    if not rows:
        return [delta] * num_clients

    m = len(rows)
    A = np.zeros((m, num_clients))
    b = np.zeros(m)
    W = np.zeros(m)
    for r, (sub, s, val, weight) in enumerate(rows):
        for i in range(num_clients):
            A[r, i] = (1.0 if i in sub else 0.0) - s / num_clients
        b[r] = (val - v_empty) - delta * s
        W[r] = weight

    Aw = A.T * W                                   # A^T W  [K, m]
    gram = Aw @ A + 1e-6 * np.eye(num_clients)     # A^T W A + 岭正则
    rhs = Aw @ b
    phi_perp = np.linalg.solve(gram, rhs)
    phi = phi_perp + delta
    return phi.tolist()


def compute_shapley_values(client_gradients, client_sizes, global_test_data, global_model, args, device, pca, budget=None):
    """基于杠杆分数采样的 Shapley 值估计（FedCWA 算法 3）。

    用 Musco & Witter (2024) 的杠杆分数估计器替换精确的 2^K 联盟枚举：
    以正比于 tau_S = 1 / C(K, |S|) 的概率采样 O(K log K) 个联盟，用尺寸加权的
    客户端更新（式 10）构建虚拟模型并在评估集上计算效用，最后通过带效率投影的
    加权最小二乘恢复 Shapley 向量。
    """
    num_clients = len(client_gradients)
    if num_clients == 0:
        return []
    if num_clients == 1:
        return [1.0]

    total_size = sum(client_sizes)
    if total_size <= 0:
        total_size = 1.0

    # 尺寸加权的客户端更新：(|D_j| / sum |D|) * Delta_j（算法 3 第 12、15 行）。
    # 维度适配器 kept local、不参与联邦，其梯度不进入效用/Shapley 估计。
    weighted_grads = []
    for j in range(num_clients):
        factor = client_sizes[j] / total_size
        weighted_grads.append({
            k: g * factor for k, g in client_gradients[j].items()
            if not (k.startswith('dim_adapter') or k.startswith('adapter_norm'))
        })

    base_state = deepcopy(global_model.state_dict())
    eval_model = deepcopy(global_model).to(device)
    eval_model.eval()

    def utility(coalition):
        state = {k: base_state[k].clone() for k in base_state}
        for j in coalition:
            for k in weighted_grads[j]:
                state[k] = state[k] + weighted_grads[j][k]
        eval_model.load_state_dict(state)
        losses = evaluate(eval_model, global_test_data, args, device)
        score = (3.0 * losses[0] + 3.0 * losses[1] + 8.0 * losses[2]) / 14.0
        return -1.0 * (score.item() if hasattr(score, "item") else float(score))

    # 采样预算 O(K log K)，上限为精确枚举的 2^K。
    if budget is None:
        budget = min(2 ** num_clients, num_clients * num_clients.bit_length() + 2)

    coalitions, realized = _leverage_score_sampling(num_clients, budget)

    v_empty = utility(())                       # v(∅)
    v_grand = utility(tuple(range(num_clients)))  # v([K])
    values = [utility(sub) for sub, _ in coalitions]

    shapley_values = _solve_shapley_wls(coalitions, values, realized, num_clients, v_empty, v_grand)

    print("Shapley values (Leverage-Score estimation):")
    for idx, sv in enumerate(shapley_values):
        print(f"  Client {idx}: Shapley={sv:.6f}")
    return shapley_values


def compute_aggregation_weights(client_models, client_sizes, client_ids, aggregation_strategy='fedavg', val_data=None, args=None, device=None, pca=None, global_test_data=None, client_gradients=None, global_model=None):
    """
    计算客户端模型的聚合权重

    Args:
        client_models: 客户端模型列表
        client_sizes: 客户端数据大小列表
        client_ids: 参与训练的客户端ID列表
        aggregation_strategy: 聚合策略 ('fedavg', 'performance_based')
        val_data: 全局验证数据集
        args: 参数对象
        device: 设备
        pca: PCA对象

    Returns:
        weights: 归一化的聚合权重列表
    """
    if aggregation_strategy == 'fedavg':
        total_size = sum(client_sizes)
        weights = [size / total_size for size in client_sizes]

    elif aggregation_strategy == 'performance_based':
        if global_test_data is None or client_gradients is None or args is None or device is None or pca is None or global_model is None:
            print("警告：performance_based策略缺少必要信息，回退到均匀权重")
            num_clients = len(client_models)
            weights = [1.0 / num_clients] * num_clients
        else:
            print("正在计算基于性能的聚合权重（Leverage-Score Shapley 估计）...")
            shapley_values = compute_shapley_values(client_gradients, client_sizes, global_test_data, global_model, args, device, pca)
            # 对 Shapley 值做 softmax 归一化得到聚合权重（算法 3 第 24-26 行）。
            phi = np.asarray(shapley_values, dtype=float)
            phi = phi - phi.max()  # 数值稳定，不改变 softmax 结果
            exp_phi = np.exp(phi)
            weights = (exp_phi / exp_phi.sum()).tolist()
            print("Aggregation weights (performance_based):")
            for cid, sv, w in zip(client_ids, shapley_values, weights):
                print(f"  Client {cid}: Shapley={sv:.6f}, Weight={w:.6f}")

    else:
        raise ValueError(f"未知的聚合策略: {aggregation_strategy}")

    return weights

def aggregate_models_v4(global_model, client_models, client_sizes, client_ids, aggregation_strategy='fedavg', val_data=None, args=None, device=None, pca=None, global_test_data=None, client_gradients=None):
    """
    使用可配置的聚合策略聚合客户端模型（V4版本）

    Args:
        global_model: 全局模型
        client_models: 客户端模型列表
        client_sizes: 客户端数据大小列表
        client_ids: 参与训练的客户端ID列表
        aggregation_strategy: 聚合策略
        val_data: 全局验证数据集
        args: 参数对象
        device: 设备
        pca: PCA对象

    Returns:
        global_model: 更新后的全局模型
    """
    weights = compute_aggregation_weights(client_models, client_sizes, client_ids, aggregation_strategy, val_data, args, device, pca, global_test_data, client_gradients, global_model)

    # 执行模型聚合
    # 维度适配器 kept local、不参与联邦（论文 3.2），聚合时跳过相关参数。
    global_state = global_model.state_dict()
    for key in global_state.keys():
        if key.startswith('dim_adapter') or key.startswith('adapter_norm'):
            continue
        global_state[key] = torch.zeros_like(global_state[key])
        for i, client_model in enumerate(client_models):
            global_state[key] += client_model.state_dict()[key] * weights[i]

    global_model.load_state_dict(global_state)
    return global_model

def train_client(model, client_data, args, device, optimizer, criterion, target, weights, pca):
    """在客户端本地数据上训练模型"""
    model.train()
    total_loss, cnt = 0, 0
    sum_loss_recall = 0.0
    sum_loss_qps = 0.0
    sum_loss_distcomp = 0.0
    sum_loss_combined = 0.0
    total_dataset_cnt = 0
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
            loss_recall = criterion(out[:, 0], targets[0])
            loss_qps = criterion(out[:, 1], targets[1])
            loss_distcomp = criterion(out[:, 2], targets[2])
            loss = weights[0] * loss_recall + weights[1] * loss_qps + weights[2] * loss_distcomp
            loss.backward()
            train_loss += loss.detach().item()
            sum_loss_recall += loss_recall.detach().item()
            sum_loss_qps += loss_qps.detach().item()
            sum_loss_distcomp += loss_distcomp.detach().item()
            sum_loss_combined += (3 * loss_recall + 3 * loss_qps + 8 * loss_distcomp).detach().item()
            total_dataset_cnt += 1

        optimizer.step()
        client_pbar.update(1)
        train_loss /= dataset_cnt
        total_loss += train_loss
        cnt += 1
        client_pbar.set_postfix({'批次时间': f'{time.time() - t_start:.2f}s', '损失': f'{train_loss:.4f}'})

    client_pbar.close()
    avg_client_loss = total_loss / cnt if cnt > 0 else 0
    if total_dataset_cnt > 0:
        avg_recall = sum_loss_recall / total_dataset_cnt
        avg_qps = sum_loss_qps / total_dataset_cnt
        avg_distcomp = sum_loss_distcomp / total_dataset_cnt
        avg_combined = sum_loss_combined / total_dataset_cnt
        print(f"客户端平均损失: recall={avg_recall:.4f}, qps={avg_qps:.4f}, distcomp={avg_distcomp:.4f}, combined={avg_combined:.4f}")
    else:
        avg_recall = 0.0
        avg_qps = 0.0
        avg_distcomp = 0.0
        avg_combined = 0.0
    return avg_client_loss, avg_recall, avg_qps, avg_distcomp, avg_combined

def evaluate_test_by_dataset(model, test, args, device, pca, save_path=None):
    bases = test.dataset.unique()
    rows = []
    for dataset in bases:
        now_mb = test[test.dataset == dataset]
        if len(now_mb) < 1:
            continue
        target0 = now_mb['recall']
        target1 = now_mb['qps']
        target2 = now_mb['distcomp']
        now = now_mb.drop('dataset', axis=1)
        now = now.drop(['recall', 'qps', 'distcomp'], axis=1)
        now = torch.from_numpy(now.values).float().to(device)
        target0 = torch.from_numpy(target0.values).float().to(device)
        target1 = torch.from_numpy(target1.values).float().to(device)
        target2 = torch.from_numpy(target2.values).float().to(device)
        feature_path = f"{args.path}/dataset/data/{dataset}/{dataset}_base_{args.input}.fvecs"
        if feature_path not in my_dict:
            node_feat = load_or_generate_node_features(dataset, args.path, args.input, pca, device)
        else:
            node_feat = my_dict[feature_path]
        adj_path = f"{args.path}/dataset/data/{dataset}/{dataset}_{args.knn_num}_adj_matrix.txt"
        if adj_path not in my_dict:
            adj = load_or_generate_adj_matrix(dataset, args.path, args.knn_num, device, args.directed)
        else:
            adj = my_dict[adj_path]
        with torch.no_grad():
            out = model(node_feat, adj, now)
            loss0 = torch.mean(torch.abs(out[:, 0] - target0))
            loss1 = torch.mean(torch.abs(out[:, 1] - target1))
            loss2 = torch.mean(torch.abs(out[:, 2] - target2))
        combined = (3 * loss0 + 3 * loss1 + 8 * loss2) / 14
        pred0 = out[:, 0].detach().cpu().numpy()
        pred1 = out[:, 1].detach().cpu().numpy()
        pred2 = out[:, 2].detach().cpu().numpy()
        t0 = target0.detach().cpu().numpy()
        t1 = target1.detach().cpu().numpy()
        t2 = target2.detach().cpu().numpy()
        pearson0 = pearson_corr(pred0, t0)
        pearson1 = pearson_corr(pred1, t1)
        pearson2 = pearson_corr(pred2, t2)
        spearman0 = spearman_corr(pred0, t0)
        spearman1 = spearman_corr(pred1, t1)
        spearman2 = spearman_corr(pred2, t2)
        rows.append({
            'dataset': dataset,
            'loss_recall': float(loss0.cpu().item()),
            'loss_qps': float(loss1.cpu().item()),
            'loss_distcomp': float(loss2.cpu().item()),
            'loss_combined': float(combined.cpu().item()),
            'pearson_recall': pearson0,
            'pearson_qps': pearson1,
            'pearson_distcomp': pearson2,
            'spearman_recall': spearman0,
            'spearman_qps': spearman1,
            'spearman_distcomp': spearman2
        })
    if save_path is not None and rows:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    'dataset',
                    'loss_recall',
                    'loss_qps',
                    'loss_distcomp',
                    'loss_combined',
                    'pearson_recall',
                    'pearson_qps',
                    'pearson_distcomp',
                    'spearman_recall',
                    'spearman_qps',
                    'spearman_distcomp'
                ]
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description='联邦学习训练流程 v4 - 本地验证集设计')
    parser_add_main_args(parser)
    parser.add_argument('--comm_rounds', type=int, default=50, help='通信轮次')
    parser.add_argument('--client_fraction', type=float, default=1.0, help='每轮训练的客户端比例')
    parser.add_argument('--split_version', type=int, default=4,
                       help='数据集划分版本：1=12个客户端(按数据集类型), 2=5个客户端(分组), 3=51个客户端(每个数据集实例), 4=3个客户端(按模态：图像/文本/音频)')
    parser.add_argument('--aggregation_strategy', type=str, default='fedavg',
                       choices=['fedavg', 'performance_based'],
                       help='聚合策略：fedavg(传统), performance_based(基于性能)')
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test'])
    args = parser.parse_args()

    # 根据split_version设置客户端数量
    if args.split_version == 1:
        args.num_clients = 12
    elif args.split_version == 2:
        args.num_clients = 5
    elif args.split_version == 3:
        # v3版本：动态确定客户端数量（等于数据集实例数）
        # 先临时设置为51，后面根据实际数据调整
        args.num_clients = 51
    elif args.split_version == 4:
        args.num_clients = 3
    else:
        raise ValueError(f"不支持的split_version: {args.split_version}")

    print(args)

    fix_seed(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    if args.phase == 'train':
        log_dir = f"{args.path}/model/summary_pic{args.id}"
        writer = SummaryWriter(log_dir)
        logger = TrainingLogger(training_id=args.id)
        logger.log_parameters(args)
        print(f"TensorBoard 日志目录: {log_dir}")
        print(f"训练日志目录: {logger.log_dir}")
        print(f"聚合策略: {args.aggregation_strategy}")

        mb = read_dataset(args.data_dir)
        mb = mb[mb['size'] <= 50000]
        mb = mb[~(mb['dataset'].str.startswith("mnist", na=False) & (mb['size'] > 8000))]

        columns_to_normalize = ['size', 'dim', 'K', 'L', 'ITER', 'S', 'R', 'L_refine', 'R_refine', 'C', 'k', 'qps', 'distcomp']
        min_values = {col: mb[col].min() for col in columns_to_normalize}
        max_values = {col: mb[col].max() for col in columns_to_normalize}
        max_values = {col: max_values[col] - min_values[col] for col in columns_to_normalize}

        mb = preprocess_data(mb, min_values, max_values, columns_to_normalize)

        print(f"总数据大小: {mb.shape}")

        mb_for_clients, global_test_for_performance = train_test_split(mb, test_size=0.15, shuffle=True)
        print(f"用于客户端分配的数据大小: {mb_for_clients.shape}")
        print(f"全局测试集大小(15%训练数据): {global_test_for_performance.shape}")

        client_datasets = split_dataset(mb_for_clients, args.num_clients)

        if args.split_version == 3:
            args.num_clients = len(client_datasets)

        print(f"\n第1步：数据已分配到 {args.num_clients} 个客户端")
        if args.split_version == 1:
            print(f"  划分策略：v1 - 每个数据集类型一个客户端")
        elif args.split_version == 2:
            print(f"  划分策略：v2 - 相关数据集分组")
        elif args.split_version == 3:
            print(f"  划分策略：v3 - 每个数据集实例一个客户端（最细粒度）")
        elif args.split_version == 4:
            print(f"  划分策略：v4 - 三类模态客户端（图像/文本/音频）")

        client_train_datasets = []
        client_val_datasets = []

        print("\n第2步：各客户端全部数据作为训练集（不再划分验证集）")
        for i, client_data in enumerate(client_datasets):
            if len(client_data) == 0:
                print(f"  客户端 {i}: 数据为空，跳过")
                client_train_datasets.append(pd.DataFrame())
                client_val_datasets.append(pd.DataFrame())
                continue
            client_train_datasets.append(client_data)
            client_val_datasets.append(pd.DataFrame())
            print(f"  客户端 {i}: 训练数据={len(client_data)}, 验证集=0")

        global_val_for_performance = pd.DataFrame(columns=mb_for_clients.columns)
        print(f"\n第3步：当前配置不使用验证集，global_val_for_performance 为空，shape={global_val_for_performance.shape}")

        total_train = sum(len(d) for d in client_train_datasets)
        print(f"\n数据统计：")
        print(f"  训练数据总量: {total_train}")
        print(f"  验证数据总量: 0")
        print("="*80 + "\n")

        pca = PCA(n_components=args.input)
        d_para = mb.shape[1] - 4
        global_model, suffix = parse_method(args, args.input, args.output, d_para, device)
        print(f"全局模型: {global_model}")
        print(f"后缀: {suffix}")

        target = ['recall', 'qps', 'distcomp']
        weights = [3, 3, 8]

        global_val_for_performance = global_val_for_performance.reset_index(drop=True)
        global_test_for_performance = global_test_for_performance.reset_index(drop=True)

        for round in range(args.comm_rounds):
            print(f"\n通信轮次 {round + 1}/{args.comm_rounds}")
            selected_clients = random.sample(range(args.num_clients),
                                            max(1, int(args.client_fraction * args.num_clients)))
            client_models = []
            client_sizes = []
            client_ids = []
            client_gradients = []
            global_state_before = deepcopy(global_model.state_dict())

            for client_id in selected_clients:
                client_train_data = client_train_datasets[client_id]
                client_size = len(client_train_data)
                if client_size == 0:
                    continue
                client_model = deepcopy(global_model).to(device)
                optimizer = torch.optim.Adam(client_model.parameters(),
                                           weight_decay=args.weight_decay, lr=args.lr)
                criterion = combined_loss
                client_loss, avg_recall, avg_qps, avg_distcomp, avg_combined = train_client(
                    client_model, client_train_data, args, device, optimizer,
                    criterion, target, weights, pca
                )
                logger.log_training_info(
                    f"客户端 {client_id} 训练损失: {client_loss:.4f} | 训练recall损失: {avg_recall:.4f} | 训练qps损失: {avg_qps:.4f} | 训练distcomp损失: {avg_distcomp:.4f} | 训练综合损失: {avg_combined:.4f}"
                )
                client_state = client_model.state_dict()
                gradient_state = {}
                for key in global_state_before.keys():
                    gradient_state[key] = client_state[key] - global_state_before[key]
                client_gradients.append(gradient_state)
                client_models.append(client_model)
                client_sizes.append(client_size)
                client_ids.append(client_id)

            if client_models:
                global_model = aggregate_models_v4(
                    global_model, client_models, client_sizes,
                    client_ids, args.aggregation_strategy,
                    val_data=None,
                    args=args, device=device, pca=pca,
                    global_test_data=global_test_for_performance if args.aggregation_strategy == 'performance_based' else None,
                    client_gradients=client_gradients if args.aggregation_strategy == 'performance_based' else None
                )
                logger.log_training_info(f"全局模型已更新 (策略: {args.aggregation_strategy})")
                global_model.eval()
                test_losses = evaluate(global_model, global_test_for_performance, args, device)
                test_recall = test_losses[0].item() if hasattr(test_losses[0], "item") else float(test_losses[0])
                test_qps = test_losses[1].item() if hasattr(test_losses[1], "item") else float(test_losses[1])
                test_distcomp = test_losses[2].item() if hasattr(test_losses[2], "item") else float(test_losses[2])
                metrics = {
                    'test_recall': test_recall,
                    'test_qps': test_qps,
                    'test_distcomp': test_distcomp,
                }
                logger.log_metrics(round + 1, metrics)

        if args.save_model:
            save_path = f"{args.model_dir}/summary_pic{args.id}/{args.method}_{args.id}.pkl"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(global_model.state_dict(), save_path)
            logger.log_training_info(f'最终模型已保存到: {save_path}')

        writer.close()

    elif args.phase == 'test':
        mb = read_dataset(args.data_dir)
        test = read_dataset(args.test_dir)
        mb = mb[mb['size'] <= 50000]
        mb = mb[~(mb['dataset'].str.startswith("mnist", na=False) & (mb['size'] > 8000))]

        columns_to_normalize = ['size', 'dim', 'K', 'L', 'ITER', 'S', 'R', 'L_refine', 'R_refine', 'C', 'k', 'qps', 'distcomp']
        min_values = {col: mb[col].min() for col in columns_to_normalize}
        max_values = {col: mb[col].max() for col in columns_to_normalize}
        max_values = {col: max_values[col] - min_values[col] for col in columns_to_normalize}

        mb = preprocess_data(mb, min_values, max_values, columns_to_normalize)
        test = preprocess_data(test, min_values, max_values, columns_to_normalize)

        print(f"测试集大小: {test.shape}")

        pca = PCA(n_components=args.input)
        d_para = mb.shape[1] - 4
        global_model, suffix = parse_method(args, args.input, args.output, d_para, device)

        model_path = f"{args.model_dir}/summary_pic{args.id}/{args.method}_{args.id}.pkl"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"未找到模型文件: {model_path}")
        state = torch.load(model_path, map_location=device)
        global_model.load_state_dict(state)
        global_model.to(device)
        global_model.eval()

        result_path = os.path.join(args.path, "experiments", "fedv4_aggregation_comparison",
                                   f"test_results_id{args.id}_{args.aggregation_strategy}.csv")
        rows = evaluate_test_by_dataset(global_model, test, args, device, pca, save_path=result_path)
        print(f"测试已完成，结果保存到: {result_path}")

if __name__ == "__main__":
    main()
