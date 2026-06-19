"""诊断客户端数据分配问题"""
import pandas as pd
import argparse
import random
import numpy as np
from sklearn.model_selection import train_test_split
from dataset import read_dataset
from parse import parser_add_main_args

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def preprocess_data(df, min_values, max_values, columns_to_normalize):
    df = pd.get_dummies(df, columns=['graph_type'], prefix='graph_type')
    for col in columns_to_normalize:
        df[col] = (df[col] - min_values[col]) / (max_values[col] - min_values[col])
    df = df.fillna(1)
    return df

def split_dataset_v2(df):
    client_datasets = []
    client_1 = df[df['dataset'].str.startswith(('sift1M', 'deep1M', 'gist1M'))]
    client_datasets.append(client_1)

    client_2 = df[df['dataset'].str.startswith(('crawl1M', 'synthetic1M', 'uqv1M'))]
    client_datasets.append(client_2)

    client_3 = df[df['dataset'].str.startswith(('Histogram', 'Moments', 'Texture'))]
    client_datasets.append(client_3)

    client_4 = df[df['dataset'].str.startswith('audio')]
    client_datasets.append(client_4)

    client_5 = df[df['dataset'].str.startswith(('mnist6W', 'msong10W'))]
    client_datasets.append(client_5)

    return client_datasets

# 创建参数解析器
parser = argparse.ArgumentParser()
parser_add_main_args(parser)
parser.add_argument('--comm_rounds', type=int, default=50)
parser.add_argument('--client_fraction', type=float, default=1.0)
parser.add_argument('--split_version', type=int, default=2)
parser.add_argument('--aggregation_strategy', type=str, default='fedavg')
args = parser.parse_args()

fix_seed(args.seed)
args.num_clients = 5  # 版本2使用5个客户端

print("="*80)
print("诊断信息")
print("="*80)
print(f"参数设置：")
print(f"  seed: {args.seed}")
print(f"  client_fraction: {args.client_fraction}")
print(f"  num_clients: {args.num_clients}")
print(f"  计算的选择客户端数: max(1, int({args.client_fraction} * {args.num_clients})) = {max(1, int(args.client_fraction * args.num_clients))}")
print()

# 加载数据
mb = read_dataset(args.data_dir)
mb = mb[mb['size'] <= 50000]
mb = mb[~(mb['dataset'].str.startswith("mnist", na=False) & (mb['size'] > 8000))]

columns_to_normalize = ['size', 'dim', 'K', 'L', 'ITER', 'S', 'R', 'L_refine', 'R_refine', 'C', 'k', 'qps', 'distcomp']
min_values = {col: mb[col].min() for col in columns_to_normalize}
max_values = {col: mb[col].max() for col in columns_to_normalize}
max_values = {col: max_values[col] - min_values[col] for col in columns_to_normalize}

mb = preprocess_data(mb, min_values, max_values, columns_to_normalize)

print(f"总数据大小: {mb.shape}")

# 分配数据到客户端
client_datasets = split_dataset_v2(mb)

# 划分训练集和验证集
client_train_datasets = []
client_val_datasets = []

print("\n客户端数据分配情况：")
print("-"*80)
for i, client_data in enumerate(client_datasets):
    if len(client_data) == 0:
        print(f"客户端 {i}: 数据为空 ❌")
        client_train_datasets.append(pd.DataFrame())
        client_val_datasets.append(pd.DataFrame())
        continue

    client_train, client_val = train_test_split(client_data, test_size=0.25, shuffle=True, random_state=args.seed)
    client_train_datasets.append(client_train)
    client_val_datasets.append(client_val)

    print(f"客户端 {i}: 总数据={len(client_data):6d}, 训练集={len(client_train):6d}, 验证集={len(client_val):6d} ✓")

# 模拟客户端选择
print("\n" + "="*80)
print("模拟客户端选择（第1轮）：")
print("="*80)
selected_clients = random.sample(range(args.num_clients), max(1, int(args.client_fraction * args.num_clients)))
print(f"选择的客户端ID: {sorted(selected_clients)}")

actually_training = []
for client_id in selected_clients:
    client_train_data = client_train_datasets[client_id]
    client_size = len(client_train_data)
    if client_size == 0:
        print(f"  客户端 {client_id}: 数据为空，跳过 ❌")
        continue
    actually_training.append(client_id)
    print(f"  客户端 {client_id}: 训练数据量={client_size} ✓")

print(f"\n实际参与训练的客户端数: {len(actually_training)}")
print(f"实际参与训练的客户端ID: {sorted(actually_training)}")

if len(actually_training) < len(selected_clients):
    print(f"\n⚠️  警告: 选择了 {len(selected_clients)} 个客户端，但只有 {len(actually_training)} 个实际参与训练！")
    skipped = set(selected_clients) - set(actually_training)
    print(f"   被跳过的客户端: {sorted(skipped)}")
else:
    print(f"\n✓ 所有选择的客户端都参与了训练")

print("="*80)
