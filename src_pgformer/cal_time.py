import os
import time
import argparse
import random
import faiss
import numpy as np
import torch
print(torch.__version__)
print(torch.version.cuda)
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from torch.cuda.amp import autocast, GradScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph

from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
# import torch_sparse
from tyk.pgformer.src_pgformer.eval import evaluate
from model_test import evaluate_test
#from logger import Logger
from dataset import *
from parse import parse_method, parser_add_main_args
from struct import pack
import warnings
warnings.filterwarnings('ignore')

import networkx as nx
from scipy.sparse import csr_matrix  
# import cudf

from torch.utils.tensorboard import SummaryWriter
import pandas as pd

def read_fvecs(fname):
    #print(fname)
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]

def read_dataset(data_dir):
    mb = pd.read_csv(data_dir)
    # mb = mb[mb['dim'] > 127]
    return mb

test=read_dataset('/mnt/HDD1/lc23/tyk/pgformer/data/test.csv')
bases = test.dataset.unique()

for dataset in bases:
    print(dataset)
    tyk0=time.time()
    # 构建knn图
    node=read_fvecs('/mnt/HDD1/lc23/tyk/pgformer/dataset/data/'+dataset+'/'+dataset+'_base.fvecs')

    # if node.shape[0] <= 100000:
    #     adj_knn = kneighbors_graph(node, n_neighbors=args.knn_num,  metric='euclidean', include_self=False)
    #     #print(adj_knn.shape) #(10000,10000)
    #     edge_index = torch.tensor(adj_knn.nonzero(), dtype=torch.long)
    # else:
    # 确保输入数据为C连续且float32类型
    node = np.ascontiguousarray(node).astype('float32')  # 修复内存布局和类型

    dim = node.shape[1]

    # 创建HNSW索引（M=32，L2距离）
    index = faiss.index_factory(dim, "HNSW32", faiss.METRIC_L2)

    # 分块添加数据（避免内存溢出）
    batch_size = 100000
    for i in range(0, len(node), batch_size):
        chunk = node[i:min(i+batch_size,len(node))]
        index.add(chunk)  # 此时chunk已确保C连续

    index.hnsw.efSearch = 50  # 调整搜索范围，类似IVF的nprobe概念

    # 执行搜索（注意结果包含自身，需排除）
    distances, indices = index.search(node, k=2+1)

    # 生成稀疏邻接矩阵（保持与原始代码一致）
    rows = np.repeat(np.arange(len(indices)), 2)  # 排除自身
    cols = indices[:, 1:].flatten()
    edge_index = torch.stack([torch.LongTensor(rows), torch.LongTensor(cols)]) 

    tyk1=time.time()
    print(tyk1-tyk0)
