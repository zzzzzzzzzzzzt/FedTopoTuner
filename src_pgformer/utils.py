# utils.py
import os
import time
import numpy as np
import torch
import faiss
from sklearn.decomposition import PCA
from struct import pack
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops

# 全局缓存字典
my_dict = {}

def read_fvecs(fname):
    """Read fvecs file format."""
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]

def write_fvecs(fname, vecs):
    """Write vectors to fvecs file format."""
    print(f"Saving to {fname}")
    with open(fname, "wb") as f:
        for vec in vecs:
            dim = len(vec)
            f.write(pack('<i', dim))
            f.write(pack('f' * dim, *list(vec)))

def load_or_generate_node_features(dataset, path, input_dim, pca, device):
    """Load or generate node features with PCA transformation."""
    fvecs_path = f"{path}/dataset/data/{dataset}/{dataset}_base_{input_dim}.fvecs"
    if fvecs_path in my_dict:
        """print(f"Loaded from cache: {fvecs_path}")"""
        return my_dict[fvecs_path]
    
    if os.path.exists(fvecs_path):
        """print(f"Loading existing file: {fvecs_path}")"""
        node_feat = read_fvecs(fvecs_path)
    else:
        """print(f"Generating new features for: {fvecs_path}")"""
        node_feat = read_fvecs(f"{path}/dataset/data/{dataset}/{dataset}_base.fvecs")
        if node_feat.shape[1] > input_dim:
            node_feat = pca.fit_transform(node_feat)
        elif node_feat.shape[1] < input_dim:
            node_feat = np.concatenate([node_feat, np.zeros((node_feat.shape[0], input_dim - node_feat.shape[1]))], axis=1)
        write_fvecs(fvecs_path, node_feat)
    
    node_feat = torch.from_numpy(node_feat).float().to(device)
    my_dict[fvecs_path] = node_feat
    return node_feat

def load_or_generate_adj_matrix(dataset, path, knn_num, device, directed=False):
    """Load or generate adjacency matrix using k-NN graph."""
    adj_path = f"{path}/dataset/data/{dataset}/{dataset}_{knn_num}_adj_matrix.txt"
    if adj_path in my_dict:
        """print(f"Loading adjacency matrix from cache: {adj_path}")"""
        return my_dict[adj_path]
    
    if os.path.exists(adj_path):
        """print(f"Loading existing adjacency matrix: {adj_path}")"""
        adj = np.loadtxt(adj_path, dtype=int)
        adj = torch.from_numpy(adj).to(device)
    else:
        """print("Generating adjacency matrix")"""
        t_start = time.time()
        node = read_fvecs(f"{path}/dataset/data/{dataset}/{dataset}_base.fvecs")
        node = np.ascontiguousarray(node).astype('float32')
        dim = node.shape[1]

        index = faiss.index_factory(dim, "HNSW32", faiss.METRIC_L2)
        batch_size = 10000
        for i in range(0, len(node), batch_size):
            index.add(node[i:min(i + batch_size, len(node))])
        index.hnsw.efSearch = 50

        distances, indices = index.search(node, k=knn_num + 1)
        rows = np.repeat(np.arange(len(indices)), knn_num)
        cols = indices[:, 1:].flatten()
        edge_index = torch.stack([torch.LongTensor(rows), torch.LongTensor(cols)])

        if not directed:
            print("Converting to undirected graph")
            edge_index = to_undirected(edge_index)
        
        adj, _ = remove_self_loops(edge_index)
        adj, _ = add_self_loops(adj, num_nodes=node.shape[0])
        np.savetxt(adj_path, adj.cpu().numpy(), fmt='%d')
        adj = adj.to(device)
        
        print(f"Adjacency matrix generation time: {time.time() - t_start:.2f}s")
    
    my_dict[adj_path] = adj
    return adj