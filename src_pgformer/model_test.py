import os
import torch
import numpy as np
import faiss
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from struct import pack
from dataset import *
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from utils import load_or_generate_node_features, load_or_generate_adj_matrix

def read_fvecs(fname):
    #print(fname)
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]

def write_fvecs(fname, vecs):
    print(fname)
    with open(fname, "wb") as f:
        for vec in vecs:
            dim = len(vec)
            f.write(pack('<i', dim))
            f.write(pack('f' * dim, *list(vec)))

target=['recall','qps','distcomp']

@torch.no_grad()
def evaluate_test(model,test,args,device,pca):
    loss_recall=0
    loss_qps=0
    loss_distcomp=0
        
    bases = test.dataset.unique()
    dataset_cnt=0
    for dataset in bases:
        # if int(dataset.split('_')[-1])>100000:
        #     continue
        
        now_mb=test[test.dataset == dataset]
        
        if len(now_mb)<1:
            continue
            
        dataset_cnt+=1
        
        target0=now_mb[target[0]]
        target1=now_mb[target[1]]
        target2=now_mb[target[2]]
        
        now=now_mb.drop('dataset', axis=1)
        now=now.drop(target , axis=1)
        
        now = torch.from_numpy(now.values).float()
        now=now.to(device)
        
        target0 = torch.from_numpy(target0.values)
        target0=target0.to(device)
        target1 = torch.from_numpy(target1.values)
        target1=target1.to(device)
        target2 = torch.from_numpy(target2.values)
        target2=target2.to(device)



        if os.path.exists(args.path+'/dataset/data/'+dataset+'/'+dataset+f'_base_{args.input}.fvecs'):
            road=args.path+'/dataset/data/'+dataset+'/'+dataset+f'_base_{args.input}.fvecs'
            # node_feat=get_fvecs(road)
            if road not in my_dict:
                print(f"Warning: Node features for {road} not found, generating now...")
                my_dict[road] = load_or_generate_node_features(dataset, args.path, args.input, pca, device)
            node_feat=my_dict[road]
        else:
            node_feat=read_fvecs(args.path+'/dataset/data/'+dataset+'/'+dataset+'_base.fvecs')
            if node_feat.shape[1] > args.input:#维度高于输入层神经元个数
                node_feat = pca.fit_transform(node_feat)
            elif node_feat.shape[1] < args.input:#维度低于输入层神经元个数
                node_feat = np.concatenate([node_feat, np.zeros((node_feat.shape[0], args.input - node_feat.shape[1]))], axis=1)
                # node_feat = np.concatenate([node_feat, 0.5 * np.ones((node_feat.shape[0], args.input - node_feat.shape[1]))], axis=1)
            write_fvecs(args.path+'/dataset/data/'+dataset+'/'+dataset+f'_base_{args.input}.fvecs',node_feat)
            node_feat = torch.from_numpy(node_feat).float()
            node_feat=node_feat.to(device)
            my_dict[args.path+'/dataset/data/'+dataset+'/'+dataset+f'_base_{args.input}.fvecs']=node_feat
            
        n=node_feat.shape[0]



        # adjs=None
        if os.path.exists(args.path+'/dataset/data/'+dataset+'/'+dataset+f'_{args.knn_num}_adj_matrix.txt'):
            # 加载邻接矩阵
            road=args.path+'/dataset/data/'+dataset+'/'+dataset+f'_{args.knn_num}_adj_matrix.txt'
            # adj = np.loadtxt(road, dtype=int)
            # adj=torch.from_numpy(adj)
            # adj=get_adjs(road)
            if road not in my_dict:
                print(f"Warning: Node features for {road} not found, generating now...")
                my_dict[road] = load_or_generate_adj_matrix(dataset, args.path, args.knn_num, device, args.directed)
            adj=my_dict[road]
        else:
            print("生成邻接矩阵")
            # 构建knn图
            node=read_fvecs(args.path+'/dataset/data/'+dataset+'/'+dataset+'_base.fvecs')
            
            # if node.shape[0] <= 100000:
            #     adj_knn = kneighbors_graph(node, n_neighbors=args.knn_num,  metric='euclidean', include_self=False)
            #     #print(adj_knn.shape) #(10000,10000)
            #     edge_index = torch.tensor(adj_knn.nonzero(), dtype=torch.long)
            # else:
            # 确保输入数据为C连续且float32类型
            node = np.ascontiguousarray(node).astype('float32')  # 修复内存布局和类型
            # 加载数据时转换（假设 node_feat 是 np.memmap）
            # node = torch.from_numpy(np.ascontiguousarray(node)).float()
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
            distances, indices = index.search(node, k=args.knn_num+1)

            # 生成稀疏邻接矩阵（保持与原始代码一致）
            rows = np.repeat(np.arange(len(indices)), args.knn_num)  # 排除自身
            cols = indices[:, 1:].flatten()
            edge_index = torch.stack([torch.LongTensor(rows), torch.LongTensor(cols)])
            # edge_index=torch.from_numpy(edge_index) 

            # edge_index=edge_index+1
            # vector0 = np.zeros(n, dtype=np.int32)  
            
            # vector1=[]
            # for i in range(1,length+1):
            #     vector1.append(i)

            # # 把聚合信息点放到最前面
            # edge_index=edge_index.numpy()
            # edge_index = np.concatenate((edge_index, np.vstack((vector0, vector1))), axis=1)
            # edge_index=torch.from_numpy(edge_index)
            
            if not args.directed:
                print("无向图")
                edge_index = to_undirected(edge_index)
            
            ### Adj storage for relational bias ###
            adj, _ = remove_self_loops(edge_index)
            adj, _ = add_self_loops(adj, num_nodes=n)
            
            #print(adj.shape) #(2,24246)
            
            # 将稀疏矩阵转换为密集矩阵
            # adj_matrix_dense = csr_matrix(adj.numpy()).toarray()
            # 保存邻接矩阵    
            np.savetxt(args.path+'/dataset/data/'+dataset+'/'+dataset+f'_{args.knn_num}_adj_matrix.txt', adj, fmt='%d')
            adj=adj.to(device)
            my_dict[args.path+'/dataset/data/'+dataset+'/'+dataset+f'_{args.knn_num}_adj_matrix.txt']=adj
        ### Adj storage for relational bias ###
        # adjs = []
        # adjs.append(adj)
        
        #adj为邻接矩阵 邻接矩阵高阶乘 可以链接到多跳邻居
        # for i in range(args.rb_order - 1): # edge_index of high order adjacency
        #     adj = adj_mul(adj, adj, n)
        #     adj=adj.to(device)
        #     adjs.append(adj)
        
        #由于adjs为list, 里面包含python对象，不能直接使用to(device)方法
        # adjs = [obj.to(device) for obj in adjs]

        # vector = np.zeros(128)
        # # 使用np.vstack添加新的行向量  
        # node_feat = np.vstack((vector, node_feat))

        

        out = model(node_feat, adj, now)
        # out = torch.clamp(out, min=0, max=1)
        
        loss0=abs(out[:,0]-target0)
        loss0=torch.mean(loss0)
        loss1=abs(out[:,1]-target1)
        loss1=torch.mean(loss1)
        loss2=abs(out[:,2]-target2)
        loss2=torch.mean(loss2)
        
        # out=out.squeeze(1)
        # target=target.squeeze(1)
        # target=target.to(torch.float)
        
        # out=out.cpu()
        # if args.target=='recall':
        #     out = np.clip(out, 0, 1)  
        
        # loss=criterion(out,target)
        
        # loss_recall+=loss.detach().item()
        loss_recall+=loss0
        loss_qps+=loss1
        loss_distcomp+=loss2
    loss_recall/=dataset_cnt
    loss_qps/=dataset_cnt
    loss_distcomp/=dataset_cnt
    
    return loss_recall,loss_qps,loss_distcomp

