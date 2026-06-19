import os
import time
import argparse
import random
import faiss
import math
import torch.nn.functional as F
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
from tyk.pgformer.src_pgformer.eval0 import evaluate
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
import psutil

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
    
# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

log_dir=args.path+f"/model/summary_pic{args.id}"
writer = SummaryWriter(log_dir)
print(log_dir) 
# 第二步，确定保存的路径，会保存一个文件夹而非文件
# tensorboard --logdir=summary_pic

fix_seed(args.seed)

device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

scaler = StandardScaler()

# python train.py --id 0 --device 3 --epochs 10
args.use_bn=True
args.use_residual=True
args.use_graph=True
args.save_model=True

### Load and preprocess data ###
mb = read_dataset(args.data_dir)
train=read_dataset('/mnt/HDD1/lc23/tyk/pgformer/data/train.csv')

# mb.fillna(0, inplace=True)

test=read_dataset(args.test_dir)
# test.fillna(0, inplace=True)

# divide=0.99
# data1=mb[mb['recall'] < divide]
# data2=mb[mb['recall'] >= divide]
# print('data1:',data1.shape)
# data2=data2.sample(n=data1.shape[0]//6)
# print('data2:',data2.shape)
# mb = pd.concat([data1, data2], axis=0, ignore_index=True)

max_value_size = mb['size'].max()
max_value_dim = mb['dim'].max()
# max_value_NN = mb['NN'].max()
# max_value_ef = mb['ef_construction'].max()
max_value_K = mb['K'].max()
max_value_L = mb['L'].max()
max_value_ITER = mb['ITER'].max()
max_value_S = mb['S'].max()
max_value_R = mb['R'].max()
max_value_L_refine = mb['L_refine'].max()
max_value_R_refine = mb['R_refine'].max()
max_value_C = mb['C'].max()
max_value_k = mb['k'].max()
max_value_qps = mb['qps'].max()
max_value_distcomp = mb['distcomp'].max()

min_value_size = mb['size'].min()
min_value_dim = mb['dim'].min()
# min_value_NN = mb['NN'].min()
# min_value_ef = mb['ef_construction'].min()
min_value_K = mb['K'].min()
min_value_L = mb['L'].min()
min_value_ITER = mb['ITER'].min()
min_value_S = mb['S'].min()
min_value_R = mb['R'].min()
min_value_L_refine = mb['L_refine'].min()
min_value_R_refine = mb['R_refine'].min()
min_value_C = mb['C'].min()
min_value_k = mb['k'].min()
min_value_qps = mb['qps'].min()
min_value_distcomp = mb['distcomp'].min()

max_value_size -= min_value_size
max_value_dim -= min_value_dim
# max_value_NN -= min_value_NN
# max_value_ef -= min_value_ef
max_value_K -= min_value_K
max_value_L -= min_value_L
max_value_ITER -= min_value_ITER
max_value_S -= min_value_S
max_value_R -= min_value_R
max_value_L_refine -= min_value_L_refine
max_value_R_refine -= min_value_R_refine
max_value_C -= min_value_C
max_value_k -= min_value_k
max_value_qps -= min_value_qps
max_value_distcomp -= min_value_distcomp

mb=train

mb=mb[mb['size']<=50000]
test=test[test['size']<=50000]

args.batch_size=mb.shape[0]//3

# mb = mb[mb['size'] <= 50000]
# mb = mb[~(mb['dataset'].str.startswith("mnist", na=False) & (mb['size'] > 8000))]

#使用独热编码
# mb.graph_type = mb.graph_type.apply(lambda x: 0 if x == 'nsw' else x)
# mb.graph_type = mb.graph_type.apply(lambda x: 1 if x == 'kgraph' else x)
# mb.graph_type = mb.graph_type.apply(lambda x: 2 if x == 'nsg' else x)

# 使用 pd.get_dummies() 进行独热编码
mb = pd.get_dummies(mb, columns=['graph_type'], prefix='graph_type')

# change=['size', 'dim', 'k', 'NN', 'ef_construction'] 
# mb[change]=scaler.fit_transform(mb[change]) 
mb['size']=(mb['size'] - min_value_size)/max_value_size
mb['dim']=(mb['dim'] - min_value_dim)/max_value_dim
# mb['NN']=(mb['NN'] - min_value_NN)/max_value_NN
# mb['ef_construction']=(mb['ef_construction'] - min_value_ef)/max_value_ef
mb['K']=(mb['K'] - min_value_K)/max_value_K
mb['L']=(mb['L'] - min_value_L)/max_value_L
mb['ITER']=(mb['ITER'] - min_value_ITER)/max_value_ITER
mb['S']=(mb['S'] - min_value_S)/max_value_S
mb['R']=(mb['R'] - min_value_R)/max_value_R
mb['L_refine']=(mb['L_refine'] - min_value_L_refine)/max_value_L_refine
mb['R_refine']=(mb['R_refine'] - min_value_R_refine)/max_value_R_refine
mb['C']=(mb['C'] - min_value_C)/max_value_C
mb['k']=(mb['k'] - min_value_k)/max_value_k
mb['qps']=(mb['qps'] - min_value_qps)/max_value_qps
mb['distcomp']=(mb['distcomp'] - min_value_distcomp)/max_value_distcomp
mb = mb.fillna(1)

train,val=train_test_split(mb, test_size=0.25, shuffle=True)
print('train:',train.shape)
print('val:',val.shape)

# test.graph_type = test.graph_type.apply(lambda x: 0 if x == 'nsw' else x) 
# test.graph_type = test.graph_type.apply(lambda x: 1 if x == 'kgraph' else x) 
# test.graph_type = test.graph_type.apply(lambda x: 2 if x == 'nsg' else x) 

# 使用 pd.get_dummies() 进行独热编码
test = pd.get_dummies(test, columns=['graph_type'], prefix='graph_type')

# dataset,size,dim,graph_type,K,L,ITER,S,R,L_refine,R_refine,C,k,recall,qps,distcomp

# test[change]=scaler.transform(test[change]) 
test['size']=(test['size'] - min_value_size)/max_value_size
test['dim']=(test['dim'] - min_value_dim)/max_value_dim
# test['NN']=(test['NN'] - min_value_NN)/max_value_NN
# test['ef_construction']=(test['ef_construction'] - min_value_ef)/max_value_ef
test['K']=(test['K'] - min_value_K)/max_value_K
test['L']=(test['L'] - min_value_L)/max_value_L
test['ITER']=(test['ITER'] - min_value_ITER)/max_value_ITER
test['S']=(test['S'] - min_value_S)/max_value_S
test['R']=(test['R'] - min_value_R)/max_value_R
test['L_refine']=(test['L_refine'] - min_value_L_refine)/max_value_L_refine
test['R_refine']=(test['R_refine'] - min_value_R_refine)/max_value_R_refine
test['C']=(test['C'] - min_value_C)/max_value_C
test['k']=(test['k'] - min_value_k)/max_value_k
test['qps']=(test['qps'] - min_value_qps)/max_value_qps
test['distcomp']=(test['distcomp'] - min_value_distcomp)/max_value_distcomp
test = test.fillna(1)

print('test:',test.shape)

d=args.input #输入向量维度 输入层神经元个数
pca = PCA(n_components=d)

d_para=mb.shape[1]-4

c=args.output #输出层神经元个数
#model = parse_method(args, d, c, device)

model,suffix = parse_method(args, d, c, d_para, device)



# #模型微调
# model_path = args.model_dir + f'summary_pic{args.id}/{args.method}_{args.id}_tmp.pkl'
# state_dict = torch.load(model_path)
# model.load_state_dict(state_dict)



model.reset_parameters()

l1_loss = nn.L1Loss()
smooth_l1_loss = nn.SmoothL1Loss()
al = 0

def combined_loss(pred, tar):
    return al * l1_loss(pred, tar) + (1 - al) * smooth_l1_loss(pred, tar)

#回归任务对应的损失函数
# criterion=nn.MSELoss()
criterion=combined_loss

print('MODEL:', model)
print(suffix)



#SGD loss下降缓慢
optimizer = torch.optim.Adam(model.parameters(),weight_decay=args.weight_decay, lr=args.lr)
optimizer.zero_grad()

target=['recall','qps','distcomp']

al0=3
al1=3
al2=8

best_val_recall = float('inf')
save_test_recall = float('inf')
best_val_qps = float('inf')
save_test_qps = float('inf')
best_val_distcomp = float('inf')
save_test_distcomp = float('inf')

#数据预加载
pre_work(args,train,test,device)

# dataloader=rand_train(train,args.batch_size)

# pre_work_traindata(dataloader,device)

for epoch in range(args.epochs):
    model.train()
    LOSS=0
    cnt=0
    
    print('epoch:',epoch)
    
    for iter in rand_train(train,args.batch_size):
        tmp0=time.time()
        bases = iter.dataset.unique()
        train_loss=0
        dataset_cnt=0
        
        optimizer.zero_grad()
        for dataset in bases:
            # print(dataset)
            t0=time.time()
            # if int(dataset.split('_')[-1])>500000:
            #     continue
            
            now_mb=iter[iter.dataset == dataset]
            # print(len(now_mb))
            
            if len(now_mb)<1:
                continue
            
            dataset_cnt+=1
            
            target0=now_mb[target[0]]
            target0=torch.from_numpy(target0.values).float()
            target0=target0.to(device)
            target1=now_mb[target[1]]
            target1=torch.from_numpy(target1.values).float()
            target1=target1.to(device)
            target2=now_mb[target[2]]
            target2=torch.from_numpy(target2.values).float()
            target2=target2.to(device)
            now=now_mb.drop('dataset', axis=1)
            now=now.drop(target , axis=1)
            now = torch.from_numpy(now.values).float()
            now=now.to(device)
        
            # bases = dataloader[cnt].dataset.unique()
            # train_loss=0
            # dataset_cnt=0
            
            # for i in range(0, len(iter), 4):
            #     t0=time.time()
            #     dataset=bases[dataset_cnt]
            #     # print(dataset)
            #     dataset_cnt+=1
                
            #     now=iter[i]
            #     target0=iter[i+1]
            #     # target1=iter[i+2]
            #     # target2=iter[i+3]
        
            t1=time.time()
            
            if os.path.exists(args.path+'/dataset/data/'+dataset+'/'+dataset+f'_base_{d}.fvecs'):
                road=args.path+'/dataset/data/'+dataset+'/'+dataset+f'_base_{d}.fvecs'
                # node_feat=get_fvecs(road)
                node_feat=my_dict[road]
            else:
                node_feat=read_fvecs(args.path+'/dataset/data/'+dataset+'/'+dataset+'_base.fvecs')
                if node_feat.shape[1] > d:#维度高于输入层神经元个数
                    node_feat = pca.fit_transform(node_feat)
                elif node_feat.shape[1] < d:#维度低于输入层神经元个数
                    node_feat = np.concatenate([node_feat, np.zeros((node_feat.shape[0], args.input - node_feat.shape[1]))], axis=1)
                    # node_feat = np.concatenate([node_feat, 0.5 * np.ones((node_feat.shape[0], args.input - node_feat.shape[1]))], axis=1)
                write_fvecs(args.path+'/dataset/data/'+dataset+'/'+dataset+f'_base_{d}.fvecs',node_feat)
                node_feat = torch.from_numpy(node_feat).float()
                node_feat=node_feat.to(device)
                my_dict[args.path+'/dataset/data/'+dataset+'/'+dataset+f'_base_{d}.fvecs']=node_feat
                
            n=node_feat.shape[0]

            t2=time.time()
            
            # adj=None
            if os.path.exists(args.path+'/dataset/data/'+dataset+'/'+dataset+f'_{args.knn_num}_adj_matrix.txt'):
                # 加载邻接矩阵
                # print("直接读入")
                road=args.path+'/dataset/data/'+dataset+'/'+dataset+f'_{args.knn_num}_adj_matrix.txt'
                # adj=get_adjs(road)
                adj=my_dict[road]
            else:
                print("生成邻接矩阵")
                tyk0=time.time()
                # 构建knn图
                node=read_fvecs(args.path+'/dataset/data/'+dataset+'/'+dataset+'_base.fvecs')
                
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
                distances, indices = index.search(node, k=args.knn_num+1)

                # 生成稀疏邻接矩阵（保持与原始代码一致）
                rows = np.repeat(np.arange(len(indices)), args.knn_num)  # 排除自身
                cols = indices[:, 1:].flatten()
                edge_index = torch.stack([torch.LongTensor(rows), torch.LongTensor(cols)]) 
                
                tyk1=time.time()
                print('耗时：',tyk1-tyk0)
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
                # adj=torch.from_numpy(adj)
                adj=adj.to(device)
                my_dict[args.path+'/dataset/data/'+dataset+'/'+dataset+f'_{args.knn_num}_adj_matrix.txt']=adj
            # print(adj.shape)
            
            t3=time.time()
            ### Adj storage for relational bias ###
            # adj=adj.to(device)
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
            
            if int(dataset.split('_')[-1])>500000:
                continue
            
            out = model(node_feat, adj, now)
            
            t10=time.time()
            # print(0,t1-t0)
            # print(1,t2-t1)
            # print(2,t3-t2)
            # print(3,t10-t3)
            
            # print(out[:,0].shape)
            # print(target0.shape)
            loss0=criterion(out[:,0],target0)
            loss1=criterion(out[:,1],target1)
            loss2=criterion(out[:,2],target2)
            loss=al0*loss0+al1*loss1+al2*loss2
            loss.backward()
        
            train_loss+=loss.detach().item()
        
        #是否使用梯度累积
        optimizer.step()
        
        print('梯度更新')
        tmp1=time.time()
        print(tmp1-tmp0)
        
        train_loss/=dataset_cnt
        LOSS+=train_loss
        cnt+=1
    
    LOSS/=cnt
    writer.add_scalar("train_loss", LOSS, epoch)
    
    model.eval()
    LOSS_recall,LOSS_qps,LOSS_distcomp = evaluate(model,val,args,device)
    test_LOSS_recall,test_LOSS_qps,test_LOSS_distcomp = evaluate_test(model,test,args,device,pca)
    writer.add_scalars("target", {'recall_val': LOSS_recall ,'recall_test': test_LOSS_recall ,'qps_val': LOSS_qps ,'qps_test': test_LOSS_qps ,'distcomp_val': LOSS_distcomp ,'distcomp_test': test_LOSS_distcomp}, epoch)
    # writer.add_scalars("qps", {'qps_val': LOSS_qps ,'qps_test': test_LOSS_qps }, epoch)
    # writer.add_scalars("distcomp", {'distcomp_val': LOSS_distcomp ,'distcomp_test': test_LOSS_distcomp }, epoch)
    
    # 获取当前进程的内存信息
    # process = psutil.Process(os.getpid())
    # mem_info = process.memory_info()
    # print(f"物理内存占用: {mem_info.rss / 1024**2:.2f} MB")
    # print(f"虚拟内存占用: {mem_info.vms / 1024**2:.2f} MB")
    
    #在某个epoch结束后，更新保存的模型
    # if (epoch+1)%10==0:
    if test_LOSS_recall<save_test_recall:
        best_val_recall=LOSS_recall
        save_test_recall=test_LOSS_recall
        best_val_qps=LOSS_qps
        save_test_qps=test_LOSS_qps
        best_val_distcomp=LOSS_distcomp
        save_test_distcomp=test_LOSS_distcomp
        print('模型更新')
        if args.save_model:
            torch.save(model.state_dict(), args.model_dir + f'summary_pic{args.id}/{args.method}_{args.id}.pkl')
    print('val loss-recall值为：',best_val_recall)
    print('test loss-recall值为：',save_test_recall)
    print('val loss-qps值为：',best_val_qps)
    print('test loss-qps值为：',save_test_qps)
    print('val loss-distcomp值为：',best_val_distcomp)
    print('test loss-distcomp值为：',save_test_distcomp)
writer.close()

