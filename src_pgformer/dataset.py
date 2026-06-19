import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split 
import functools

def read_fvecs(fname):
    #print(fname)
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]

@functools.lru_cache(maxsize=None)
def get_fvecs(road):
    return read_fvecs(road)

@functools.lru_cache(maxsize=None)
def get_adjs(road):
    adj = np.loadtxt(road, dtype=int)
    return torch.from_numpy(adj)

my_dict={}

def pre_work(args,train,test,device):
    bases = train.dataset.unique()
    for dataset in bases:
        if os.path.exists(args.path+'/dataset/data/'+dataset+'/'+dataset+f'_base_{args.input}.fvecs'):
            road=args.path+'/dataset/data/'+dataset+'/'+dataset+f'_base_{args.input}.fvecs'
            # print(road)
            node_feat=get_fvecs(road)
            
            node_feat=torch.Tensor(node_feat)
            node_feat=node_feat.to(device)
            my_dict[road]=node_feat
    
        if os.path.exists(args.path+'/dataset/data/'+dataset+'/'+dataset+f'_{args.knn_num}_adj_matrix.txt'):
            road=args.path+'/dataset/data/'+dataset+'/'+dataset+f'_{args.knn_num}_adj_matrix.txt'
            adj=get_adjs(road)
            
            adj=adj.to(device)
            my_dict[road]=adj
    
    bases = test.dataset.unique()
    for dataset in bases:
        if os.path.exists(args.path+'/dataset/data/'+dataset+'/'+dataset+f'_base_{args.input}.fvecs'):
            road=args.path+'/dataset/data/'+dataset+'/'+dataset+f'_base_{args.input}.fvecs'
            node_feat=get_fvecs(road)
            
            node_feat=torch.Tensor(node_feat)
            node_feat=node_feat.to(device)
            my_dict[road]=node_feat
    
        if os.path.exists(args.path+'/dataset/data/'+dataset+'/'+dataset+f'_{args.knn_num}_adj_matrix.txt'):
            road=args.path+'/dataset/data/'+dataset+'/'+dataset+f'_{args.knn_num}_adj_matrix.txt'
            adj=get_adjs(road)
            
            adj=adj.to(device)
            my_dict[road]=adj

def rand_train(train,batch_size):
    train = train.sample(frac=1)
    length=train.shape[0]
    
    ans=[]
    l=0
    r=min(batch_size,length)
    
    while l<length:
        ans.append(train.iloc[l:r])
        l=r
        r=min(r+batch_size,length)
    
    return ans

target=['recall','qps','distcomp']
my_traindata = []

def pre_work_traindata(dataloader,device):
    for iter in dataloader:
        tmp = []
        bases = iter.dataset.unique()
        
        for dataset in bases:
            if int(dataset.split('_')[-1])>100000:
                continue
            
            now_mb=iter[iter.dataset == dataset]

            if len(now_mb)<1:
                continue
            
            target0=now_mb[target[0]]
            target1=now_mb[target[1]]
            target2=now_mb[target[2]]
            
            target0 = torch.from_numpy(target0.values)
            target0=target0.to(device)
            target1 = torch.from_numpy(target1.values)
            target1=target1.to(device)
            target2 = torch.from_numpy(target2.values)
            target2=target2.to(device)
            
            now=now_mb.drop('dataset', axis=1)
            now=now.drop(target , axis=1)
            now = torch.from_numpy(now.values).float()
            now=now.to(device)
            
            tmp.append(now)
            tmp.append(target0)
            tmp.append(target1)
            tmp.append(target2)
        my_traindata.append(tmp)
            

def load_dataset(data_dir):
    mb = pd.read_csv(data_dir)
    
    # divide=0.99
    # data1=mb[mb.recall < divide]
    # data2=mb[mb.recall >= divide]
    # data2=data2.sample(n=data1.shape[0]//2)
    # mb = pd.concat([data1, data2], axis=0, ignore_index=True)
    
    # mb = mb[mb['dim'] > 127]  
    
    
    # #META_TARGETS = ['DistComp', 'Recall', 'QueryTime']
    # META_TARGETS = ['Recall']
    #划分train,test
    #返回划分后的csv格式数据
    train,val=train_test_split(mb, test_size=0.25, shuffle=True)
    
    # get_xy = lambda x: (x.drop(META_TARGETS, axis=1).copy(), x.loc[:, META_TARGETS].copy())
    # #将元数据库中图配置数据和元目标分离
    # train_x,train_y=get_xy(train)
    # test_x,test_y=get_xy(test)
    
    return train,val

def read_dataset(data_dir):
    mb = pd.read_csv(data_dir)
    # mb = mb[mb['dim'] > 127]
    # # mb = mb[mb['size'] < 500000]
    # mb = mb[mb['dataset'].str.startswith("syn")]
    # mb = mb[mb['size'] == 5000]
    
    return mb

def adj_mul(adj_i, adj, N):
    adj_i_sp = torch.sparse_coo_tensor(adj_i, torch.ones(adj_i.shape[1], dtype=torch.float).to(adj.device), (N, N))
    adj_sp = torch.sparse_coo_tensor(adj, torch.ones(adj.shape[1], dtype=torch.float).to(adj.device), (N, N))
    adj_j = torch.sparse.mm(adj_i_sp, adj_sp)
    adj_j = adj_j.coalesce().indices()
    return adj_j
