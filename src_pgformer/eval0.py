import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from struct import pack
from dataset import *

def read_fvecs(fname):
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
def evaluate(model,val,args,device):
    loss_recall=0
    loss_qps=0
    loss_distcomp=0
        
    bases = val.dataset.unique()
    dataset_cnt=0
    for dataset in bases:

        now_mb=val[val.dataset == dataset]
        
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

        road=args.path+'/dataset/data/'+dataset+'/'+dataset+f'_base_{args.input}.fvecs'
        node_feat=my_dict[road]
        n=node_feat.shape[0]
        road=args.path+'/dataset/data/'+dataset+'/'+dataset+f'_{args.knn_num}_adj_matrix.txt'
        adj=my_dict[road]
        
        out = model(node_feat, adj, now)
        
        loss0=abs(out[:,0]-target0)
        loss0=torch.mean(loss0)
        loss1=abs(out[:,1]-target1)
        loss1=torch.mean(loss1)
        loss2=abs(out[:,2]-target2)
        loss2=torch.mean(loss2)
        
        loss_recall+=loss0
        loss_qps+=loss1
        loss_distcomp+=loss2
    loss_recall/=dataset_cnt
    loss_qps/=dataset_cnt
    loss_distcomp/=dataset_cnt
    
    return loss_recall,loss_qps,loss_distcomp

