import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from struct import pack
from dataset import *
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from parse import parse_method, parser_add_main_args
import argparse
import pandas as pd
import time
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
    results = []
    
    loss_recall=0
    loss_qps=0
    loss_distcomp=0
        
    bases = test.dataset.unique()
    dataset_cnt=0
    for dataset in bases:
        result_df = pd.DataFrame()
        print("dataset:",dataset)
        
        start_time = time.time()
        
        if int(dataset.split('_')[-1])>100000:
            continue
        
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
            node_feat=my_dict[road]
        else:
            node_feat=read_fvecs(args.path+'/dataset/data/'+dataset+'/'+dataset+'_base.fvecs')
            if node_feat.shape[1] > args.input:#维度高于输入层神经元个数
                node_feat = pca.fit_transform(node_feat)
            elif node_feat.shape[1] < args.input:#维度低于输入层神经元个数
                node_feat = np.concatenate([node_feat, np.zeros((node_feat.shape[0], args.input - node_feat.shape[1]))], axis=1)
                # node_feat = np.concatenate([node_feat, 0.5 * np.ones((node_feat.shape[0], args.input - node_feat.shape[1]))], axis=1)
            write_fvecs(args.path+'/dataset/data/'+dataset+'/'+dataset+f'_base_{args.input}.fvecs',node_feat)
        
        n=node_feat.shape[0]



        # adjs=None
        if os.path.exists(args.path+'/dataset/data/'+dataset+'/'+dataset+f'_{args.knn_num}_adj_matrix.txt'):
            # 加载邻接矩阵
            road=args.path+'/dataset/data/'+dataset+'/'+dataset+f'_{args.knn_num}_adj_matrix.txt'
            # adj = np.loadtxt(road, dtype=int)
            # adj=torch.from_numpy(adj)
            # adj=get_adjs(road)
            adj=my_dict[road]
        else:
            # 构建knn图
            node=read_fvecs(args.path+'/dataset/data/'+dataset+'/'+dataset+'_base.fvecs')
            adj_knn = kneighbors_graph(node, n_neighbors=args.knn_num,  metric='euclidean', include_self=False)
            #print(adj_knn.shape) #(10000,10000)
            edge_index = torch.tensor(adj_knn.nonzero(), dtype=torch.long)
            
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
            
        out = model(node_feat, adj, now)
        
        end_time = time.time()
        search_time = (end_time - start_time)*1000
        search_time = round(search_time, 1)
        
        loss0=abs(out[:,0]-target0)
        loss0=torch.mean(loss0)
        loss1=abs(out[:,1]-target1)
        loss1=torch.mean(loss1)
        loss2=abs(out[:,2]-target2)
        loss2=torch.mean(loss2)
        
        results.append({
            'dataset': dataset,
            'search_time(ms)': search_time,
            'loss_recall': loss0,
            'loss_qps': loss1,
            'loss_distcomp': loss2,
        })
        
        loss_recall+=loss0
        loss_qps+=loss1
        loss_distcomp+=loss2
    loss_recall/=dataset_cnt
    loss_qps/=dataset_cnt
    loss_distcomp/=dataset_cnt
    result = pd.DataFrame(results)
    return loss_recall,loss_qps,loss_distcomp,result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='General Training Pipeline')
    # dataset and evaluation
    parser.add_argument('--data_dir', type=str, default='/mnt/HDD1/lc23/tyk/pgformer/data/mb.csv')
    parser.add_argument('--test_dir', type=str, default='/mnt/HDD1/lc23/tyk/pgformer/data/test.csv')
    parser.add_argument('--id', type=int, default=54)
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--method', type=str, default='difformer')
    parser.add_argument('--model_dir', type=str, default='/mnt/HDD1/lc23/tyk/pgformer/model/')
    parser.add_argument('--input', type=int, default=128, help='input channel')
    parser.add_argument('--output', type=int, default=3, help='output channel')
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2,
                    help='number of layers for deep methods')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='number of heads for attention')
    parser.add_argument('--use_bn', action='store_true', help='use layernorm')
    parser.add_argument('--alpha', type=float, default=0.5, help='weight for residual link')
    parser.add_argument('--use_residual', action='store_true', help='use residual link for each GNN layer')
    parser.add_argument('--use_graph', action='store_true', help='use pos emb')
    parser.add_argument('--use_weight', action='store_true', help='use weight for GNN convolution')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=720)
    parser.add_argument('--knn_num', type=int, default=2, help='number of k for KNN graph')
    parser.add_argument('--kernel', type=str, default='simple', choices=['simple', 'sigmoid'])
    parser.add_argument('--path', type=str, default='/mnt/HDD1/lc23/tyk/pgformer')
    
    args = parser.parse_args()
    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    mb = read_dataset(args.data_dir)
    
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
    
    # 使用 pd.get_dummies() 进行独热编码
    mb = pd.get_dummies(mb, columns=['graph_type'], prefix='graph_type')
    
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
    graph_type = 'kgraph'
    # test_dir = '/mnt/HDD1/lc23/tyk/pgformer/data/test_{}.csv'.format(graph_type)
    
    test=read_dataset(args.test_dir)
    
    test = pd.get_dummies(test, columns=['graph_type'], prefix='graph_type')
    
    test = test[test['graph_type_{}'.format(graph_type)] == 1]
    
    
    print(test)
    
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
    print('test:',test.shape)
    
    d=args.input #输入向量维度 输入层神经元个数
    pca = PCA(n_components=d)

    d_para=mb.shape[1]-4

    c=args.output #输出层神经元个数
    train,val=train_test_split(mb, test_size=0.25, shuffle=True)
    pre_work(args,train,test,device)
    model,suffix = parse_method(args, d, c, d_para, device)
    model.load_state_dict(torch.load(args.model_dir + f'summary_pic{args.id}/{args.method}_{args.id}.pkl'))
    # print("model:",model)
    model.eval()
    test_LOSS_recall,test_LOSS_qps,test_LOSS_distcomp,result = evaluate_test(model,test,args,device,pca)
    print("test_LOSS_recall:",test_LOSS_recall)
    print("test_LOSS_qps:",test_LOSS_qps)
    print("test_LOSS_distcomp:",test_LOSS_distcomp)
    result.to_csv('/mnt/HDD1/lc23/sfy/pg-recommendation/pg-recommendation/src/result/pgformer_{}_result.csv'.format(graph_type), index=False)