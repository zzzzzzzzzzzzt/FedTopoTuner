import os
import time
import argparse
import random
import faiss
import numpy as np
import torch
print(torch.__version__)
print(torch.version.cuda)
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph

from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from dataset import *
from parse import parse_method, parser_add_main_args
from struct import pack
import warnings
warnings.filterwarnings('ignore')
import csv
import pandas as pd
from sklearn.metrics import r2_score

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

### Parse args ###
parser = argparse.ArgumentParser(description='General Test Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)



### Load and preprocess data ###
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

test=read_dataset(args.test_dir)
# test.fillna(0, inplace=True)

# 使用 pd.get_dummies() 进行独热编码
test = pd.get_dummies(test, columns=['graph_type'], prefix='graph_type')

test['size']=(test['size'] - min_value_size)/max_value_size
test['dim']=(test['dim'] - min_value_dim)/max_value_dim
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



# device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")

d=args.input #输入层神经元个数
pca = PCA(n_components=d)

d_para=test.shape[1]-4
c=args.output #输出层神经元个数
model,suffix = parse_method(args, d, c, d_para, device)
print('MODEL:', model)

model_path = args.model_dir + f'summary_pic{args.id}/{args.method}_{args.id}.pkl'
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)

model.eval()

#对测试集进行测试
# print('预测数据集 模型预测时间 recall误差 qps误差 distcomp误差')

target=['recall','qps','distcomp']

k=30
recall_val=0.9
    
#数据预加载
pre_work(args,test,test,device)

with open("tmp.txt", "a") as f:
    print("dataset,K,L,ITER,S,R,L_refine,R_refine,C,recall,qps,distcomp",file=f)

data_entries = []

bases = test.dataset.unique()
for i in range(1):
    for dataset in bases:
        data=test[test.dataset == dataset]
        data.drop(target, axis=1, inplace=True)
            
        now_mb=test[test.dataset == dataset]
        
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
            # print("直接读入")
            road=args.path+'/dataset/data/'+dataset+'/'+dataset+f'_{args.knn_num}_adj_matrix.txt'
            # adj=get_adjs(road)
            adj=my_dict[road]
        else:
            print("生成邻接矩阵")
            # 构建knn图
            node=read_fvecs(args.path+'/dataset/data/'+dataset+'/'+dataset+'_base.fvecs')
            
            if node.shape[0] <= 100000:
                adj_knn = kneighbors_graph(node, n_neighbors=args.knn_num,  metric='euclidean', include_self=False)
                #print(adj_knn.shape) #(10000,10000)
                edge_index = torch.tensor(adj_knn.nonzero(), dtype=torch.long)
            else:
                # 数据标准化
                node = node.astype('float32')
                faiss.normalize_L2(node)  # 可选：L2归一化
                
                # 构建索引并搜索
                index = faiss.IndexFlatL2(node.shape[1])
                index.add(node)
                distances, indices = index.search(node, k=args.knn_num + 1)  # +1因包含自身
                
                # 生成边索引
                rows = np.repeat(np.arange(node.shape[0]), args.knn_num)
                cols = indices[:, 1:].flatten()  # 排除自身
                edge_index = torch.tensor([rows, cols], dtype=torch.long)
            
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
            adj=torch.from_numpy(adj)
            adj=adj.to(device)
            
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
        
        time0=time.time()
        with torch.no_grad():
            out = model(node_feat, adj, now)
        time1=time.time()
        # print(dataset)
        
        # numpy_array = feature.cpu().numpy()  # 从GPU转移到CPU并转为NumPy数组
        # reshaped_data = numpy_array.reshape(1, -1)
        
        # print(numpy_array)
        # print(numpy_array.shape)
        
        # with open("tmp.txt", "a") as f:
            # print(dataset,numpy_array,file=f)
            
#         # 添加数据
#         data_entries.append({
#             "dataset": dataset,
#             "features": numpy_array
#         })
        
# # 将每个浮点数组展开为多列，并添加数据集名称
# rows = []
# for entry in data_entries:
#     row = {"dataset": entry["dataset"]}
#     for i, val in enumerate(entry["features"]):
#         row[f"feature_{i+1}"] = val
#     rows.append(row)

# df = pd.DataFrame(rows)

# df.to_csv(
#     "tensor.csv",
#     index=False,
#     float_format="%.6f"  # 保留6位小数
# )
        
        
        # data['recall']=out[:,0].cpu().detach().numpy()
        # data['qps']=out[:,1].cpu().detach().numpy()
        # data['distcomp']=out[:,2].cpu().detach().numpy()
        
        
        # data['k']=data['k'] * max_value_k + min_value_k
        # data['k'] = data['k'].astype(int)
        # data['qps']=data['qps'] * max_value_qps + min_value_qps
        # data['distcomp']=data['distcomp'] * max_value_distcomp + min_value_distcomp
        
        # now = data[(data['k'] == k) & (data['recall'] > recall_val)]
        # # if dataset == "mnist6W_2000":
        # #     print(now)
        
        # now = now.sort_values(
        #     by=['distcomp'],
        #     ascending=[True],
        #     na_position='first'  # 将NaN放在最前
        # )
        
        # # if dataset == "mnist6W_16000":
        # #     print(now)
        
        # if now.shape[0] == 0:
        #     now = data[(data['k'] == k)]
            
        #     now = now.sort_values(
        #         by=['recall', 'distcomp'],
        #         ascending=[False, True],
        #         na_position='first'  # 将NaN放在最前
        #     )
        
        # tmp=now.iloc[0]
        # ans=test[(test['dataset'] == dataset) & (round(test['k']* max_value_k + min_value_k) == k) & (test['K'] == tmp['K']) & (test['L'] == tmp['L']) & (test['ITER'] == tmp['ITER']) & (test['S'] == tmp['S']) & (test['R'] == tmp['R']) & (test['L_refine'] == tmp['L_refine']) & (test['R_refine'] == tmp['R_refine']) & (test['C'] == tmp['C'])]
        
        # ans=ans.iloc[0]
        # with open("tmp.txt", "a") as f:
        #     ans['K']=ans['K']*max_value_K+min_value_K
        #     ans['K']=ans['K'].astype(int)
        #     ans['L']=ans['L']*max_value_L+min_value_L
        #     ans['L']=ans['L'].astype(int)
        #     ans['ITER']=ans['ITER']*max_value_ITER+min_value_ITER
        #     ans['ITER']=ans['ITER'].astype(int)
        #     ans['S']=ans['S']*max_value_S+min_value_S
        #     ans['S']=ans['S'].astype(int)
        #     ans['R']=ans['R']*max_value_R+min_value_R
        #     ans['R']=ans['R'].astype(int)
        #     ans['L_refine']=ans['L_refine']*max_value_L_refine+min_value_L_refine
        #     ans['L_refine']=ans['L_refine'].astype(int)
        #     ans['R_refine']=ans['R_refine']*max_value_R_refine+min_value_R_refine
        #     ans['R_refine']=ans['R_refine'].astype(int)
        #     ans['C']=ans['C']*max_value_C+min_value_C
        #     ans['C']=ans['C'].astype(int)
            
        #     ans['qps']=ans['qps']*max_value_qps+min_value_qps
        #     ans['distcomp']=ans['distcomp']*max_value_distcomp+min_value_distcomp
        #     print(dataset,ans['K'],ans['L'],ans['ITER'],ans['S'],ans['R'],ans['L_refine'],ans['R_refine'],ans['C'],ans['recall'],ans['qps'],ans['distcomp'],file=f)
        
        
        
        # out=out.item()
        # target0=target0.item()
        # target1=target1.item()
        # target2=target2.item()
        
        loss0=abs(out[:,0]-target0)
        loss0=torch.mean(loss0)
        
        # target0=target0.cpu().detach().numpy()
        # out0=out[:,0].cpu().detach().numpy()
        # r2 = r2_score(target0, out0)
        # print('recall ',f"{r2:.2f}")
        
        loss1=abs(out[:,1]-target1)
        loss1=torch.mean(loss1)
        
        # target1=target1.cpu().detach().numpy()
        # out1=out[:,1].cpu().detach().numpy()
        # r2 = r2_score(target1, out1)
        # print('qps ',f"{r2:.2f}")
        
        loss2=abs(out[:,2]-target2)
        loss2=torch.mean(loss2)
        
        # target2=target2.cpu().detach().numpy()
        # out2=out[:,2].cpu().detach().numpy()  
        # r2 = r2_score(target2, out2)
        # print('distcomp ',f"{r2:.2f}")
        
        print(dataset,loss0.item(),loss1.item()*max_value_qps+min_value_qps,loss2.item()*max_value_distcomp+min_value_distcomp)
        # print(dataset,time1-time0)
    
    
    #     python /mnt/HDD1/lc23/tyk/pgformer/src_difformer/generate.py --id 54 --device 3 --output 3 --epochs 1000 --batch_size 20000 \
    #    --method difformer --lr 0.008 --num_layers 2 --num_heads 4    \
    #           --knn_num 2 --hidden_channels 128 --dropout 0.5 --weight_decay 1e-5 --alpha 0.5 --use_bn --use_residual --use_graph --save_model
    