from flask import Flask, render_template, request, jsonify
import time
import numpy as np
import os
import torch
print(torch.__version__)
print(torch.version.cuda)

from struct import pack
import warnings
warnings.filterwarnings('ignore')
import csv
import pandas as pd
from dataset import *
import argparse
from parse import parse_method, parser_add_main_args
from shlex import split
from sklearn.decomposition import PCA
import itertools
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import base64
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# 添加全局缓存，减少重复计算
image_features_cache = {}
resnet_model = None

# 获取ResNet模型，只初始化一次
def get_resnet_model():
    global resnet_model
    if resnet_model is None:
        # 加载预训练模型
        resnet_model = models.resnet50(pretrained=True)
        resnet_model.fc = torch.nn.Identity()
        resnet_model.eval()
    return resnet_model

# 提取ResNet-50特征的函数
def extract_resnet50_features(image, image_path=None):
    # 检查缓存
    if image_path and image_path in image_features_cache:
        return image_features_cache[image_path]
        
    # 定义预处理
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # 预处理图像
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # 获取模型
    model = get_resnet_model()

    # 提取特征
    with torch.no_grad():
        features = model(input_batch)
    
    # 转换为NumPy数组
    features_np = features.squeeze(0).numpy()
    
    # 存入缓存
    if image_path:
        image_features_cache[image_path] = features_np
        
    return features_np

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
fixed_str="--id 57 --device 3 --output 3 --epochs 2000 --batch_size 6000 \
    --method pgformer --lr 0.01 --num_layers 1 --num_heads 4    \
    --knn_num 2 --hidden_channels 256 --dropout 0.5 --weight_decay 1e-5 --alpha 0.5 --use_bn --use_residual --use_graph --save_model"

parser = argparse.ArgumentParser(description='General Test Pipeline')
parser_add_main_args(parser)
args = parser.parse_args(split(fixed_str))
# args = parser.parse_args()
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
print(test)

device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")



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

target=['recall','qps','distcomp']

#数据预加载
pre_work(args,test,test,device)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/model')
def model_page():
    return render_template('model.html')

@app.route('/image_query')
def image_query():
    return render_template('image_query.html')

# 处理图像查询请求
@app.route('/process_image_query', methods=['POST'])
def process_image_query():
    try:
        # 检查是否有图像文件上传
        if 'image' not in request.files:
            return jsonify({"error": "未上传图片"}), 400
        
        image_file = request.files['image']
        index_file = request.files.get('index_file', None)
        
        # 检查文件类型
        if not image_file.filename:
            return jsonify({"error": "文件名不能为空"}), 400
            
        # 如果提供了索引文件，检查格式
        if index_file and index_file.filename and not index_file.filename.endswith('.index'):
            return jsonify({"error": "索引文件必须是.index格式"}), 400
        
        # 读取查询图像并提取特征
        query_image = Image.open(image_file).convert('RGB')
        query_features = extract_resnet50_features(query_image)
        
        # 如果有索引文件，使用索引文件进行检索（这里保留接口，但暂不实现）
        if index_file and index_file.filename:
            print(f"检测到索引文件: {index_file.filename}，但当前版本暂不支持索引文件检索")
            # TODO: 实现索引文件检索逻辑
        
        # 获取dog文件夹中的所有图像
        dog_folder = os.path.join(args.path, 'dog')
        dog_images = [f for f in os.listdir(dog_folder) if f.endswith('.jpg')]
        
        print(f"开始处理 {len(dog_images)} 张图像进行相似度计算...")
        
        # 提取所有图像的特征并计算相似度
        similarities = []
        for img_name in dog_images:
            img_path = os.path.join(dog_folder, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                features = extract_resnet50_features(img, image_path=img_path)
                
                # 计算余弦相似度
                similarity = cosine_similarity([query_features], [features])[0][0]
                
                # 转换图像为base64以在前端显示
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                
                similarities.append({
                    "image_name": img_name,
                    "similarity": float(similarity),
                    "image_data": f"data:image/jpeg;base64,{img_base64}"
                })
                
                # 每处理10张图像输出一次进度
                if len(similarities) % 10 == 0:
                    print(f"已处理 {len(similarities)}/{len(dog_images)} 张图像")
                    
            except Exception as e:
                print(f"处理图像 {img_name} 时出错: {str(e)}")
        
        print(f"图像处理完成，共处理 {len(similarities)} 张图像")
        
        # 按相似度降序排序并取前11个
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        # 排除相似度为100%的图像（即查询图像自身）
        # 查找最接近1.0的相似度
        max_similarity = max(similarities, key=lambda x: x["similarity"])["similarity"]
        top_results = [s for s in similarities if abs(s["similarity"] - max_similarity) > 0.00001][:10]
        
        # 如果没有找到足够的结果，则取前10个
        if len(top_results) < 10:
            top_results = similarities[1:11]  # 跳过第一个（自身）
        
        return jsonify({
            "status": "success",
            "results": top_results
        }), 200
        
    except Exception as e:
        print(f"图像查询处理时出错: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/load_datasets', methods=['GET'])
def load_datasets():
    try:
        # 读取CSV文件中的dataset列
        csv_path = args.data_dir  # 使用args.data_dir直接获取mb.csv的路径
        
        # 使用pandas读取CSV文件
        df = pd.read_csv(csv_path)
        
        # 获取dataset列并去重
        if 'dataset' in df.columns:
            datasets = df['dataset'].unique().tolist()
        else:
            # 如果没有dataset列，尝试猜测可能的列名
            possible_columns = ['dataset', 'Dataset', 'data', 'Data', 'name', 'Name']
            for col in possible_columns:
                if col in df.columns:
                    datasets = df[col].unique().tolist()
                    break
            else:
                # 如果找不到合适的列，返回错误
                return jsonify({"error": "未找到dataset列"}), 400
        
        # 返回去重后的数据集列表
        return jsonify(datasets), 200
    
    except Exception as e:
        print(f"加载数据集时出错: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/load_graph_types', methods=['GET'])
def load_graph_types():
    try:
        # 读取CSV文件
        csv_path = args.data_dir
        df = pd.read_csv(csv_path)
        
        # 获取graph_type列并去重
        if 'graph_type' in df.columns:
            graph_types = df['graph_type'].unique().tolist()
        else:
            return jsonify({"error": "未找到graph_type列"}), 400
        
        # 构建结果数据
        result = []
        
        # 根据不同图索引类型获取对应的数据和参数
        if 'nsw' in graph_types:
            nsw_data = df[df['graph_type'] == 'nsw']
            if not nsw_data.empty:
                parameters = []
                # 对于nsw，只有K和L是有意义的参数
                for param in ['K', 'L']:
                    valid_data = nsw_data[param][(nsw_data[param] != 0) & (~nsw_data[param].isna())]
                    if len(valid_data) > 0:
                        min_val = max(1, int(valid_data.min()))
                        max_val = max(min_val + 10, int(valid_data.max()))
                        parameters.append({
                            "name": param,
                            "min": min_val,
                            "max": max_val
                        })
                
                result.append({
                    "type": "nsw",
                    "parameters": parameters
                })
        
        if 'kgraph' in graph_types:
            kgraph_data = df[df['graph_type'] == 'kgraph']
            if not kgraph_data.empty:
                parameters = []
                # 对于kgraph，需要K、L、ITER、S、R参数
                for param in ['K', 'L', 'ITER', 'S', 'R']:
                    valid_data = kgraph_data[param][(kgraph_data[param] != 0) & (~kgraph_data[param].isna())]
                    if len(valid_data) > 0:
                        min_val = max(1, int(valid_data.min()))
                        max_val = max(min_val + 10, int(valid_data.max()))
                        parameters.append({
                            "name": param,
                            "min": min_val,
                            "max": max_val
                        })
                
                result.append({
                    "type": "kgraph",
                    "parameters": parameters
                })
        
        if 'nsg' in graph_types:
            nsg_data = df[df['graph_type'] == 'nsg']
            if not nsg_data.empty:
                parameters = []
                # 对于nsg，需要所有参数
                for param in ['K', 'L', 'ITER', 'S', 'R', 'L_refine', 'R_refine', 'C']:
                    valid_data = nsg_data[param][(nsg_data[param] != 0) & (~nsg_data[param].isna())]
                    if len(valid_data) > 0:
                        min_val = max(1, int(valid_data.min()))
                        max_val = max(min_val + 10, int(valid_data.max()))
                        parameters.append({
                            "name": param,
                            "min": min_val,
                            "max": max_val
                        })
                
                result.append({
                    "type": "nsg",
                    "parameters": parameters
                })
        
        # 添加其他不在预设列表中的图索引类型
        for graph_type in [gt for gt in graph_types if gt not in ['nsw', 'kgraph', 'nsg']]:
            type_data = df[df['graph_type'] == graph_type]
            parameters = []
            
            # 对于其他类型，检查所有可能的参数列
            for param in ['K', 'L', 'ITER', 'S', 'R', 'L_refine', 'R_refine', 'C']:
                if param in df.columns:
                    valid_data = type_data[param][(type_data[param] != 0) & (~type_data[param].isna())]
                    if len(valid_data) > 0:
                        min_val = max(1, int(valid_data.min()))
                        max_val = max(min_val + 10, int(valid_data.max()))
                        parameters.append({
                            "name": param,
                            "min": min_val,
                            "max": max_val
                        })
            
            result.append({
                "type": graph_type,
                "parameters": parameters
            })
        
        return jsonify(result), 200
    
    except Exception as e:
        print(f"加载图索引类型时出错: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/train_model', methods=['POST'])
def train_model():
    try:
        # 获取请求中的数据
        data = request.get_json()
        selected_datasets = data.get('datasets', [])
        training_command = data.get('command', '')
        model_info = data.get('model_info', {})
        
        if not selected_datasets:
            return jsonify({"error": "未选择任何数据集"}), 400
        
        # 读取mb.csv文件
        csv_path = args.data_dir
        df = pd.read_csv(csv_path)
        
        # 筛选出所选数据集对应的行
        filtered_df = df[df['dataset'].isin(selected_datasets)]
        
        if filtered_df.empty:
            return jsonify({"error": "未找到所选数据集对应的数据"}), 404
        
        # 确定train.csv的路径（与mb.csv在同一目录）
        train_csv_path = os.path.join(os.path.dirname(csv_path), 'train.csv')
        
        # 将筛选后的数据保存到train.csv
        filtered_df.to_csv(train_csv_path, index=False)
        
        # 创建日志目录
        log_dir = os.path.join(args.path, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建一个唯一的训练ID
        train_id = int(time.time())
        
        # 日志文件路径
        log_file = os.path.join(log_dir, f'training_log_{train_id}.txt')
        
        # 处理训练命令，确保正确定位到src_pgformer目录
        if not training_command.strip():
            # 如果没有提供训练命令，使用默认命令
            training_command = f"python {args.path}/src_pgformer/main.py --data_dir {train_csv_path} --id {train_id % 100} --device 0 --epochs 1000 --batch_size 720 --method pgformer --save_model"
        else:
            # 处理命令中的相对路径
            # 如果命令以python开头，并且后面跟着一个相对路径
            import re
            if re.match(r'^python\s+([^/].+\.py)', training_command):
                # 将相对路径替换为绝对路径，指向src_pgformer目录
                training_command = re.sub(
                    r'^python\s+([^/].+\.py)', 
                    f'python {args.path}/src_pgformer/\\1', 
                    training_command
                )
            
            # 确保命令包含data_dir参数
            if '--data_dir' not in training_command:
                training_command += f" --data_dir {train_csv_path}"
        
        # 将模型信息与训练ID保存到临时文件，等训练完成后再写入CSV
        if model_info:
            # 保存模型信息到临时文件
            model_info_temp = os.path.join(log_dir, f'model_info_temp_{train_id}.json')
            
            # 只保留命令的参数部分
            model_command = model_info.get('command', '')
            if model_command:
                # 使用正则表达式匹配并提取参数部分
                param_match = re.search(r'(?:python\s+\S+\.py\s+)?(--\S+.*)$', model_command)
                if param_match:
                    model_command = param_match.group(1).strip()
                else:
                    # 如果无法匹配，尝试直接获取第一个'--'之后的所有内容
                    pos = model_command.find('--')
                    if pos >= 0:
                        model_command = model_command[pos:].strip()
            
            # 准备模型信息
            model_data = {
                'id': model_info.get('id', ''),
                'command': model_command,
                'supported_graph_types': model_info.get('supported_graph_types', '')
            }
            
            # 保存到临时文件
            with open(model_info_temp, 'w') as f:
                import json
                json.dump(model_data, f)
        
        # 将训练ID保存到会话中，以便前端获取日志时使用
        with open(os.path.join(log_dir, 'current_train.txt'), 'w') as f:
            f.write(str(train_id))
        
        # 记录命令和日志位置
        with open(log_file, 'w') as f:
            f.write(f"训练开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据集: {', '.join(selected_datasets)}\n")
            f.write(f"训练数据: {train_csv_path}\n")
            f.write(f"执行命令: {training_command}\n")
            f.write(f"模型保存位置: {args.model_dir}/summary_pic{train_id % 100}\n")
            f.write("\n--- 训练日志 ---\n\n")
        
        # 准备在后台执行命令，确保输出实时写入日志文件
        train_cmd = f"cd {args.path} && {training_command} > {log_file} 2>&1 &"
        
        # 执行命令
        os.system(train_cmd)
        
        return jsonify({
            "success": True,
            "message": f"已将所选数据集的数据保存到 {train_csv_path}，并启动训练",
            "row_count": len(filtered_df),
            "log_file": log_file,
            "train_id": train_id
        }), 200
    
    except Exception as e:
        print(f"训练模型时出错: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_training_log', methods=['GET'])
def get_training_log():
    try:
        # 获取训练ID
        train_id = request.args.get('id')
        # 获取ensure_save参数，如果前端传入此参数为true，则强制保存模型信息
        ensure_save = request.args.get('ensure_save', 'false').lower() == 'true'
        
        # 如果没有提供ID，尝试从文件读取当前训练ID
        if not train_id:
            log_dir = os.path.join(args.path, 'logs')
            current_train_file = os.path.join(log_dir, 'current_train.txt')
            if os.path.exists(current_train_file):
                with open(current_train_file, 'r') as f:
                    train_id = f.read().strip()
            else:
                return jsonify({"error": "未找到训练ID"}), 404
        
        # 构建日志文件路径
        log_dir = os.path.join(args.path, 'logs')
        log_file = os.path.join(log_dir, f'training_log_{train_id}.txt')
        
        # 检查日志文件是否存在
        if not os.path.exists(log_file):
            return jsonify({"error": "未找到日志文件"}), 404
        
        # 读取日志文件的最后n行
        n = int(request.args.get('lines', 50))  # 默认读取最后50行
        
        # 读取文件的最后n行
        lines = []
        with open(log_file, 'r') as f:
            lines = f.readlines()
            # 只取最后n行
            lines = lines[-n:]
        
        # 检查训练是否已经完成
        training_completed = False

        # 定义训练完成标记列表
        completion_markers = [
            "Training completed", "训练完成", "Model saved", "模型已保存", 
            "Finished", "完成", "Done", "结束", "训练已完成", "成功保存模型",
            "training finished", "finished training", "done"
        ]
        
        # 检查日志内容是否包含完成标记
        for line in reversed(lines):
            if any(marker.lower() in line.lower() for marker in completion_markers):
                training_completed = True
                with open(log_file, 'a') as f:
                    f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 检测到训练完成标记，准备保存模型信息\n")
                break
        
        # 检查文件是否已经很长时间没有更新
        if not training_completed:
            log_file_stat = os.stat(log_file)
            last_modified = log_file_stat.st_mtime
            if (time.time() - last_modified) > 300:  # 如果超过5分钟没有更新
                training_completed = True
                with open(log_file, 'a') as f:
                    f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 训练日志已超过5分钟未更新，视为训练完成\n")
        
        # 如果前端传入ensure_save=true参数，强制认为训练已完成
        if ensure_save:
            training_completed = True
            with open(log_file, 'a') as f:
                f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 前端请求强制保存模型信息\n")
        
        # 如果训练已完成，将模型信息从临时文件写入CSV
        if training_completed:
            # 检查是否存在临时的模型信息文件
            model_info_temp = os.path.join(log_dir, f'model_info_temp_{train_id}.json')
            
            # 记录调试信息到日志文件
            with open(log_file, 'a') as f:
                f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 检查临时文件: {model_info_temp}, 存在: {os.path.exists(model_info_temp)}\n")
            
            if os.path.exists(model_info_temp):
                try:
                    # 读取临时文件的模型信息
                    import json
                    with open(model_info_temp, 'r') as f:
                        model_data = json.load(f)
                    
                    # 记录读取到的模型数据
                    with open(log_file, 'a') as f:
                        f.write(f"读取到的模型数据: {model_data}\n")
                    
                    # 添加训练完成时间
                    model_data['training_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
                    
                    # 模型信息CSV文件路径
                    model_info_csv = os.path.join(log_dir, 'model_info.csv')
                    file_exists = os.path.isfile(model_info_csv)
                    
                    try:
                        if file_exists:
                            # 读取现有CSV文件为DataFrame
                            df_model_info = pd.read_csv(model_info_csv)
                            
                            # 检查是否有相同ID的行
                            if 'id' in df_model_info.columns and model_data['id'] in df_model_info['id'].astype(str).values:
                                # 删除相同ID的行
                                df_model_info = df_model_info[df_model_info['id'].astype(str) != model_data['id']]
                                with open(log_file, 'a') as f:
                                    f.write(f"已删除相同ID的记录: {model_data['id']}\n")
                            
                            # 创建新的DataFrame行
                            new_row = pd.DataFrame([model_data])
                            
                            # 合并原有DataFrame和新行
                            df_model_info = pd.concat([df_model_info, new_row], ignore_index=True)
                            
                            # 保存更新后的DataFrame
                            df_model_info.to_csv(model_info_csv, index=False)
                            with open(log_file, 'a') as f:
                                f.write(f"已将模型信息更新到现有CSV文件: {model_info_csv}\n")
                        else:
                            # 文件不存在，创建新文件
                            df_model_info = pd.DataFrame([model_data])
                            df_model_info.to_csv(model_info_csv, index=False)
                            with open(log_file, 'a') as f:
                                f.write(f"已创建新的CSV文件: {model_info_csv}\n")
                        
                        # 删除临时文件
                        if os.path.exists(model_info_temp):
                            os.remove(model_info_temp)
                            with open(log_file, 'a') as f:
                                f.write(f"已删除临时文件: {model_info_temp}\n")
                        
                    except Exception as e:
                        with open(log_file, 'a') as f:
                            f.write(f"保存模型信息到CSV时出错: {str(e)}\n")
                        print(f"保存模型信息到CSV时出错: {str(e)}")
                        # 即使出错，也尝试创建新文件
                        try:
                            df_model_info = pd.DataFrame([model_data])
                            df_model_info.to_csv(model_info_csv, index=False)
                            with open(log_file, 'a') as f:
                                f.write(f"已创建新的CSV文件(错误恢复): {model_info_csv}\n")
                            
                            # 删除临时文件
                            if os.path.exists(model_info_temp):
                                os.remove(model_info_temp)
                                with open(log_file, 'a') as f:
                                    f.write(f"已删除临时文件: {model_info_temp}\n")
                        except Exception as inner_e:
                            with open(log_file, 'a') as f:
                                f.write(f"恢复创建CSV文件也失败: {str(inner_e)}\n")
                    
                except Exception as e:
                    print(f"读取模型信息临时文件时出错: {str(e)}")
                    with open(log_file, 'a') as f:
                        f.write(f"读取模型信息临时文件时出错: {str(e)}\n")
            else:
                with open(log_file, 'a') as f:
                    f.write(f"未找到模型信息临时文件: {model_info_temp}\n")
        
        return jsonify({
            "log_content": "".join(lines),
            "completed": training_completed
        }), 200
    
    except Exception as e:
        print(f"获取训练日志时出错: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/choose_file', methods=['POST'])
def choose_file():
    data = request.get_json()
    
    name=data['name']
    name = name.split(".")[0]
    tmp0=name.split("_")[0]
    tmp1=name.split("_")[1]
    
    if tmp0=="mnist":
        name=tmp0+"6W_"+tmp1
    else:
        name=tmp0+"1M_"+tmp1
    
    road=args.path+'/dataset/data/'+name+'/'+name+f'_base.fvecs'
    result=read_fvecs(road)
    
    road1=args.path+'/dataset/data/'+name+'/'+name+f'_query.fvecs'
    result1=read_fvecs(road1)
    
    # 构造返回结果
    return jsonify({
        "status": "success",
        "num": int(result.shape[0]+result1.shape[0]),
        "dim": int(result.shape[1])
    }), 200

def numpy_linspace(start, stop, num=50, endpoint=True):
    return np.linspace(start, stop, num=num, endpoint=endpoint).tolist()

@app.route('/generate_index', methods=['POST'])
def generate_index():
    # try:
    start_time = time.time()
    data = request.get_json()
    
    print(data)
    
    # 验证必需参数
    if not data or 'type' not in data:
        return jsonify({"error": "Invalid request"}), 400
    
    # 处理不同索引类型
    index_type = data['type']
    parameters = data.get('parameters', {})
    
    K0=K1=0
    L0=L1=0
    ITER0=ITER1=0
    S0=S1=0
    R0=R1=0
    LR0=LR1=0
    RR0=RR1=0
    C0=C1=0
    
    K0=parameters['K'][0]
    K1=parameters['K'][1]
    L0=parameters['L'][0]
    L1=parameters['L'][1]
    if index_type != 'nsw':
        ITER0=parameters['ITER'][0]
        ITER1=parameters['ITER'][1]
        S0=parameters['S'][0]
        S1=parameters['S'][1]
        R0=parameters['R'][0]
        R1=parameters['R'][1]
    
    if index_type == 'nsg':
        LR0=parameters['L_refine'][0]
        LR1=parameters['L_refine'][1]
        RR0=parameters['R_refine'][0]
        RR1=parameters['R_refine'][1]
        C0=parameters['C'][0]
        C1=parameters['C'][1]    
        
    k=data['search_k']
    recall_val=data['recall_min']
    optimize_target=data['optimize_target']
    name=data['name']
    name = name.split("：")[1]
    tmp0=name.split("_")[0]
    tmp1=name.split("_")[1]
    
    if tmp0=="mnist":
        name=tmp0+"6W_"+tmp1
    else:
        name=tmp0+"1M_"+tmp1    
    
    dataset=name
    
    
    
    K=numpy_linspace(K0,K1,5)
    K = [int(x) for x in K]
    print("K",K)
    
    L=numpy_linspace(L0,L1,5)
    L = [int(x) for x in L]
    print("L",L)
    
    ITER=numpy_linspace(ITER0,ITER1,5)
    ITER = [int(x) for x in ITER]
    print("ITER",ITER)
    
    S=numpy_linspace(S0,S1,5)
    S = [int(x) for x in S]
    print("S",S)

    R=numpy_linspace(R0,R1,5)
    R = [int(x) for x in R]
    print("R",R)

    LR=numpy_linspace(LR0,LR1,5)
    LR = [int(x) for x in LR]
    print("LR",LR)

    RR=numpy_linspace(RR0,RR1,5)
    RR = [int(x) for x in RR]
    print("RR",RR)
    
    C=numpy_linspace(C0,C1,5)
    C = [int(x) for x in C]
    print("C",C)
    
    # 生成笛卡尔积
    cartesian_product = itertools.product(K, L, ITER, S, R, LR, RR, C)

    # 转换为DataFrame
    df = pd.DataFrame(cartesian_product, columns=['K', 'L', 'ITER', 'S', 'R', 'L_refine', 'R_refine', 'C'])
    
    road=args.path+'/dataset/data/'+dataset+'/'+dataset+f'_base.fvecs'
    node_feat=read_fvecs(road)
    
    prefix_df = pd.DataFrame({
        'dataset': [dataset]*len(df),
        'size': [tmp1]*len(df),
        'dim': [node_feat.shape[1]]*len(df)
    })

    df = pd.concat([prefix_df, df], axis=1)
    
    back_df = pd.DataFrame({
        'k': [k]*len(df)
    })

    df = pd.concat([df, back_df], axis=1)
    
    if index_type=='nsg':
        back_df = pd.DataFrame({
            'graph_type_kgraph': [0]*len(df),
            'graph_type_nsg': [1]*len(df),
            'graph_type_nsw': [0]*len(df)
        })
        df = pd.concat([df, back_df], axis=1)
    if index_type=='nsw':
        back_df = pd.DataFrame({
            'graph_type_kgraph': [0]*len(df),
            'graph_type_nsg': [0]*len(df),
            'graph_type_nsw': [1]*len(df)
        })
        df = pd.concat([df, back_df], axis=1)
    if index_type=='kgraph':
        back_df = pd.DataFrame({
            'graph_type_kgraph': [1]*len(df),
            'graph_type_nsg': [0]*len(df),
            'graph_type_nsw': [0]*len(df)
        })
        df = pd.concat([df, back_df], axis=1)
    
    df['size']=(df['size'].astype(int) - min_value_size)/max_value_size
    df['dim']=(df['dim'].astype(int) - min_value_dim)/max_value_dim
    df['K']=(df['K'].astype(int) - min_value_K)/max_value_K
    df['L']=(df['L'].astype(int) - min_value_L)/max_value_L
    df['ITER']=(df['ITER'].astype(int) - min_value_ITER)/max_value_ITER
    df['S']=(df['S'].astype(int) - min_value_S)/max_value_S
    df['R']=(df['R'].astype(int) - min_value_R)/max_value_R
    df['L_refine']=(df['L_refine'].astype(int) - min_value_L_refine)/max_value_L_refine
    df['R_refine']=(df['R_refine'].astype(int) - min_value_R_refine)/max_value_R_refine
    df['C']=(df['C'].astype(int) - min_value_C)/max_value_C
    df['k']=(df['k'].astype(int) - min_value_k)/max_value_k
    
    print(df)
    
    now_mb=df
    
    # if index_type=='nsg':
    #     data=test[(test.dataset == dataset) & (test.graph_type_nsg == 1)]
    #     now_mb=test[(test.dataset == dataset) & (test.graph_type_nsg == 1)]
    # if index_type=='nsw':
    #     data=test[(test.dataset == dataset) & (test.graph_type_nsw == 1)]
    #     now_mb=test[(test.dataset == dataset) & (test.graph_type_nsw == 1)]
    # if index_type=='kgraph':
    #     data=test[(test.dataset == dataset) & (test.graph_type_kgraph == 1)]
    #     now_mb=test[(test.dataset == dataset) & (test.graph_type_kgraph == 1)]
    
    # data.drop(target, axis=1, inplace=True)
    
    # target0=now_mb[target[0]]
    # target1=now_mb[target[1]]
    # target2=now_mb[target[2]]
    now=df
    now_mb=now_mb.drop('dataset', axis=1)
    # now=now.drop(target , axis=1)
    
    print("now",now_mb.shape)
    
    now_mb = torch.from_numpy(now_mb.values).float()
    now_mb=now_mb.to(device)
    
    # target0 = torch.from_numpy(target0.values)
    # target0=target0.to(device)
    # target1 = torch.from_numpy(target1.values)
    # target1=target1.to(device)
    # target2 = torch.from_numpy(target2.values)
    # target2=target2.to(device)

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

        if not args.directed:
            print("无向图")
            edge_index = to_undirected(edge_index)
        
        ### Adj storage for relational bias ###
        adj, _ = remove_self_loops(edge_index)
        adj, _ = add_self_loops(adj, num_nodes=n)
        #print(adj.shape) #(2,24246)
        
        # 保存邻接矩阵
        np.savetxt(args.path+'/dataset/data/'+dataset+'/'+dataset+f'_{args.knn_num}_adj_matrix.txt', adj, fmt='%d')
        adj=torch.from_numpy(adj)
        adj=adj.to(device)
    
    with torch.no_grad():
        out = model(node_feat, adj, now_mb)
    
    
    
    back_now = pd.DataFrame({
        'recall': out[:,0].cpu().detach().numpy(),
        'qps': out[:,1].cpu().detach().numpy(),
        'distcomp': out[:,2].cpu().detach().numpy()
    })
    now = pd.concat([now, back_now], axis=1)
    
    now['qps']=now['qps'] * max_value_qps + min_value_qps
    now['distcomp']=now['distcomp'] * max_value_distcomp + min_value_distcomp
    
    data=now
    now = now[now['recall'] > recall_val]
    
    if optimize_target=='内存使用':
        now = now.sort_values(
            by=['K'],
            ascending=[True],
            na_position='first'  # 将NaN放在最前
        )
        
        if now.shape[0] == 0:
            now = data
            
            now = now.sort_values(
                by=['recall', 'K'],
                ascending=[False, True],
                na_position='first'  # 将NaN放在最前
            )
    elif optimize_target=='查询时间':
        now = now.sort_values(
            by=['qps'],
            ascending=[False],
            na_position='first'  # 将NaN放在最前
        )
        
        if now.shape[0] == 0:
            now = data
            
            now = now.sort_values(
                by=['recall', 'qps'],
                ascending=[False, False],
                na_position='first'  # 将NaN放在最前
            )
    else:
        now = now.sort_values(
            by=['distcomp'],
            ascending=[True],
            na_position='first'  # 将NaN放在最前
        )
        
        if now.shape[0] == 0:
            now = data
            
            now = now.sort_values(
                by=['recall', 'distcomp'],
                ascending=[False, True],
                na_position='first'  # 将NaN放在最前
            )
    
    tmp=now.iloc[0]
    # ans=test[(test['dataset'] == dataset) & (round(test['k']* max_value_k + min_value_k) == k) & (test['K'] == tmp['K']) & (test['L'] == tmp['L']) & (test['ITER'] == tmp['ITER']) & (test['S'] == tmp['S']) & (test['R'] == tmp['R']) & (test['L_refine'] == tmp['L_refine']) & (test['R_refine'] == tmp['R_refine']) & (test['C'] == tmp['C'])]
    # ans=ans.iloc[0]
    ans=tmp
    
    ans['K']=ans['K']*max_value_K+min_value_K
    ans['K']=ans['K'].astype(int)
    ans['L']=ans['L']*max_value_L+min_value_L
    ans['L']=ans['L'].astype(int)
    ans['ITER']=ans['ITER']*max_value_ITER+min_value_ITER
    ans['ITER']=ans['ITER'].astype(int)
    ans['S']=ans['S']*max_value_S+min_value_S
    ans['S']=ans['S'].astype(int)
    ans['R']=ans['R']*max_value_R+min_value_R
    ans['R']=ans['R'].astype(int)
    ans['L_refine']=ans['L_refine']*max_value_L_refine+min_value_L_refine
    ans['L_refine']=ans['L_refine'].astype(int)
    ans['R_refine']=ans['R_refine']*max_value_R_refine+min_value_R_refine
    ans['R_refine']=ans['R_refine'].astype(int)
    ans['C']=ans['C']*max_value_C+min_value_C
    ans['C']=ans['C'].astype(int)
    
    # ans['qps']=ans['qps']*max_value_qps+min_value_qps
    # ans['distcomp']=ans['distcomp']*max_value_distcomp+min_value_distcomp
    
    print(dataset,ans['K'],ans['L'],ans['ITER'],ans['S'],ans['R'],ans['L_refine'],ans['R_refine'],ans['C'],ans['recall'],ans['qps'],ans['distcomp'])
    
    if index_type=='nsg':           
        parameters = [
            {"key": "K", "value": int(ans['K'])},
            {"key": "L", "value": int(ans['L'])},
            {"key": "ITER", "value": int(ans['ITER'])},
            {"key": "S", "value": int(ans['S'])},
            {"key": "R", "value": int(ans['R'])},
            {"key": "L_refine", "value": int(ans['L_refine'])},
            {"key": "R_refine", "value": int(ans['R_refine'])},
            {"key": "C", "value": int(ans['C'])}
        ]
    if index_type=='nsw':
        parameters = [
            {"key": "K", "value": int(ans['K'])},
            {"key": "L", "value": int(ans['L'])}
        ]
    if index_type=='kgraph':
        parameters = [
            {"key": "K", "value": int(ans['K'])},
            {"key": "L", "value": int(ans['L'])},
            {"key": "ITER", "value": int(ans['ITER'])},
            {"key": "S", "value": int(ans['S'])},
            {"key": "R", "value": int(ans['R'])}
        ]
    
    # 构造返回结果
    return jsonify({
        "status": "success",
        # "type": index_type,
        "parameters": parameters,
        "recall": float(ans['recall']),
        # "file_path": f"/data/{index_type}_index_{datetime.now().strftime('%Y%m%d%H%M%S')}.index",
        "time_cost": float(round(time.time() - start_time, 2))
    }), 200
        
    # except Exception as e:
    #     return jsonify({
    #         "status": "error",
    #         "message": str(e)
    #     }), 500

if __name__ == '__main__':
    app.run(port=5001, debug=False)
