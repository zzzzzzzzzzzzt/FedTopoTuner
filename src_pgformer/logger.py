import os
import json
import time
from datetime import datetime

class TrainingLogger:
    def __init__(self, log_dir='/root/autodl-tmp/tyk/pgformer/logs', training_id=None):
        self.base_log_dir = log_dir
        self.timestamp = int(time.time())
        
        # 如果提供了training_id，创建以id命名的子文件夹
        if training_id is not None:
            self.log_dir = os.path.join(log_dir, f'training_{training_id}')
        else:
            self.log_dir = log_dir
            
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f'training_log_{self.timestamp}.txt')
        self.model_info_file = os.path.join(self.log_dir, f'model_info_temp_{self.timestamp}.json')
        
    def log_parameters(self, args):
        """记录训练参数"""
        with open(self.model_info_file, 'w', encoding='utf-8') as f:
            params = {
                'timestamp': self.timestamp,
                'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_parameters': {
                    'method': args.method,
                    'hidden_channels': args.hidden_channels,
                    'num_layers': args.num_layers,
                    'num_heads': args.num_heads,
                    'dropout': args.dropout,
                    'alpha': args.alpha,
                    'kernel': args.kernel,
                    'use_bn': args.use_bn,
                    'use_residual': args.use_residual,
                    'use_graph': args.use_graph,
                    'use_weight': args.use_weight
                },
                'training_parameters': {
                    'split_version': getattr(args, 'split_version', None),
                    'num_clients': getattr(args, 'num_clients', None),
                    'comm_rounds': getattr(args, 'comm_rounds', None),
                    'client_fraction': getattr(args, 'client_fraction', None),
                    'epochs': args.epochs,
                    'batch_size': args.batch_size,
                    'learning_rate': args.lr,
                    'weight_decay': args.weight_decay
                },
                'data_parameters': {
                    'input_dim': args.input,
                    'output_dim': args.output,
                    'knn_num': args.knn_num,
                    'data_dir': args.data_dir,
                    'test_dir': args.test_dir
                },
                'other_parameters': {
                    'seed': args.seed,
                    'device': args.device,
                    'save_model': args.save_model,
                    'model_dir': args.model_dir
                }
            }
            json.dump(params, f, indent=4, ensure_ascii=False)
            
    def log_training_info(self, message):
        """记录训练过程信息"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f'[{timestamp}] {message}\n')
            
    def log_metrics(self, round_num, metrics):
        """记录训练指标"""
        message = f'Round {round_num} - '
        message += ' | '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        self.log_training_info(message)