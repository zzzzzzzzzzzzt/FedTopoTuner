# PGFormer - Performance Graph Transformer for Federated Learning

## 项目简介

PGFormer是一个基于图神经网络和联邦学习的性能预测框架，专门用于预测近似最近邻搜索算法的性能指标（召回率、QPS、距离计算次数）。

## 主要特性

- **联邦学习框架**: 实现了多种聚合策略，包括FedAvg、performance_based等
- **图神经网络**: 支持多种GNN模型（GCN、GAT、SGC等）
- **性能预测**: 预测ANN算法的recall、qps、distcomp等关键指标
- **多数据集支持**: 支持多种向量数据集的性能分析

## 项目结构

```
pgformer/
├── src_pgformer/          # 核心源代码
│   ├── fedv3.py          # 联邦学习主要实现
│   ├── gnns.py           # 图神经网络模型
│   ├── utils.py          # 工具函数
│   ├── pgformer.py       # PGFormer模型
│   └── ...
├── data/                 # 数据文件
├── logs/                 # 训练日志
├── model/                # 模型保存目录
├── tools/                # 工具脚本
└── demo_performance_based.py  # 性能基础聚合策略演示
```

## 安装依赖

```bash
pip install torch torch-geometric
pip install numpy pandas scikit-learn
pip install faiss-cpu  # 或 faiss-gpu
```

## 使用方法

### 1. 运行联邦学习训练

```bash
cd src_pgformer
python fedv3.py --aggregation_strategy performance_based --num_rounds 20
```

### 2. 测试性能基础聚合策略

```bash
python test_performance_based.py
```

### 3. 运行演示脚本

```bash
python demo_performance_based.py
```

## 聚合策略

### Performance-Based 聚合策略

该策略根据客户端模型在测试数据集上的性能来分配聚合权重：

- 评估每个客户端模型的recall、qps、distcomp性能
- 计算加权性能分数（权重：[3, 3, 8]）
- 使用性能分数的倒数作为聚合权重
- 进行归一化处理

详细说明请参考：[PERFORMANCE_BASED_STRATEGY.md](PERFORMANCE_BASED_STRATEGY.md)

## 主要文件说明

- `fedv3.py`: 联邦学习主框架，包含多种聚合策略
- `gnns.py`: 图神经网络模型实现
- `utils.py`: 数据加载、图构建等工具函数
- `pgformer.py`: PGFormer Transformer模型
- `demo_performance_based.py`: 性能基础聚合策略演示
- `test_performance_based.py`: 性能基础聚合策略测试

## 实验结果

项目包含多种模型训练对比图表，展示了不同聚合策略和模型的性能表现。

## 贡献

欢迎提交Issue和Pull Request来改进项目。

## 许可证

本项目采用MIT许可证。