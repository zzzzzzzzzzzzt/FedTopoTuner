pgformer 文档说明
==================

本目录用于存放与 pgformer 相关的文档与公式推导。

当前文件
--------
- Shapley.pdf  
  - 参考文献/方法说明：关于 Shapley 值的原理介绍。
- PerformanceBased_ShAP_Contribution.md  
  - 描述在 fedv4 中实现的基于 Shapley 的贡献度计算方法和 performance_based 聚合策略。

fedv4 联邦训练与测试使用说明
-----------------------------

fedv4 支持将“训练”和“测试”拆分为两个独立阶段，通过参数 `--phase` 控制：

- `--phase train`：只进行联邦训练，不在每一轮后使用 test.csv 做推理
- `--phase test`：只进行测试，在训练完成后对 test.csv 中每个 dataset 单独计算 Loss

### 1. 训练阶段（phase=train）

- 作用：
  - 从 `mb.csv` 中加载训练数据，按 `split_version` 划分到多个客户端。
  - 对每个客户端内部再划分本地训练/验证集。
  - 若使用 `performance_based`，会从训练数据中均匀抽取 15% 作为“全局测试集”用于 Shapley 值计算。
  - 运行若干轮联邦训练，仅在训练结束后（若指定 `--save_model`）保存最终模型。

- 示例命令（fedavg）：

  ```bash
  python src_pgformer/fedv4.py \
    --phase train \
    --id 201 \
    --aggregation_strategy fedavg \
    --save_model \
    --model_dir /root/autodl-tmp/tyk/pgformer/experiments/fedv4_aggregation_comparison/models
  ```

- 示例命令（performance_based）：

  ```bash
  python src_pgformer/fedv4.py \
    --phase train \
    --id 202 \
    --aggregation_strategy performance_based \
    --save_model \
    --model_dir /root/autodl-tmp/tyk/pgformer/experiments/fedv4_aggregation_comparison/models
  ```

### 2. 测试阶段（phase=test）

- 作用：
  - 从 `model_dir/summary_pic{id}/{method}_{id}.pkl` 加载训练好的模型。
  - 再次读取并归一化 `mb.csv` 和 `test.csv`。
  - 对 test.csv 中的每一个 dataset（例如 mnist6W_*, deep1M_*, synthetic1M_* 等）单独计算 Loss。
  - 将每个 dataset 的损失写入一个表格（CSV），便于后续画曲线和对比。

- 输出表格路径（示例）：
  - `/root/autodl-tmp/tyk/pgformer/experiments/fedv4_aggregation_comparison/test_results_id201_fedavg.csv`
  - `/root/autodl-tmp/tyk/pgformer/experiments/fedv4_aggregation_comparison/test_results_id202_performance_based.csv`

  表格列包括：
  - `dataset`
  - `loss_recall`
  - `loss_qps`
  - `loss_distcomp`
  - `loss_combined`（3:3:8 加权组合）

- 示例命令（使用训练阶段保存的模型进行测试）：

  ```bash
  python src_pgformer/fedv4.py \
    --phase test \
    --id 201 \
    --aggregation_strategy fedavg \
    --model_dir /root/autodl-tmp/tyk/pgformer/experiments/fedv4_aggregation_comparison/models
  ```

  ```bash
  python src_pgformer/fedv4.py \
    --phase test \
    --id 202 \
    --aggregation_strategy performance_based \
    --model_dir /root/autodl-tmp/tyk/pgformer/experiments/fedv4_aggregation_comparison/models
  ```

通过上述方式，可以：
- 在训练阶段专注于联邦聚合策略的优化；
- 在测试阶段统一、细粒度地评估每个 dataset 上的性能，并把结果整理成表格用于绘图和对比分析。

