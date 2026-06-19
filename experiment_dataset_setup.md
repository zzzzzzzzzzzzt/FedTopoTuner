# PGFormer 实验数据设置（模态与数据集划分）

本文档说明当前 PGFormer 实验中训练数据集与测试数据集的来源文件，以及它们按模态（图像 / 文本 / 音频 / 合成）和数据集前缀划分后的具体数据集列表。所有路径均基于仓库根目录 `pgformer/`。

---

## 1. 数据源文件

- 训练数据源（联邦训练的母数据集）
  - 文件路径：`data/mb.csv`
  - 用途：作为联邦训练的整体数据池，在 `fedv4.py` 中读取并按不同划分策略分配给各客户端。
  - 读取方式（示意）：
    - `mb = read_dataset(args.data_dir)`，其中 `args.data_dir` 通常为 `/root/autodl-tmp/tyk/pgformer/data/mb.csv`。

- 测试数据源（集中测试数据集）
  - 文件路径：`data/test.csv`
  - 用途：在训练结束后，用于整体模型评估（`evaluate_test`）。
  - 读取方式（示意）：
    - `test = read_dataset(args.test_dir)`，其中 `args.test_dir` 通常为 `/root/autodl-tmp/tyk/pgformer/data/test.csv`。

---

## 2. 模态与前缀定义

根据常用 ANN benchmark 语义和现有代码中的划分逻辑，当前实验中使用的模态与前缀映射如下：

- 图像模态（Image）
  - 前缀：`sift1M`, `deep1M`, `gist1M`
  - 前缀：`Histogram`, `Moments`, `Texture`
  - 前缀：`mnist6W`

- 文本模态（Text）
  - 前缀：`crawl1M`, `uqv1M`

- 音频模态（Audio）
  - 前缀：`audio`, `msong10W`

- 合成数据（Synthetic / 其它）
  - 前缀：`synthetic1M`

在 `fedv4.py` 中新增的模态划分函数 `split_dataset_by_modality` 使用的是上述前缀映射，将图像、文本、音频三类分别划分到三个客户端。

---

## 3. 训练数据集（data/mb.csv）

- 文件：`data/mb.csv`
- 总行数：`121485`
- 唯一 `dataset` 数量：`70`
- 说明：包含图像 / 文本 / 音频三类模态的多种规模配置，不包含 `deep1M_*`、`mnist6W_*`、`synthetic1M_*`。

### 3.1 图像模态（Image）

当前训练数据源中的图像模态数据集来自以下前缀：

- 前缀 `Histogram`
  - `Histogram_500`
  - `Histogram_1000`
  - `Histogram_2000`
  - `Histogram_4000`
  - `Histogram_8000`
  - `Histogram_16000`
  - `Histogram_32000`
  - `Histogram_64000`
  - `Histogram_68040`

- 前缀 `Moments`
  - `Moments_500`
  - `Moments_1000`
  - `Moments_2000`
  - `Moments_4000`
  - `Moments_8000`
  - `Moments_16000`
  - `Moments_32000`
  - `Moments_64000`
  - `Moments_68040`

- 前缀 `Texture`
  - `Texture_500`
  - `Texture_1000`
  - `Texture_2000`
  - `Texture_4000`
  - `Texture_8000`
  - `Texture_16000`
  - `Texture_32000`
  - `Texture_64000`
  - `Texture_68040`

- 前缀 `sift1M`
  - `sift1M_1000`
  - `sift1M_5000`
  - `sift1M_10000`
  - `sift1M_50000`
  - `sift1M_100000`
  - `sift1M_500000`
  - `sift1M_1000000`

- 前缀 `gist1M`
  - `gist1M_1000`
  - `gist1M_5000`
  - `gist1M_10000`
  - `gist1M_50000`
  - `gist1M_100000`

### 3.2 文本模态（Text）

训练数据源中的文本模态数据集来自以下前缀：

- 前缀 `crawl1M`
  - `crawl1M_1000`
  - `crawl1M_5000`
  - `crawl1M_10000`
  - `crawl1M_50000`
  - `crawl1M_100000`
  - `crawl1M_500000`
  - `crawl1M_1000000`

- 前缀 `uqv1M`
  - `uqv1M_1000`
  - `uqv1M_5000`
  - `uqv1M_10000`
  - `uqv1M_50000`
  - `uqv1M_100000`
  - `uqv1M_500000`
  - `uqv1M_1000000`

### 3.3 音频模态（Audio）

训练数据源中的音频模态数据集来自以下前缀：

- 前缀 `audio`
  - `audio_500`
  - `audio_1000`
  - `audio_2000`
  - `audio_4000`
  - `audio_8000`
  - `audio_16000`
  - `audio_32000`
  - `audio_53387`

- 前缀 `msong10W`
  - `msong10W_500`
  - `msong10W_1000`
  - `msong10W_2000`
  - `msong10W_4000`
  - `msong10W_8000`
  - `msong10W_16000`
  - `msong10W_32000`
  - `msong10W_64000`
  - `msong10W_100000`

---

## 4. 测试数据集（data/test.csv）

- 文件：`data/test.csv`
- 总行数：`34344`
- 唯一 `dataset` 数量：`22`
- 说明：测试数据主要覆盖图像模态的 `deep1M` 和 `mnist6W`，以及合成模态的 `synthetic1M`，不包含文本或音频前缀。

### 4.1 图像模态（Image）

测试数据源中的图像模态数据集来自以下前缀：

- 前缀 `deep1M`
  - `deep1M_1000`
  - `deep1M_5000`
  - `deep1M_10000`
  - `deep1M_50000`
  - `deep1M_100000`
  - `deep1M_500000`
  - `deep1M_1000000`

- 前缀 `mnist6W`
  - `mnist6W_500`
  - `mnist6W_1000`
  - `mnist6W_2000`
  - `mnist6W_4000`
  - `mnist6W_8000`
  - `mnist6W_16000`
  - `mnist6W_32000`
  - `mnist6W_60000`

### 4.2 合成数据（Synthetic）

测试数据源中的合成数据集来自以下前缀：

- 前缀 `synthetic1M`
  - `synthetic1M_1000`
  - `synthetic1M_5000`
  - `synthetic1M_10000`
  - `synthetic1M_50000`
  - `synthetic1M_100000`
  - `synthetic1M_500000`
  - `synthetic1M_1000000`

---

## 5. 与联邦划分策略的关系（fedv4）

在 `src_pgformer/fedv4.py` 中，数据集的模态和前缀被用于多种数据划分策略：

- `split_dataset_v1`：按数据集类型划分为 12 个客户端（每种前缀大致对应一个客户端）。
- `split_dataset_v2`：按前缀分组划分为 5 个客户端（高维向量、文本/合成、图像特征、音频、其它）。
- `split_dataset_v3`：每个具体 `dataset` 实例单独作为一个客户端。
- `split_dataset_by_modality`：按模态划分为 3 个客户端（图像 / 文本 / 音频）。

通过这些策略，可以在同一数据源配置下，研究不同模态和前缀组合对联邦学习收敛行为和性能的影响。

