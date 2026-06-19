#!/usr/bin/env python3
"""
实验对比脚本：对比集中式训练和多种联邦学习策略的性能
支持对比：实验104(集中式)、实验105(FedSGD)、实验106(FedAvg)、实验107(Performance-Based)
"""

import os
import re
import glob
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

def parse_training_log(log_path):
    """解析训练日志文件，提取关键指标"""
    metrics = []

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 尝试匹配包含train_loss的格式（main.py）
            match = re.search(r'Round (\d+) - train_loss: ([\d.]+) \| val_recall: ([\d.]+) \| val_qps: ([\d.]+) \| val_distcomp: ([\d.]+) \| test_recall: ([\d.]+) \| test_qps: ([\d.]+) \| test_distcomp: ([\d.]+)', line)
            if match:
                round_num = int(match.group(1))
                metrics.append({
                    'round': round_num,
                    'train_loss': float(match.group(2)),
                    'val_recall': float(match.group(3)),
                    'val_qps': float(match.group(4)),
                    'val_distcomp': float(match.group(5)),
                    'test_recall': float(match.group(6)),
                    'test_qps': float(match.group(7)),
                    'test_distcomp': float(match.group(8))
                })
                continue

            # 尝试匹配不包含train_loss的格式（fedv3.py）
            match = re.search(r'Round (\d+) - val_recall: ([\d.]+) \| test_recall: ([\d.]+) \| val_qps: ([\d.]+) \| test_qps: ([\d.]+) \| val_distcomp: ([\d.]+) \| test_distcomp: ([\d.]+)', line)
            if match:
                round_num = int(match.group(1))
                metrics.append({
                    'round': round_num,
                    'train_loss': None,
                    'val_recall': float(match.group(2)),
                    'test_recall': float(match.group(3)),
                    'val_qps': float(match.group(4)),
                    'test_qps': float(match.group(5)),
                    'val_distcomp': float(match.group(6)),
                    'test_distcomp': float(match.group(7))
                })
                continue

            # 尝试匹配只有测试集指标的格式（新的联邦学习日志）
            match = re.search(r'Round (\d+) - test_recall: ([\d.]+) \| test_qps: ([\d.]+) \| test_distcomp: ([\d.]+)', line)
            if match:
                round_num = int(match.group(1))
                metrics.append({
                    'round': round_num,
                    'train_loss': None,
                    'val_recall': None,
                    'test_recall': float(match.group(2)),
                    'val_qps': None,
                    'test_qps': float(match.group(3)),
                    'val_distcomp': None,
                    'test_distcomp': float(match.group(4))
                })

    return metrics

def generate_comparison_report(experiments, save_path):
    """生成对比报告"""
    report = []
    report.append("=" * 100)
    report.append("EXPERIMENT COMPARISON REPORT")
    report.append("=" * 100)
    report.append("")

    # 显示实验概况
    report.append("## Experiments Overview")
    report.append("-" * 100)
    for exp in experiments:
        report.append(f"\n{exp['name']}:")
        report.append(f"  Type:        {exp['type']}")
        report.append(f"  Rounds:      {len(exp['data'])}")
        report.append(f"  Log File:    {exp['log_path']}")

    # 1. 最终性能对比
    report.append("\n\n## 1. Final Performance Comparison (Round 20)")
    report.append("-" * 100)

    metrics = [
        ('test_recall', 'Recall (Test)'),
        ('test_qps', 'QPS (Test)'),
        ('test_distcomp', 'DistComp (Test)')
    ]

    for metric, name in metrics:
        report.append(f"\n{name}:")

        values = []
        for exp in experiments:
            if exp['data']:
                val = exp['data'][-1][metric]
                values.append((exp['name'], val))
                report.append(f"  {exp['name']:20s}: {val:.6f}")

        # 找出最优值
        if values:
            best = min(values, key=lambda x: x[1])
            report.append(f"  {'Best':20s}: {best[0]} ({best[1]:.6f})")

    # 2. 收敛速度对比
    report.append("\n\n## 2. Convergence Speed Comparison")
    report.append("-" * 100)

    def rounds_to_target(data, metric, target_ratio=0.9):
        """找到达到最终性能90%所需的轮次"""
        if not data:
            return None
        best_val = min([d[metric] for d in data])
        target_val = best_val / target_ratio
        for d in data:
            if d[metric] <= target_val:
                return d['round']
        return len(data)

    report.append("\nRounds to reach 90% of best performance:")
    for metric, name in [('test_recall', 'Recall')]:
        report.append(f"\n  {name}:")
        for exp in experiments:
            rounds = rounds_to_target(exp['data'], metric)
            if rounds:
                report.append(f"    {exp['name']:20s}: {int(rounds)} rounds")

    # 3. 训练稳定性
    report.append("\n\n## 3. Training Stability (Std Dev of Last 10 Rounds)")
    report.append("-" * 100)

    for metric, name in metrics:
        report.append(f"\n{name}:")
        stds = []
        for exp in experiments:
            if len(exp['data']) >= 10:
                last_10 = [d[metric] for d in exp['data'][-10:]]
                std = (sum((x - sum(last_10)/len(last_10))**2 for x in last_10) / len(last_10)) ** 0.5
                stds.append((exp['name'], std))
                report.append(f"  {exp['name']:20s}: {std:.6f}")

        if stds:
            best = min(stds, key=lambda x: x[1])
            report.append(f"  {'Most Stable':20s}: {best[0]} ({best[1]:.6f})")

    # 4. 总结
    report.append("\n\n## 4. Summary")
    report.append("-" * 100)

    # 计算综合性能分数（使用权重 3:3:8）
    def compute_score(data):
        if not data:
            return None
        recall = data[-1]['test_recall']
        qps = data[-1]['test_qps']
        distcomp = data[-1]['test_distcomp']
        return (3 * recall + 3 * qps + 8 * distcomp) / 14

    report.append(f"\nWeighted Performance Score (weights: 3:3:8 for recall:qps:distcomp):")
    scores = []
    for exp in experiments:
        score = compute_score(exp['data'])
        if score:
            scores.append((exp['name'], score))
            report.append(f"  {exp['name']:20s}: {score:.6f}")

    if scores:
        best = min(scores, key=lambda x: x[1])
        report.append(f"\n✓ Best Overall: {best[0]} (score: {best[1]:.6f})")

    report.append("\n" + "=" * 100)

    # 保存报告
    report_text = "\n".join(report)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n✅ Comparison report saved to: {save_path}")
    print("\n" + report_text)

    return report_text

def plot_convergence_curves(experiments, output_dir):
    """绘制收敛曲线对比图"""
    metrics = [
        ('test_recall', 'Recall (Test)', 'lower'),
        ('test_qps', 'QPS (Test)', 'lower'),
        ('test_distcomp', 'DistComp (Test)', 'lower')
    ]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']

    # 为每个指标创建单独的图
    for metric, title, _ in metrics:
        plt.figure(figsize=(12, 8))

        for idx, exp in enumerate(experiments):
            if not exp['data']:
                continue

            rounds = [d['round'] for d in exp['data']]
            values = [d[metric] for d in exp['data']]

            plt.plot(rounds, values,
                    marker=markers[idx % len(markers)],
                    color=colors[idx % len(colors)],
                    label=exp['name'],
                    linewidth=2.5,
                    markersize=8,
                    alpha=0.8)

        plt.xlabel('Epoch / Communication Round', fontsize=14, fontweight='bold')
        plt.ylabel(title, fontsize=14, fontweight='bold')
        plt.title(f'{title} - Convergence Comparison', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12, loc='best', framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()

        # 保存图片
        save_path = f"{output_dir}/convergence_{metric}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved: {save_path}")
        plt.close()

    # 创建综合对比图（2x2子图）
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comprehensive Convergence Comparison', fontsize=18, fontweight='bold')

    # 三个指标 + 综合得分
    plot_configs = [
        ('test_recall', 'Recall (Test)', axes[0, 0]),
        ('test_qps', 'QPS (Test)', axes[0, 1]),
        ('test_distcomp', 'DistComp (Test)', axes[1, 0]),
    ]

    for metric, title, ax in plot_configs:
        for idx, exp in enumerate(experiments):
            if not exp['data']:
                continue

            rounds = [d['round'] for d in exp['data']]
            values = [d[metric] for d in exp['data']]

            ax.plot(rounds, values,
                   marker=markers[idx % len(markers)],
                   color=colors[idx % len(colors)],
                   label=exp['name'],
                   linewidth=2,
                   markersize=6,
                   alpha=0.8)

        ax.set_xlabel('Round', fontsize=12, fontweight='bold')
        ax.set_ylabel(title, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    # 第四个子图：综合得分
    ax = axes[1, 1]
    for idx, exp in enumerate(experiments):
        if not exp['data']:
            continue

        rounds = [d['round'] for d in exp['data']]
        scores = [(3*d['test_recall'] + 3*d['test_qps'] + 8*d['test_distcomp'])/14
                  for d in exp['data']]

        ax.plot(rounds, scores,
               marker=markers[idx % len(markers)],
               color=colors[idx % len(colors)],
               label=exp['name'],
               linewidth=2,
               markersize=6,
               alpha=0.8)

    ax.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax.set_ylabel('Weighted Score', fontsize=12, fontweight='bold')
    ax.set_title('Comprehensive Score (3:3:8)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = f"{output_dir}/convergence_comprehensive.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Comprehensive plot saved: {save_path}")
    plt.close()

def plot_final_comparison_bars(experiments, output_dir):
    """绘制最终性能柱状图对比"""
    import numpy as np

    metrics = ['test_recall', 'test_qps', 'test_distcomp']
    metric_names = ['Recall', 'QPS', 'DistComp']

    # 提取数据
    exp_names = [exp['name'] for exp in experiments]
    data = {metric: [exp['data'][-1][metric] if exp['data'] else 0
                     for exp in experiments]
            for metric in metrics}

    x = np.arange(len(metric_names))
    width = 0.20  # 调整为 4 个实验
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, ax = plt.subplots(figsize=(12, 8))

    for i, exp_name in enumerate(exp_names):
        values = [data[metric][i] for metric in metrics]
        offset = (i - len(exp_names)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width,
                     label=exp_name,
                     color=colors[i % len(colors)],
                     alpha=0.8)

        # 在柱子上标注数值
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=9,
                       rotation=0)

    ax.set_xlabel('Metrics', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss Value', fontsize=14, fontweight='bold')
    ax.set_title('Final Performance Comparison (Lower is Better)',
                fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=12)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = f"{output_dir}/final_performance_bars.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Bar chart saved: {save_path}")
    plt.close()

    # 创建综合得分对比
    fig, ax = plt.subplots(figsize=(10, 6))

    scores = [(3*data['test_recall'][i] + 3*data['test_qps'][i] + 8*data['test_distcomp'][i])/14
              for i in range(len(exp_names))]

    bars = ax.bar(exp_names, scores, color=colors[:len(exp_names)], alpha=0.8, width=0.6)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=12,
                   fontweight='bold')

    ax.set_ylabel('Weighted Score (3:3:8)', fontsize=14, fontweight='bold')
    ax.set_title('Comprehensive Performance Score Comparison\n(Lower is Better)',
                fontsize=16, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = f"{output_dir}/final_comprehensive_score.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Comprehensive score chart saved: {save_path}")
    plt.close()

def save_metrics_csv(experiments, output_dir):
    """保存指标数据为CSV格式"""
    for exp in experiments:
        csv_path = f"{output_dir}/{exp['id']}_metrics.csv"

        with open(csv_path, 'w', encoding='utf-8') as f:
            # 写入表头
            if exp['data']:
                headers = list(exp['data'][0].keys())
                f.write(','.join(headers) + '\n')

                # 写入数据
                for row in exp['data']:
                    values = [str(row.get(h, '')) for h in headers]
                    f.write(','.join(values) + '\n')

        print(f"✅ Metrics saved to: {csv_path}")

def main():
    # 配置实验信息
    experiments_config = [
        {
            'id': '104',
            'name': 'Exp104_Centralized',
            'type': 'Centralized Training (main.py)',
            'log_path': '/root/autodl-tmp/tyk/pgformer/src_pgformer/104/training_log_1763001113.txt'
        },
        {
            'id': '105',
            'name': 'Exp105_FedSGD',
            'type': 'Federated Learning (FedSGD)',
            'log_path': '/root/autodl-tmp/tyk/pgformer/logs/training_105/training_log_1763005576.txt'
        },
        {
            'id': '106',
            'name': 'Exp106_FedAvg',
            'type': 'Federated Learning (FedAvg)',
            'log_path': '/root/autodl-tmp/tyk/pgformer/logs/training_106/training_log_1763018048.txt'
        },
        {
            'id': '107',
            'name': 'Exp107_Performance_Based',
            'type': 'Federated Learning (Performance-Based)',
            'log_path': '/root/autodl-tmp/tyk/pgformer/logs/training_107/training_log_1763021784.txt'
        }
    ]

    print("=" * 100)
    print("STARTING FOUR-WAY COMPARISON ANALYSIS")
    print("=" * 100)
    print()

    # 解析所有实验的日志
    experiments = []
    for config in experiments_config:
        if not os.path.exists(config['log_path']):
            print(f"⚠️  Warning: Log file not found for {config['name']}: {config['log_path']}")
            continue

        print(f"📊 Parsing {config['name']}...")
        data = parse_training_log(config['log_path'])

        if not data:
            print(f"❌ Failed to parse {config['name']}")
            continue

        experiments.append({
            **config,
            'data': data
        })
        print(f"✅ {config['name']}: {len(data)} rounds")

    if len(experiments) < 2:
        print("\n❌ Error: At least 2 experiments are required for comparison")
        return

    print(f"\n✓ Successfully loaded {len(experiments)} experiments")
    print()

    # 生成输出目录
    output_dir = "/root/autodl-tmp/tyk/pgformer/comparison_results"
    os.makedirs(output_dir, exist_ok=True)

    # 生成对比报告
    print("📝 Generating comparison report...")
    generate_comparison_report(experiments, f"{output_dir}/comparison_report.txt")

    # 绘制对比图表
    print("\n📈 Generating comparison plots...")
    plot_convergence_curves(experiments, output_dir)
    plot_final_comparison_bars(experiments, output_dir)

    # 保存CSV数据
    print("\n💾 Saving metrics to CSV...")
    save_metrics_csv(experiments, output_dir)

    print("\n" + "=" * 100)
    print("COMPARISON ANALYSIS COMPLETED!")
    print("=" * 100)
    print(f"\nResults saved in: {output_dir}/")
    print("  - comparison_report.txt: Detailed comparison report")
    print("  - convergence_*.png: Individual metric convergence curves")
    print("  - convergence_comprehensive.png: Comprehensive 4-panel comparison")
    print("  - final_performance_bars.png: Final performance bar charts")
    print("  - final_comprehensive_score.png: Overall weighted score comparison")
    print("  - *_metrics.csv: Individual experiment metrics data")

if __name__ == "__main__":
    main()
