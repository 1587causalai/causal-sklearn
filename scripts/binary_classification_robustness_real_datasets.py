#!/usr/bin/env python3
"""
二分类算法真实数据集噪声鲁棒性测试脚本

🎯 目标：在真实二分类数据集上测试 OvR 和 single_score 模式的噪声鲁棒性
🔬 核心：基于 classification_robustness_real_datasets.py，仅修改为测试二分类

主要特性：
- 真实数据集：Breast Cancer (二分类数据集)
- 算法对比：sklearn MLP, CausalEngine OvR, CausalEngine single_score
- 噪声级别：0%, 10%, 20%, ..., 100% (11个级别)
- 完整指标：Accuracy, Precision, Recall, F1
- 折线图可视化：清晰展示算法在真实数据上的鲁棒性对比

使用方法：
python scripts/binary_classification_robustness_real_datasets.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import sys
import warnings
from tqdm import tqdm
import pandas as pd

# 设置matplotlib后端，避免弹出窗口
plt.switch_backend('Agg')

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入分类器
from causal_sklearn.classifier import MLPCausalClassifier
from causal_sklearn.data_processing import inject_shuffle_noise

warnings.filterwarnings('ignore')

# =============================================================================
# 配置部分 - 在这里修改实验参数
# =============================================================================

BINARY_ROBUSTNESS_CONFIG = {
    # 数据集选择 - 可选多个二分类数据集
    'dataset': 'spambase',  # 可选: 'breast_cancer', 'adult', 'bank_marketing', 'spambase'
    
    # 噪声级别设置
    'noise_levels': np.linspace(0, 1, 6),  # 0%, 10%, 20%, ..., 100%
    
    # 数据参数
    'random_state': 42,     # 固定随机种子
    'test_size': 0.2,       # 测试集比例
    
    # 网络结构（所有算法统一）
    'hidden_layers': (128, 64, 64),      # 保持网络结构
    'max_iter': 3000,               # 最大迭代次数
    'learning_rate': 0.001,         # 学习率
    'patience': 100,                # 早停耐心
    'tol': 1e-4,                    # 收敛容忍度
    'batch_size': None,             # 批处理大小
    
    # 稳定性改进参数
    'n_runs': 2,                     # 5次运行
    'base_random_seed': 42,          # 基础随机种子
    
    # 额外稳定性参数
    'validation_fraction': 0.2,     # 验证集比例（早停用）
    'early_stopping': True,          # 确保早停开启
    
    # 输出控制
    'output_dir': 'results/binary_classification_robustness',
    'save_plots': True,
    'save_data': True,
    'verbose': True,
    'figure_dpi': 300
}

# =============================================================================
# 工具函数
# =============================================================================

def _ensure_output_dir(output_dir):
    """确保输出目录存在"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

def _get_output_path(output_dir, filename):
    """获取输出文件路径"""
    return os.path.join(output_dir, filename)

def load_binary_dataset(dataset_name='breast_cancer', max_samples=10000):
    """加载二分类数据集
    
    支持的数据集：
    - breast_cancer: 乳腺癌检测（简单，干净）
    - adult: 成人收入预测（中等难度，有缺失值）
    - bank_marketing: 银行营销（不平衡）
    - spambase: 垃圾邮件检测（高维）
    """
    
    if dataset_name == 'breast_cancer':
        data = load_breast_cancer()
        X, y = data.data, data.target
        desc = "569 samples, 30 features, 2 classes (balanced, clean)"
        
    elif dataset_name == 'adult':
        # Adult dataset - 收入预测
        print("📥 从 OpenML 加载 Adult 数据集...")
        try:
            data = fetch_openml('adult', version=2, as_frame=True, parser='auto')
            X = data.data
            y = (data.target == '>50K').astype(int)  # 转为二分类
            
            # 处理分类特征
            from sklearn.preprocessing import LabelEncoder
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            X_encoded = X.copy()
            for col in categorical_cols:
                le = LabelEncoder()
                # 先转换为字符串处理缺失值
                X_encoded[col] = X_encoded[col].astype(str).fillna('missing')
                X_encoded[col] = le.fit_transform(X_encoded[col])
            X = X_encoded
            
            X = X.values
            y = y.values
            
            # 限制样本数
            if len(X) > max_samples:
                idx = np.random.choice(len(X), max_samples, replace=False)
                X, y = X[idx], y[idx]
            
            desc = f"{len(X)} samples, {X.shape[1]} features, imbalanced (~24% positive)"
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            print("📋 使用备用的 breast_cancer 数据集")
            return load_binary_dataset('breast_cancer')
    
    elif dataset_name == 'bank_marketing':
        # Bank Marketing dataset - 银行营销
        print("📥 从 OpenML 加载 Bank Marketing 数据集...")
        try:
            data = fetch_openml('bank-marketing', version=1, as_frame=True, parser='auto')
            X = data.data
            y = (data.target == '2').astype(int)  # '2' 表示订阅
            
            # 处理分类特征
            from sklearn.preprocessing import LabelEncoder
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            X_encoded = X.copy()
            for col in categorical_cols:
                le = LabelEncoder()
                # 先转换为字符串处理缺失值
                X_encoded[col] = X_encoded[col].astype(str).fillna('missing')
                X_encoded[col] = le.fit_transform(X_encoded[col])
            X = X_encoded
            
            X = X.values
            y = y.values
            
            # 限制样本数
            if len(X) > max_samples:
                idx = np.random.choice(len(X), max_samples, replace=False)
                X, y = X[idx], y[idx]
            
            desc = f"{len(X)} samples, {X.shape[1]} features, highly imbalanced (~11% positive)"
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            print("📋 使用备用的 breast_cancer 数据集")
            return load_binary_dataset('breast_cancer')
    
    elif dataset_name == 'spambase':
        # Spambase dataset - 垃圾邮件检测
        print("📥 从 OpenML 加载 Spambase 数据集...")
        try:
            data = fetch_openml('spambase', version=1, as_frame=True, parser='auto')
            X = data.data.values
            y = (data.target == '1').astype(int)  # '1' 表示垃圾邮件
            
            desc = f"{len(X)} samples, {X.shape[1]} features, slightly imbalanced (~39% spam)"
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            print("📋 使用备用的 breast_cancer 数据集")
            return load_binary_dataset('breast_cancer')
    
    else:
        print(f"❌ 未知数据集: {dataset_name}")
        print("📋 使用默认的 breast_cancer 数据集")
        return load_binary_dataset('breast_cancer')
    
    print(f"📊 加载数据集: {dataset_name}")
    print(f"📋 数据集描述: {desc}")
    print(f"📐 实际数据形状: {X.shape}, 类别分布: {np.bincount(y)}")
    print(f"📊 正类比例: {np.mean(y):.2%}")
    
    return X, y

# =============================================================================
# 二分类鲁棒性测试
# =============================================================================

def test_binary_classification_noise_robustness(config):
    """测试二分类算法的噪声鲁棒性"""
    print("\n" + "="*80)
    print("🎯 二分类算法噪声鲁棒性测试")
    print("="*80)
    
    noise_levels = config['noise_levels']
    results = {}
    
    # 定义要测试的算法
    algorithms = {
        'sklearn_mlp': ('sklearn MLP', None, None),
        'causal_ovr_det': ('CausalEngine OvR (det)', 'ovr', 'deterministic'),
        'causal_ovr_std': ('CausalEngine OvR (std)', 'ovr', 'standard'),
        'causal_single_det': ('CausalEngine Single (det)', 'single_score', 'deterministic'),
        'causal_single_std': ('CausalEngine Single (std)', 'single_score', 'standard')
    }
    
    # 初始化结果字典
    for algo_key, (algo_name, _, _) in algorithms.items():
        results[algo_key] = {
            'name': algo_name,
            'noise_levels': [],
            'accuracy': [], 'precision': [], 'recall': [], 'f1': []
        }
    
    # 加载数据集
    X, y = load_binary_dataset(config.get('dataset', 'breast_cancer'))
    
    # 分割数据
    X_train, X_test, y_train_clean, y_test = train_test_split(
        X, y, test_size=config['test_size'], random_state=config['random_state']
    )
    
    # 标准化特征
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # 在不同噪声级别下测试
    for noise_level in tqdm(noise_levels, desc="噪声级别"):
        print(f"\n📊 测试噪声级别: {noise_level:.1%}")
        
        # 对训练标签注入噪声
        if noise_level > 0:
            y_train_noisy, noise_indices = inject_shuffle_noise(
                y_train_clean,
                noise_ratio=noise_level,
                random_state=config['random_state']
            )
        else:
            y_train_noisy = y_train_clean.copy()
        
        # 测试每个算法
        for algo_key, (algo_name, binary_mode, causal_mode) in algorithms.items():
            try:
                if config['verbose']:
                    print(f"  🔧 训练 {algo_name}...")
                
                # 创建和训练模型
                if algo_key == 'sklearn_mlp':
                    model = MLPClassifier(
                        hidden_layer_sizes=config['hidden_layers'],
                        max_iter=config['max_iter'],
                        learning_rate_init=config['learning_rate'],
                        random_state=config['random_state'],
                        early_stopping=config['early_stopping'],
                        validation_fraction=config['validation_fraction'],
                        n_iter_no_change=config['patience'],
                        tol=config['tol']
                    )
                    model.fit(X_train_scaled, y_train_noisy)
                    
                else:  # CausalEngine variants
                    model = MLPCausalClassifier(
                        perception_hidden_layers=config['hidden_layers'],
                        binary_mode=binary_mode,
                        mode=causal_mode,
                        max_iter=config['max_iter'],
                        learning_rate=config['learning_rate'],
                        batch_size=config['batch_size'],
                        random_state=config['random_state'],
                        early_stopping=config['early_stopping'],
                        validation_fraction=config['validation_fraction'],
                        verbose=False
                    )
                    model.fit(X_train_scaled, y_train_noisy)
                
                # 在测试集上评估
                y_pred = model.predict(X_test_scaled)
                
                # 计算指标
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
                recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
                
                # 存储结果
                results[algo_key]['noise_levels'].append(noise_level)
                results[algo_key]['accuracy'].append(accuracy)
                results[algo_key]['precision'].append(precision)
                results[algo_key]['recall'].append(recall)
                results[algo_key]['f1'].append(f1)
                
                if config['verbose']:
                    print(f"    Acc: {accuracy:.4f}, F1: {f1:.4f}")
                
            except Exception as e:
                print(f"    ❌ {algo_name} 训练失败: {str(e)}")
                # 添加NaN值保持数组长度一致
                results[algo_key]['noise_levels'].append(noise_level)
                results[algo_key]['accuracy'].append(np.nan)
                results[algo_key]['precision'].append(np.nan)
                results[algo_key]['recall'].append(np.nan)
                results[algo_key]['f1'].append(np.nan)
    
    return results

# =============================================================================
# 可视化函数
# =============================================================================

def create_binary_robustness_plots(results, config):
    """创建二分类鲁棒性分析折线图"""
    if not config.get('save_plots', False):
        return
    
    _ensure_output_dir(config['output_dir'])
    
    print("\n📊 创建二分类鲁棒性分析图表")
    print("-" * 50)
    
    # 设置图表风格
    plt.style.use('seaborn-v0_8')
    colors = plt.cm.Set3(np.linspace(0, 1, 12))  # 12种不同颜色
    
    # 创建分类鲁棒性图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    n_runs = config.get('n_runs', 1)
    title_suffix = f' (Averaged over {n_runs} runs)' if n_runs > 1 else ''
    fig.suptitle(f'Binary Classification Algorithms Noise Robustness{title_suffix}', 
                 fontsize=16, fontweight='bold')
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i//2, i%2]
        
        color_idx = 0
        for algo_key, data in results.items():
            if data[metric]:  # 确保有数据
                noise_levels = np.array(data['noise_levels']) * 100  # 转换为百分比
                values = np.array(data[metric])
                
                # 过滤NaN值
                valid_mask = ~np.isnan(values)
                if valid_mask.any():
                    # 判断是否为因果算法
                    is_causal = algo_key.startswith('causal_')
                    is_single = 'single' in algo_key
                    
                    # 设置线型：sklearn虚线，causal OvR细实线，causal single粗实线
                    if not is_causal:
                        linestyle = '--'
                        linewidth = 2
                    elif is_single:
                        linestyle = '-'
                        linewidth = 3  # 粗线突出single_score
                    else:
                        linestyle = '-'
                        linewidth = 2
                    
                    ax.plot(noise_levels[valid_mask], values[valid_mask], 
                           marker='o', linewidth=linewidth, markersize=4, linestyle=linestyle,
                           label=data['name'], color=colors[color_idx])
                    color_idx += 1
        
        ax.set_xlabel('Noise Level (%)')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} vs Noise Level')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 动态设置y轴范围以更好地展示差异
        # 获取所有有效数据的最小值
        all_valid_values = []
        for algo_key, data in results.items():
            if data[metric]:
                values = np.array(data[metric])
                valid_values = values[~np.isnan(values)]
                if len(valid_values) > 0:
                    all_valid_values.extend(valid_values)
        
        if all_valid_values:
            min_value = min(all_valid_values)
            # 如果最小值大于0.5，从0.5开始；否则留一些边距
            if min_value > 0.5:
                y_min = 0.5
            else:
                y_min = max(0, min_value - 0.05)
            ax.set_ylim(y_min, 1.05)
    
    plt.tight_layout()
    
    # 生成文件名，动态包含数据集名称
    dataset_key = config.get('dataset', 'unknown_dataset')
    plot_filename = f'binary_classification_robustness_{dataset_key}.png'
    plot_path = _get_output_path(config['output_dir'], plot_filename)
    
    plt.savefig(plot_path, dpi=config['figure_dpi'], bbox_inches='tight')
    print(f"📊 二分类鲁棒性图表已保存为 {plot_path}")
    plt.close()

# =============================================================================
# 主函数
# =============================================================================

def run_single_binary_robustness_analysis(config, run_idx=0):
    """运行单次二分类鲁棒性分析"""
    if config['verbose']:
        print(f"\n🔄 第 {run_idx + 1}/{config['n_runs']} 次运行 (随机种子: {config['random_state']})")
    
    # 运行二分类鲁棒性测试
    results = test_binary_classification_noise_robustness(config)
    
    return results

def aggregate_binary_results(all_results):
    """聚合多次运行的二分类结果"""
    aggregated_results = {}
    
    if all_results and len(all_results) > 0:
        first_result = all_results[0]
        if first_result:
            for algo_key in first_result.keys():
                aggregated_results[algo_key] = {
                    'name': first_result[algo_key]['name'],
                    'noise_levels': first_result[algo_key]['noise_levels'],
                    'accuracy': [], 'precision': [], 'recall': [], 'f1': [],
                    'accuracy_std': [], 'precision_std': [], 'recall_std': [], 'f1_std': []
                }
                
                # 收集所有运行的结果
                metrics = ['accuracy', 'precision', 'recall', 'f1']
                for metric in metrics:
                    all_values = []
                    for run_result in all_results:
                        if run_result and algo_key in run_result:
                            all_values.append(run_result[algo_key][metric])
                    
                    if all_values:
                        # 计算每个噪声级别的均值和标准差
                        all_values = np.array(all_values)  # shape: (n_runs, n_noise_levels)
                        means = np.nanmean(all_values, axis=0)
                        stds = np.nanstd(all_values, axis=0)
                        
                        aggregated_results[algo_key][metric] = means.tolist()
                        aggregated_results[algo_key][f'{metric}_std'] = stds.tolist()
    
    return aggregated_results

def run_binary_classification_robustness_analysis(config=None):
    """运行完整的多次二分类鲁棒性分析"""
    if config is None:
        config = BINARY_ROBUSTNESS_CONFIG
    
    print("🚀 二分类算法噪声鲁棒性分析")
    print("=" * 70)
    print(f"数据集: {config.get('dataset', 'breast_cancer')}")
    print(f"噪声级别: {config['noise_levels'][0]:.0%} - {config['noise_levels'][-1]:.0%} ({len(config['noise_levels'])}个级别)")
    print(f"运行次数: {config['n_runs']}次 (随机种子: {config['base_random_seed']} - {config['base_random_seed'] + config['n_runs'] - 1})")
    
    all_results = []
    
    # 多次运行
    for run_idx in range(config['n_runs']):
        # 为每次运行设置不同的随机种子
        run_config = config.copy()
        run_config['random_state'] = config['base_random_seed'] + run_idx
        
        result = run_single_binary_robustness_analysis(run_config, run_idx)
        all_results.append(result)
    
    # 聚合结果
    print(f"\n📊 聚合 {config['n_runs']} 次运行的结果...")
    aggregated_results = aggregate_binary_results(all_results)
    
    # 创建可视化（使用聚合后的结果）
    create_binary_robustness_plots(aggregated_results, config)
    
    # 保存结果数据
    if config.get('save_data', True):
        _ensure_output_dir(config['output_dir'])
        
        # 动态生成数据文件名
        dataset_key = config.get('dataset', 'unknown_dataset')
        data_filename = f'binary_classification_results_{dataset_key}_aggregated.npy'
        data_path = _get_output_path(config['output_dir'], data_filename)
        
        np.save(data_path, aggregated_results)
        print(f"📊 聚合结果已保存为 {data_path}")
    
    print(f"\n✅ 二分类鲁棒性分析完成!")
    print(f"📊 结果保存在: {config['output_dir']}")
    
    return aggregated_results

# =============================================================================
# 入口点
# =============================================================================

if __name__ == '__main__':
    # 运行二分类鲁棒性分析
    results = run_binary_classification_robustness_analysis()