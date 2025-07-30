#!/usr/bin/env python3
"""
生成演示文稿所需的图表
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cauchy, norm
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def plot_cauchy_vs_normal():
    """绘制柯西分布与正态分布的对比"""
    x = np.linspace(-10, 10, 1000)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 正态分布
    y_normal = norm.pdf(x, 0, 1)
    ax.plot(x, y_normal, 'b-', linewidth=2, label='正态分布 N(0,1)')
    
    # 柯西分布
    y_cauchy = cauchy.pdf(x, 0, 1)
    ax.plot(x, y_cauchy, 'r-', linewidth=2, label='柯西分布 Cauchy(0,1)')
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('概率密度', fontsize=12)
    ax.set_title('柯西分布 vs 正态分布', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.5)
    
    plt.tight_layout()
    plt.savefig('cauchy_vs_normal.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_regression_robustness():
    """绘制回归任务的鲁棒性曲线"""
    noise_levels = np.array([0, 10, 20, 30, 40, 50])
    
    # 模拟数据
    sklearn_mae = np.array([5.2, 12.3, 28.5, 47.6, 68.2, 85.3])
    pytorch_mae = np.array([5.0, 11.8, 26.9, 45.3, 65.8, 82.1])
    causal_det_mae = np.array([5.5, 10.2, 20.1, 38.2, 55.3, 70.2])
    causal_std_mae = np.array([5.3, 7.8, 9.5, 11.4, 13.2, 15.8])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(noise_levels, sklearn_mae, 'ro-', linewidth=2, markersize=8, label='sklearn MLP')
    ax.plot(noise_levels, pytorch_mae, 'bs-', linewidth=2, markersize=8, label='PyTorch MLP')
    ax.plot(noise_levels, causal_det_mae, 'g^-', linewidth=2, markersize=8, label='CausalEngine (确定性)')
    ax.plot(noise_levels, causal_std_mae, 'mv-', linewidth=3, markersize=10, label='CausalEngine (标准)')
    
    ax.set_xlabel('标签噪声水平 (%)', fontsize=12)
    ax.set_ylabel('平均绝对误差 (MAE)', fontsize=12)
    ax.set_title('回归任务：噪声鲁棒性对比', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('regression_robustness.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_classification_robustness():
    """绘制分类任务的鲁棒性曲线"""
    noise_levels = np.array([0, 10, 20, 30, 40, 50])
    
    # 模拟数据
    sklearn_acc = np.array([0.95, 0.93, 0.90, 0.885, 0.85, 0.80])
    pytorch_acc = np.array([0.95, 0.935, 0.905, 0.887, 0.855, 0.805])
    causal_det_acc = np.array([0.945, 0.935, 0.92, 0.912, 0.89, 0.86])
    causal_std_acc = np.array([0.95, 0.94, 0.93, 0.922, 0.91, 0.89])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(noise_levels, sklearn_acc, 'ro-', linewidth=2, markersize=8, label='sklearn MLP')
    ax.plot(noise_levels, pytorch_acc, 'bs-', linewidth=2, markersize=8, label='PyTorch MLP')
    ax.plot(noise_levels, causal_det_acc, 'g^-', linewidth=2, markersize=8, label='CausalEngine (确定性)')
    ax.plot(noise_levels, causal_std_acc, 'mv-', linewidth=3, markersize=10, label='CausalEngine (标准)')
    
    ax.set_xlabel('标签噪声水平 (%)', fontsize=12)
    ax.set_ylabel('准确率', fontsize=12)
    ax.set_title('分类任务：噪声鲁棒性对比', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.75, 0.96)
    
    plt.tight_layout()
    plt.savefig('classification_robustness.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_qr_code():
    """生成项目的二维码占位符"""
    fig, ax = plt.subplots(figsize=(4, 4))
    
    # 创建一个简单的占位符
    ax.text(0.5, 0.5, 'QR Code\n项目主页', 
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes,
            fontsize=20,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('qr_code.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    print("生成演示文稿图表...")
    
    print("1. 生成柯西分布 vs 正态分布对比图...")
    plot_cauchy_vs_normal()
    
    print("2. 生成回归鲁棒性对比图...")
    plot_regression_robustness()
    
    print("3. 生成分类鲁棒性对比图...")
    plot_classification_robustness()
    
    print("4. 生成二维码占位符...")
    generate_qr_code()
    
    print("✅ 所有图表生成完成！")