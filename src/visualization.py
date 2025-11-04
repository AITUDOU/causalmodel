"""
可视化工具
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional
from scipy import stats
from dowhy import gcm

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置seaborn样式
sns.set_style('whitegrid')
sns.set_palette('husl')


def plot_causal_graph(graph: nx.DiGraph,
                     title: str = '因果图',
                     figsize: tuple = (20, 16),
                     node_size: int = 3000,
                     font_size: int = 10,
                     layout: str = 'spring'):
    """
    绘制因果图
    
    Parameters:
    -----------
    graph : nx.DiGraph
        因果图
    title : str
        图表标题
    figsize : tuple
        图形大小
    node_size : int
        节点大小
    font_size : int
        字体大小
    layout : str
        布局算法 ('spring', 'kamada_kawai', 'hierarchical')
        
    Returns:
    --------
    matplotlib.figure.Figure
        图形对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 选择布局
    if layout == 'hierarchical':
        # 层次化布局（适合有向无环图）
        try:
            pos = nx.nx_agraph.graphviz_layout(graph, prog='dot')
        except:
            pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(graph)
    else:  # spring
        pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)
    
    # 绘制边
    nx.draw_networkx_edges(
        graph, pos,
        edge_color='gray',
        width=1.5,
        alpha=0.6,
        arrows=True,
        arrowsize=20,
        arrowstyle='->',
        connectionstyle='arc3,rad=0.1',
        ax=ax
    )
    
    # 绘制节点
    nx.draw_networkx_nodes(
        graph, pos,
        node_color='lightblue',
        node_size=node_size,
        alpha=0.9,
        edgecolors='black',
        linewidths=2,
        ax=ax
    )
    
    # 绘制标签
    nx.draw_networkx_labels(
        graph, pos,
        font_size=font_size,
        font_family='sans-serif',
        font_weight='bold',
        ax=ax
    )
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    plt.tight_layout()
    
    return fig


def plot_attribution_results(attribution_df: pd.DataFrame = None,
                            contributions: Dict = None, 
                            uncertainties: Dict = None,
                            title: str = '归因分析结果',
                            xlabel: str = '对目标变化的贡献',
                            figsize: tuple = (12, 8),
                            top_n: int = 15):
    """
    绘制归因分析结果
    
    Parameters:
    -----------
    attribution_df : pd.DataFrame
        归因结果DataFrame（包含Variable, Contribution, Lower_CI, Upper_CI列）
    contributions : Dict
        贡献字典（可选，与attribution_df二选一）
    uncertainties : Dict
        不确定性字典（可选）
    title : str
        图表标题
    xlabel : str
        x轴标签
    figsize : tuple
        图形大小
    top_n : int
        显示前N个最重要的因素
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if attribution_df is not None:
        # 使用DataFrame模式
        df = attribution_df.copy()
        df = df.dropna(subset=['Contribution', 'Lower_CI', 'Upper_CI'])
        df = df.head(top_n)
        
        # 计算误差条
        errors = [
            df['Contribution'] - df['Lower_CI'],
            df['Upper_CI'] - df['Contribution']
        ]
        
        # 绘制柱状图
        colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in df['Contribution']]
        bars = ax.barh(df['Variable'], df['Contribution'], 
                       xerr=errors,
                       capsize=5,
                       color=colors,
                       alpha=0.7,
                       edgecolor='black',
                       linewidth=1.2)
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('变量', fontsize=12)
        
    else:
        # 使用字典模式（原有逻辑）
        if contributions is not None and uncertainties is not None:
            gcm.util.bar_plot(contributions, uncertainties, xlabel, figure_size=figsize)
    
    # 添加零线
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_intervention_results(result_df: pd.DataFrame,
                             title: str = '干预效果分析',
                             figsize: tuple = (12, 8),
                             top_n: Optional[int] = None):
    """
    绘制干预分析结果
    
    Parameters:
    -----------
    result_df : pd.DataFrame
        干预结果数据框
    title : str
        图表标题
    figsize : tuple
        图形大小
    top_n : int, optional
        只显示前N个最有效的干预
    """
    df = result_df.copy()
    
    # 只保留有效数据
    df = df.dropna(subset=['Causal_Effect', 'Lower_CI', 'Upper_CI'])
    
    if top_n:
        df = df.head(top_n)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 计算误差条
    errors = [
        df['Causal_Effect'] - df['Lower_CI'],
        df['Upper_CI'] - df['Causal_Effect']
    ]
    
    # 绘制柱状图
    bars = ax.barh(df['Variable'], df['Causal_Effect'], 
                   xerr=errors, 
                   capsize=5,
                   color=['#2ecc71' if x > 0 else '#e74c3c' for x in df['Causal_Effect']],
                   alpha=0.7,
                   edgecolor='black',
                   linewidth=1.2)
    
    # 添加零线
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('因果效应（单位变化对目标的影响）', fontsize=12)
    ax.set_ylabel('干预变量', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_metric_distributions(df_new: pd.DataFrame, 
                              df_old: pd.DataFrame, 
                              metric_columns: List[str],
                              figsize: tuple = (12, 4)):
    """
    绘制两个时期的指标分布对比
    
    Parameters:
    -----------
    df_new : pd.DataFrame
        新时期数据
    df_old : pd.DataFrame
        旧时期数据
    metric_columns : List[str]
        要绘制的指标列表
    figsize : tuple
        图形大小
    """
    for metric_column in metric_columns:
        fig, ax = plt.subplots(figsize=figsize)
        
        # 计算KDE
        kde_new = stats.gaussian_kde(df_new[metric_column].dropna())
        kde_old = stats.gaussian_kde(df_old[metric_column].dropna())
        
        x_range = np.linspace(
            min(df_new[metric_column].min(), df_old[metric_column].min()),
            max(df_new[metric_column].max(), df_old[metric_column].max()),
            1000
        )
        
        # 绘制分布
        ax.plot(x_range, kde_new(x_range), color='#FF6B6B', lw=2.5, label='改进后')
        ax.plot(x_range, kde_old(x_range), color='#4ECDC4', lw=2.5, label='改进前')
        
        ax.fill_between(x_range, kde_new(x_range), alpha=0.3, color='#FF6B6B')
        ax.fill_between(x_range, kde_old(x_range), alpha=0.3, color='#4ECDC4')
        
        ax.set_xlabel(metric_column, fontsize=12)
        ax.set_ylabel('密度', fontsize=12)
        ax.set_title(f'{metric_column} 分布对比', fontsize=14, fontweight='bold')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=11, loc='best')
        
        plt.tight_layout()
        plt.show()


def create_summary_report(contributions: Dict,
                         uncertainties: Dict,
                         intervention_results: pd.DataFrame,
                         metrics_comparison: pd.DataFrame) -> str:
    """
    创建文本摘要报告
    
    Parameters:
    -----------
    contributions : Dict
        归因贡献
    uncertainties : Dict
        不确定性
    intervention_results : pd.DataFrame
        干预结果
    metrics_comparison : pd.DataFrame
        指标对比
        
    Returns:
    --------
    str
        报告文本
    """
    report = []
    report.append("=" * 80)
    report.append("混凝土集料因果分析报告")
    report.append("=" * 80)
    report.append("")
    
    # 1. 指标变化总结
    report.append("一、关键指标变化")
    report.append("-" * 80)
    for _, row in metrics_comparison.iterrows():
        report.append(f"  {row['Metric']}:")
        if pd.notna(row.get('Mean_Change_%')):
            report.append(f"    - 均值变化: {row['Mean_Change_%']:.2f}%")
        if pd.notna(row.get('Median_Change_%')):
            report.append(f"    - 中位数变化: {row['Median_Change_%']:.2f}%")
    report.append("")
    
    # 2. 归因分析结果
    report.append("二、归因分析：变化的主要驱动因素")
    report.append("-" * 80)
    sorted_contributions = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    for node, contrib in sorted_contributions[:10]:  # 只显示前10个
        if node in uncertainties:
            lb, ub = uncertainties[node]
            report.append(f"  {node}: {contrib:.2f} (95% CI: [{lb:.2f}, {ub:.2f}])")
    report.append("")
    
    # 3. 干预建议
    report.append("三、优化建议：最有效的干预措施")
    report.append("-" * 80)
    top_interventions = intervention_results.nlargest(5, 'Causal_Effect')
    for _, row in top_interventions.iterrows():
        if pd.notna(row['Causal_Effect']) and row['Causal_Effect'] > 0:
            report.append(f"  {row['Variable']}:")
            report.append(f"    - 因果效应: {row['Causal_Effect']:.4f}")
            if pd.notna(row['Lower_CI']):
                report.append(f"    - 95% CI: [{row['Lower_CI']:.4f}, {row['Upper_CI']:.4f}]")
            report.append(f"    - 建议: 增加该参数可显著提升目标指标")
            report.append("")
    
    report.append("=" * 80)
    
    return "\n".join(report)
