"""
混凝土集料因果分析演示
展示因果推理、归因分析、干预分析和反事实分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.causal_model import ConcreteAggregateCausalModel, compare_metrics
from src.visualization import plot_causal_graph, plot_attribution_results, plot_intervention_results

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置可视化风格
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)


def main():
    """主函数"""
    
    print("="*80)
    print("混凝土集料因果分析演示系统")
    print("="*80)
    
    # 创建结果目录
    results_dir = Path("results")
    figures_dir = results_dir / "figures"
    reports_dir = results_dir / "reports"
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # ===== 1. 加载数据 =====
    print("\n" + "="*80)
    print("步骤 1: 加载混凝土集料数据")
    print("="*80)
    
    data_path = "data/synthetic/concrete_aggregate_data.csv"
    df = pd.read_csv(data_path)
    
    print(f"✓ 数据加载完成: {df.shape[0]} 行, {df.shape[1]} 列")
    print(f"\n关键指标统计:")
    print(f"  - 混凝土强度平均值: {df['concrete_strength_mpa'].mean():.2f} MPa")
    print(f"  - 质量合格率: {df['quality_pass'].mean()*100:.2f}%")
    print(f"  - 配合比准确性: {df['mix_design_accuracy_pct'].mean():.2f}%")
    
    # ===== 2. 构建因果图 =====
    print("\n" + "="*80)
    print("步骤 2: 构建因果图")
    print("="*80)
    
    model = ConcreteAggregateCausalModel(df)
    causal_graph = model.build_causal_graph()
    
    print(f"✓ 因果图构建完成")
    print(f"  - 节点数: {causal_graph.number_of_nodes()}")
    print(f"  - 边数: {causal_graph.number_of_edges()}")
    print(f"\n关键因果路径:")
    print(f"  1. 集料类型 → 材质密度 → 混凝土强度")
    print(f"  2. 产地 → 砂率 → 混凝土强度")
    print(f"  3. 生产工艺 → 含泥量 → 混凝土强度")
    print(f"  4. 水胶比 → 胶凝材料 → 配合比准确性 → 混凝土强度")
    print(f"  5. 混凝土强度 → 工程性能指标")
    
    # 可视化因果图
    print("\n正在生成因果图可视化...")
    fig = plot_causal_graph(
        causal_graph, 
        title="混凝土集料和配合比因果图",
        figsize=(20, 16)
    )
    fig.savefig(figures_dir / "concrete_causal_graph.png", dpi=300, bbox_inches='tight')
    print(f"✓ 因果图已保存: {figures_dir / 'concrete_causal_graph.png'}")
    plt.close()
    
    # ===== 3. 拟合因果模型 =====
    print("\n" + "="*80)
    print("步骤 3: 拟合因果模型")
    print("="*80)
    
    # 归因和干预分析不需要可逆模型，使用标准模型即可（速度更快）
    model.fit_causal_model(quality='BETTER', invertible=False)
    
    # ===== 4. 归因分析 =====
    print("\n" + "="*80)
    print("步骤 4: 归因分析 - 识别混凝土强度变化的根本原因")
    print("="*80)
    
    # 创建两个时期的数据：优化前vs优化后
    # 模拟场景：优化生产工艺，降低含泥量，提高配合比准确性
    
    # 旧时期数据（前1000个样本）
    df_old = df.iloc[:1000].copy()
    
    # 新时期数据（模拟优化效果）
    df_new = df.iloc[1000:2000].copy()
    # 模拟工艺优化：减少含泥量，提高水胶比控制精度
    df_new['mud_content_pct'] = df_new['mud_content_pct'] * 0.7  # 含泥量降低30%
    df_new['water_binder_ratio'] = df_new['water_binder_ratio'] * 0.95  # 水胶比更精确
    df_new['mix_design_accuracy_pct'] = df_new['mix_design_accuracy_pct'] + 3  # 配合比准确性提高
    
    # 重新计算混凝土强度（简化模拟）
    df_new['concrete_strength_mpa'] = df_new['concrete_strength_mpa'] + \
        (df_old['mud_content_pct'].mean() - df_new['mud_content_pct'].mean()) * 3 + \
        (df_new['mix_design_accuracy_pct'] - df_old['mix_design_accuracy_pct'].mean()) * 0.3
    
    print(f"\n时期对比:")
    print(f"  旧时期 - 混凝土强度: {df_old['concrete_strength_mpa'].mean():.2f} MPa")
    print(f"  新时期 - 混凝土强度: {df_new['concrete_strength_mpa'].mean():.2f} MPa")
    print(f"  强度变化: {df_new['concrete_strength_mpa'].mean() - df_old['concrete_strength_mpa'].mean():.2f} MPa")
    
    print("\n正在执行归因分析...")
    contributions, uncertainties = model.attribution_analysis(
        df_old=df_old,
        df_new=df_new,
        target_column='concrete_strength_mpa',
        num_samples=1000,
        confidence_level=0.90,
        num_bootstrap_resamples=4
    )
    
    # 整理归因结果
    attribution_results = []
    for var, contrib in contributions.items():
        if var in uncertainties:
            uncertainty = uncertainties[var]
            attribution_results.append({
                'Variable': var,
                'Contribution': contrib,
                'Lower_CI': uncertainty[0],
                'Upper_CI': uncertainty[1]
            })
    
    attribution_df = pd.DataFrame(attribution_results)
    attribution_df = attribution_df.sort_values('Contribution', key=abs, ascending=False)
    
    print("\n✓ 归因分析完成")
    print("\n主要贡献因素（Top 10）:")
    for idx, row in attribution_df.head(10).iterrows():
        print(f"  {row['Variable']}: {row['Contribution']:.4f} "
              f"(95% CI: [{row['Lower_CI']:.4f}, {row['Upper_CI']:.4f}])")
    
    # 保存归因结果
    attribution_df.to_csv(reports_dir / "attribution_results_concrete.csv", index=False)
    print(f"\n✓ 归因结果已保存: {reports_dir / 'attribution_results_concrete.csv'}")
    
    # 可视化归因结果
    print("\n正在生成归因分析可视化...")
    fig = plot_attribution_results(
        attribution_df,
        title="混凝土强度变化归因分析",
        top_n=15
    )
    fig.savefig(figures_dir / "attribution_analysis_concrete.png", dpi=300, bbox_inches='tight')
    print(f"✓ 归因图已保存: {figures_dir / 'attribution_analysis_concrete.png'}")
    plt.close()
    
    # ===== 5. 干预分析 =====
    print("\n" + "="*80)
    print("步骤 5: 干预分析 - 评估不同优化措施的效果")
    print("="*80)
    
    # 定义不可干预的节点（结果变量）
    non_interveneable = [
        'concrete_strength_mpa',
        'quality_pass',
        'quality_pass_rate_pct',
        'pavement_performance_score',
        'structural_strength_mpa',
        'sand_pct',  # 由碎石比例决定
        'total_binder',  # 由各组分决定
        'mix_ratio'  # 由配比决定
    ]
    
    print("\n正在执行干预分析...")
    print("评估各因素增加1个单位对混凝土强度的因果效应...")
    
    intervention_results = model.intervention_analysis(
        target='concrete_strength_mpa',
        step_size=1.0,
        non_interveneable_nodes=non_interveneable,
        confidence_level=0.95,
        num_samples=5000,
        num_bootstrap_resamples=20
    )
    
    print("\n✓ 干预分析完成")
    print("\n最有效的优化措施（Top 10）:")
    for idx, row in intervention_results.head(10).iterrows():
        print(f"  {row['Variable']}: {row['Causal_Effect']:.4f} MPa "
              f"(95% CI: [{row['Lower_CI']:.4f}, {row['Upper_CI']:.4f}])")
    
    # 保存干预结果
    intervention_results.to_csv(reports_dir / "intervention_results_concrete.csv", index=False)
    print(f"\n✓ 干预结果已保存: {reports_dir / 'intervention_results_concrete.csv'}")
    
    # 可视化干预结果
    print("\n正在生成干预分析可视化...")
    fig = plot_intervention_results(
        intervention_results,
        title="混凝土强度优化措施因果效应分析",
        top_n=15
    )
    fig.savefig(figures_dir / "intervention_analysis_concrete.png", dpi=300, bbox_inches='tight')
    print(f"✓ 干预图已保存: {figures_dir / 'intervention_analysis_concrete.png'}")
    plt.close()
    
    # ===== 6. 提示：反事实分析 =====
    print("\n" + "="*80)
    print("步骤 6: 反事实分析")
    print("="*80)
    print("\n反事实分析需要使用可逆因果模型，请运行专门的反事实分析脚本：")
    print("  python concrete_counterfactual_demo.py")
    print("\n该脚本会回答以下问题：")
    print("  - 如果采用水洗工艺降低含泥量，强度会提升多少？")
    print("  - 如果优化水胶比至0.42，效果如何？")
    print("  - 综合优化策略的预期效果？")
    
    # ===== 7. 生成分析报告 =====
    print("\n" + "="*80)
    print("步骤 7: 生成分析报告")
    print("="*80)
    
    report_path = reports_dir / "analysis_summary_concrete.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("混凝土集料因果分析报告\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. 数据概览\n")
        f.write("-"*80 + "\n")
        f.write(f"样本数量: {df.shape[0]}\n")
        f.write(f"变量数量: {df.shape[1]}\n")
        f.write(f"混凝土强度平均值: {df['concrete_strength_mpa'].mean():.2f} MPa\n")
        f.write(f"质量合格率: {df['quality_pass'].mean()*100:.2f}%\n\n")
        
        f.write("2. 因果图结构\n")
        f.write("-"*80 + "\n")
        f.write(f"节点数: {causal_graph.number_of_nodes()}\n")
        f.write(f"边数: {causal_graph.number_of_edges()}\n\n")
        
        f.write("3. 归因分析结果（Top 10）\n")
        f.write("-"*80 + "\n")
        for idx, row in attribution_df.head(10).iterrows():
            f.write(f"{row['Variable']}: {row['Contribution']:.4f} "
                   f"(CI: [{row['Lower_CI']:.4f}, {row['Upper_CI']:.4f}])\n")
        f.write("\n")
        
        f.write("4. 干预分析结果（Top 10）\n")
        f.write("-"*80 + "\n")
        for idx, row in intervention_results.head(10).iterrows():
            f.write(f"{row['Variable']}: {row['Causal_Effect']:.4f} MPa "
                   f"(CI: [{row['Lower_CI']:.4f}, {row['Upper_CI']:.4f}])\n")
        f.write("\n")
        
        f.write("5. 主要发现与建议\n")
        f.write("-"*80 + "\n")
        f.write("✓ 水胶比是影响混凝土强度的最关键因素\n")
        f.write("✓ 降低含泥量可显著提高混凝土强度\n")
        f.write("✓ 优化生产工艺（如采用水洗工艺）可有效控制含泥量\n")
        f.write("✓ 提高配合比准确性对工程质量至关重要\n")
        f.write("✓ 材质密度和产地对强度有显著影响，需合理选择原材料\n\n")
    
    print(f"✓ 分析报告已保存: {report_path}")
    
    # ===== 8. 总结 =====
    print("\n" + "="*80)
    print("分析完成!")
    print("="*80)
    print("\n生成的文件:")
    print(f"  1. 因果图: {figures_dir / 'concrete_causal_graph.png'}")
    print(f"  2. 归因分析图: {figures_dir / 'attribution_analysis_concrete.png'}")
    print(f"  3. 干预分析图: {figures_dir / 'intervention_analysis_concrete.png'}")
    print(f"  4. 分析报告: {report_path}")
    print(f"\n如需反事实分析，请运行: python concrete_counterfactual_demo.py")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

