"""
混凝土集料反事实分析
回答问题："如果采用不同的工艺参数和原材料，混凝土强度会如何？"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.causal_model import ConcreteAggregateCausalModel

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def main():
    """主函数"""
    
    print("="*80)
    print("混凝土集料反事实分析")
    print("基于因果模型回答：如果当时做了不同的选择，结果会怎样？")
    print("="*80)
    
    # 创建结果目录
    results_dir = Path("results")
    figures_dir = results_dir / "figures"
    reports_dir = results_dir / "reports"
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # ===== 1. 加载数据并构建模型 =====
    print("\n" + "="*80)
    print("步骤 1: 加载数据并构建因果模型")
    print("="*80)
    
    data_path = "data/synthetic/concrete_aggregate_data.csv"
    df = pd.read_csv(data_path)
    
    print(f"✓ 数据加载完成: {df.shape[0]} 行, {df.shape[1]} 列")
    
    # 构建因果模型（使用可逆模型以支持反事实分析）
    model = ConcreteAggregateCausalModel(df)
    causal_graph = model.build_causal_graph()
    model.fit_causal_model(quality='BETTER', invertible=True)
    
    print(f"✓ 因果模型构建完成")
    
    # ===== 2. 场景设定：选择一个不达标样本 =====
    print("\n" + "="*80)
    print("步骤 2: 场景设定 - 选择一个质量不达标的批次")
    print("="*80)
    
    # 找到强度不达标的样本
    low_strength_threshold = 30  # MPa
    low_strength_samples = df[df['concrete_strength_mpa'] < low_strength_threshold]
    
    if len(low_strength_samples) == 0:
        print("⚠ 没有找到不达标样本，选择一个中等强度样本进行分析")
        observed_sample = df[df['concrete_strength_mpa'].between(35, 40)].iloc[0:1]
    else:
        # 选择一个含泥量较高的不达标样本
        observed_sample = low_strength_samples.nlargest(1, 'mud_content_pct')
    
    print("\n实际观测的批次:")
    print("-"*80)
    print(f"  集料类型: {observed_sample['aggregate_type'].values[0]}")
    print(f"  产地: {observed_sample['origin'].values[0]}")
    print(f"  生产工艺: {observed_sample['production_process'].values[0]}")
    print(f"  材质密度: {observed_sample['material_density'].values[0]:.3f} g/cm³")
    print(f"  含泥量: {observed_sample['mud_content_pct'].values[0]:.2f}%")
    print(f"  泥块含量: {observed_sample['clay_lump_pct'].values[0]:.2f}%")
    print(f"  压碎值: {observed_sample['crushing_value_pct'].values[0]:.2f}%")
    print(f"  针片状含量: {observed_sample['flaky_particle_pct'].values[0]:.2f}%")
    print(f"  粒形指数: {observed_sample['particle_shape_index'].values[0]:.3f}")
    print(f"  水胶比: {observed_sample['water_binder_ratio'].values[0]:.3f}")
    print(f"  配合比准确性: {observed_sample['mix_design_accuracy_pct'].values[0]:.2f}%")
    print(f"  砂率: {observed_sample['sand_rate'].values[0]:.3f}")
    print(f"  → 混凝土强度: {observed_sample['concrete_strength_mpa'].values[0]:.2f} MPa")
    
    actual_strength = observed_sample['concrete_strength_mpa'].values[0]
    
    if actual_strength < low_strength_threshold:
        print(f"\n✗ 当前强度 {actual_strength:.2f} MPa < {low_strength_threshold} MPa (不达标)")
    else:
        print(f"\n✓ 当前强度 {actual_strength:.2f} MPa (中等)")
    
    # ===== 3. 反事实场景 1: 采用水洗工艺降低含泥量 =====
    print("\n" + "="*80)
    print("场景 1: 如果采用水洗工艺，降低含泥量")
    print("="*80)
    print(f"问题：如果当时采用水洗工艺，将含泥量从 {observed_sample['mud_content_pct'].values[0]:.2f}% 降至 1.0%，")
    print(f"      混凝土强度会提升到多少？")
    
    interventions_1 = {
        'mud_content_pct': 1.0,  # 水洗后含泥量降至1%
        'clay_lump_pct': 0.3     # 相应的泥块含量也降低
    }
    
    try:
        cf_result_1 = model.counterfactual_analysis(
            observed_data=observed_sample,
            interventions=interventions_1,
            target='concrete_strength_mpa'
        )
        
        cf_strength_1 = cf_result_1['counterfactual_mean']
        improvement_1 = cf_result_1['causal_effect']
        
        print(f"\n反事实预测结果:")
        print(f"  含泥量: {observed_sample['mud_content_pct'].values[0]:.2f}% → 1.00%")
        print(f"  泥块含量: {observed_sample['clay_lump_pct'].values[0]:.2f}% → 0.30%")
        print(f"  混凝土强度: {actual_strength:.2f} MPa → {cf_strength_1:.2f} MPa")
        print(f"  强度提升: {improvement_1:+.2f} MPa ({(improvement_1/actual_strength)*100:+.1f}%)")
        
        if improvement_1 > 0:
            print(f"\n✓ 结论：采用水洗工艺可显著提升强度 {improvement_1:.2f} MPa")
            if cf_strength_1 >= low_strength_threshold:
                print(f"✓ 优化后强度 {cf_strength_1:.2f} MPa 达标！")
        
    except Exception as e:
        print(f"\n✗ 场景1分析失败: {e}")
        cf_strength_1 = None
        improvement_1 = 0
    
    # ===== 4. 反事实场景 2: 优化水胶比 =====
    print("\n" + "="*80)
    print("场景 2: 优化水胶比")
    print("="*80)
    print(f"问题：如果将水胶比从 {observed_sample['water_binder_ratio'].values[0]:.3f} 优化至 0.42，")
    print(f"      混凝土强度会如何变化？")
    
    interventions_2 = {
        'water_binder_ratio': 0.42  # 优化水胶比
    }
    
    try:
        cf_result_2 = model.counterfactual_analysis(
            observed_data=observed_sample,
            interventions=interventions_2,
            target='concrete_strength_mpa'
        )
        
        cf_strength_2 = cf_result_2['counterfactual_mean']
        improvement_2 = cf_result_2['causal_effect']
        
        print(f"\n反事实预测结果:")
        print(f"  水胶比: {observed_sample['water_binder_ratio'].values[0]:.3f} → 0.420")
        print(f"  混凝土强度: {actual_strength:.2f} MPa → {cf_strength_2:.2f} MPa")
        print(f"  强度提升: {improvement_2:+.2f} MPa ({(improvement_2/actual_strength)*100:+.1f}%)")
        
        if improvement_2 > 0:
            print(f"\n✓ 结论：优化水胶比可提升强度 {improvement_2:.2f} MPa")
            if cf_strength_2 >= low_strength_threshold:
                print(f"✓ 优化后强度 {cf_strength_2:.2f} MPa 达标！")
        
    except Exception as e:
        print(f"\n✗ 场景2分析失败: {e}")
        cf_strength_2 = None
        improvement_2 = 0
    
    # ===== 5. 反事实场景 3: 综合优化策略 =====
    print("\n" + "="*80)
    print("场景 3: 综合优化策略")
    print("="*80)
    print("问题：如果同时采用水洗工艺、优化水胶比、提高配合比准确性，")
    print("      混凝土强度会达到什么水平？")
    
    interventions_3 = {
        'mud_content_pct': 1.0,              # 水洗降低含泥量
        'clay_lump_pct': 0.3,                # 降低泥块含量
        'water_binder_ratio': 0.42,          # 优化水胶比
        'mix_design_accuracy_pct': 96.0      # 提高配合比准确性
    }
    
    try:
        cf_result_3 = model.counterfactual_analysis(
            observed_data=observed_sample,
            interventions=interventions_3,
            target='concrete_strength_mpa'
        )
        
        cf_strength_3 = cf_result_3['counterfactual_mean']
        improvement_3 = cf_result_3['causal_effect']
        
        print(f"\n反事实预测结果:")
        print(f"  含泥量: {observed_sample['mud_content_pct'].values[0]:.2f}% → 1.00%")
        print(f"  水胶比: {observed_sample['water_binder_ratio'].values[0]:.3f} → 0.420")
        print(f"  配合比准确性: {observed_sample['mix_design_accuracy_pct'].values[0]:.2f}% → 96.00%")
        print(f"  混凝土强度: {actual_strength:.2f} MPa → {cf_strength_3:.2f} MPa")
        print(f"  强度提升: {improvement_3:+.2f} MPa ({(improvement_3/actual_strength)*100:+.1f}%)")
        
        if improvement_3 > 0:
            print(f"\n✓ 结论：综合优化策略可大幅提升强度 {improvement_3:.2f} MPa")
            if cf_strength_3 >= low_strength_threshold:
                print(f"✓ 优化后强度 {cf_strength_3:.2f} MPa 显著达标！")
        
    except Exception as e:
        print(f"\n✗ 场景3分析失败: {e}")
        cf_strength_3 = None
        improvement_3 = 0
    
    # ===== 6. 可视化对比 =====
    print("\n" + "="*80)
    print("步骤 3: 可视化对比各场景效果")
    print("="*80)
    
    scenarios = ['实际情况\n(现状)']
    strengths = [actual_strength]
    
    if cf_strength_1 is not None:
        scenarios.append('场景1\n(水洗工艺)')
        strengths.append(cf_strength_1)
    
    if cf_strength_2 is not None:
        scenarios.append('场景2\n(优化水胶比)')
        strengths.append(cf_strength_2)
    
    if cf_strength_3 is not None:
        scenarios.append('场景3\n(综合优化)')
        strengths.append(cf_strength_3)
    
    # 绘制对比图
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
    bars = ax.bar(scenarios, strengths, color=colors[:len(scenarios)], 
                  alpha=0.8, edgecolor='black', linewidth=2)
    
    # 添加数值标签
    for i, (bar, strength) in enumerate(zip(bars, strengths)):
        height = bar.get_height()
        if i == 0:
            label = f'{strength:.1f} MPa'
        else:
            diff = strength - strengths[0]
            label = f'{strength:.1f} MPa\n({diff:+.1f})'
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                label, ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 添加达标线
    ax.axhline(y=low_strength_threshold, color='red', linestyle='--', 
               linewidth=2, label=f'达标线 ({low_strength_threshold} MPa)')
    
    # 添加实际情况基准线
    ax.axhline(y=actual_strength, color='gray', linestyle=':', 
               linewidth=1.5, alpha=0.5, label='当前水平')
    
    ax.set_ylabel('混凝土强度 (MPa)', fontsize=14, fontweight='bold')
    ax.set_title('反事实分析：不同优化策略的效果对比', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim([min(strengths) - 5, max(strengths) + 10])
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax.legend(fontsize=11, loc='upper left')
    
    plt.tight_layout()
    fig.savefig(figures_dir / "concrete_counterfactual_analysis.png", dpi=300, bbox_inches='tight')
    print(f"✓ 图表已保存: {figures_dir / 'concrete_counterfactual_analysis.png'}")
    plt.close()
    
    # ===== 7. 生成决策建议 =====
    print("\n" + "="*80)
    print("反事实分析总结与决策建议")
    print("="*80)
    
    # 检查是否有成功的场景
    if all([cf_strength_1 is not None, cf_strength_2 is not None, cf_strength_3 is not None]):
        
        improvements = {
            '采用水洗工艺': improvement_1,
            '优化水胶比': improvement_2,
            '综合优化策略': improvement_3
        }
        
        # 按效果排序
        sorted_improvements = sorted(improvements.items(), key=lambda x: x[1], reverse=True)
        
        print("\n措施效果排名:")
        for i, (measure, improvement) in enumerate(sorted_improvements, 1):
            if improvement > 0:
                print(f"  {i}. {measure}: +{improvement:.2f} MPa ⭐")
            else:
                print(f"  {i}. {measure}: {improvement:.2f} MPa")
        
        print("\n推荐行动方案:")
        if sorted_improvements[0][1] > 0:
            print(f"  ✓ 最佳方案: {sorted_improvements[0][0]}")
            print(f"    预期效果: 强度提升 {sorted_improvements[0][1]:.2f} MPa")
            
            final_strength = actual_strength + sorted_improvements[0][1]
            if final_strength >= low_strength_threshold:
                print(f"    优化后强度: {final_strength:.2f} MPa (达标)")
            else:
                print(f"    优化后强度: {final_strength:.2f} MPa (仍需进一步改进)")
        else:
            print("  ⚠ 当前工艺已接近最优，无需大幅调整")
    else:
        print("\n⚠ 反事实分析未能完成，无法生成决策建议")
        print("  可能原因：模型拟合问题或数据质量问题")
        sorted_improvements = []  # 定义为空列表，避免后续引用错误
    
    # 保存分析报告
    report_path = reports_dir / "counterfactual_analysis_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("混凝土集料反事实分析报告\n")
        f.write("="*80 + "\n\n")
        
        f.write("实际观测批次:\n")
        f.write(f"  混凝土强度: {actual_strength:.2f} MPa\n")
        f.write(f"  含泥量: {observed_sample['mud_content_pct'].values[0]:.2f}%\n")
        f.write(f"  水胶比: {observed_sample['water_binder_ratio'].values[0]:.3f}\n\n")
        
        f.write("反事实场景效果:\n")
        if cf_strength_1 is not None:
            f.write(f"  场景1 (水洗工艺): {cf_strength_1:.2f} MPa ({improvement_1:+.2f} MPa)\n")
        if cf_strength_2 is not None:
            f.write(f"  场景2 (优化水胶比): {cf_strength_2:.2f} MPa ({improvement_2:+.2f} MPa)\n")
        if cf_strength_3 is not None:
            f.write(f"  场景3 (综合优化): {cf_strength_3:.2f} MPa ({improvement_3:+.2f} MPa)\n")
        
        f.write("\n推荐措施:\n")
        if sorted_improvements[0][1] > 0:
            f.write(f"  最佳方案: {sorted_improvements[0][0]}\n")
            f.write(f"  预期提升: {sorted_improvements[0][1]:.2f} MPa\n")
    
    print(f"\n✓ 分析报告已保存: {report_path}")
    
    print("\n" + "="*80)
    print("核心价值:")
    print("  ✓ 无需实际试验即可预测不同决策的效果")
    print("  ✓ 为混凝土配合比优化提供定量依据")
    print("  ✓ 降低试错成本，提高工程质量")
    print("  ✓ 支持\"如果当时...会怎样\"的反事实推理")
    print("="*80)


if __name__ == "__main__":
    main()

