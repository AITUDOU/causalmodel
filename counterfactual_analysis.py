"""
筛分试验反事实分析
回答问题："如果当时采用了不同的工艺参数，质量会如何？"
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from dowhy import gcm

print("=" * 80)
print("筛分试验反事实分析")
print("=" * 80)
print()

# ============================================================================
# 场景设定
# ============================================================================
print("场景设定:")
print("-" * 80)
print("""
某生产批次的筛分试验已完成：
- 集料类型：中砂
- 初始含水量：4.2%
- 烘干时间：28分钟
- 筛分效率：72%
- 最终产品质量：75分

现在我们想知道：
1. 如果当时烘干时间延长到35分钟，质量会提升到多少？
2. 如果当时选用的是优质集料，质量会有多大改善？
3. 如果什么改进都不做，但初始含水量降低到3%，效果如何？
""")
print()

# ============================================================================
# 第一步：加载数据并构建可逆因果模型
# ============================================================================
print("第一步：构建可逆因果模型...")
print("-" * 80)

# 加载完整数据集
df = pd.read_csv('data/synthetic/screening_data_v2.csv')

# 只保留核心变量（简化模型）
core_vars = ['aggregate_type', 'aggregate_quality', 'initial_moisture', 
             'drying_time', 'residual_moisture', 'screening_efficiency',
             'fineness_modulus', 'product_quality_score']

df_model = df[core_vars].copy()

# 处理分类变量
categorical_cols = ['aggregate_type', 'aggregate_quality']
for col in categorical_cols:
    df_model[col] = pd.Categorical(df_model[col]).codes

print(f"✓ 使用 {len(df_model)} 条历史数据训练模型")
print(f"✓ 核心变量: {', '.join(core_vars)}")
print()

# 构建因果图（简化版，只保留主要路径）
edges = [
    ('aggregate_quality', 'screening_efficiency'),
    ('aggregate_type', 'screening_efficiency'),
    ('initial_moisture', 'drying_time'),
    ('drying_time', 'residual_moisture'),
    ('residual_moisture', 'screening_efficiency'),
    ('screening_efficiency', 'fineness_modulus'),
    ('screening_efficiency', 'product_quality_score'),
    ('fineness_modulus', 'product_quality_score'),
    ('residual_moisture', 'product_quality_score'),
]

causal_graph = nx.DiGraph(edges)

print("因果图结构:")
print(f"  节点数: {len(causal_graph.nodes)}")
print(f"  边数: {len(causal_graph.edges)}")
print()

# 使用可逆结构因果模型（关键！）
print("创建可逆结构因果模型...")
causal_model = gcm.InvertibleStructuralCausalModel(causal_graph)

# 自动分配因果机制
print("自动分配因果机制...")
gcm.auto.assign_causal_mechanisms(causal_model, df_model)

# 拟合模型
print("拟合模型...")
gcm.fit(causal_model, df_model)

print("✓ 可逆因果模型构建完成！")
print()

# ============================================================================
# 第二步：定义观测到的实际情况
# ============================================================================
print("\n第二步：定义实际观测场景...")
print("-" * 80)

# 创建实际观测的批次数据
# 这是已经发生的真实情况
observed_batch = pd.DataFrame({
    'aggregate_type': [1],  # 中砂（编码后）
    'aggregate_quality': [1],  # 良（编码后）
    'initial_moisture': [4.2],
    'drying_time': [28.0],
    'residual_moisture': [0.42],
    'screening_efficiency': [72.0],
    'fineness_modulus': [2.85],
    'product_quality_score': [75.0]
})

print("实际观测的批次数据:")
print("-" * 40)
print(f"  集料类型: 中砂")
print(f"  集料质量: 良")
print(f"  初始含水量: {observed_batch['initial_moisture'].values[0]:.1f}%")
print(f"  烘干时间: {observed_batch['drying_time'].values[0]:.0f}分钟")
print(f"  残留含水量: {observed_batch['residual_moisture'].values[0]:.2f}%")
print(f"  筛分效率: {observed_batch['screening_efficiency'].values[0]:.0f}%")
print(f"  产品质量评分: {observed_batch['product_quality_score'].values[0]:.1f}分")
print()

# ============================================================================
# 第三步：反事实场景1 - 如果延长烘干时间
# ============================================================================
print("\n第三步：反事实场景1 - 延长烘干时间")
print("-" * 80)
print("问题：如果当时烘干时间从28分钟延长到35分钟，质量会提升多少？")
print()

try:
    counterfactual_longer_drying = gcm.counterfactual_samples(
        causal_model,
        {'drying_time': lambda x: 35.0},  # 干预：设定烘干时间为35分钟
        observed_data=observed_batch
    )
    
    actual_quality = observed_batch['product_quality_score'].values[0]
    cf_quality_1 = counterfactual_longer_drying['product_quality_score'].values[0]
    improvement_1 = cf_quality_1 - actual_quality
    
    print("反事实结果:")
    print(f"  烘干时间: 28分钟 → 35分钟")
    print(f"  残留含水量: {observed_batch['residual_moisture'].values[0]:.2f}% → {counterfactual_longer_drying['residual_moisture'].values[0]:.2f}%")
    print(f"  筛分效率: {observed_batch['screening_efficiency'].values[0]:.1f}% → {counterfactual_longer_drying['screening_efficiency'].values[0]:.1f}%")
    print(f"  产品质量: {actual_quality:.1f}分 → {cf_quality_1:.1f}分")
    print(f"  质量提升: {improvement_1:+.2f}分 ({(improvement_1/actual_quality)*100:+.1f}%)")
    print()
    
    if improvement_1 > 0:
        print(f"✓ 结论：延长烘干时间可以提升质量 {improvement_1:.2f}分")
    else:
        print(f"✗ 结论：延长烘干时间反而会降低质量 {improvement_1:.2f}分（可能过度烘干）")
    print()
    
except Exception as e:
    print(f"场景1分析失败: {e}")
    counterfactual_longer_drying = None

# ============================================================================
# 第四步：反事实场景2 - 如果使用优质集料
# ============================================================================
print("\n第四步：反事实场景2 - 使用优质集料")
print("-" * 80)
print("问题：如果当时选用优质集料（而非良等），质量会提升多少？")
print()

try:
    counterfactual_better_quality = gcm.counterfactual_samples(
        causal_model,
        {'aggregate_quality': lambda x: 0},  # 干预：优质集料（编码0）
        observed_data=observed_batch
    )
    
    actual_quality = observed_batch['product_quality_score'].values[0]
    cf_quality_2 = counterfactual_better_quality['product_quality_score'].values[0]
    improvement_2 = cf_quality_2 - actual_quality
    
    print("反事实结果:")
    print(f"  集料质量: 良 → 优")
    print(f"  筛分效率: {observed_batch['screening_efficiency'].values[0]:.1f}% → {counterfactual_better_quality['screening_efficiency'].values[0]:.1f}%")
    print(f"  产品质量: {actual_quality:.1f}分 → {cf_quality_2:.1f}分")
    print(f"  质量提升: {improvement_2:+.2f}分 ({(improvement_2/actual_quality)*100:+.1f}%)")
    print()
    
    if improvement_2 > 0:
        print(f"✓ 结论：选用优质集料可以提升质量 {improvement_2:.2f}分")
    else:
        print(f"✗ 结论：集料质量提升对该批次影响不大")
    print()
    
except Exception as e:
    print(f"场景2分析失败: {e}")
    counterfactual_better_quality = None

# ============================================================================
# 第五步：反事实场景3 - 如果降低初始含水量
# ============================================================================
print("\n第五步：反事实场景3 - 降低初始含水量")
print("-" * 80)
print("问题：如果初始含水量从4.2%降低到3.0%，质量会如何变化？")
print()

try:
    counterfactual_lower_moisture = gcm.counterfactual_samples(
        causal_model,
        {'initial_moisture': lambda x: 3.0},  # 干预：降低初始含水量
        observed_data=observed_batch
    )
    
    actual_quality = observed_batch['product_quality_score'].values[0]
    cf_quality_3 = counterfactual_lower_moisture['product_quality_score'].values[0]
    improvement_3 = cf_quality_3 - actual_quality
    
    print("反事实结果:")
    print(f"  初始含水量: 4.2% → 3.0%")
    print(f"  烘干时间: {observed_batch['drying_time'].values[0]:.1f}分钟 → {counterfactual_lower_moisture['drying_time'].values[0]:.1f}分钟")
    print(f"  残留含水量: {observed_batch['residual_moisture'].values[0]:.2f}% → {counterfactual_lower_moisture['residual_moisture'].values[0]:.2f}%")
    print(f"  筛分效率: {observed_batch['screening_efficiency'].values[0]:.1f}% → {counterfactual_lower_moisture['screening_efficiency'].values[0]:.1f}%")
    print(f"  产品质量: {actual_quality:.1f}分 → {cf_quality_3:.1f}分")
    print(f"  质量提升: {improvement_3:+.2f}分 ({(improvement_3/actual_quality)*100:+.1f}%)")
    print()
    
    if improvement_3 > 0:
        print(f"✓ 结论：降低初始含水量可以提升质量 {improvement_3:.2f}分")
    else:
        print(f"✗ 结论：初始含水量降低对质量影响不大")
    print()
    
except Exception as e:
    print(f"场景3分析失败: {e}")
    counterfactual_lower_moisture = None

# ============================================================================
# 第六步：可视化对比
# ============================================================================
print("\n第六步：可视化对比...")
print("-" * 80)

try:
    # 准备数据
    scenarios = ['实际情况\n(现状)']
    quality_scores = [observed_batch['product_quality_score'].values[0]]
    
    if counterfactual_longer_drying is not None:
        scenarios.append('场景1\n(延长烘干)')
        quality_scores.append(counterfactual_longer_drying['product_quality_score'].values[0])
    
    if counterfactual_better_quality is not None:
        scenarios.append('场景2\n(优质集料)')
        quality_scores.append(counterfactual_better_quality['product_quality_score'].values[0])
    
    if counterfactual_lower_moisture is not None:
        scenarios.append('场景3\n(降低含水量)')
        quality_scores.append(counterfactual_lower_moisture['product_quality_score'].values[0])
    
    # 绘制对比图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    bars = ax.bar(scenarios, quality_scores, color=colors[:len(scenarios)], alpha=0.7, edgecolor='black', linewidth=2)
    
    # 添加数值标签
    for i, (bar, score) in enumerate(zip(bars, quality_scores)):
        height = bar.get_height()
        if i == 0:
            label = f'{score:.1f}'
        else:
            diff = score - quality_scores[0]
            label = f'{score:.1f}\n({diff:+.1f})'
        ax.text(bar.get_x() + bar.get_width()/2., height,
                label, ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 添加基准线
    ax.axhline(y=quality_scores[0], color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    
    ax.set_ylabel('Product Quality Score', fontsize=13, fontweight='bold')
    ax.set_title('Counterfactual Analysis: What-If Scenarios', fontsize=15, fontweight='bold', pad=20)
    ax.set_ylim([min(quality_scores) - 5, max(quality_scores) + 5])
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('results/figures', exist_ok=True)
    plt.savefig('results/figures/counterfactual_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ 图表已保存至: results/figures/counterfactual_analysis.png")
    plt.close()
    
except Exception as e:
    print(f"可视化失败: {e}")

# ============================================================================
# 第七步：生成决策建议
# ============================================================================
print("\n" + "=" * 80)
print("反事实分析总结与决策建议")
print("=" * 80)
print()

print("问题：对于类似的批次，我们应该采取什么措施？")
print("-" * 80)

if all([counterfactual_longer_drying is not None, 
        counterfactual_better_quality is not None,
        counterfactual_lower_moisture is not None]):
    
    improvements = {
        '延长烘干时间': cf_quality_1 - actual_quality,
        '使用优质集料': cf_quality_2 - actual_quality,
        '降低初始含水量': cf_quality_3 - actual_quality
    }
    
    # 按效果排序
    sorted_improvements = sorted(improvements.items(), key=lambda x: x[1], reverse=True)
    
    print("\n措施效果排名:")
    for i, (measure, improvement) in enumerate(sorted_improvements, 1):
        if improvement > 0:
            print(f"  {i}. {measure}: +{improvement:.2f}分 ⭐")
        else:
            print(f"  {i}. {measure}: {improvement:.2f}分")
    
    print("\n推荐行动方案:")
    if sorted_improvements[0][1] > 0:
        print(f"  ✓ 优先措施: {sorted_improvements[0][0]}")
        print(f"    预期效果: 质量提升 {sorted_improvements[0][1]:.2f}分")
        
        if sorted_improvements[1][1] > 0:
            print(f"  ✓ 次要措施: {sorted_improvements[1][0]}")
            print(f"    预期效果: 质量提升 {sorted_improvements[1][1]:.2f}分")
            print(f"  ✓ 如果同时采取两项措施，预期累计提升约 {sorted_improvements[0][1] + sorted_improvements[1][1]:.2f}分")
    else:
        print("  ⚠ 当前工艺已接近最优，无需调整")

print()
print("=" * 80)
print("核心价值：")
print("  ✓ 无需实际试验即可预测不同决策的效果")
print("  ✓ 为工艺优化提供定量依据")
print("  ✓ 降低试错成本，提高决策效率")
print("=" * 80)

