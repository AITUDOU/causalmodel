"""
快速测试脚本 - 因果分析智能体系统
单个查询测试，用于快速验证系统功能

应用场景：混凝土配合比质量控制与优化
数据来源：UCI Machine Learning Repository (Yeh 1998)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
import warnings
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# 加载 .env 配置
load_dotenv()

# 检查 API Key
if not os.getenv('OPENAI_API_KEY'):
    print("⚠️  错误：未找到 OPENAI_API_KEY")
    print("   请确保 .env 文件存在且包含正确的配置")
    sys.exit(1)

print(f"✓ 使用模型: {os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}")
print(f"✓ API地址: {os.getenv('OPENAI_API_BASE', 'default')}\n")

from src.causal_agent_system import (
    initialize_causal_model,
    create_causal_agent_graph
)

# ============================================================================
# 初始化系统
# ============================================================================

print("🚀 因果分析智能体系统 - 快速测试\n")

print("🔧 初始化因果模型...")
try:
    # 优先从缓存加载
    causal_model = initialize_causal_model()
    print()
except ValueError as e:
    # 如果缓存不存在，加载数据并训练
    print("⚠️  未找到缓存模型，正在训练...")
    print("📦 加载真实混凝土数据（UCI数据集）...")
    df_real = pd.read_csv('data/real/concrete_compressive_strength.csv')
    df_real.columns = df_real.columns.str.strip()  # 清理列名
    print(f"✓ 已加载 {len(df_real)} 条记录\n")
    causal_model = initialize_causal_model(df_real)
    print()
    print("💡 提示：下次运行将直接使用缓存，启动更快！")
    print()

print("🏗️  构建智能体工作流...")
agent_graph = create_causal_agent_graph()
print("✓ 系统就绪\n")

# ============================================================================
# 选择参考批次（用于反事实分析）
# ============================================================================

print("=" * 80)
print("📦 选择参考批次（反事实分析基准）")
print("=" * 80)
print()
print("反事实分析需要一个实际批次作为基准，然后模拟\"如果改变某些参数会怎样\"。")
print()

# 读取真实数据
df = pd.read_csv('data/real/concrete_compressive_strength.csv')
df.columns = df.columns.str.strip()  # 清理列名

# 提供几个典型样本供选择
print("请选择一个参考批次（真实UCI数据集）：\n")

# 计算水胶比用于选择样本（临时计算，不添加到df）
total_binder = df['cement'] + df['blast_furnace_slag'] + df['fly_ash']
water_binder_ratio = df['water'] / total_binder

# 选项1：低强度样本（28d）
low_strength_sample = df[df['age'] == 28].nsmallest(1, 'concrete_compressive_strength').iloc[0]
low_wb = low_strength_sample['water'] / (low_strength_sample['cement'] + low_strength_sample['blast_furnace_slag'] + low_strength_sample['fly_ash'])
print("1️⃣  低强度批次（28天，需要优化）")
print(f"   水泥: {low_strength_sample['cement']:.0f} | 矿渣: {low_strength_sample['blast_furnace_slag']:.0f} | 粉煤灰: {low_strength_sample['fly_ash']:.0f}")
print(f"   水: {low_strength_sample['water']:.0f} | 减水剂: {low_strength_sample['superplasticizer']:.2f}")
print(f"   水胶比≈{low_wb:.3f} | 龄期: {low_strength_sample['age']:.0f}天")
print(f"   → 强度: {low_strength_sample['concrete_compressive_strength']:.1f} MPa ⚠️\n")

# 选项2：中等强度样本（28d）
medium_samples = df[df['age'] == 28]
median_strength = medium_samples['concrete_compressive_strength'].median()
medium_strength_sample = medium_samples.iloc[(medium_samples['concrete_compressive_strength'] - median_strength).abs().argmin()]
medium_wb = medium_strength_sample['water'] / (medium_strength_sample['cement'] + medium_strength_sample['blast_furnace_slag'] + medium_strength_sample['fly_ash'])
print("2️⃣  中等强度批次（28天，标准配合比）")
print(f"   水泥: {medium_strength_sample['cement']:.0f} | 矿渣: {medium_strength_sample['blast_furnace_slag']:.0f} | 粉煤灰: {medium_strength_sample['fly_ash']:.0f}")
print(f"   水: {medium_strength_sample['water']:.0f} | 减水剂: {medium_strength_sample['superplasticizer']:.2f}")
print(f"   水胶比≈{medium_wb:.3f} | 龄期: {medium_strength_sample['age']:.0f}天")
print(f"   → 强度: {medium_strength_sample['concrete_compressive_strength']:.1f} MPa ✓\n")

# 选项3：高强度样本（28d）
high_strength_sample = df[df['age'] == 28].nlargest(1, 'concrete_compressive_strength').iloc[0]
high_wb = high_strength_sample['water'] / (high_strength_sample['cement'] + high_strength_sample['blast_furnace_slag'] + high_strength_sample['fly_ash'])
print("3️⃣  高强度批次（28天，优质配合比）")
print(f"   水泥: {high_strength_sample['cement']:.0f} | 矿渣: {high_strength_sample['blast_furnace_slag']:.0f} | 粉煤灰: {high_strength_sample['fly_ash']:.0f}")
print(f"   水: {high_strength_sample['water']:.0f} | 减水剂: {high_strength_sample['superplasticizer']:.2f}")
print(f"   水胶比≈{high_wb:.3f} | 龄期: {high_strength_sample['age']:.0f}天")
print(f"   → 强度: {high_strength_sample['concrete_compressive_strength']:.1f} MPa 🌟\n")

# 选项4：接近图片配合比的样本（水胶比≈0.43, 28天）
target_samples_28d = df[df['age'] == 28].copy()
target_wb_ratios = target_samples_28d['water'] / (target_samples_28d['cement'] + target_samples_28d['blast_furnace_slag'] + target_samples_28d['fly_ash'])
target_sample_idx = (target_wb_ratios - 0.43).abs().idxmin()
target_sample = df.loc[target_sample_idx]
target_wb_val = target_sample['water'] / (target_sample['cement'] + target_sample['blast_furnace_slag'] + target_sample['fly_ash'])
print("4️⃣  图片配合比（28天，水胶比≈0.43）")
print(f"   水泥: {target_sample['cement']:.0f} | 矿渣: {target_sample['blast_furnace_slag']:.0f} | 粉煤灰: {target_sample['fly_ash']:.0f}")
print(f"   水: {target_sample['water']:.0f} | 减水剂: {target_sample['superplasticizer']:.2f}")
print(f"   水胶比≈{target_wb_val:.3f} | 龄期: {target_sample['age']:.0f}天")
print(f"   → 强度: {target_sample['concrete_compressive_strength']:.1f} MPa\n")

choice = input("请选择参考批次 (1-4) 或按回车随机选择: ").strip()

if choice == '1':
    selected_sample = low_strength_sample
    sample_idx = low_strength_sample.name
elif choice == '2':
    selected_sample = medium_strength_sample
    sample_idx = medium_strength_sample.name
elif choice == '3':
    selected_sample = high_strength_sample
    sample_idx = high_strength_sample.name
elif choice == '4':
    selected_sample = target_sample
    sample_idx = target_sample_idx
else:
    # 随机选择
    sample_idx = np.random.randint(0, len(df))
    selected_sample = df.iloc[sample_idx]
    print(f"→ 随机选择批次 #{sample_idx}")

selected_wb = selected_sample['water'] / (selected_sample['cement'] + selected_sample['blast_furnace_slag'] + selected_sample['fly_ash'])

print("\n" + "="*80)
print("✅ 已选择参考批次")
print("="*80)
print(f"批次编号: #{sample_idx}")
print(f"水泥: {selected_sample['cement']:.0f} kg/m³")
print(f"矿渣: {selected_sample['blast_furnace_slag']:.0f} kg/m³")
print(f"粉煤灰: {selected_sample['fly_ash']:.0f} kg/m³")
print(f"水: {selected_sample['water']:.0f} kg/m³")
print(f"减水剂: {selected_sample['superplasticizer']:.2f} kg/m³")
print(f"粗骨料: {selected_sample['coarse_aggregate']:.0f} kg/m³")
print(f"细骨料: {selected_sample['fine_aggregate']:.0f} kg/m³")
print(f"龄期: {selected_sample['age']:.0f} 天")
print(f"水胶比≈{selected_wb:.3f}")
print(f"→ 抗压强度: {selected_sample['concrete_compressive_strength']:.1f} MPa")
print("="*80)
print()

# 保存选择的样本索引到全局变量（供反事实分析使用）
selected_sample_index = sample_idx

# ============================================================================
# 交互式查询
# ============================================================================

print("=" * 80)
print("💬 交互式因果分析 - 混凝土配合比优化")
print("=" * 80)
print()
print("您可以提问：")
print()
print("  📊 归因分析（问题诊断）：")
print("     • \"为什么抗压强度下降了？\"")
print("     • \"强度变化的主要驱动因素是什么？\"")
print("     • \"是什么导致强度不达标？\"")
print()
print("  🔧 干预分析（方案优化）：")
print("     • \"如何提高混凝土强度？\"")
print("     • \"哪些配合比参数对强度影响最大？\"")
print("     • \"如何在保证强度的前提下降低成本？\"")
print()
print("  🔮 反事实分析（效果预测，基于选择的批次）：")
print(f"     • \"如果水用量从{selected_sample['water']:.0f}降到150会怎样？\"")
print("     • \"增加水泥用量50 kg/m³能提升多少强度？\"")
print(f"     • \"如果添加矿渣100 kg/m³，强度会改善吗？\"")
print(f"     • \"龄期延长到90天，强度能达到多少？\"")
print()
print("💡 提示：反事实分析将基于您选择的参考批次进行模拟")
print(f"   当前参考批次 #{sample_idx}: 水胶比≈{selected_wb:.3f}, 强度{selected_sample['concrete_compressive_strength']:.1f} MPa")
print()
print("输入 'quit' 或 'exit' 退出")
print("=" * 80)
print()

while True:
    # 获取用户输入
    user_query = input("👤 您的问题: ").strip()
    
    if not user_query:
        continue
    
    if user_query.lower() in ['quit', 'exit', 'q']:
        print("\n👋 感谢使用！")
        break
    
    try:
        # 执行分析（传入参考批次索引）
        print()
        result = agent_graph.invoke({
            "user_query": user_query,
            "reference_sample_index": int(sample_idx)
        })
        
        # 显示结果
        print("\n" + "=" * 80)
        print("📊 分析结果")
        print("=" * 80)
        print(f"\n🎯 分析类型: {result['analysis_type']}")
        print(f"📈 目标变量: {result['target_variable']}")
        
        # 如果有详细结果，显示关键数据
        if 'causal_results' in result and result['causal_results']:
            causal_results = result['causal_results']
            
            if causal_results.get('type') == 'attribution':
                print(f"\n📋 主要影响因素（Top 3）:")
                for i, factor in enumerate(causal_results.get('top_factors', [])[:3], 1):
                    print(f"   {i}. {factor['variable']}: 贡献 {factor['contribution']:.4f}")
            
            elif causal_results.get('type') == 'intervention':
                interventions = causal_results.get('interventions', [])
                if interventions:
                    sorted_interventions = sorted(interventions, key=lambda x: abs(x['causal_effect']), reverse=True)
                    print(f"\n📋 最有效的干预措施（Top 3）:")
                    for i, inter in enumerate(sorted_interventions[:3], 1):
                        print(f"   {i}. {inter['variable']}: 效应 {inter['causal_effect']:.4f}")
            
            elif causal_results.get('type') == 'counterfactual':
                interventions = causal_results.get('interventions', [])
                print(f"\n📋 反事实模拟:")
                for interv in interventions:
                    orig_val = interv.get('original_value')
                    new_val = interv.get('new_value')
                    if orig_val is not None:
                        print(f"   {interv.get('variable')}: {orig_val:.4f} → {new_val:.4f}")
                    else:
                        print(f"   {interv.get('variable')}: → {new_val:.4f}")
                print(f"   观测值: {causal_results.get('observed_mean', 0):.4f}")
                print(f"   反事实值: {causal_results.get('counterfactual_mean', 0):.4f}")
                print(f"   预期变化: {causal_results.get('causal_effect', 0):.4f}")
        
        print(f"\n💡 决策建议:\n{result['recommendations']}")
        print("\n" + "=" * 80)
        print()
        
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，退出程序")
        break
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        print()

