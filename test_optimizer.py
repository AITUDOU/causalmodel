#!/usr/bin/env python3
"""
测试新的精确目标优化功能
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.causal_agent_system import initialize_causal_model, create_causal_agent_graph
import pandas as pd

# 初始化因果模型
print("="*80)
print("初始化因果模型...")
print("="*80)

df = pd.read_csv('data/real/concrete_compressive_strength.csv')
causal_model = initialize_causal_model(df, force_retrain=False)

# 创建agent图
print("\n创建Agent工作流...")
agent_graph = create_causal_agent_graph()

# 测试用例1：要求提升10%
print("\n" + "="*80)
print("测试用例1: 要求提升10%")
print("="*80)

state = {
    "user_query": "如果我想强度提升10%，应该如何调整配合比？",
    "reference_sample_index": 100,
    "observed_config": None
}

result = agent_graph.invoke(state)

print("\n📊 最终结果：")
print(f"  分析类型: {result['analysis_type']}")
print(f"  目标提升: {result.get('target_improvement')}%")
print(f"  实际提升: {((result.get('predicted_strength', 0) - 35.0) / 35.0 * 100):.1f}%")
print(f"  预测强度: {result.get('predicted_strength', 0):.2f} MPa")

# 测试用例2：要求提升5%
print("\n" + "="*80)
print("测试用例2: 要求提升5%")
print("="*80)

state = {
    "user_query": "如果我想强度提升5%，应该如何调整配合比？",
    "reference_sample_index": 100,
    "observed_config": None
}

result = agent_graph.invoke(state)

print("\n📊 最终结果：")
print(f"  分析类型: {result['analysis_type']}")
print(f"  目标提升: {result.get('target_improvement')}%")
if result.get('optimized_config'):
    base_strength = result.get('optimized_config', {}).get('concrete_compressive_strength', 35.0)
    actual_improvement = ((result.get('predicted_strength', 0) - base_strength) / base_strength * 100)
    print(f"  实际提升: {actual_improvement:.1f}%")
print(f"  预测强度: {result.get('predicted_strength', 0):.2f} MPa")

# 测试用例3：要求提升20%
print("\n" + "="*80)
print("测试用例3: 要求提升20%")
print("="*80)

state = {
    "user_query": "如果我想强度提升20%，应该如何调整配合比？",
    "reference_sample_index": 100,
    "observed_config": None
}

result = agent_graph.invoke(state)

print("\n📊 最终结果：")
print(f"  分析类型: {result['analysis_type']}")
print(f"  目标提升: {result.get('target_improvement')}%")
if result.get('optimized_config'):
    base_strength = result.get('optimized_config', {}).get('concrete_compressive_strength', 35.0)
    actual_improvement = ((result.get('predicted_strength', 0) - base_strength) / base_strength * 100)
    print(f"  实际提升: {actual_improvement:.1f}%")
print(f"  预测强度: {result.get('predicted_strength', 0):.2f} MPa")

print("\n✅ 测试完成！")


