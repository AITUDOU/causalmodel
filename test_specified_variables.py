"""
测试用户指定变量的干预分析
验证系统能否按照用户指定的变量（如水泥和粉煤灰）进行优化
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.causal_agent_system import (
    initialize_causal_model,
    create_causal_agent_graph
)

print("="*80)
print("测试用户指定变量的干预分析")
print("="*80)
print()

# 初始化模型
print("📦 加载因果模型...")
causal_model = initialize_causal_model()
print("✓ 模型加载完成\n")

# 创建智能体
print("🏗️  构建智能体工作流...")
agent_graph = create_causal_agent_graph()
print("✓ 工作流构建完成\n")

# 测试查询：用户明确指定要调整水泥和粉煤灰
query = "现在我想强度达到45，水泥和粉煤灰应该怎么调？"

print("="*80)
print(f"🔍 测试查询: {query}")
print("="*80)
print()

# 执行分析
result = agent_graph.invoke({
    "user_query": query
})

print("\n" + "="*80)
print("📊 分析结果")
print("="*80)
print()

print(f"分析类型: {result.get('analysis_type')}")
print(f"目标变量: {result.get('target_variable')}")
print(f"路由推理: {result.get('routing_reasoning')}")

if result.get('target_value'):
    print(f"目标值: {result.get('target_value')} MPa")

if result.get('specified_variables'):
    print(f"用户指定变量: {', '.join(result.get('specified_variables'))}")

print()
print("优化配比摘要:")
print(result.get('optimization_summary', ''))

print()
print("决策建议:")
print(result.get('recommendations', ''))

print("\n" + "="*80)
print("✅ 测试完成")
print("="*80)

