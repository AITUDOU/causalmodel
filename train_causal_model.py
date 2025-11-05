"""
因果模型训练脚本
一次性训练并保存模型，后续使用时直接加载
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.causal_model import ConcreteAggregateCausalModel

print("=" * 80)
print("🔧 因果模型训练脚本")
print("=" * 80)
print()

# 配置
DATA_FILE = 'data/synthetic/concrete_aggregate_data.csv'
MODEL_DIR = Path('models')
MODEL_FILE = MODEL_DIR / 'causal_model.pkl'

# 创建模型目录
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 加载数据
print("📦 加载混凝土集料数据...")
df = pd.read_csv(DATA_FILE)
print(f"✓ 数据加载完成：{len(df)} 条记录，{len(df.columns)} 个变量")
print()

# 训练模型
print("🔧 开始训练因果模型（这需要1-2分钟）...")
print("-" * 80)

model = ConcreteAggregateCausalModel(df)

print("1/3 构建因果图...")
model.build_causal_graph()
print(f"    ✓ 节点数: {model.causal_graph.number_of_nodes()}")
print(f"    ✓ 边数: {model.causal_graph.number_of_edges()}")

print("\n2/3 拟合因果模型...")
model.fit_causal_model(quality='BETTER', invertible=True)
print("    ✓ 模型拟合完成")

print("\n3/3 保存模型...")
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(model, f)
print(f"    ✓ 模型已保存至: {MODEL_FILE}")

print()
print("=" * 80)
print("✅ 训练完成！")
print("=" * 80)
print()
print("模型信息：")
print(f"  • 文件路径: {MODEL_FILE}")
print(f"  • 文件大小: {MODEL_FILE.stat().st_size / 1024 / 1024:.2f} MB")
print(f"  • 因果图节点: {model.causal_graph.number_of_nodes()}")
print(f"  • 因果图边: {model.causal_graph.number_of_edges()}")
print()
print("现在可以运行智能体系统：")
print("  python3 quick_agent_test.py")
print("  python3 causal_agent_demo.py")
print()

