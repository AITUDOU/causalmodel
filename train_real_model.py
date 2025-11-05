"""
训练真实数据因果模型
使用 Kaggle 混凝土抗压强度数据集
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.causal_model import ConcreteStrengthCausalModel

print("=" * 80)
print("🔧 训练真实数据因果模型")
print("=" * 80)
print()

# 配置
DATA_FILE = 'data/real/concrete_compressive_strength.csv'
MODEL_DIR = Path('models')
MODEL_FILE = MODEL_DIR / 'causal_model.pkl'

# 创建模型目录
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 加载真实数据
print("📦 加载真实混凝土抗压强度数据...")
df = pd.read_csv(DATA_FILE)

# 清理列名（去除空格）
df.columns = df.columns.str.strip()
print(f"✓ 数据加载完成：{len(df)} 条记录，{len(df.columns)} 个原始变量")
print("✓ 列名已清理")

print("\n数据字段:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

print(f"\n基本统计:")
print(df.describe())

# 训练模型
print("\n" + "=" * 80)
print("🔧 开始训练因果模型（这需要1-2分钟）...")
print("-" * 80)

model = ConcreteStrengthCausalModel(df)

print("\n1/3 构建因果图...")
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
print(f"  • 数据来源: UCI Machine Learning Repository (Kaggle)")
print(f"  • 样本数: {len(df)}")
print(f"  • 文件路径: {MODEL_FILE}")
print(f"  • 文件大小: {MODEL_FILE.stat().st_size / 1024 / 1024:.2f} MB")
print(f"  • 因果图节点: {model.causal_graph.number_of_nodes()}")
print(f"  • 因果图边: {model.causal_graph.number_of_edges()}")
print()
print("数据字段（含衍生变量）：")
for col in sorted(model.df.columns):
    if col in model.causal_graph.nodes:
        print(f"  ✓ {col}")
print()
print("现在可以运行智能体系统：")
print("  python3 api_server.py")
print("  python3 quick_agent_test.py")
print()

