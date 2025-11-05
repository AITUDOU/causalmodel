"""
下载真实混凝土抗压强度数据集
来源：Kaggle - Concrete Compressive Strength Dataset
"""

import kagglehub
import pandas as pd
import shutil
from pathlib import Path

print("=" * 80)
print("下载真实混凝土抗压强度数据集")
print("=" * 80)
print()

# 下载数据集
print("📦 正在从 Kaggle 下载数据集...")
path = kagglehub.dataset_download("elikplim/concrete-compressive-strength-data-set")

print(f"✓ 数据集下载完成")
print(f"路径: {path}")
print()

# 查找CSV文件
print("📂 查找数据文件...")
dataset_path = Path(path)
csv_files = list(dataset_path.glob("*.csv"))

if not csv_files:
    # 可能在子目录中
    csv_files = list(dataset_path.glob("**/*.csv"))

print(f"找到 {len(csv_files)} 个CSV文件:")
for f in csv_files:
    print(f"  • {f.name}")

# 加载主数据文件
if csv_files:
    main_file = csv_files[0]
    print(f"\n📊 加载数据文件: {main_file.name}")
    df = pd.read_csv(main_file)
    
    print(f"\n数据概览:")
    print(f"  样本数: {len(df)}")
    print(f"  字段数: {len(df.columns)}")
    print(f"\n列名:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    
    print(f"\n前5行数据:")
    print(df.head())
    
    print(f"\n数据统计:")
    print(df.describe())
    
    print(f"\n数据类型:")
    print(df.dtypes)
    
    print(f"\n缺失值:")
    print(df.isnull().sum())
    
    # 复制到项目数据目录
    target_dir = Path("data/real")
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / "concrete_compressive_strength.csv"
    
    shutil.copy(main_file, target_file)
    print(f"\n✓ 数据已复制到: {target_file}")
    
else:
    print("❌ 未找到CSV文件")

print("\n" + "=" * 80)
print("✅ 下载完成")
print("=" * 80)

