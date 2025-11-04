"""
探索数据内容
"""

import pandas as pd

# 读取清洗后的数据
df = pd.read_csv("/Users/superkang/Desktop/causalmodel/data/processed/cleaned_data.csv")

print("="*80)
print("检测参数类型统计:")
print("="*80)
print(df['检测参数'].value_counts().head(30))

print("\n" + "="*80)
print("品种统计:")
print("="*80)
print(df['品种'].value_counts().head(20))

print("\n" + "="*80)
print("检测结果样例 (非空):")
print("="*80)
result_samples = df[df['检测结果'].notna()]['检测结果'].head(10)
for i, result in enumerate(result_samples, 1):
    print(f"{i}. {result}")

print("\n" + "="*80)
print("检测结论统计:")
print("="*80)
print(df['检测结论'].value_counts())

print("\n" + "="*80)
print("是否合格统计:")
print("="*80)
print(df['是否合格'].value_counts())

# 查看与集料相关的数据
print("\n" + "="*80)
print("与集料相关的数据筛选:")
print("="*80)
aggregate_keywords = ['集料', '砂', '石', '碎石', '砂率', '含泥量', '压碎', '针片状', '粒形']
aggregate_data = df[df['品种'].str.contains('|'.join(aggregate_keywords), na=False)]
print(f"找到 {len(aggregate_data)} 条集料相关数据")
print("\n集料相关的检测参数:")
print(aggregate_data['检测参数'].value_counts().head(20))

# 查看与混凝土相关的数据
print("\n" + "="*80)
print("与混凝土相关的数据筛选:")
print("="*80)
concrete_keywords = ['混凝土', '水泥', '强度', '抗压']
concrete_data = df[df['品种'].str.contains('|'.join(concrete_keywords), na=False)]
print(f"找到 {len(concrete_data)} 条混凝土相关数据")
if len(concrete_data) > 0:
    print("\n混凝土相关的检测参数:")
    print(concrete_data['检测参数'].value_counts().head(20))

# 查看样例数据
print("\n" + "="*80)
print("样例数据 (集料相关):")
print("="*80)
if len(aggregate_data) > 0:
    sample_cols = ['品种', '检测参数', '检测结果', '是否合格', '生产厂家']
    print(aggregate_data[sample_cols].head(10).to_string())

