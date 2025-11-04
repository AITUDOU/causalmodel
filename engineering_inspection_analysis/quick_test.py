"""
快速测试脚本 - 验证环境和数据
"""

import sys
from pathlib import Path

print("="*80)
print("环境和数据检查")
print("="*80)

# 检查Python版本
print(f"\nPython版本: {sys.version}")

# 检查依赖包
print("\n检查依赖包...")
required_packages = [
    'pandas', 'numpy', 'scipy', 'networkx', 
    'matplotlib', 'seaborn', 'sklearn', 'dowhy', 'torch'
]

missing_packages = []
for package in required_packages:
    try:
        __import__(package)
        print(f"  ✓ {package}")
    except ImportError:
        print(f"  ✗ {package} (未安装)")
        missing_packages.append(package)

if missing_packages:
    print(f"\n警告: 缺少以下依赖包: {', '.join(missing_packages)}")
    print("请运行: pip install -r requirements.txt")
else:
    print("\n✓ 所有依赖包已安装")

# 检查数据文件
print("\n" + "="*80)
print("检查数据文件")
print("="*80)

data_path = Path("../data/processed/cleaned_data.csv")
if data_path.exists():
    import pandas as pd
    df = pd.read_csv(data_path)
    print(f"✓ 数据文件存在: {data_path}")
    print(f"  - 行数: {len(df)}")
    print(f"  - 列数: {len(df.columns)}")
    
    # 检查关键列
    key_cols = ['是否合格', '价格', '施工单位', '生产厂家', '品种', '强度等级(级别)']
    print(f"\n关键列检查:")
    for col in key_cols:
        if col in df.columns:
            print(f"  ✓ {col}")
        else:
            print(f"  ✗ {col} (缺失)")
else:
    print(f"✗ 数据文件不存在: {data_path}")

# 检查目录结构
print("\n" + "="*80)
print("检查目录结构")
print("="*80)

dirs_to_check = ['src', 'data', 'results', 'results/figures', 'results/reports']
for dir_name in dirs_to_check:
    dir_path = Path(dir_name)
    if dir_path.exists():
        print(f"  ✓ {dir_name}/")
    else:
        print(f"  ✗ {dir_name}/ (不存在)")
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"    → 已创建")

print("\n" + "="*80)
print("检查完成!")
print("="*80)
print("\n如果所有检查都通过，可以运行:")
print("  python inspection_analysis_demo.py")
print("或者:")
print("  ./run_analysis.sh")
print("")

