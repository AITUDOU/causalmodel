"""
数据清洗模块
用于清洗混凝土集料和配合比原始数据
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    加载原始 Excel 数据
    
    Parameters:
    -----------
    file_path : str
        Excel 文件路径
        
    Returns:
    --------
    pd.DataFrame
        原始数据
    """
    try:
        df = pd.read_excel(file_path)
        print(f"成功加载数据: {df.shape[0]} 行, {df.shape[1]} 列")
        print(f"\n列名:\n{df.columns.tolist()}")
        print(f"\n前5行数据:\n{df.head()}")
        print(f"\n数据类型:\n{df.dtypes}")
        print(f"\n缺失值统计:\n{df.isnull().sum()}")
        print(f"\n基本统计:\n{df.describe()}")
        return df
    except Exception as e:
        print(f"加载数据失败: {e}")
        raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗数据
    
    Parameters:
    -----------
    df : pd.DataFrame
        原始数据
        
    Returns:
    --------
    pd.DataFrame
        清洗后的数据
    """
    df_clean = df.copy()
    
    print("\n" + "="*60)
    print("开始数据清洗...")
    print("="*60)
    
    # 1. 删除完全重复的行
    n_duplicates = df_clean.duplicated().sum()
    if n_duplicates > 0:
        df_clean = df_clean.drop_duplicates()
        print(f"✓ 删除 {n_duplicates} 条重复记录")
    
    # 2. 处理列名（去除空格、统一命名）
    df_clean.columns = df_clean.columns.str.strip()
    print(f"✓ 清理列名")
    
    # 3. 处理缺失值
    print("\n缺失值处理:")
    for col in df_clean.columns:
        missing_count = df_clean[col].isnull().sum()
        if missing_count > 0:
            missing_pct = (missing_count / len(df_clean)) * 100
            print(f"  - {col}: {missing_count} ({missing_pct:.2f}%)")
            
            # 根据缺失比例决定处理策略
            if missing_pct > 50:
                print(f"    警告: {col} 缺失率过高 ({missing_pct:.2f}%)，建议删除该列")
            elif missing_pct > 20:
                print(f"    注意: {col} 缺失率较高 ({missing_pct:.2f}%)，保留但标记")
    
    # 4. 处理异常值（根据业务逻辑）
    print("\n异常值检测:")
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
        if outliers > 0:
            outliers_pct = (outliers / len(df_clean)) * 100
            print(f"  - {col}: {outliers} 个异常值 ({outliers_pct:.2f}%)")
            print(f"    范围: [{df_clean[col].min():.2f}, {df_clean[col].max():.2f}]")
            print(f"    建议范围: [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    # 5. 数据类型转换
    print("\n数据类型转换:")
    for col in df_clean.columns:
        # 如果列名包含"类型"、"产地"、"工艺"等关键词，转为分类型
        if any(keyword in col for keyword in ['类型', '产地', '工艺', '方向', '形状']):
            if df_clean[col].dtype != 'category':
                df_clean[col] = df_clean[col].astype('category')
                print(f"  - {col}: 转换为类别型")
    
    print("\n" + "="*60)
    print(f"数据清洗完成: {df_clean.shape[0]} 行, {df_clean.shape[1]} 列")
    print("="*60)
    
    return df_clean


def save_cleaned_data(df: pd.DataFrame, output_path: str):
    """
    保存清洗后的数据
    
    Parameters:
    -----------
    df : pd.DataFrame
        清洗后的数据
    output_path : str
        输出文件路径
    """
    # 创建输出目录
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 保存为 CSV
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ 数据已保存到: {output_path}")


def main():
    """主函数"""
    # 文件路径
    raw_data_path = "/Users/superkang/Desktop/causalmodel/data/raw/raw_data.xlsx"
    cleaned_data_path = "/Users/superkang/Desktop/causalmodel/data/processed/cleaned_data.csv"
    
    # 加载数据
    df_raw = load_raw_data(raw_data_path)
    
    # 清洗数据
    df_clean = clean_data(df_raw)
    
    # 保存清洗后的数据
    save_cleaned_data(df_clean, cleaned_data_path)
    
    return df_clean


if __name__ == "__main__":
    df_cleaned = main()

