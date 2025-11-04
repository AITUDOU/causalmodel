"""
工具函数
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime


def load_data(filepath: str) -> pd.DataFrame:
    """
    加载数据
    
    Parameters:
    -----------
    filepath : str
        文件路径
        
    Returns:
    --------
    pd.DataFrame
        数据框
    """
    df = pd.read_csv(filepath)
    
    # 转换日期列
    if 'test_date' in df.columns:
        df['test_date'] = pd.to_datetime(df['test_date'])
    
    return df


def split_by_period(df: pd.DataFrame,
                   split_date: str = None,
                   new_year: int = 2024,
                   old_year: int = 2023,
                   new_months: List[int] = None,
                   old_months: List[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    按时期分割数据
    
    Parameters:
    -----------
    df : pd.DataFrame
        数据框
    split_date : str, optional
        分割日期
    new_year : int
        新时期年份
    old_year : int
        旧时期年份
    new_months : List[int], optional
        新时期月份
    old_months : List[int], optional
        旧时期月份
        
    Returns:
    --------
    df_old, df_new : Tuple[pd.DataFrame, pd.DataFrame]
        旧时期和新时期数据
    """
    if split_date:
        split_dt = pd.to_datetime(split_date)
        df_old = df[df['test_date'] < split_dt].copy()
        df_new = df[df['test_date'] >= split_dt].copy()
    else:
        if new_months is None:
            new_months = list(range(1, 13))
        if old_months is None:
            old_months = list(range(1, 13))
        
        new_conditions = (df['year'] == new_year) & (df['month'].isin(new_months))
        old_conditions = (df['year'] == old_year) & (df['month'].isin(old_months))
        
        df_new = df[new_conditions].copy()
        df_old = df[old_conditions].copy()
    
    return df_old, df_new


def calculate_statistics(df: pd.DataFrame, 
                         columns: List[str]) -> pd.DataFrame:
    """
    计算描述性统计
    
    Parameters:
    -----------
    df : pd.DataFrame
        数据框
    columns : List[str]
        要计算的列
        
    Returns:
    --------
    pd.DataFrame
        统计结果
    """
    stats_dict = {}
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            stats_dict[col] = {
                '均值': df[col].mean(),
                '中位数': df[col].median(),
                '标准差': df[col].std(),
                '最小值': df[col].min(),
                '最大值': df[col].max(),
                '缺失值': df[col].isna().sum()
            }
    
    return pd.DataFrame(stats_dict).T


def detect_outliers(df: pd.DataFrame, 
                   column: str, 
                   method: str = 'iqr',
                   threshold: float = 1.5) -> pd.Series:
    """
    检测异常值
    
    Parameters:
    -----------
    df : pd.DataFrame
        数据框
    column : str
        列名
    method : str
        方法 ('iqr' 或 'zscore')
    threshold : float
        阈值
        
    Returns:
    --------
    pd.Series
        异常值布尔索引
    """
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (df[column] < lower_bound) | (df[column] > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs(stats.zscore(df[column].dropna()))
        return z_scores > threshold
    
    else:
        raise ValueError("method 必须是 'iqr' 或 'zscore'")


def clean_data(df: pd.DataFrame, 
               outlier_columns: List[str] = None,
               fill_method: str = 'mean') -> pd.DataFrame:
    """
    清洗数据
    
    Parameters:
    -----------
    df : pd.DataFrame
        数据框
    outlier_columns : List[str], optional
        需要检测异常值的列
    fill_method : str
        缺失值填充方法
        
    Returns:
    --------
    pd.DataFrame
        清洗后的数据
    """
    df_clean = df.copy()
    
    # 处理异常值
    if outlier_columns:
        for col in outlier_columns:
            if col in df_clean.columns:
                outliers = detect_outliers(df_clean, col)
                if outliers.sum() > 0:
                    print(f"列 {col} 检测到 {outliers.sum()} 个异常值")
                    # 用中位数替换异常值
                    median_val = df_clean[col].median()
                    df_clean.loc[outliers, col] = median_val
    
    # 处理缺失值
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    if fill_method == 'mean':
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
    elif fill_method == 'median':
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
    elif fill_method == 'forward':
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(method='ffill')
    
    return df_clean


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    特征工程
    
    Parameters:
    -----------
    df : pd.DataFrame
        数据框
        
    Returns:
    --------
    pd.DataFrame
        增强后的数据框
    """
    df_enhanced = df.copy()
    
    # 创建交互特征
    if 'sieve_amplitude' in df.columns and 'sieve_frequency' in df.columns:
        df_enhanced['sieve_power'] = df['sieve_amplitude'] * df['sieve_frequency']
    
    # 创建比率特征
    if 'initial_moisture' in df.columns and 'drying_time' in df.columns:
        df_enhanced['moisture_drying_ratio'] = df['initial_moisture'] / (df['drying_time'] + 1)
    
    # 创建时间特征
    if 'test_date' in df.columns:
        df_enhanced['day_of_week'] = pd.to_datetime(df['test_date']).dt.dayofweek
        df_enhanced['is_weekend'] = df_enhanced['day_of_week'].isin([5, 6]).astype(int)
    
    return df_enhanced


def export_results(results: Dict, 
                  output_dir: str,
                  prefix: str = 'result'):
    """
    导出结果
    
    Parameters:
    -----------
    results : Dict
        结果字典
    output_dir : str
        输出目录
    prefix : str
        文件前缀
    """
    import os
    import json
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for key, value in results.items():
        if isinstance(value, pd.DataFrame):
            filepath = os.path.join(output_dir, f'{prefix}_{key}_{timestamp}.csv')
            value.to_csv(filepath, index=False)
            print(f"已保存: {filepath}")
        
        elif isinstance(value, dict):
            filepath = os.path.join(output_dir, f'{prefix}_{key}_{timestamp}.json')
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(value, f, ensure_ascii=False, indent=2)
            print(f"已保存: {filepath}")


def validate_data_quality(df: pd.DataFrame) -> Dict:
    """
    验证数据质量
    
    Parameters:
    -----------
    df : pd.DataFrame
        数据框
        
    Returns:
    --------
    Dict
        数据质量报告
    """
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': {},
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict()
    }
    
    # 检查缺失值
    for col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            report['missing_values'][col] = {
                'count': int(missing_count),
                'percentage': float(missing_count / len(df) * 100)
            }
    
    # 检查数值范围
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    report['numeric_ranges'] = {}
    for col in numeric_cols:
        report['numeric_ranges'][col] = {
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'mean': float(df[col].mean())
        }
    
    return report


def format_contribution_table(contributions: Dict,
                              uncertainties: Dict) -> pd.DataFrame:
    """
    格式化贡献表格
    
    Parameters:
    -----------
    contributions : Dict
        贡献字典
    uncertainties : Dict
        不确定性字典
        
    Returns:
    --------
    pd.DataFrame
        格式化的表格
    """
    rows = []
    for node, contrib in contributions.items():
        row = {
            '变量': node,
            '贡献值': contrib
        }
        if node in uncertainties:
            row['下界'] = uncertainties[node][0]
            row['上界'] = uncertainties[node][1]
            row['显著性'] = '是' if (uncertainties[node][0] > 0 and uncertainties[node][1] > 0) or \
                                    (uncertainties[node][0] < 0 and uncertainties[node][1] < 0) else '否'
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values('贡献值', key=abs, ascending=False)
    
    return df

