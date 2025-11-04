"""
工程检测数据预处理模块
处理缺失值、编码分类变量、特征工程
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from sklearn.preprocessing import LabelEncoder


class InspectionDataPreprocessor:
    """工程检测数据预处理器"""
    
    def __init__(self):
        self.label_encoders = {}
        self.feature_stats = {}
        
    def preprocess(self, df: pd.DataFrame, 
                   target_cols: List[str] = None) -> pd.DataFrame:
        """
        完整预处理流程
        
        Args:
            df: 原始数据
            target_cols: 目标变量列名列表
            
        Returns:
            预处理后的数据
        """
        if target_cols is None:
            target_cols = ['是否合格', '价格']
            
        df_processed = df.copy()
        
        # 1. 处理缺失值
        df_processed = self._handle_missing_values(df_processed)
        
        # 2. 清理和标准化文本
        df_processed = self._clean_text_columns(df_processed)
        
        # 3. 提取时间特征
        df_processed = self._extract_time_features(df_processed)
        
        # 4. 编码分类变量
        df_processed = self._encode_categorical(df_processed, target_cols)
        
        # 5. 特征工程
        df_processed = self._feature_engineering(df_processed)
        
        # 6. 过滤有效数据
        df_processed = self._filter_valid_data(df_processed)
        
        return df_processed
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        df = df.copy()
        
        # 删除完全为空的列
        empty_cols = ['分项工程(分)', '地址', '见证号', '见证人', '分项工程']
        df = df.drop(columns=[col for col in empty_cols if col in df.columns], errors='ignore')
        
        # 关键字段缺失则填充默认值
        if '生产厂家' in df.columns:
            df['生产厂家'] = df['生产厂家'].fillna('未知厂家')
            
        if '代表数量' in df.columns:
            df['代表数量'] = df['代表数量'].fillna('0')
            
        if '强度等级(级别)' in df.columns:
            df['强度等级(级别)'] = df['强度等级(级别)'].fillna('未指定')
            
        if '规格(线芯结构)' in df.columns:
            df['规格(线芯结构)'] = df['规格(线芯结构)'].fillna('0')
            
        if '结构部位' in df.columns:
            df['结构部位'] = df['结构部位'].fillna('未指定')
            
        if '委托人' in df.columns:
            df['委托人'] = df['委托人'].fillna('未知')
            
        if '监理单位' in df.columns:
            df['监理单位'] = df['监理单位'].fillna('无监理')
            
        if '施工单位' in df.columns:
            df['施工单位'] = df['施工单位'].fillna('未知')
            
        if '产品标准' in df.columns:
            df['产品标准'] = df['产品标准'].fillna('无标准')
            
        return df
    
    def _clean_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理和标准化文本列"""
        df = df.copy()
        
        # 去除文本列的首尾空格
        text_cols = df.select_dtypes(include=['object']).columns
        for col in text_cols:
            df[col] = df[col].astype(str).str.strip()
            # 替换常见的空值表示
            df[col] = df[col].replace(['-----', '    -  -  ', 'nan', 'None', ''], '未知')
            
        return df
    
    def _extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取时间特征"""
        df = df.copy()
        
        # 转换日期列
        date_cols = ['委托日期', '检测开始日期', '检测完成日期', '审核日期', '签发日期']
        
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # 计算检测周期（检测开始到完成）
        if '检测开始日期' in df.columns and '检测完成日期' in df.columns:
            df['检测周期_天'] = (df['检测完成日期'] - df['检测开始日期']).dt.days
            df['检测周期_天'] = df['检测周期_天'].fillna(0).clip(lower=0)
        
        # 计算审核周期（检测完成到审核）
        if '检测完成日期' in df.columns and '审核日期' in df.columns:
            df['审核周期_天'] = (df['审核日期'] - df['检测完成日期']).dt.days
            df['审核周期_天'] = df['审核周期_天'].fillna(0).clip(lower=0)
        
        # 提取年份、月份、季度
        if '委托日期' in df.columns:
            df['委托年份'] = df['委托日期'].dt.year.fillna(0).astype(int)
            df['委托月份'] = df['委托日期'].dt.month.fillna(0).astype(int)
            df['委托季度'] = df['委托日期'].dt.quarter.fillna(0).astype(int)
            df['委托星期'] = df['委托日期'].dt.dayofweek.fillna(0).astype(int)
        
        return df
    
    def _encode_categorical(self, df: pd.DataFrame, 
                           target_cols: List[str]) -> pd.DataFrame:
        """编码分类变量"""
        df = df.copy()
        
        # 二值编码：是否合格
        if '是否合格' in df.columns:
            df['是否合格_编码'] = (df['是否合格'] == '合格').astype(int)
        
        # 二值编码：检测状态
        if '检测状态' in df.columns:
            df['检测完成_编码'] = (df['检测状态'] == '已完成').astype(int)
        
        # 标签编码：高基数分类变量
        high_cardinality_cols = [
            '委托单位', '工程名称', '建设单位', '监理单位', '施工单位',
            '生产厂家', '品种', '强度等级(级别)', '结构部位', '检验类别',
            '来样方式', '任务来源'
        ]
        
        for col in high_cardinality_cols:
            if col in df.columns and col not in target_cols:
                le = LabelEncoder()
                valid_mask = df[col].notna()
                if valid_mask.sum() > 0:
                    df.loc[valid_mask, f'{col}_编码'] = le.fit_transform(df.loc[valid_mask, col])
                    df[f'{col}_编码'] = df[f'{col}_编码'].fillna(-1).astype(int)
                    self.label_encoders[col] = le
        
        return df
    
    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """特征工程"""
        df = df.copy()
        
        # 1. 提取规格数值（从字符串中提取）
        if '规格(线芯结构)' in df.columns:
            # 尝试提取数字（如"25"从"25mm"中提取）
            df['规格_数值'] = df['规格(线芯结构)'].astype(str).str.extract(r'(\d+)').astype(float).fillna(0)
        
        # 2. 提取代表数量数值
        if '代表数量' in df.columns:
            df['代表数量_数值'] = df['代表数量'].astype(str).str.extract(r'(\d+)').astype(float).fillna(0)
        
        # 3. 价格分组（低/中/高）
        if '价格' in df.columns:
            df['价格分组'] = pd.cut(df['价格'], 
                                   bins=[0, 100, 300, 1000], 
                                   labels=['低', '中', '高'])
            df['价格分组_编码'] = df['价格分组'].map({'低': 0, '中': 1, '高': 2}).fillna(1).astype(int)
        
        # 4. 施工单位合格率（历史统计特征）
        if '施工单位' in df.columns and '是否合格_编码' in df.columns:
            unit_quality = df.groupby('施工单位')['是否合格_编码'].transform('mean')
            df['施工单位历史合格率'] = unit_quality
        
        # 5. 生产厂家合格率
        if '生产厂家' in df.columns and '是否合格_编码' in df.columns:
            manufacturer_quality = df.groupby('生产厂家')['是否合格_编码'].transform('mean')
            df['生产厂家历史合格率'] = manufacturer_quality
        
        # 6. 检验类别风险评分（监督抽检风险高）
        if '检验类别' in df.columns:
            risk_map = {
                '委托': 0,
                '工程现场抽检': 1,
                '监督抽检': 2
            }
            df['检验风险等级'] = df['检验类别'].map(risk_map).fillna(0).astype(int)
        
        return df
    
    def _filter_valid_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """过滤有效数据"""
        df = df.copy()
        
        # 保留检测状态为"已完成"或"未完成"的记录
        if '检测状态' in df.columns:
            df = df[df['检测状态'].isin(['已完成', '未完成'])]
        
        # 保留价格合理的记录
        if '价格' in df.columns:
            df = df[(df['价格'] >= 0) & (df['价格'] <= 10000)]
        
        # 删除关键字段全为空的行
        if '是否合格' in df.columns:
            df = df[df['是否合格'].isin(['合格', '不合格'])]
        
        return df
    
    def get_feature_importance_cols(self) -> List[str]:
        """获取建模所需的关键特征列"""
        return [
            # 编码后的分类特征
            '施工单位_编码', '生产厂家_编码', '品种_编码', 
            '强度等级(级别)_编码', '结构部位_编码', '检验类别_编码',
            '监理单位_编码', '建设单位_编码',
            
            # 数值特征
            '价格', '规格_数值', '代表数量_数值',
            '检测周期_天', '审核周期_天',
            '委托年份', '委托月份', '委托季度', '委托星期',
            
            # 统计特征
            '施工单位历史合格率', '生产厂家历史合格率', '检验风险等级',
            '价格分组_编码',
            
            # 目标变量
            '是否合格_编码', '检测完成_编码'
        ]
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str):
        """保存预处理后的数据"""
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✓ 预处理数据已保存至: {output_path}")
        print(f"  - 样本数: {len(df)}")
        print(f"  - 特征数: {len(df.columns)}")


def load_and_preprocess_data(data_path: str, 
                             output_path: str = None) -> pd.DataFrame:
    """
    加载并预处理工程检测数据
    
    Args:
        data_path: 原始数据路径
        output_path: 输出路径（可选）
        
    Returns:
        预处理后的数据
    """
    print("="*80)
    print("开始数据预处理")
    print("="*80)
    
    # 加载数据
    print(f"\n正在加载数据: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✓ 原始数据: {df.shape[0]} 行, {df.shape[1]} 列")
    
    # 预处理
    preprocessor = InspectionDataPreprocessor()
    df_processed = preprocessor.preprocess(df)
    
    print(f"\n✓ 预处理完成: {df_processed.shape[0]} 行, {df_processed.shape[1]} 列")
    
    # 显示关键统计
    if '是否合格_编码' in df_processed.columns:
        quality_rate = df_processed['是否合格_编码'].mean() * 100
        print(f"\n合格率: {quality_rate:.2f}%")
        print(f"合格样本: {df_processed['是否合格_编码'].sum()}")
        print(f"不合格样本: {(1 - df_processed['是否合格_编码']).sum()}")
    
    # 保存
    if output_path:
        preprocessor.save_processed_data(df_processed, output_path)
    
    return df_processed


if __name__ == "__main__":
    # 测试预处理
    data_path = "../data/processed/cleaned_data.csv"
    output_path = "../data/processed/inspection_data_processed.csv"
    
    df_processed = load_and_preprocess_data(data_path, output_path)
    
    print("\n" + "="*80)
    print("预处理完成")
    print("="*80)

