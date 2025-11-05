"""
混凝土配合比因果数据生成器 V2
基于真实配合比数据，使用正态分布生成仿真数据
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class ConcreteDataGeneratorV2:
    """混凝土配合比因果数据生成器（基于真实数据）"""
    
    def __init__(self, n_samples: int = 2000, random_seed: int = 42):
        """
        初始化数据生成器
        
        Parameters:
        -----------
        n_samples : int
            样本数量
        random_seed : int
            随机种子
        """
        self.n_samples = n_samples
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def generate_data(self) -> pd.DataFrame:
        """
        生成混凝土配合比因果数据
        
        Returns:
        --------
        pd.DataFrame
            生成的数据
        """
        data = {}
        
        print("="*80)
        print("开始生成混凝土配合比数据（基于真实数据分布）")
        print("="*80)
        
        # ===== 第一层：根节点（设计参数）=====
        
        # 1. 水泥品种（根节点）
        data['cement_type'] = np.random.choice(
            ['P.O42.5', 'P.O52.5', 'P.C32.5', 'P.S32.5'], 
            size=self.n_samples,
            p=[0.50, 0.30, 0.15, 0.05]  # P.O42.5最常用
        )
        
        # 2. 砂品种（根节点）
        data['sand_type'] = np.random.choice(
            ['Ⅰ区细砂', 'Ⅱ区中砂', 'Ⅲ区粗砂'], 
            size=self.n_samples,
            p=[0.20, 0.60, 0.20]  # Ⅱ区中砂最常用
        )
        
        # 3. 碎石粒径（根节点）
        data['stone_size'] = np.random.choice(
            ['5-10mm', '5-25mm', '5-31.5mm', '10-20mm'], 
            size=self.n_samples,
            p=[0.15, 0.50, 0.25, 0.10]  # 5-25mm最常用
        )
        
        # 4. 外加剂类型（根节点）
        data['admixture_type'] = np.random.choice(
            ['聚羧酸高性能减水剂', '萘系减水剂', '脂肪族减水剂'], 
            size=self.n_samples,
            p=[0.70, 0.20, 0.10]  # 聚羧酸最常用
        )
        
        # 5. 砂产地（根节点）
        data['sand_origin'] = np.random.choice(
            ['湖田县湘豪砂石', '安徽甲港', '江苏沿海', '本地河砂'], 
            size=self.n_samples,
            p=[0.30, 0.25, 0.25, 0.20]
        )
        
        # 6. 石料产地（根节点）
        data['stone_origin'] = np.random.choice(
            ['安徽甲港', '江苏金峰', '本地石场', '外省调入'], 
            size=self.n_samples,
            p=[0.35, 0.30, 0.25, 0.10]
        )
        
        # ===== 第二层：配合比设计（核心因果关系）=====
        
        # 水胶比（最关键参数）- 基于真实数据：均值0.43, 范围0.38-0.48
        data['water_binder_ratio'] = np.random.normal(0.43, 0.03, self.n_samples)
        data['water_binder_ratio'] = np.clip(data['water_binder_ratio'], 0.35, 0.55)
        
        # 水泥用量 (kg/m³) - 基于真实数据：均值383, 范围340-430
        cement_base = np.random.normal(383, 30, self.n_samples)
        
        # 水泥品种影响
        cement_type_effect = pd.Series(data['cement_type']).map({
            'P.O42.5': 0,
            'P.O52.5': -20,  # 高标号用量少
            'P.C32.5': 30,   # 低标号用量多
            'P.S32.5': 25
        }).values
        
        data['cement_content'] = cement_base + cement_type_effect
        data['cement_content'] = np.clip(data['cement_content'], 300, 500)
        
        # 粉煤灰掺量 - 一般为胶凝材料的15-25%
        fly_ash_ratio = np.random.normal(0.20, 0.05, self.n_samples)
        fly_ash_ratio = np.clip(fly_ash_ratio, 0.10, 0.30)
        
        # 矿渣粉掺量 - 一般为胶凝材料的10-20%
        slag_ratio = np.random.normal(0.15, 0.04, self.n_samples)
        slag_ratio = np.clip(slag_ratio, 0.05, 0.25)
        
        # 胶凝材料总量
        data['total_binder'] = data['cement_content'] / (1 - fly_ash_ratio - slag_ratio)
        data['fly_ash_content'] = data['total_binder'] * fly_ash_ratio
        data['slag_powder_content'] = data['total_binder'] * slag_ratio
        
        # 水用量 (kg/m³) - 基于水胶比计算，真实数据约163
        data['water_content'] = data['total_binder'] * data['water_binder_ratio']
        data['water_content'] = np.clip(data['water_content'], 150, 200)
        
        # 细集料(砂)用量 (kg/m³) - 基于真实数据：均值773, 范围738-807
        sand_base = np.random.normal(773, 25, self.n_samples)
        
        # 砂品种影响
        sand_type_effect = pd.Series(data['sand_type']).map({
            'Ⅰ区细砂': 20,   # 细砂用量多
            'Ⅱ区中砂': 0,
            'Ⅲ区粗砂': -15   # 粗砂用量少
        }).values
        
        data['sand_content'] = sand_base + sand_type_effect
        data['sand_content'] = np.clip(data['sand_content'], 700, 850)
        
        # 粗集料(石)用量 (kg/m³) - 基于真实数据：均值1112, 范围1107-1115
        stone_base = np.random.normal(1112, 10, self.n_samples)
        
        # 粒径影响
        stone_size_effect = pd.Series(data['stone_size']).map({
            '5-10mm': -20,
            '5-25mm': 0,
            '5-31.5mm': 10,
            '10-20mm': -10
        }).values
        
        data['stone_content'] = stone_base + stone_size_effect
        data['stone_content'] = np.clip(data['stone_content'], 1050, 1180)
        
        # 外加剂掺量 (kg/m³) - 基于真实数据：均值4.6, 范围4.08-5.16
        admixture_base = data['total_binder'] * 0.01  # 一般为胶凝材料的0.8-1.2%
        
        admixture_type_effect = pd.Series(data['admixture_type']).map({
            '聚羧酸高性能减水剂': 0,
            '萘系减水剂': 0.5,
            '脂肪族减水剂': 0.3
        }).values
        
        data['admixture_content'] = admixture_base + admixture_type_effect + np.random.normal(0, 0.3, self.n_samples)
        data['admixture_content'] = np.clip(data['admixture_content'], 3, 8)
        
        # ===== 第三层：砂率计算 =====
        
        # 砂率 = 砂/(砂+石)
        data['sand_rate'] = data['sand_content'] / (data['sand_content'] + data['stone_content'])
        data['sand_rate'] = np.clip(data['sand_rate'], 0.35, 0.45)
        
        # ===== 第四层：材料质量指标（受产地和品种影响）=====
        
        # 1. 砂含泥量(%) - 受产地和品种影响
        mud_content_base = np.random.normal(2.0, 0.8, self.n_samples)
        
        sand_origin_mud_effect = pd.Series(data['sand_origin']).map({
            '湖田县湘豪砂石': -0.5,
            '安徽甲港': -0.3,
            '江苏沿海': 0.2,
            '本地河砂': 0.5
        }).values
        
        sand_type_mud_effect = pd.Series(data['sand_type']).map({
            'Ⅰ区细砂': 0.3,
            'Ⅱ区中砂': 0,
            'Ⅲ区粗砂': -0.2
        }).values
        
        data['mud_content_pct'] = mud_content_base + sand_origin_mud_effect + sand_type_mud_effect
        data['mud_content_pct'] = np.clip(data['mud_content_pct'], 0.5, 5.0)
        
        # 2. 石料压碎值(%) - 受产地和粒径影响
        crushing_value_base = np.random.normal(12, 3, self.n_samples)
        
        stone_origin_crushing_effect = pd.Series(data['stone_origin']).map({
            '安徽甲港': -1.5,
            '江苏金峰': -1.0,
            '本地石场': 0.5,
            '外省调入': 1.0
        }).values
        
        data['crushing_value_pct'] = crushing_value_base + stone_origin_crushing_effect
        data['crushing_value_pct'] = np.clip(data['crushing_value_pct'], 6, 20)
        
        # 3. 针片状含量(%) - 受粒径和产地影响
        flaky_base = np.random.normal(8, 2, self.n_samples)
        
        stone_size_flaky_effect = pd.Series(data['stone_size']).map({
            '5-10mm': -1,
            '5-25mm': 0,
            '5-31.5mm': 0.5,
            '10-20mm': -0.5
        }).values
        
        data['flaky_particle_pct'] = flaky_base + stone_size_flaky_effect + stone_origin_crushing_effect * 0.5
        data['flaky_particle_pct'] = np.clip(data['flaky_particle_pct'], 3, 15)
        
        # 4. 砂细度模数 - 受砂品种影响
        fineness_modulus_map = {
            'Ⅰ区细砂': np.random.normal(2.0, 0.2, self.n_samples),
            'Ⅱ区中砂': np.random.normal(2.6, 0.2, self.n_samples),
            'Ⅲ区粗砂': np.random.normal(3.2, 0.2, self.n_samples)
        }
        
        data['fineness_modulus'] = np.zeros(self.n_samples)
        for sand_type, values in fineness_modulus_map.items():
            mask = pd.Series(data['sand_type']) == sand_type
            data['fineness_modulus'][mask] = values[mask]
        
        data['fineness_modulus'] = np.clip(data['fineness_modulus'], 1.6, 3.7)
        
        # ===== 第五层：配合比准确性 =====
        
        # 配合比准确性(%) - 受设计和施工控制影响
        mix_accuracy_base = 95.0
        
        # 水胶比偏差影响（偏离0.43越远，准确性越低）
        water_binder_deviation_effect = -abs(data['water_binder_ratio'] - 0.43) * 15
        
        # 外加剂掺量影响
        admixture_effect = (data['admixture_content'] - 4.6) * 0.5
        
        data['mix_design_accuracy_pct'] = (mix_accuracy_base + water_binder_deviation_effect + 
                                           admixture_effect + np.random.normal(0, 2, self.n_samples))
        data['mix_design_accuracy_pct'] = np.clip(data['mix_design_accuracy_pct'], 80, 100)
        
        # ===== 第六层：工作性能指标 =====
        
        # 1. 坍落度(mm) - 基于真实数据：均值193, 范围190-200
        slump_base = np.random.normal(193, 10, self.n_samples)
        
        # 水胶比影响（水胶比越大，坍落度越大）
        water_binder_slump_effect = (data['water_binder_ratio'] - 0.43) * 100
        
        # 外加剂影响
        admixture_slump_effect = (data['admixture_content'] - 4.6) * 5
        
        # 砂率影响
        sand_rate_slump_effect = (data['sand_rate'] - 0.40) * 50
        
        data['slump_mm'] = (slump_base + water_binder_slump_effect + 
                           admixture_slump_effect + sand_rate_slump_effect)
        data['slump_mm'] = np.clip(data['slump_mm'], 150, 230)
        
        # 2. 2小时坍落度损失(mm) - 基于真实数据：均值8, 范围5-10
        slump_loss_base = np.random.normal(8, 2, self.n_samples)
        
        # 外加剂影响（聚羧酸损失小）
        admixture_loss_effect = pd.Series(data['admixture_type']).map({
            '聚羧酸高性能减水剂': -2,
            '萘系减水剂': 1,
            '脂肪族减水剂': 0
        }).values
        
        data['slump_loss_2h_mm'] = slump_loss_base + admixture_loss_effect
        data['slump_loss_2h_mm'] = np.clip(data['slump_loss_2h_mm'], 0, 20)
        
        # 3. 表观密度(kg/m³) - 基于真实数据：均值2423, 范围2420-2430
        apparent_density_base = data['cement_content'] + data['sand_content'] + data['stone_content'] + data['water_content']
        apparent_density_base = apparent_density_base * 1.02  # 考虑空气和化学反应
        
        data['apparent_density'] = apparent_density_base + np.random.normal(0, 20, self.n_samples)
        data['apparent_density'] = np.clip(data['apparent_density'], 2300, 2500)
        
        # 4. 含气量(%) - 一般1-4%
        data['air_content_pct'] = np.random.normal(2.5, 0.8, self.n_samples)
        data['air_content_pct'] = np.clip(data['air_content_pct'], 1.0, 5.0)
        
        # 5. 泌水率(%) - 一般小于1%
        bleeding_base = np.random.normal(0.3, 0.2, self.n_samples)
        
        # 水胶比影响（水胶比越大，泌水越严重）
        bleeding_water_effect = (data['water_binder_ratio'] - 0.43) * 2
        
        # 砂率影响（砂率低，泌水严重）
        bleeding_sand_effect = (0.40 - data['sand_rate']) * 1.5
        
        data['bleeding_rate_pct'] = bleeding_base + bleeding_water_effect + bleeding_sand_effect
        data['bleeding_rate_pct'] = np.clip(data['bleeding_rate_pct'], 0, 2.0)
        
        # ===== 第七层：凝结时间 =====
        
        # 初凝时间(分钟) - 一般6-10小时
        initial_setting_base = np.random.normal(450, 60, self.n_samples)  # 约7.5小时
        
        # 温度影响（假设恒温）
        # 水胶比影响（水多，凝结慢）
        setting_water_effect = (data['water_binder_ratio'] - 0.43) * 100
        
        # 外加剂影响（减水剂会延缓凝结）
        setting_admixture_effect = (data['admixture_content'] - 4.6) * 10
        
        data['initial_setting_time_min'] = (initial_setting_base + setting_water_effect + 
                                            setting_admixture_effect)
        data['initial_setting_time_min'] = np.clip(data['initial_setting_time_min'], 360, 600)
        
        # 终凝时间(分钟) - 比初凝晚1-3小时
        final_setting_offset = np.random.normal(100, 20, self.n_samples)
        data['final_setting_time_min'] = data['initial_setting_time_min'] + final_setting_offset
        data['final_setting_time_min'] = np.clip(data['final_setting_time_min'], 450, 720)
        
        # ===== 第八层：抗压强度（最关键结果变量）=====
        
        # 7天抗压强度 - 基于真实数据：均值39.5, 范围32.9-46.3 MPa
        strength_7d_base = 40.0
        
        # 水胶比影响（最关键，负相关）
        water_binder_strength_effect = (0.43 - data['water_binder_ratio']) * 120
        
        # 水泥用量影响
        cement_strength_effect = (data['cement_content'] - 383) * 0.08
        
        # 砂率影响（砂率接近0.40最优）
        sand_rate_strength_effect = -(abs(data['sand_rate'] - 0.40) * 30)
        
        # 含泥量影响（负向）
        mud_strength_effect = -(data['mud_content_pct'] - 2.0) * 1.5
        
        # 压碎值影响（负向）
        crushing_strength_effect = -(data['crushing_value_pct'] - 12) * 0.5
        
        # 针片状影响（负向）
        flaky_strength_effect = -(data['flaky_particle_pct'] - 8) * 0.8
        
        # 配合比准确性影响
        mix_accuracy_strength_effect = (data['mix_design_accuracy_pct'] - 95) * 0.3
        
        # 外加剂影响
        admixture_strength_effect = (data['admixture_content'] - 4.6) * 0.5
        
        # 水泥品种影响
        cement_type_strength_effect = pd.Series(data['cement_type']).map({
            'P.O42.5': 0,
            'P.O52.5': 8,
            'P.C32.5': -6,
            'P.S32.5': -4
        }).values
        
        data['strength_7d_mpa'] = (strength_7d_base + water_binder_strength_effect + 
                                   cement_strength_effect + sand_rate_strength_effect +
                                   mud_strength_effect + crushing_strength_effect +
                                   flaky_strength_effect + mix_accuracy_strength_effect +
                                   admixture_strength_effect + cement_type_strength_effect +
                                   np.random.normal(0, 3, self.n_samples))
        
        data['strength_7d_mpa'] = np.clip(data['strength_7d_mpa'], 25, 60)
        
        # 28天抗压强度 - 基于真实数据：均值47.7, 范围42.7-52.5 MPa
        # 28d强度约为7d强度的1.2-1.3倍
        strength_28d_ratio = np.random.normal(1.22, 0.05, self.n_samples)
        strength_28d_ratio = np.clip(strength_28d_ratio, 1.15, 1.35)
        
        data['strength_28d_mpa'] = data['strength_7d_mpa'] * strength_28d_ratio
        data['strength_28d_mpa'] = np.clip(data['strength_28d_mpa'], 30, 80)
        
        # ===== 第九层：工程性能评价 =====
        
        # 质量合格率 - 以28d强度≥30 MPa为标准
        strength_pass_threshold = 30  # MPa
        data['quality_pass'] = (data['strength_28d_mpa'] >= strength_pass_threshold).astype(int)
        data['quality_pass_rate_pct'] = (data['strength_28d_mpa'] / strength_pass_threshold * 100)
        data['quality_pass_rate_pct'] = np.clip(data['quality_pass_rate_pct'], 80, 150)
        
        # 综合性能评分 (0-100)
        performance_base = 80.0
        performance_strength_effect = (data['strength_28d_mpa'] - 47.7) * 0.5
        performance_slump_effect = -(abs(data['slump_mm'] - 193) * 0.05)
        performance_mud_effect = -(data['mud_content_pct'] - 2.0) * 2
        
        data['performance_score'] = (performance_base + performance_strength_effect + 
                                     performance_slump_effect + performance_mud_effect +
                                     np.random.normal(0, 3, self.n_samples))
        data['performance_score'] = np.clip(data['performance_score'], 60, 100)
        
        # 转换为DataFrame
        df = pd.DataFrame(data)
        
        print(f"\n✓ 数据生成完成: {df.shape[0]} 行, {df.shape[1]} 列")
        
        return df
    
    def save_data(self, df: pd.DataFrame, output_path: str):
        """保存数据到CSV文件"""
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✓ 数据已保存到: {output_path}")


def main():
    """主函数"""
    # 生成数据
    generator = ConcreteDataGeneratorV2(n_samples=2000, random_seed=42)
    df = generator.generate_data()
    
    # 显示数据信息
    print("\n" + "="*80)
    print("数据统计信息")
    print("="*80)
    print(f"\n关键变量统计:")
    key_vars = ['water_binder_ratio', 'cement_content', 'sand_content', 'stone_content',
                'water_content', 'admixture_content', 'slump_mm', 'strength_7d_mpa', 'strength_28d_mpa']
    print(df[key_vars].describe())
    
    print(f"\n前5行数据:")
    print(df.head())
    
    # 保存数据
    output_path = "data/synthetic/concrete_aggregate_data.csv"
    generator.save_data(df, output_path)
    
    # 显示对比
    print("\n" + "="*80)
    print("与真实数据对比")
    print("="*80)
    print(f"水胶比 - 真实: 0.38-0.48 | 生成: {df['water_binder_ratio'].min():.2f}-{df['water_binder_ratio'].max():.2f} (均值{df['water_binder_ratio'].mean():.2f})")
    print(f"水泥 - 真实: 340-430 | 生成: {df['cement_content'].min():.0f}-{df['cement_content'].max():.0f} (均值{df['cement_content'].mean():.0f})")
    print(f"砂 - 真实: 738-807 | 生成: {df['sand_content'].min():.0f}-{df['sand_content'].max():.0f} (均值{df['sand_content'].mean():.0f})")
    print(f"石 - 真实: 1107-1115 | 生成: {df['stone_content'].min():.0f}-{df['stone_content'].max():.0f} (均值{df['stone_content'].mean():.0f})")
    print(f"坍落度 - 真实: 190-200 | 生成: {df['slump_mm'].min():.0f}-{df['slump_mm'].max():.0f} (均值{df['slump_mm'].mean():.0f})")
    print(f"7d强度 - 真实: 32.9-46.3 | 生成: {df['strength_7d_mpa'].min():.1f}-{df['strength_7d_mpa'].max():.1f} (均值{df['strength_7d_mpa'].mean():.1f})")
    print(f"28d强度 - 真实: 42.7-52.5 | 生成: {df['strength_28d_mpa'].min():.1f}-{df['strength_28d_mpa'].max():.1f} (均值{df['strength_28d_mpa'].mean():.1f})")
    
    return df


if __name__ == "__main__":
    df = main()

