"""
混凝土集料因果数据生成器
基于混凝土集料和配合比的因果关系生成仿真数据
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class ConcreteAggregateDataGenerator:
    """混凝土集料因果数据生成器"""
    
    def __init__(self, n_samples: int = 1000, random_seed: int = 42):
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
        生成混凝土集料因果数据
        
        Returns:
        --------
        pd.DataFrame
            生成的数据
        """
        data = {}
        
        # ===== 第一层：根节点（外生变量）=====
        
        # 1. 集料类型（根节点）
        data['aggregate_type'] = np.random.choice(
            ['花岗岩', '石灰岩', '玄武岩', '砂岩'], 
            size=self.n_samples,
            p=[0.35, 0.30, 0.25, 0.10]
        )
        
        # 2. 产地（根节点）
        data['origin'] = np.random.choice(
            ['产地A', '产地B', '产地C', '产地D'], 
            size=self.n_samples,
            p=[0.30, 0.25, 0.25, 0.20]
        )
        
        # 3. 生产工艺（根节点）
        data['production_process'] = np.random.choice(
            ['破碎工艺', '筛分工艺', '水洗工艺', '干法工艺'], 
            size=self.n_samples,
            p=[0.30, 0.25, 0.30, 0.15]
        )
        
        # 4. 环境温度（根节点）
        data['ambient_temp_c'] = np.random.normal(25, 5, self.n_samples)
        
        # 5. 环境湿度（根节点）
        data['ambient_humidity_pct'] = np.random.normal(60, 10, self.n_samples)
        
        # ===== 第二层：材质特性（受产地和集料类型影响）=====
        
        # 材质密度（受产地和集料类型影响）
        material_density_base = np.random.normal(2.65, 0.05, self.n_samples)
        
        # 产地影响
        origin_effect = pd.Series(data['origin']).map({
            '产地A': 0.05,
            '产地B': 0.02,
            '产地C': -0.02,
            '产地D': -0.03
        }).values
        
        # 集料类型影响
        aggregate_effect = pd.Series(data['aggregate_type']).map({
            '花岗岩': 0.10,
            '石灰岩': 0.00,
            '玄武岩': 0.08,
            '砂岩': -0.05
        }).values
        
        data['material_density'] = material_density_base + origin_effect + aggregate_effect
        
        # ===== 第三层：砂石比例（受集料类型影响）=====
        
        # 碎石比例
        data['coarse_aggregate_pct'] = np.random.normal(70, 5, self.n_samples)
        
        # 砂比例
        data['sand_pct'] = 100 - data['coarse_aggregate_pct']
        
        # 砂率（受产地影响）
        sand_rate_base = data['sand_pct'] / 100
        sand_rate_origin_effect = pd.Series(data['origin']).map({
            '产地A': 0.02,
            '产地B': 0.01,
            '产地C': -0.01,
            '产地D': 0.00
        }).values
        data['sand_rate'] = sand_rate_base + sand_rate_origin_effect + np.random.normal(0, 0.02, self.n_samples)
        data['sand_rate'] = np.clip(data['sand_rate'], 0.25, 0.45)
        
        # ===== 第四层：质量指标（受材质、工艺、环境影响）=====
        
        # 1. 粒形指数（受材质和生产工艺影响）
        particle_shape_base = np.random.normal(0.7, 0.1, self.n_samples)
        
        material_shape_effect = data['material_density'] * 0.15
        process_shape_effect = pd.Series(data['production_process']).map({
            '破碎工艺': -0.05,
            '筛分工艺': 0.05,
            '水洗工艺': 0.08,
            '干法工艺': 0.02
        }).values
        
        data['particle_shape_index'] = particle_shape_base + material_shape_effect + process_shape_effect
        data['particle_shape_index'] = np.clip(data['particle_shape_index'], 0.5, 1.0)
        
        # 2. 压碎值（受材质影响）
        crushing_value_base = np.random.normal(15, 3, self.n_samples)
        crushing_material_effect = (2.7 - data['material_density']) * 10
        
        data['crushing_value_pct'] = crushing_value_base + crushing_material_effect
        data['crushing_value_pct'] = np.clip(data['crushing_value_pct'], 8, 25)
        
        # 3. 针片状含量（受材质和工艺影响）
        flaky_base = np.random.normal(8, 2, self.n_samples)
        flaky_material_effect = (2.7 - data['material_density']) * 5
        flaky_process_effect = pd.Series(data['production_process']).map({
            '破碎工艺': 2,
            '筛分工艺': -1,
            '水洗工艺': -1.5,
            '干法工艺': 0
        }).values
        
        data['flaky_particle_pct'] = flaky_base + flaky_material_effect + flaky_process_effect
        data['flaky_particle_pct'] = np.clip(data['flaky_particle_pct'], 3, 15)
        
        # 4. 含泥量（受材质、工艺和环境影响）
        mud_content_base = np.random.normal(1.5, 0.5, self.n_samples)
        
        mud_process_effect = pd.Series(data['production_process']).map({
            '破碎工艺': 0.5,
            '筛分工艺': 0.3,
            '水洗工艺': -0.8,
            '干法工艺': 0.2
        }).values
        
        mud_humidity_effect = (data['ambient_humidity_pct'] - 60) * 0.01
        
        data['mud_content_pct'] = mud_content_base + mud_process_effect + mud_humidity_effect
        data['mud_content_pct'] = np.clip(data['mud_content_pct'], 0.2, 4.0)
        
        # 5. 泥块含量（受含泥量和工艺影响）
        clay_lump_base = data['mud_content_pct'] * 0.3
        clay_process_effect = pd.Series(data['production_process']).map({
            '破碎工艺': 0.1,
            '筛分工艺': 0.05,
            '水洗工艺': -0.15,
            '干法工艺': 0.05
        }).values
        
        data['clay_lump_pct'] = clay_lump_base + clay_process_effect + np.random.normal(0, 0.1, self.n_samples)
        data['clay_lump_pct'] = np.clip(data['clay_lump_pct'], 0, 1.5)
        
        # ===== 第五层：水胶比系统（配合比设计）=====
        
        # 水胶比（目标设计值）
        data['water_binder_ratio'] = np.random.normal(0.45, 0.08, self.n_samples)
        data['water_binder_ratio'] = np.clip(data['water_binder_ratio'], 0.30, 0.65)
        
        # 水泥用量（kg/m³）
        data['cement_content'] = np.random.normal(350, 50, self.n_samples)
        data['cement_content'] = np.clip(data['cement_content'], 250, 500)
        
        # 粉煤灰掺量（受水胶比影响）
        fly_ash_ratio = 0.15 + (0.50 - data['water_binder_ratio']) * 0.2
        data['fly_ash_content'] = data['cement_content'] * fly_ash_ratio
        data['fly_ash_content'] = np.clip(data['fly_ash_content'], 30, 120)
        
        # 矿渣粉掺量
        slag_ratio = 0.10 + (0.50 - data['water_binder_ratio']) * 0.15
        data['slag_powder_content'] = data['cement_content'] * slag_ratio
        data['slag_powder_content'] = np.clip(data['slag_powder_content'], 20, 100)
        
        # 胶凝材料总量
        data['total_binder'] = data['cement_content'] + data['fly_ash_content'] + data['slag_powder_content']
        
        # 掺配比例
        data['mix_ratio'] = data['cement_content'] / data['total_binder']
        
        # 水用量
        data['water_content'] = data['total_binder'] * data['water_binder_ratio']
        
        # 外加剂掺量（受水胶比影响）
        admixture_ratio = 0.015 - (data['water_binder_ratio'] - 0.45) * 0.01
        data['admixture_content'] = data['total_binder'] * admixture_ratio
        data['admixture_content'] = np.clip(data['admixture_content'], 3, 15)
        
        # ===== 第六层：配合比准确性（受各组分影响）=====
        
        mix_accuracy_base = 95.0
        
        # 水胶比偏差影响
        water_binder_deviation = abs(data['water_binder_ratio'] - 0.45) * 10
        
        # 掺配比例影响
        mix_ratio_effect = (data['mix_ratio'] - 0.70) * 5
        
        data['mix_design_accuracy_pct'] = (mix_accuracy_base - water_binder_deviation + 
                                           mix_ratio_effect + np.random.normal(0, 2, self.n_samples))
        data['mix_design_accuracy_pct'] = np.clip(data['mix_design_accuracy_pct'], 80, 100)
        
        # ===== 第七层：混凝土强度（最终结果变量）=====
        
        # 基础强度
        strength_base = 35.0
        
        # 水胶比影响（最关键因素）
        water_binder_effect = (0.50 - data['water_binder_ratio']) * 80
        
        # 材质密度影响
        density_effect = (data['material_density'] - 2.65) * 50
        
        # 产地影响
        origin_strength_effect = pd.Series(data['origin']).map({
            '产地A': 3,
            '产地B': 1,
            '产地C': -1,
            '产地D': -2
        }).values
        
        # 粒形影响
        shape_effect = (data['particle_shape_index'] - 0.7) * 15
        
        # 压碎值影响（负向）
        crushing_effect = -(data['crushing_value_pct'] - 15) * 0.5
        
        # 针片状影响（负向）
        flaky_effect = -(data['flaky_particle_pct'] - 8) * 0.8
        
        # 含泥量影响（负向）
        mud_effect = -(data['mud_content_pct'] - 1.5) * 3
        
        # 泥块影响（负向）
        clay_effect = -(data['clay_lump_pct'] - 0.5) * 5
        
        # 配合比准确性影响
        mix_accuracy_effect = (data['mix_design_accuracy_pct'] - 95) * 0.3
        
        # 砂率影响
        sand_rate_effect = -(abs(data['sand_rate'] - 0.35) * 20)
        
        data['concrete_strength_mpa'] = (strength_base + water_binder_effect + density_effect + 
                                         origin_strength_effect + shape_effect + crushing_effect + 
                                         flaky_effect + mud_effect + clay_effect + 
                                         mix_accuracy_effect + sand_rate_effect +
                                         np.random.normal(0, 3, self.n_samples))
        
        data['concrete_strength_mpa'] = np.clip(data['concrete_strength_mpa'], 20, 80)
        
        # ===== 第八层：工程性能指标（受强度影响）=====
        
        # 质量合格率
        strength_pass_threshold = 30  # MPa
        data['quality_pass'] = (data['concrete_strength_mpa'] >= strength_pass_threshold).astype(int)
        data['quality_pass_rate_pct'] = (data['concrete_strength_mpa'] / strength_pass_threshold * 100)
        data['quality_pass_rate_pct'] = np.clip(data['quality_pass_rate_pct'], 60, 120)
        
        # 路面性能（受强度、粒形、针片状影响）
        pavement_base = 80.0
        pavement_strength_effect = (data['concrete_strength_mpa'] - 35) * 0.5
        pavement_shape_effect = (data['particle_shape_index'] - 0.7) * 10
        pavement_flaky_effect = -(data['flaky_particle_pct'] - 8) * 0.5
        
        data['pavement_performance_score'] = (pavement_base + pavement_strength_effect + 
                                              pavement_shape_effect + pavement_flaky_effect +
                                              np.random.normal(0, 3, self.n_samples))
        data['pavement_performance_score'] = np.clip(data['pavement_performance_score'], 60, 100)
        
        # 结构强度（受混凝土强度和配合比准确性影响）
        structural_base = data['concrete_strength_mpa'] * 0.95
        structural_mix_effect = (data['mix_design_accuracy_pct'] - 95) * 0.2
        structural_aggregate_effect = pd.Series(data['aggregate_type']).map({
            '花岗岩': 2,
            '石灰岩': 0,
            '玄武岩': 3,
            '砂岩': -2
        }).values
        
        data['structural_strength_mpa'] = (structural_base + structural_mix_effect + 
                                           structural_aggregate_effect +
                                           np.random.normal(0, 2, self.n_samples))
        data['structural_strength_mpa'] = np.clip(data['structural_strength_mpa'], 20, 80)
        
        # 转换为DataFrame
        df = pd.DataFrame(data)
        
        return df
    
    def save_data(self, df: pd.DataFrame, output_path: str):
        """
        保存数据到CSV文件
        
        Parameters:
        -----------
        df : pd.DataFrame
            数据
        output_path : str
            输出路径
        """
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✓ 数据已保存到: {output_path}")


def main():
    """主函数"""
    # 生成数据
    generator = ConcreteAggregateDataGenerator(n_samples=2000, random_seed=42)
    df = generator.generate_data()
    
    # 显示数据信息
    print("="*80)
    print("混凝土集料因果数据生成完成")
    print("="*80)
    print(f"\n数据形状: {df.shape}")
    print(f"\n列名:\n{df.columns.tolist()}")
    print(f"\n前5行数据:\n{df.head()}")
    print(f"\n基本统计:\n{df.describe()}")
    
    # 保存数据
    output_path = "/Users/superkang/Desktop/causalmodel/data/synthetic/concrete_aggregate_data.csv"
    generator.save_data(df, output_path)
    
    return df


if __name__ == "__main__":
    df = main()

