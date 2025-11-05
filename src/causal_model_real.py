"""
混凝土抗压强度因果模型 - 基于真实Kaggle数据集
数据来源：Concrete Compressive Strength Dataset (UCI Machine Learning Repository)
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable
from functools import partial

from dowhy import gcm
from scipy import stats


class ConcreteStrengthCausalModel:
    """混凝土抗压强度因果模型（真实数据）"""
    
    def __init__(self, df: pd.DataFrame):
        """
        初始化因果模型
        
        Parameters:
        -----------
        df : pd.DataFrame
            混凝土抗压强度数据
            
        数据字段（9个）：
        - cement: 水泥 (kg/m³)
        - blast_furnace_slag: 高炉矿渣 (kg/m³)
        - fly_ash: 粉煤灰 (kg/m³)
        - water: 水 (kg/m³)
        - superplasticizer: 高效减水剂 (kg/m³)
        - coarse_aggregate: 粗骨料 (kg/m³)
        - fine_aggregate: 细骨料 (kg/m³)
        - age: 龄期 (天)
        - concrete_compressive_strength: 抗压强度 (MPa)
        """
        self.df = df
        self.causal_graph = None
        self.causal_model = None
    
    def build_causal_graph(self) -> nx.DiGraph:
        """
        构建混凝土强度因果图（仅使用9个原始变量）
        
        基于Yeh 1998论文和混凝土材料科学理论
        
        核心因果关系：
        1. **水-减水剂负相关**（r = -0.66）：减水剂越多，所需水越少
        2. **水泥主导强度**：活性最高的胶凝材料
        3. **矿物掺合料协同**：矿渣和粉煤灰的长期强度贡献
        4. **龄期效应**：强度随时间发展
        5. **骨料影响**：细骨料和粗骨料对强度的影响
        
        参考文献：
        - Yeh, I-Cheng (1998). "Modeling of strength of high-performance concrete using ANN"
        - UCI Machine Learning Repository
        
        Returns:
        --------
        nx.DiGraph
            因果图（仅9个原始变量）
        """
        edges = []
        
        print("构建混凝土强度因果图（仅9个原始变量）...")
        
        # ===== 根节点（外生变量）=====
        # cement, blast_furnace_slag, fly_ash, superplasticizer, 
        # coarse_aggregate, fine_aggregate, age
        
        # ===== 关键因果关系 =====
        
        # 1. 减水剂 → 水（负相关，r = -0.66）
        #    减水剂分散水泥颗粒，减少所需水量
        edges.append(('superplasticizer', 'water'))
        
        # 2. 水泥 → 强度（最关键的胶凝材料）⭐⭐⭐
        edges.append(('cement', 'concrete_compressive_strength'))
        
        # 3. 高炉矿渣 → 强度（提高密实度和耐久性）⭐⭐
        edges.append(('blast_furnace_slag', 'concrete_compressive_strength'))
        
        # 4. 粉煤灰 → 强度（火山灰反应，长期强度）⭐⭐
        edges.append(('fly_ash', 'concrete_compressive_strength'))
        
        # 5. 水 → 强度（水胶比效应的体现，Abrams定律）⭐⭐⭐
        #    水越多，强度越低（负相关）
        edges.append(('water', 'concrete_compressive_strength'))
        
        # 6. 减水剂 → 强度（改善密实度）⭐
        edges.append(('superplasticizer', 'concrete_compressive_strength'))
        
        # 7. 粗骨料 → 强度（骨架作用）
        edges.append(('coarse_aggregate', 'concrete_compressive_strength'))
        
        # 8. 细骨料 → 强度（填充作用）
        edges.append(('fine_aggregate', 'concrete_compressive_strength'))
        
        # 9. 龄期 → 强度（时间效应，水化反应持续）⭐⭐⭐
        edges.append(('age', 'concrete_compressive_strength'))
        
        self.causal_graph = nx.DiGraph(edges)
        
        print(f"✓ 因果图构建完成（仅9个原始变量，无衍生变量）")
        print(f"  节点数: {self.causal_graph.number_of_nodes()}")
        print(f"  边数: {self.causal_graph.number_of_edges()}")
        print(f"\n核心因果路径：")
        print(f"  ⭐⭐⭐ 水泥 → 强度")
        print(f"  ⭐⭐⭐ 水 → 强度（Abrams定律：水越多强度越低）")
        print(f"  ⭐⭐⭐ 龄期 → 强度（时间效应）")
        print(f"  ⭐⭐  矿渣 → 强度")
        print(f"  ⭐⭐  粉煤灰 → 强度")
        print(f"  ⭐   减水剂 → 水（负相关）+ 减水剂 → 强度")
        
        return self.causal_graph
    
    def fit_causal_model(self, quality: str = 'BETTER', invertible: bool = True):
        """拟合因果模型"""
        if self.causal_graph is None:
            raise ValueError("请先构建因果图")
        
        print(f"\n拟合因果模型...")
        print(f"  质量模式: {quality}")
        print(f"  可逆模型: {invertible}")
        
        # 只保留因果图中的列
        cols_in_graph = [col for col in self.df.columns if col in self.causal_graph.nodes]
        df_processed = self.df[cols_in_graph].copy()
        
        # 根据是否需要反事实分析选择模型类型
        if invertible:
            self.causal_model = gcm.InvertibleStructuralCausalModel(self.causal_graph)
            print("  使用可逆因果模型（支持反事实分析）")
        else:
            self.causal_model = gcm.StructuralCausalModel(self.causal_graph)
            print("  使用标准因果模型")
        
        quality_map = {
            'GOOD': gcm.auto.AssignmentQuality.GOOD,
            'BETTER': gcm.auto.AssignmentQuality.BETTER,
            'BEST': gcm.auto.AssignmentQuality.BEST
        }
        
        print("\n  分配因果机制...")
        gcm.auto.assign_causal_mechanisms(
            self.causal_model, 
            df_processed, 
            quality=quality_map.get(quality, gcm.auto.AssignmentQuality.BETTER)
        )
        
        print("  拟合模型...")
        gcm.fit(self.causal_model, df_processed)
        print("\n✓ 因果模型拟合完成!")
    
    def attribution_analysis(self,
                            df_old: pd.DataFrame,
                            df_new: pd.DataFrame,
                            target_column: str,
                            difference_func: Callable = None,
                            num_samples: int = 2000,
                            confidence_level: float = 0.90,
                            num_bootstrap_resamples: int = 4) -> Tuple[Dict, Dict]:
        """归因分析"""
        if self.causal_model is None:
            raise ValueError("请先拟合因果模型")
        
        if difference_func is None:
            difference_func = lambda x1, x2: np.mean(x2) - np.mean(x1)
        
        # 只保留因果图中的列
        cols_in_graph = [col for col in df_old.columns if col in self.causal_graph.nodes]
        df_old_processed = df_old[cols_in_graph]
        df_new_processed = df_new[cols_in_graph]
        
        contributions, uncertainties = gcm.confidence_intervals(
            lambda: gcm.distribution_change(
                self.causal_model,
                df_old_processed,
                df_new_processed,
                target_column,
                num_samples=num_samples,
                difference_estimation_func=difference_func,
                shapley_config=gcm.shapley.ShapleyConfig(
                    approximation_method=gcm.shapley.ShapleyApproximationMethods.PERMUTATION,
                    num_permutations=50
                )
            ),
            confidence_level=confidence_level,
            num_bootstrap_resamples=num_bootstrap_resamples
        )
        
        return contributions, uncertainties
    
    def intervention_analysis(self,
                             target: str,
                             step_size: float = 1,
                             non_interveneable_nodes: List[str] = None,
                             confidence_level: float = 0.95,
                             num_samples: int = 10000,
                             num_bootstrap_resamples: int = 40) -> pd.DataFrame:
        """干预分析"""
        if self.causal_model is None:
            raise ValueError("请先拟合因果模型")
        
        if non_interveneable_nodes is None:
            non_interveneable_nodes = []
        
        causal_effects = {}
        causal_effects_ci = {}
        
        for node in self.causal_graph.nodes:
            if node in non_interveneable_nodes or node == target:
                continue
            
            # 定义干预
            def intervention(x):
                return x + step_size
            
            def non_intervention(x):
                return x
            
            interventions_alternative = {node: intervention}
            interventions_reference = {node: non_intervention}
            
            try:
                effect = gcm.confidence_intervals(
                    partial(gcm.average_causal_effect,
                           causal_model=self.causal_model,
                           target_node=target,
                           interventions_alternative=interventions_alternative,
                           interventions_reference=interventions_reference,
                           num_samples_to_draw=num_samples),
                    num_bootstrap_resamples=num_bootstrap_resamples,
                    confidence_level=confidence_level
                )
                
                causal_effects[node] = effect[0][0] if isinstance(effect[0], np.ndarray) else effect[0]
                causal_effects_ci[node] = effect[1].squeeze() if isinstance(effect[1], np.ndarray) else effect[1]
            
            except Exception as e:
                print(f"  跳过 {node}: {str(e)[:80]}")
                continue
        
        # 构建结果DataFrame
        result_df = pd.DataFrame({
            'Variable': list(causal_effects.keys()),
            'Causal_Effect': list(causal_effects.values()),
            'Lower_CI': [causal_effects_ci[k][0] if isinstance(causal_effects_ci[k], (list, np.ndarray)) else np.nan 
                        for k in causal_effects.keys()],
            'Upper_CI': [causal_effects_ci[k][1] if isinstance(causal_effects_ci[k], (list, np.ndarray)) else np.nan 
                        for k in causal_effects.keys()]
        })
        
        result_df = result_df.sort_values('Causal_Effect', key=abs, ascending=False)
        
        return result_df
    
    def counterfactual_analysis(self,
                                observed_data: pd.DataFrame,
                                interventions: Dict[str, float],
                                target: str,
                                num_samples: int = 1000) -> Dict:
        """反事实分析"""
        if self.causal_model is None:
            raise ValueError("请先拟合因果模型")
        
        # 只保留因果图中的列
        cols_in_graph = [col for col in observed_data.columns if col in self.causal_graph.nodes]
        observed_processed = observed_data[cols_in_graph]
        
        # 构建干预函数
        intervention_funcs = {}
        for var, value in interventions.items():
            intervention_funcs[var] = lambda x, v=value: v
        
        # 执行反事实查询
        counterfactual_samples_result = gcm.counterfactual_samples(
            self.causal_model,
            intervention_funcs,
            observed_data=observed_processed
        )
        
        # 计算结果
        result = {
            'observed_mean': observed_processed[target].mean(),
            'counterfactual_mean': counterfactual_samples_result[target].mean(),
            'causal_effect': counterfactual_samples_result[target].mean() - observed_processed[target].mean(),
            'counterfactual_samples': counterfactual_samples_result[target]
        }
        
        return result


# 为了兼容性，提供别名
ConcreteAggregateCausalModel = ConcreteStrengthCausalModel

