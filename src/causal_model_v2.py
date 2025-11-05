"""
混凝土配合比因果模型 V2
基于真实配合比数据的因果图结构
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable
from functools import partial

from dowhy import gcm
from scipy import stats


class ConcreteConfigurationCausalModel:
    """混凝土配合比因果模型（基于真实数据）"""
    
    def __init__(self, df: pd.DataFrame):
        """
        初始化因果模型
        
        Parameters:
        -----------
        df : pd.DataFrame
            混凝土配合比数据
        """
        self.df = df
        self.causal_graph = None
        self.causal_model = None
        
    def build_causal_graph(self) -> nx.DiGraph:
        """
        构建混凝土配合比因果图（根据图片简化版）
        
        核心因果关系（参考图片）：
        1. 水胶比层：水胶比 → 胶凝材料、水、外加剂
        2. 材料层：砂、石 → 砂率
        3. 质量层：产地 → 材质 → 质量指标（粒形、压碎值、针片状、含泥量、泥块含量）
        4. 强度层：水胶比 + 5大材料 + 质量指标 → 混凝土强度（7d → 28d）
        
        Returns:
        --------
        nx.DiGraph
            因果图
        """
        edges = []
        
        print("构建简化因果图（基于图片）...")
        
        # ===== 根节点 =====
        # cement_type, sand_type, stone_size, sand_origin, stone_origin, admixture_type, water_binder_ratio
        
        # ===== 配合比设计：水胶比 → 5大材料 =====
        
        # 1. 水泥
        edges.append(('water_binder_ratio', 'cement_content'))
        
        # 2. 水
        edges.append(('water_binder_ratio', 'water_content'))
        
        # 3. 砂（细集料）
        edges.append(('sand_type', 'sand_content'))
        
        # 4. 石（粗集料）
        edges.append(('stone_size', 'stone_content'))
        
        # 5. 外加剂
        edges.append(('water_binder_ratio', 'admixture_content'))
        edges.append(('admixture_type', 'admixture_content'))
        
        # 砂石 → 砂率
        edges.append(('sand_content', 'sand_rate'))
        edges.append(('stone_content', 'sand_rate'))
        
        # ===== 产地 → 材质（图片左侧） =====
        
        # 产地影响砂率（图片中产地有箭头指向砂率）
        edges.append(('sand_origin', 'sand_rate'))
        
        # ===== 材质 → 质量指标（图片下方） =====
        
        # 产地 → 含泥量、泥块含量
        edges.append(('sand_origin', 'mud_content_pct'))
        
        # 产地 → 压碎值
        edges.append(('stone_origin', 'crushing_value_pct'))
        
        # 产地/粒径 → 针片状
        edges.append(('stone_origin', 'flaky_particle_pct'))
        edges.append(('stone_size', 'flaky_particle_pct'))
        
        # ===== 混凝土强度（图片核心结果） =====
        
        # 水胶比（最关键！）
        edges.append(('water_binder_ratio', 'strength_7d_mpa'))
        
        # 水泥
        edges.append(('cement_content', 'strength_7d_mpa'))
        
        # 砂率（图片中砂率指向混凝土强度）
        edges.append(('sand_rate', 'strength_7d_mpa'))
        
        # 质量指标 → 强度（图片中所有质量指标都指向混凝土强度）
        edges.append(('mud_content_pct', 'strength_7d_mpa'))
        edges.append(('crushing_value_pct', 'strength_7d_mpa'))
        edges.append(('flaky_particle_pct', 'strength_7d_mpa'))
        
        # 外加剂
        edges.append(('admixture_content', 'strength_7d_mpa'))
        
        # 7d → 28d强度
        edges.append(('strength_7d_mpa', 'strength_28d_mpa'))
        
        # 水胶比也直接影响28d
        edges.append(('water_binder_ratio', 'strength_28d_mpa'))
        
        self.causal_graph = nx.DiGraph(edges)
        
        print(f"✓ 因果图构建完成")
        print(f"  节点数: {self.causal_graph.number_of_nodes()}")
        print(f"  边数: {self.causal_graph.number_of_edges()}")
        
        return self.causal_graph
    
    def fit_causal_model(self, quality: str = 'BETTER', invertible: bool = True):
        """
        拟合因果模型
        
        Parameters:
        -----------
        quality : str
            模型质量 ('GOOD', 'BETTER', 'BEST')
        invertible : bool
            是否使用可逆模型（用于反事实分析）
        """
        if self.causal_graph is None:
            raise ValueError("请先构建因果图")
        
        print(f"\n拟合因果模型...")
        print(f"  质量模式: {quality}")
        print(f"  可逆模型: {invertible}")
        
        # 准备数据：处理分类变量
        df_processed = self.df.copy()
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        
        # 对于因果图中的分类变量，使用标签编码
        for col in categorical_cols:
            if col in self.causal_graph.nodes:
                df_processed[col] = pd.Categorical(df_processed[col]).codes
        
        # 只保留因果图中的列
        cols_in_graph = [col for col in df_processed.columns if col in self.causal_graph.nodes]
        df_processed = df_processed[cols_in_graph]
        
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
    
    def intervention_analysis(self,
                             target: str,
                             step_size: float = 1,
                             non_interveneable_nodes: List[str] = None,
                             confidence_level: float = 0.95,
                             num_samples: int = 10000,
                             num_bootstrap_resamples: int = 40) -> pd.DataFrame:
        """
        干预分析：评估不同干预措施的效果
        
        Parameters:
        -----------
        target : str
            目标KPI
        step_size : float
            干预步长
        non_interveneable_nodes : List[str]
            不可干预的节点
        confidence_level : float
            置信水平
        num_samples : int
            采样数
        num_bootstrap_resamples : int
            自助法重采样次数
            
        Returns:
        --------
        pd.DataFrame
            干预效果结果
        """
        if self.causal_model is None:
            raise ValueError("请先拟合因果模型")
        
        if non_interveneable_nodes is None:
            non_interveneable_nodes = []
        
        causal_effects = {}
        causal_effects_ci = {}
        
        for node in self.causal_graph.nodes:
            if node in non_interveneable_nodes or node == target:
                continue
            
            # 跳过分类变量
            if self.df[node].dtype == 'object':
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
        """
        反事实分析：评估"如果做不同操作会怎样"
        
        Parameters:
        -----------
        observed_data : pd.DataFrame
            观测数据
        interventions : Dict[str, float]
            干预变量及其目标值
        target : str
            目标变量
        num_samples : int
            采样数
            
        Returns:
        --------
        Dict
            反事实分析结果
        """
        if self.causal_model is None:
            raise ValueError("请先拟合因果模型")
        
        # 处理分类变量
        observed_processed = observed_data.copy()
        categorical_cols = observed_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in self.causal_graph.nodes:
                observed_processed[col] = pd.Categorical(observed_processed[col]).codes
        
        # 只保留因果图中的列
        cols_in_graph = [col for col in observed_processed.columns if col in self.causal_graph.nodes]
        observed_processed = observed_processed[cols_in_graph]
        
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


# 为了兼容性，保留旧的类名作为别名
ConcreteAggregateCausalModel = ConcreteConfigurationCausalModel

