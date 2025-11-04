"""
混凝土集料和配合比因果模型
基于 DoWhy 进行因果分析
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable
from functools import partial

from dowhy import gcm
from dowhy.utils.plotting import plot
from scipy import stats
from statsmodels.stats.multitest import multipletests


class ConcreteAggregateCausalModel:
    """混凝土集料和配合比因果模型"""
    
    def __init__(self, df: pd.DataFrame):
        """
        初始化因果模型
        
        Parameters:
        -----------
        df : pd.DataFrame
            混凝土集料数据
        """
        self.df = df
        self.causal_graph = None
        self.causal_model = None
        
    def build_causal_graph(self) -> nx.DiGraph:
        """
        构建混凝土集料因果图
        
        基于图片中的因果关系：
        1. 集料系统：集料类型 → 材质 → 质量指标 → 混凝土强度
        2. 产地影响：产地 → 砂率、材质 → 强度
        3. 工艺影响：生产工艺 → 含泥量、泥块含量
        4. 配合比系统：水胶比 → 胶凝材料 → 配合比准确性 → 强度
        5. 工程性能：混凝土强度 → 质量合格率、路面性能、结构强度
        
        Returns:
        --------
        nx.DiGraph
            因果图
        """
        edges = []
        
        # ===== 集料系统因果关系 =====
        
        # 1. 集料类型 → 材质密度
        edges.append(('aggregate_type', 'material_density'))
        
        # 2. 产地 → 材质密度、砂率
        edges.append(('origin', 'material_density'))
        edges.append(('origin', 'sand_rate'))
        
        # 3. 集料 → 砂石比例
        edges.append(('coarse_aggregate_pct', 'sand_pct'))
        edges.append(('sand_pct', 'sand_rate'))
        
        # 4. 材质 → 质量指标
        edges.append(('material_density', 'particle_shape_index'))
        edges.append(('material_density', 'crushing_value_pct'))
        edges.append(('material_density', 'flaky_particle_pct'))
        
        # 5. 生产工艺 → 质量指标
        edges.append(('production_process', 'particle_shape_index'))
        edges.append(('production_process', 'flaky_particle_pct'))
        edges.append(('production_process', 'mud_content_pct'))
        edges.append(('production_process', 'clay_lump_pct'))
        
        # 6. 环境因素 → 含泥量
        edges.append(('ambient_humidity_pct', 'mud_content_pct'))
        
        # 7. 含泥量 → 泥块含量
        edges.append(('mud_content_pct', 'clay_lump_pct'))
        
        # ===== 配合比系统因果关系 =====
        
        # 8. 水胶比 → 胶凝材料用量
        edges.append(('water_binder_ratio', 'fly_ash_content'))
        edges.append(('water_binder_ratio', 'slag_powder_content'))
        edges.append(('water_binder_ratio', 'water_content'))
        edges.append(('water_binder_ratio', 'admixture_content'))
        
        # 9. 胶凝材料 → 总胶凝材料量
        edges.append(('cement_content', 'total_binder'))
        edges.append(('fly_ash_content', 'total_binder'))
        edges.append(('slag_powder_content', 'total_binder'))
        
        # 10. 胶凝材料 → 掺配比例
        edges.append(('cement_content', 'mix_ratio'))
        edges.append(('total_binder', 'mix_ratio'))
        
        # 11. 掺配比例和水胶比 → 配合比准确性
        edges.append(('water_binder_ratio', 'mix_design_accuracy_pct'))
        edges.append(('mix_ratio', 'mix_design_accuracy_pct'))
        
        # ===== 混凝土强度因果关系 =====
        
        # 12. 质量指标 → 混凝土强度
        edges.append(('particle_shape_index', 'concrete_strength_mpa'))
        edges.append(('crushing_value_pct', 'concrete_strength_mpa'))
        edges.append(('flaky_particle_pct', 'concrete_strength_mpa'))
        edges.append(('mud_content_pct', 'concrete_strength_mpa'))
        edges.append(('clay_lump_pct', 'concrete_strength_mpa'))
        
        # 13. 材质和产地 → 混凝土强度
        edges.append(('material_density', 'concrete_strength_mpa'))
        edges.append(('origin', 'concrete_strength_mpa'))
        
        # 14. 配合比系统 → 混凝土强度
        edges.append(('water_binder_ratio', 'concrete_strength_mpa'))
        edges.append(('mix_design_accuracy_pct', 'concrete_strength_mpa'))
        
        # 15. 砂率 → 混凝土强度
        edges.append(('sand_rate', 'concrete_strength_mpa'))
        
        # ===== 工程性能因果关系 =====
        
        # 16. 混凝土强度 → 质量合格率
        edges.append(('concrete_strength_mpa', 'quality_pass'))
        edges.append(('concrete_strength_mpa', 'quality_pass_rate_pct'))
        
        # 17. 混凝土强度和质量指标 → 路面性能
        edges.append(('concrete_strength_mpa', 'pavement_performance_score'))
        edges.append(('particle_shape_index', 'pavement_performance_score'))
        edges.append(('flaky_particle_pct', 'pavement_performance_score'))
        
        # 18. 混凝土强度和配合比 → 结构强度
        edges.append(('concrete_strength_mpa', 'structural_strength_mpa'))
        edges.append(('mix_design_accuracy_pct', 'structural_strength_mpa'))
        edges.append(('aggregate_type', 'structural_strength_mpa'))
        
        self.causal_graph = nx.DiGraph(edges)
        return self.causal_graph
    
    def test_causal_minimality(self, 
                               target: str, 
                               method: str = 'kernel',
                               significance_level: float = 0.10,
                               fdr_control_method: str = 'fdr_bh') -> List[str]:
        """
        测试因果最小性，识别不显著的边
        
        Parameters:
        -----------
        target : str
            目标节点
        method : str
            独立性检验方法
        significance_level : float
            显著性水平
        fdr_control_method : str
            错误发现率控制方法
            
        Returns:
        --------
        List[str]
            不显著的父节点列表
        """
        if self.causal_graph is None:
            raise ValueError("请先构建因果图")
        
        p_vals = []
        all_parents = list(self.causal_graph.predecessors(target))
        
        for node in all_parents:
            tmp_conditioning_set = list(all_parents)
            tmp_conditioning_set.remove(node)
            
            # 处理分类变量
            if self.df[node].dtype == 'object':
                # 使用one-hot编码
                node_data = pd.get_dummies(self.df[node], drop_first=True).values
                if node_data.ndim == 1:
                    node_data = node_data.reshape(-1, 1)
            else:
                node_data = self.df[node].to_numpy()
            
            if len(tmp_conditioning_set) > 0:
                # 处理条件集中的分类变量
                conditioning_data = []
                for cond_node in tmp_conditioning_set:
                    if self.df[cond_node].dtype == 'object':
                        cond_data = pd.get_dummies(self.df[cond_node], drop_first=True).values
                        if cond_data.ndim == 1:
                            cond_data = cond_data.reshape(-1, 1)
                        conditioning_data.append(cond_data)
                    else:
                        conditioning_data.append(self.df[cond_node].to_numpy().reshape(-1, 1))
                conditioning_array = np.hstack(conditioning_data)
            else:
                conditioning_array = None
            
            try:
                if conditioning_array is not None:
                    p_val = gcm.independence_test(
                        self.df[target].to_numpy(), 
                        node_data, 
                        conditioning_array, 
                        method=method
                    )
                else:
                    p_val = gcm.independence_test(
                        self.df[target].to_numpy(), 
                        node_data, 
                        method=method
                    )
                p_vals.append(p_val)
            except Exception as e:
                print(f"警告: 节点 {node} 的独立性检验失败: {e}")
                p_vals.append(1.0)  # 保守处理，假设独立
        
        if fdr_control_method is not None and len(p_vals) > 0:
            p_vals = multipletests(p_vals, significance_level, method=fdr_control_method)[1]
        
        nodes_above_threshold = []
        nodes_below_threshold = []
        
        for i, node in enumerate(all_parents):
            if p_vals[i] < significance_level:
                nodes_above_threshold.append(node)
            else:
                nodes_below_threshold.append(node)
        
        print(f"显著连接: {[(n, target) for n in sorted(nodes_above_threshold)]}")
        print(f"不显著连接: {[(n, target) for n in sorted(nodes_below_threshold)]}")
        
        return sorted(nodes_below_threshold)
    
    def prune_graph(self, targets: List[str]):
        """
        修剪因果图，移除不显著的边
        
        Parameters:
        -----------
        targets : List[str]
            要检验的目标节点列表
        """
        if self.causal_graph is None:
            raise ValueError("请先构建因果图")
        
        for target in targets:
            if target in self.causal_graph.nodes:
                insignificant_parents = self.test_causal_minimality(target)
                for parent in insignificant_parents:
                    if self.causal_graph.has_edge(parent, target):
                        self.causal_graph.remove_edge(parent, target)
        
        # 移除孤立节点
        isolated_nodes = [node for node in self.causal_graph.nodes 
                         if self.causal_graph.in_degree(node) + self.causal_graph.out_degree(node) == 0]
        self.causal_graph.remove_nodes_from(isolated_nodes)
        
        if isolated_nodes:
            print(f"移除孤立节点: {isolated_nodes}")
    
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
            print("使用可逆因果模型（支持反事实分析）")
        else:
            self.causal_model = gcm.StructuralCausalModel(self.causal_graph)
            print("使用标准因果模型")
        
        quality_map = {
            'GOOD': gcm.auto.AssignmentQuality.GOOD,
            'BETTER': gcm.auto.AssignmentQuality.BETTER,
            'BEST': gcm.auto.AssignmentQuality.BEST
        }
        
        print(gcm.auto.assign_causal_mechanisms(
            self.causal_model, 
            df_processed, 
            quality=quality_map.get(quality, gcm.auto.AssignmentQuality.BETTER)
        ))
        
        gcm.fit(self.causal_model, df_processed)
        print("\n因果模型拟合完成!")
    
    def attribution_analysis(self,
                            df_old: pd.DataFrame,
                            df_new: pd.DataFrame,
                            target_column: str,
                            difference_func: Callable = None,
                            num_samples: int = 2000,
                            confidence_level: float = 0.90,
                            num_bootstrap_resamples: int = 4) -> Tuple[Dict, Dict]:
        """
        归因分析：找出KPI变化的根本原因
        
        Parameters:
        -----------
        df_old : pd.DataFrame
            旧时期数据
        df_new : pd.DataFrame
            新时期数据
        target_column : str
            目标KPI
        difference_func : Callable
            差异估计函数
        num_samples : int
            采样数
        confidence_level : float
            置信水平
        num_bootstrap_resamples : int
            自助法重采样次数
            
        Returns:
        --------
        contributions, uncertainties : Tuple[Dict, Dict]
            贡献和不确定性
        """
        if self.causal_model is None:
            raise ValueError("请先拟合因果模型")
        
        if difference_func is None:
            difference_func = lambda x1, x2: np.mean(x2) - np.mean(x1)
        
        # 处理分类变量
        df_old_processed = df_old.copy()
        df_new_processed = df_new.copy()
        
        categorical_cols = df_old_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in self.causal_graph.nodes:
                df_old_processed[col] = pd.Categorical(df_old_processed[col]).codes
                df_new_processed[col] = pd.Categorical(df_new_processed[col]).codes
        
        # 只保留因果图中的列
        cols_in_graph = [col for col in df_old_processed.columns if col in self.causal_graph.nodes]
        df_old_processed = df_old_processed[cols_in_graph]
        df_new_processed = df_new_processed[cols_in_graph]
        
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
                print(f"警告: 节点 {node} 的干预分析失败: {e}")
                causal_effects[node] = 0
                causal_effects_ci[node] = [np.nan, np.nan]
        
        # 构建结果DataFrame
        result_df = pd.DataFrame({
            'Variable': list(causal_effects.keys()),
            'Causal_Effect': list(causal_effects.values()),
            'Lower_CI': [causal_effects_ci[k][0] if isinstance(causal_effects_ci[k], (list, np.ndarray)) else np.nan 
                        for k in causal_effects.keys()],
            'Upper_CI': [causal_effects_ci[k][1] if isinstance(causal_effects_ci[k], (list, np.ndarray)) else np.nan 
                        for k in causal_effects.keys()]
        })
        
        result_df = result_df.sort_values('Causal_Effect', ascending=False)
        
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
        
        # 执行反事实查询（修正参数名）
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


def compare_metrics(df_new: pd.DataFrame, 
                   df_old: pd.DataFrame, 
                   metrics: List[str]) -> pd.DataFrame:
    """
    比较两个时期的指标变化
    
    Parameters:
    -----------
    df_new : pd.DataFrame
        新时期数据
    df_old : pd.DataFrame
        旧时期数据
    metrics : List[str]
        要比较的指标列表
        
    Returns:
    --------
    pd.DataFrame
        比较结果
    """
    comparison_data = []
    
    for metric in metrics:
        try:
            mean_old = df_old[metric].mean()
            median_old = df_old[metric].median()
            variance_old = df_old[metric].var()
            
            mean_new = df_new[metric].mean()
            median_new = df_new[metric].median()
            variance_new = df_new[metric].var()
            
            mean_change = ((mean_new - mean_old) / mean_old) * 100 if mean_old != 0 else None
            median_change = ((median_new - median_old) / median_old) * 100 if median_old != 0 else None
            variance_change = ((variance_new - variance_old) / variance_old) * 100 if variance_old != 0 else None
            
            comparison_data.append({
                'Metric': metric,
                'Mean_Change_%': mean_change,
                'Median_Change_%': median_change,
                'Variance_Change_%': variance_change
            })
        except KeyError as e:
            print(f"指标 {metric} 未找到: {e}")
    
    return pd.DataFrame(comparison_data)


def filter_significant_contributions(result_df: pd.DataFrame,
                                     direction: str = 'positive',
                                     lb_col: str = 'lb',
                                     ub_col: str = 'ub') -> pd.DataFrame:
    """
    筛选显著的贡献
    
    Parameters:
    -----------
    result_df : pd.DataFrame
        结果数据框
    direction : str
        方向 ('positive' 或 'negative')
    lb_col : str
        下界列名
    ub_col : str
        上界列名
        
    Returns:
    --------
    pd.DataFrame
        显著结果
    """
    if direction == 'positive':
        return result_df[(result_df[ub_col] > 0) & (result_df[lb_col] > 0)]
    elif direction == 'negative':
        return result_df[(result_df[ub_col] < 0) & (result_df[lb_col] < 0)]
    else:
        raise ValueError("direction 必须是 'positive' 或 'negative'")
