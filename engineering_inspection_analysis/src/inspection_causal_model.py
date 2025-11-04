"""
工程检测因果模型
基于 DoWhy 进行因果分析
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from functools import partial

from dowhy import gcm
from scipy import stats


class InspectionCausalModel:
    """工程检测因果模型"""
    
    def __init__(self, df: pd.DataFrame):
        """
        初始化因果模型
        
        Parameters:
        -----------
        df : pd.DataFrame
            工程检测数据（已预处理）
        """
        self.df = df
        self.causal_graph = None
        self.causal_model = None
    
    def _is_categorical_node(self, node: str) -> bool:
        """判断目标节点是否应作为分类变量建模"""
        if node not in self.df.columns:
            return False
        series = self.df[node].dropna()
        if len(series) == 0:
            return False
        # 明确的分类特征命名或目标
        categorical_name_hints = [
            '_编码', '分组_编码'
        ]
        if any(node.endswith(suffix) for suffix in categorical_name_hints):
            return True
        # 常见的二分类目标
        if node in {'是否合格_编码', '检测完成_编码'}:
            return True
        # 低基数整数型也按分类处理
        nunique = series.nunique()
        if pd.api.types.is_integer_dtype(series) and nunique <= 20:
            return True
        return False

    def _to_model_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """将数据转换为与模型机制一致的类型（分类列转字符串）。"""
        df_conv = df.copy()
        for node in self.causal_graph.nodes if self.causal_graph is not None else df_conv.columns:
            if node in df_conv.columns and self._is_categorical_node(node):
                df_conv[node] = df_conv[node].astype(str).fillna('未知')
        return df_conv
        
    def build_causal_graph(self) -> nx.DiGraph:
        """
        构建工程检测因果图
        
        因果关系：
        1. 供应链：生产厂家 → 品种、强度等级 → 质量
        2. 施工管理：施工单位 → 结构部位、代表数量 → 质量
        3. 监管体系：检验类别、监理单位 → 检测周期 → 质量
        4. 时间因素：委托时间 → 检测周期、审核周期 → 完成率
        5. 价格机制：品种、强度等级 → 价格
        
        Returns:
        --------
        nx.DiGraph
            因果图
        """
        edges = []
        
        # ===== 供应链系统 =====
        # 生产厂家 → 材料属性
        edges.append(('生产厂家_编码', '品种_编码'))
        edges.append(('生产厂家_编码', '强度等级(级别)_编码'))
        edges.append(('生产厂家_编码', '规格_数值'))
        
        # 材料属性 → 价格
        edges.append(('品种_编码', '价格'))
        edges.append(('强度等级(级别)_编码', '价格'))
        edges.append(('规格_数值', '价格'))
        
        # 材料属性 → 质量
        edges.append(('品种_编码', '是否合格_编码'))
        edges.append(('强度等级(级别)_编码', '是否合格_编码'))
        
        # 生产厂家历史质量 → 当前质量
        edges.append(('生产厂家历史合格率', '是否合格_编码'))
        
        # ===== 施工管理系统 =====
        # 施工单位 → 施工质量指标
        edges.append(('施工单位_编码', '结构部位_编码'))
        edges.append(('施工单位_编码', '代表数量_数值'))
        
        # 施工单位历史质量 → 当前质量
        edges.append(('施工单位历史合格率', '是否合格_编码'))
        
        # 结构部位 → 质量要求
        edges.append(('结构部位_编码', '强度等级(级别)_编码'))
        edges.append(('结构部位_编码', '是否合格_编码'))
        
        # 代表数量 → 抽样质量
        edges.append(('代表数量_数值', '是否合格_编码'))
        
        # ===== 监管体系 =====
        # 建设单位 → 监理单位
        edges.append(('建设单位_编码', '监理单位_编码'))
        
        # 监理单位 → 检验类别
        edges.append(('监理单位_编码', '检验类别_编码'))
        
        # 检验类别 → 检测严格程度
        edges.append(('检验类别_编码', '检验风险等级'))
        edges.append(('检验风险等级', '是否合格_编码'))
        
        # 检验类别 → 价格
        edges.append(('检验类别_编码', '价格'))
        
        # ===== 时间因素 =====
        # 委托时间 → 检测周期
        edges.append(('委托季度', '检测周期_天'))
        edges.append(('委托星期', '检测周期_天'))
        
        # 检测周期 → 审核周期
        edges.append(('检测周期_天', '审核周期_天'))
        
        # 检测周期 → 完成率
        edges.append(('检测周期_天', '检测完成_编码'))
        
        # 审核周期 → 完成率
        edges.append(('审核周期_天', '检测完成_编码'))
        
        # ===== 价格-质量关系 =====
        # 价格 → 检测资源投入 → 质量发现能力
        edges.append(('价格', '检测周期_天'))
        edges.append(('价格分组_编码', '是否合格_编码'))
        
        # 仅保留数据中存在的节点对应的边
        edges = [(u, v) for (u, v) in edges if u in self.df.columns and v in self.df.columns]
        # 创建有向图
        self.causal_graph = nx.DiGraph(edges)
        
        return self.causal_graph
    
    def fit_causal_model(self, quality: str = 'BETTER', invertible: bool = False):
        """
        拟合因果模型
        
        Parameters:
        -----------
        quality : str
            模型质量 ('BETTER' 或 'FAST')
        invertible : bool
            是否使用可逆模型（用于反事实分析）
        """
        if self.causal_graph is None:
            raise ValueError("请先调用 build_causal_graph() 构建因果图")
        
        print(f"\n正在拟合因果模型...")
        print(f"  - 节点数: {self.causal_graph.number_of_nodes()}")
        print(f"  - 边数: {self.causal_graph.number_of_edges()}")
        print(f"  - 质量模式: {quality}")
        print(f"  - 可逆模型: {invertible}")
        
        # 创建结构因果模型（反事实分析需可逆模型）
        if invertible:
            self.causal_model = gcm.InvertibleStructuralCausalModel(self.causal_graph)
            # 可逆模型：保持编码为数值，不做字符串转换
            df_fit = self.df.copy()
        else:
            self.causal_model = gcm.StructuralCausalModel(self.causal_graph)
            # 分类节点转字符串以配合 ClassifierFCM
            df_fit = self._to_model_data(self.df)
        
        # 为每个节点分配因果机制
        for node in self.causal_graph.nodes:
            if node not in self.df.columns:
                continue
                
            parents = list(self.causal_graph.predecessors(node))
            
            if len(parents) == 0:
                # 根节点：使用经验分布
                self.causal_model.set_causal_mechanism(
                    node,
                    gcm.EmpiricalDistribution()
                )
            else:
                # 非根节点：根据是否可逆选择机制
                if invertible:
                    # 可逆模型统一使用可逆的加性噪声模型（数值编码）
                    if quality == 'BETTER':
                        regressor = gcm.ml.create_hist_gradient_boost_regressor()
                    else:
                        regressor = gcm.ml.create_linear_regressor()
                    self.causal_model.set_causal_mechanism(
                        node,
                        gcm.AdditiveNoiseModel(regressor)
                    )
                else:
                    # 标准模型：分类用分类FCM，连续用加性噪声
                    if self._is_categorical_node(node):
                        if quality == 'BETTER':
                            classifier = gcm.ml.create_hist_gradient_boost_classifier()
                        else:
                            classifier = gcm.ml.create_logistic_regression_classifier()
                        self.causal_model.set_causal_mechanism(
                            node,
                            gcm.ClassifierFCM(classifier)
                        )
                    else:
                        if quality == 'BETTER':
                            regressor = gcm.ml.create_hist_gradient_boost_regressor()
                        else:
                            regressor = gcm.ml.create_linear_regressor()
                        self.causal_model.set_causal_mechanism(
                            node,
                            gcm.AdditiveNoiseModel(regressor)
                        )
        
        # 拟合模型
        gcm.fit(self.causal_model, df_fit)
        
        print("✓ 因果模型拟合完成")
    
    def attribution_analysis(
        self,
        df_old: pd.DataFrame,
        df_new: pd.DataFrame,
        target_column: str = '是否合格_编码',
        num_samples: int = 1000,
        confidence_level: float = 0.95,
        num_bootstrap_resamples: int = 10
    ) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:
        """
        归因分析：识别目标变量变化的原因
        
        Parameters:
        -----------
        df_old : pd.DataFrame
            旧时期数据
        df_new : pd.DataFrame
            新时期数据
        target_column : str
            目标变量
        num_samples : int
            采样数量
        confidence_level : float
            置信水平
        num_bootstrap_resamples : int
            Bootstrap重采样次数
            
        Returns:
        --------
        contributions : Dict[str, float]
            各变量的贡献
        uncertainties : Dict[str, Tuple[float, float]]
            置信区间
        """
        if self.causal_model is None:
            raise ValueError("请先调用 fit_causal_model() 拟合模型")
        
        print(f"\n执行归因分析...")
        print(f"  目标变量: {target_column}")
        print(f"  旧时期样本数: {len(df_old)}")
        print(f"  新时期样本数: {len(df_new)}")

        # 使用基于干预的替代方法：对每个变量将其分布从旧时期替换为新时期分布，估计对目标的影响
        df_old_model = self._to_model_data(df_old)
        df_new_model = self._to_model_data(df_new)

        baseline_mean = pd.to_numeric(df_old_model[target_column], errors='coerce').mean()

        contrib_dict: Dict[str, float] = {}
        uncertainty_dict: Dict[str, Tuple[float, float]] = {}

        # 优先评估的主要变量，避免长尾变量导致失败/卡顿
        preferred_vars = [
            '规格_数值', '价格', '检验风险等级',
            '生产厂家历史合格率', '施工单位历史合格率',
            '强度等级(级别)_编码', '品种_编码'
        ]
        candidate_vars = [v for v in preferred_vars if v in df_old_model.columns and v in self.causal_graph.nodes]
        # 如果优先变量不足，再补充其他变量
        if len(candidate_vars) < 3:
            fallback_vars = [v for v in df_old_model.columns if v != target_column and v in self.causal_graph.nodes and v not in candidate_vars]
            candidate_vars += fallback_vars[:max(0, 6 - len(candidate_vars))]

        rng = np.random.default_rng(42)

        # 采样用于主效应估计
        obs_sample = df_old_model.sample(min(len(df_old_model), min(1000, num_samples)), random_state=42)

        for var in candidate_vars:
            new_vals = df_new_model[var].values

            def draw_from_new(input_values):
                """按新时期分布重采样，保持与输入相同的形状/索引。"""
                # 处理标量
                if np.isscalar(input_values) or (isinstance(input_values, np.ndarray) and input_values.ndim == 0):
                    idx = int(rng.integers(0, len(new_vals)))
                    return new_vals[idx]
                # 处理 Series
                if isinstance(input_values, pd.Series):
                    size = len(input_values)
                    idx = rng.integers(0, len(new_vals), size=size)
                    sampled = new_vals[idx]
                    return pd.Series(sampled, index=input_values.index)
                # 处理 array-like
                size = np.size(input_values)
                idx = rng.integers(0, len(new_vals), size=size)
                sampled = new_vals[idx]
                return sampled.reshape(np.shape(input_values))

            try:
                samples = gcm.interventional_samples(
                    self.causal_model,
                    interventions={var: draw_from_new},
                    observed_data=obs_sample
                )
                effect_mean = pd.to_numeric(samples[target_column], errors='coerce').mean()
                contrib = effect_mean - baseline_mean

                # Bootstrap 置信区间
                boot_effects = []
                for _ in range(max(1, num_bootstrap_resamples)):
                    boot_obs = df_old_model.sample(min(len(df_old_model), max(100, num_samples // 5)), replace=True)
                    boot_samples = gcm.interventional_samples(
                        self.causal_model,
                        interventions={var: draw_from_new},
                        observed_data=boot_obs
                    )
                    boot_effects.append(pd.to_numeric(boot_samples[target_column], errors='coerce').mean() - pd.to_numeric(boot_obs[target_column], errors='coerce').mean())

                lower = float(np.percentile(boot_effects, (1 - confidence_level) / 2 * 100))
                upper = float(np.percentile(boot_effects, (1 + confidence_level) / 2 * 100))

                contrib_dict[var] = float(contrib)
                uncertainty_dict[var] = (lower, upper)
            except Exception as e:
                print(f"  跳过 {var}: {str(e)[:120]}")
                continue

        return contrib_dict, uncertainty_dict
    
    def intervention_analysis(
        self,
        target: str = '是否合格_编码',
        step_size: float = 1.0,
        non_interveneable_nodes: List[str] = None,
        confidence_level: float = 0.95,
        num_samples: int = 5000,
        num_bootstrap_resamples: int = 20
    ) -> pd.DataFrame:
        """
        干预分析：评估各因素对目标的因果效应
        
        Parameters:
        -----------
        target : str
            目标变量
        step_size : float
            干预步长
        non_interveneable_nodes : List[str]
            不可干预的节点
        confidence_level : float
            置信水平
        num_samples : int
            采样数量
        num_bootstrap_resamples : int
            Bootstrap重采样次数
            
        Returns:
        --------
        pd.DataFrame
            干预分析结果
        """
        if self.causal_model is None:
            raise ValueError("请先调用 fit_causal_model() 拟合模型")
        
        if non_interveneable_nodes is None:
            non_interveneable_nodes = [target]
        
        print(f"\n执行干预分析...")
        print(f"  目标变量: {target}")
        print(f"  干预步长: {step_size}")
        
        results = []

        def _to_numeric(series: pd.Series) -> pd.Series:
            """尽可能将字符串类别转为数值（用于0/1目标）。"""
            if pd.api.types.is_numeric_dtype(series):
                return series
            converted = pd.to_numeric(series, errors='coerce')
            if converted.notna().mean() > 0.9:  # 大部分可转换
                return converted
            return series
        
        # 观测数据需与模型机制一致
        df_model_obs = self._to_model_data(self.df)

        # 对每个可干预节点进行分析
        for node in self.causal_graph.nodes:
            if node not in df_model_obs.columns or node in non_interveneable_nodes:
                continue
            # 跳过分类节点（数值步长不适用）
            if self._is_categorical_node(node):
                continue
            
            try:
                # 计算干预效应
                def intervention_func(x):
                    return x + step_size
                
                effect_samples = gcm.interventional_samples(
                    self.causal_model,
                    interventions={node: intervention_func},
                    observed_data=df_model_obs.sample(min(len(df_model_obs), num_samples), random_state=42)
                )
                
                # 计算平均因果效应
                baseline_mean = _to_numeric(df_model_obs[target]).mean()
                intervention_mean = _to_numeric(effect_samples[target]).mean()
                causal_effect = intervention_mean - baseline_mean
                
                # Bootstrap 估计置信区间
                bootstrap_effects = []
                for _ in range(num_bootstrap_resamples):
                    sample_data = df_model_obs.sample(min(len(df_model_obs), num_samples // 5), replace=True)
                    effect_sample = gcm.interventional_samples(
                        self.causal_model,
                        interventions={node: intervention_func},
                        observed_data=sample_data
                    )
                    boot_effect = _to_numeric(effect_sample[target]).mean() - _to_numeric(sample_data[target]).mean()
                    bootstrap_effects.append(boot_effect)
                
                # 计算置信区间
                lower = np.percentile(bootstrap_effects, (1 - confidence_level) / 2 * 100)
                upper = np.percentile(bootstrap_effects, (1 + confidence_level) / 2 * 100)
                
                results.append({
                    'Variable': node,
                    'Causal_Effect': causal_effect,
                    'Lower_CI': lower,
                    'Upper_CI': upper,
                    'Std_Error': np.std(bootstrap_effects)
                })
                
                print(f"  ✓ {node}: {causal_effect:.6f}")
                
            except Exception as e:
                print(f"  ✗ {node}: 跳过 ({str(e)[:50]})")
                continue
        
        # 转换为DataFrame并排序
        results_df = pd.DataFrame(results)
        if len(results_df) > 0:
            results_df = results_df.sort_values('Causal_Effect', key=abs, ascending=False)
        
        return results_df
    
    def counterfactual_analysis(
        self,
        sample_index: int,
        interventions: Dict[str, float],
        target: str = '是否合格_编码'
    ) -> Dict[str, float]:
        """
        反事实分析：探索"如果...会怎样"的问题
        
        Parameters:
        -----------
        sample_index : int
            样本索引
        interventions : Dict[str, float]
            干预值 {变量名: 新值}
        target : str
            目标变量
            
        Returns:
        --------
        Dict[str, float]
            反事实结果
        """
        if self.causal_model is None:
            raise ValueError("请先调用 fit_causal_model(invertible=True) 拟合可逆模型")
        
        # 可逆模型下使用原始数值编码；非可逆模型使用字符串化数据
        is_invertible_model = isinstance(self.causal_model, gcm.InvertibleStructuralCausalModel)
        if is_invertible_model:
            if sample_index in self.df.index:
                observed_sample = self.df.loc[[sample_index]]
            else:
                observed_sample = self.df.iloc[[sample_index]]
            typed_interventions = dict(interventions)
        else:
            model_df = self._to_model_data(self.df)
            if sample_index in model_df.index:
                observed_sample = model_df.loc[[sample_index]]
            else:
                observed_sample = model_df.iloc[[sample_index]]
            typed_interventions: Dict[str, object] = {}
            for k, v in interventions.items():
                if self._is_categorical_node(k):
                    typed_interventions[k] = str(v)
                else:
                    typed_interventions[k] = v
        
        # 执行反事实推理
        # 将常量干预包装为可调用函数，兼容 gcm API
        interventions_funcs: Dict[str, object] = {}
        for k, v in typed_interventions.items():
            if callable(v):
                interventions_funcs[k] = v
            else:
                def _make_func(val):
                    def _f(x):
                        # 保持与输入相同的形状与索引
                        if isinstance(x, pd.Series):
                            return pd.Series(np.full(len(x), val), index=x.index)
                        arr = np.asarray(x)
                        return np.full(arr.shape, val)
                    return _f
                interventions_funcs[k] = _make_func(v)

        counterfactual_samples = gcm.counterfactual_samples(
            self.causal_model,
            interventions_funcs,
            observed_data=observed_sample
        )
        
        # 计算变化（转换为数值以避免字符串相减）
        obs_val = pd.to_numeric(observed_sample[target], errors='coerce').values[0]
        cf_val = pd.to_numeric(counterfactual_samples[target], errors='coerce').values[0]
        result = {
            'observed': float(obs_val) if obs_val is not None else None,
            'counterfactual': float(cf_val) if cf_val is not None else None,
            'change': float(cf_val - obs_val) if pd.notnull(cf_val) and pd.notnull(obs_val) else None
        }
        
        return result
    
    def get_graph_summary(self) -> Dict:
        """获取因果图摘要"""
        if self.causal_graph is None:
            return {}
        
        return {
            'num_nodes': self.causal_graph.number_of_nodes(),
            'num_edges': self.causal_graph.number_of_edges(),
            'nodes': list(self.causal_graph.nodes()),
            'root_nodes': [n for n in self.causal_graph.nodes() 
                          if self.causal_graph.in_degree(n) == 0],
            'leaf_nodes': [n for n in self.causal_graph.nodes() 
                          if self.causal_graph.out_degree(n) == 0]
        }


def compare_metrics(baseline: float, intervention: float, label: str = "指标"):
    """比较基线和干预结果"""
    change = intervention - baseline
    pct_change = (change / baseline * 100) if baseline != 0 else 0
    
    print(f"\n{label}:")
    print(f"  基线值: {baseline:.4f}")
    print(f"  干预后: {intervention:.4f}")
    print(f"  绝对变化: {change:+.4f}")
    print(f"  相对变化: {pct_change:+.2f}%")

