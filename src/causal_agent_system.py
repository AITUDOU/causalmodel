"""
因果驱动的智能体系统 - 基于 LangGraph
三智能体架构：Router Agent → Causal Analyst Agent → Advisor Agent

应用场景：混凝土集料质量控制与工艺优化
"""

import os
import pickle
from pathlib import Path
from typing import TypedDict, Literal, Dict, Any, Optional
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 导入混凝土集料因果模型
from causal_model import ConcreteAggregateCausalModel


# ============================================================================
# 全局配置
# ============================================================================

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_API_BASE = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')


# ============================================================================
# 第一步：定义 State
# ============================================================================

class CausalAnalysisState(TypedDict):
    """因果分析智能体系统的共享状态"""
    # 用户输入
    user_query: str                    # 原始查询
    reference_sample_index: int        # 参考批次索引（用于反事实分析，可选）
    observed_config: dict              # 用户输入的观测配比（用于反事实分析，可选，优先于reference_sample_index）
    
    # Router 输出
    analysis_type: str                 # 'attribution' | 'intervention' | 'counterfactual'
    target_variable: str               # 目标变量
    intervention_params: dict          # 干预参数
    routing_reasoning: str             # 路由推理过程
    target_improvement: float          # 目标提升幅度（百分比，如10表示提升10%）
    specified_variables: list          # 用户指定要调整的变量列表
    target_value: float                # 用户指定的目标值（如"强度达到45"中的45）
    
    # Causal Analyst 输出
    causal_results: dict               # 因果分析数值结果
    analysis_summary: str              # 分析摘要
    
    # Optimizer 输出（新增）
    optimized_config: dict             # 优化后的配比建议
    predicted_strength: float          # 预测的强度
    optimization_summary: str          # 优化摘要
    base_sample_info: dict             # 基准样本信息（当使用默认样本时）
    
    # Advisor 输出
    recommendations: str               # 决策建议
    
    # 系统元数据
    error: Optional[str]               # 错误信息


# ============================================================================
# 第二步：定义工具函数 - 包装现有因果模型
# ============================================================================

# 全局因果模型实例
_causal_model_instance: Optional[ConcreteAggregateCausalModel] = None

# 模型缓存路径
MODEL_CACHE_FILE = Path("models/causal_model.pkl")


def build_sample_info_dict(sample: pd.Series, source: str = "默认") -> dict:
    """
    构建样本信息字典
    
    Args:
        sample: pandas Series，包含样本的所有字段
        source: 样本来源说明（如"默认"、"用户选择"等）
    
    Returns:
        包含样本完整信息的字典
    """
    return {
        "source": source,
        "cement": float(sample['cement']),
        "blast_furnace_slag": float(sample['blast_furnace_slag']),
        "fly_ash": float(sample['fly_ash']),
        "water": float(sample['water']),
        "superplasticizer": float(sample['superplasticizer']),
        "coarse_aggregate": float(sample['coarse_aggregate']),
        "fine_aggregate": float(sample['fine_aggregate']),
        "age": int(sample['age']),
        "concrete_compressive_strength": float(sample['concrete_compressive_strength'])
    }


def initialize_causal_model(df: pd.DataFrame = None, force_retrain: bool = False) -> ConcreteAggregateCausalModel:
    """
    初始化混凝土集料因果模型（支持缓存加载）
    
    Args:
        df: 混凝土集料数据（如果从缓存加载则可以为 None）
        force_retrain: 是否强制重新训练（默认 False，优先使用缓存）
        
    Returns:
        初始化好的因果模型
    """
    global _causal_model_instance
    
    # 优先尝试从缓存加载
    if not force_retrain and MODEL_CACHE_FILE.exists():
        print("📦 从缓存加载因果模型...")
        try:
            with open(MODEL_CACHE_FILE, 'rb') as f:
                model = pickle.load(f)
            _causal_model_instance = model
            print(f"✓ 模型加载完成 (缓存文件: {MODEL_CACHE_FILE})")
            print(f"  • 节点数: {model.causal_graph.number_of_nodes()}")
            print(f"  • 边数: {model.causal_graph.number_of_edges()}")
            return model
        except Exception as e:
            print(f"⚠️  缓存加载失败: {e}")
            print("   将重新训练模型...")
    
    # 如果缓存不存在或加载失败，重新训练
    if df is None:
        raise ValueError(
            "未找到缓存模型且未提供数据。请先运行 train_causal_model.py 训练模型，"
            "或者在调用时提供数据。"
        )
    
    print("🔧 训练新的因果模型（首次运行需要1-2分钟）...")
    model = ConcreteAggregateCausalModel(df)
    model.build_causal_graph()
    model.fit_causal_model(quality='BETTER', invertible=True)
    
    # 保存到缓存
    MODEL_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_CACHE_FILE, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ 模型训练完成并保存至: {MODEL_CACHE_FILE}")
    
    _causal_model_instance = model
    return model


@tool
def attribution_analysis_tool(
    target_variable: str,
    old_period_start: int,
    old_period_end: int,
    new_period_start: int,
    new_period_end: int
) -> dict:
    """
    执行归因分析，识别目标变量变化的根本原因。
    
    用于回答"为什么含泥量上升了？"、"是什么导致强度下降？"等问题。
    对比两个时间段的数据，找出哪些因素对目标变量的变化贡献最大。
    
    Args:
        target_variable: 目标变量（如 'mud_content_pct', 'concrete_strength_mpa'）
        old_period_start: 旧时期起始索引
        old_period_end: 旧时期结束索引
        new_period_start: 新时期起始索引
        new_period_end: 新时期结束索引
        
    Returns:
        dict: 包含各因素贡献度的分析结果
    """
    if _causal_model_instance is None:
        return {"error": "因果模型未初始化"}
    
    try:
        df = _causal_model_instance.df
        df_old = df.iloc[old_period_start:old_period_end]
        df_new = df.iloc[new_period_start:new_period_end]
        
        contributions, uncertainties = _causal_model_instance.attribution_analysis(
            df_old=df_old,
            df_new=df_new,
            target_column=target_variable,
            num_samples=2000,
            num_bootstrap_resamples=4
        )
        
        # 按贡献度排序
        sorted_contributions = sorted(
            contributions.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        # 将 numpy 数组转换为 Python 原生类型
        def convert_to_native(val):
            """转换 numpy 类型为 Python 原生类型"""
            if val is None:
                return None
            if isinstance(val, np.ndarray):
                return val.tolist()
            if isinstance(val, (np.floating, np.integer)):
                return float(val)
            if isinstance(val, tuple):
                return tuple(convert_to_native(v) for v in val)
            return val
        
        result = {
            "type": "attribution",
            "target": target_variable,
            "top_factors": [
                {
                    "variable": var,
                    "contribution": float(contrib),
                    "confidence_interval": convert_to_native(uncertainties.get(var, (None, None)))
                }
                for var, contrib in sorted_contributions[:5]
            ],
            "old_period_size": len(df_old),
            "new_period_size": len(df_new)
        }
        
        return result
        
    except Exception as e:
        return {"error": str(e)}


@tool
def intervention_analysis_tool(
    target_variable: str,
    step_size: float = 1.0
) -> dict:
    """
    执行干预分析，评估各因素对目标变量的因果效应。
    
    用于回答"如何降低含泥量？"、"哪些工艺参数最有效？"等问题。
    计算每个可控变量增加一个单位后，对目标变量的影响程度。
    
    Args:
        target_variable: 目标变量（如 'mud_content_pct', 'concrete_strength_mpa'）
        step_size: 干预步长（默认为1.0）
        
    Returns:
        dict: 包含各变量因果效应的分析结果
    """
    if _causal_model_instance is None:
        return {"error": "因果模型未初始化"}
    
    try:
        results_df = _causal_model_instance.intervention_analysis(
            target=target_variable,
            step_size=step_size,
            num_samples=10000,
            num_bootstrap_resamples=40
        )
        
        # 转换为字典格式
        interventions = []
        for _, row in results_df.iterrows():
            interventions.append({
                "variable": row['Variable'],
                "causal_effect": float(row['Causal_Effect']),
                "confidence_interval": (float(row['Lower_CI']), float(row['Upper_CI'])),
                "std_error": float(row.get('Std_Error', 0))
            })
        
        result = {
            "type": "intervention",
            "target": target_variable,
            "step_size": step_size,
            "interventions": interventions
        }
        
        return result
        
    except Exception as e:
        return {"error": str(e)}


@tool
def math_calculator_tool(
    variable: str,
    base_value: float,
    operation: str,
    operand: float
) -> float:
    """
    数学计算工具，处理变量的加减乘除运算
    
    Args:
        variable: 变量名称
        base_value: 基准值
        operation: 运算类型 ('add'/'subtract'/'multiply'/'divide')
        operand: 操作数
        
    Returns:
        float: 计算结果
    """
    if operation == 'add':
        result = base_value + operand
    elif operation == 'subtract':
        result = base_value - operand
    elif operation == 'multiply':
        result = base_value * operand
    elif operation == 'divide':
        if operand == 0:
            raise ValueError(f"除数不能为0")
        result = base_value / operand
    else:
        raise ValueError(f"不支持的运算类型: {operation}")
    
    print(f"  🧮 计算: {variable} = {base_value} {operation} {operand} = {result}")
    return result


@tool
def counterfactual_analysis_tool(
    sample_index: int,
    interventions: dict,
    target_variable: str
) -> dict:
    """
    执行反事实分析，预测"如果改变某些变量，结果会如何"。
    
    用于回答"如果改变水胶比会怎样？"、"换个产地能达标吗？"等问题。
    针对具体的历史样本，模拟改变一个或多个变量后的结果。
    
    Args:
        sample_index: 样本索引（要分析的历史记录）
        interventions: 干预变量及其新值的字典，如 {"water_binder_ratio": 0.43, "cement_content": 379}
        target_variable: 目标变量名称
        
    Returns:
        dict: 包含实际值、反事实值和变化的结果
    """
    if _causal_model_instance is None:
        return {"error": "因果模型未初始化"}
    
    try:
        df = _causal_model_instance.df
        
        # 验证索引是否在范围内
        if sample_index < 0 or sample_index >= len(df):
            return {"error": f"样本索引 {sample_index} 超出范围 [0, {len(df)-1}]"}
        
        observed_data = df.iloc[[sample_index]]
        
        # 转换干预值为float
        interventions_float = {k: float(v) for k, v in interventions.items()}
        
        result_dict = _causal_model_instance.counterfactual_analysis(
            observed_data=observed_data,
            interventions=interventions_float,
            target=target_variable,
            num_samples=1000
        )
        
        # 转换为原生 Python 类型
        def to_float(val):
            """安全地转换为 float"""
            if val is None:
                return None
            if isinstance(val, (np.floating, np.integer, np.ndarray)):
                return float(val)
            return float(val)
        
        # 构建干预信息列表
        intervention_list = []
        for var, new_val in interventions_float.items():
            original_val = to_float(observed_data[var].values[0]) if var in observed_data.columns else None
            intervention_list.append({
                "variable": var,
                "original_value": original_val,
                "new_value": float(new_val)
            })
        
        result = {
            "type": "counterfactual",
            "sample_index": sample_index,
            "target": target_variable,
            "interventions": intervention_list,  # 现在是列表，支持多个干预
            "observed_mean": to_float(result_dict['observed_mean']),
            "counterfactual_mean": to_float(result_dict['counterfactual_mean']),
            "causal_effect": to_float(result_dict['causal_effect'])
        }
        
        return result
        
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# 第三步：定义三个智能体节点
# ============================================================================

def router_agent(state: CausalAnalysisState) -> dict:
    """
    Router Agent：理解用户查询，识别分析类型和关键参数
    
    职责：
    1. 理解自然语言查询的意图
    2. 识别查询类型（归因/干预/反事实）
    3. 提取关键信息（目标变量、干预参数等）
    
    注意：如果 state 中已经有 specified_variables 或 target_value，优先使用它们
    """
    print("\n🔍 Router Agent 正在分析您的问题...")
    
    # 检查是否已经从API传入了参数
    api_specified_variables = state.get('specified_variables')
    api_target_value = state.get('target_value')
    
    if api_specified_variables:
        print(f"  📌 检测到API传入的调整变量: {', '.join(api_specified_variables)}")
    if api_target_value:
        print(f"  📌 检测到API传入的目标值: {api_target_value}")
    
    # 使用 LLM 理解用户查询
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0.1,
        openai_api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE
    )
    
    prompt = f"""你是一个因果分析系统的路由专家。请分析用户的查询，确定应该执行哪种因果分析。

【应用场景】高性能混凝土配合比设计与强度优化（基于UCI真实数据集，Yeh 1998）

【分析类型】
1. **attribution**（归因分析）- 用于回答"为什么XX变化了？"、"原因是什么？"
   - 对比两个时期的数据，识别导致目标变量变化的根本原因

2. **intervention**（干预分析）- 用于回答"如何改进XX？"、"哪个因素最有效？"
   - 评估不同措施的效果，找出最有影响力的可控变量

3. **counterfactual**（反事实分析）- 用于回答"如果...会怎样？"
   - 针对具体案例模拟假设场景，预测改变某个变量后的结果
   - **必须从用户问题中提取**：干预变量名、原始值、新值

【因果图可用变量】（基于UCI真实数据集，仅9个原始变量）

1. cement: 水泥 (102-540 kg/m³, 均值281) **【关键材料】**
2. blast_furnace_slag: 高炉矿渣 (0-359 kg/m³, 均值74) - 提高密实度和耐久性
3. fly_ash: 粉煤灰 (0-200 kg/m³, 均值54) - 火山灰反应，长期强度
4. water: 水 (122-247 kg/m³, 均值182) **【Abrams定律：水越多强度越低】**
5. superplasticizer: 高效减水剂 (0-32 kg/m³, 均值6.2) - 与水负相关（r=-0.66）
6. coarse_aggregate: 粗骨料 (801-1145 kg/m³, 均值973) - 骨架作用
7. fine_aggregate: 细骨料 (594-993 kg/m³, 均值774) - 填充作用
8. age: 龄期 (1-365天, 均值46天) **【时间效应】**
9. concrete_compressive_strength: 抗压强度 (2.3-82.6 MPa, 均值35.8) **【目标变量】**

用户查询："{state['user_query']}"

【变量识别规则】
- 用户提到"强度"/"抗压强度"/"混凝土强度" → concrete_compressive_strength
- 用户提到"水泥"/"水泥用量" → cement
- 用户提到"矿渣"/"高炉矿渣"/"矿粉" → blast_furnace_slag  
- 用户提到"粉煤灰" → fly_ash
- 用户提到"水"/"用水量"/"拌合水" → water
- 用户提到"减水剂"/"外加剂"/"高效减水剂"/"超塑化剂" → superplasticizer
- 用户提到"粗骨料"/"石子"/"碎石" → coarse_aggregate
- 用户提到"细骨料"/"砂"/"河砂" → fine_aggregate
- 用户提到"龄期"/"养护时间"/"天数"/"龄龄" → age

注意：本模型使用真实UCI数据集的9个原始变量，未使用衍生变量

请以JSON格式回复：
{{
    "analysis_type": "attribution/intervention/counterfactual",
    "target_variable": "从上述变量列表中选择（必须是准确的变量名）",
    "reasoning": "你的推理过程（1-2句话）",
    "target_improvement": 目标提升百分比（如用户说"提升10%"则为10，如果没有明确提及则为null）,
    "target_value": 用户指定的目标值（如"强度达到45"则为45，如果没有明确提及则为null）,
    "specified_variables": ["用户明确要求调整的变量列表，如cement、fly_ash，如果没有则为空列表"],
    "extracted_info": {{
        // 如果是反事实分析：
        // 情况A - 绝对值干预（明确给出新值）：
        "intervention_variable": "变量名",
        "original_value": 原始数值,
        "intervention_value": 新数值
        
        // 情况B - 单变量数学运算：
        "intervention_variable": "变量名",
        "operation": "add/subtract/multiply/divide",
        "operand": 数值
        
        // 情况C - 多变量数学运算：
        "interventions": [
            {{"variable": "cement", "operation": "subtract", "operand": 50}},
            {{"variable": "blast_furnace_slag", "operation": "add", "operand": 100}}
        ]
        
        // 情况D - 多变量绝对值：
        "intervention_variable": {{"water": 150, "cement": 300}}
    }}
}}

示例1（绝对值）：
用户问："如果水用量从200降到150，强度会怎样？"
回复：{{"intervention_variable": "water", "original_value": 200, "intervention_value": 150}}

示例2（单变量运算）：
用户问："如果水泥增加50 kg/m³，强度会怎样？"
回复：{{"intervention_variable": "cement", "operation": "add", "operand": 50}}

示例3（多变量运算）：
用户问："添加矿渣100 kg/m³，减少水泥50 kg/m³，强度会怎样？"
回复：{{"interventions": [{{"variable": "blast_furnace_slag", "operation": "add", "operand": 100}}, {{"variable": "cement", "operation": "subtract", "operand": 50}}]}}

示例4（多变量绝对值）：
用户问："如果水泥300、水180、龄期28天，强度是多少？"
回复：{{"intervention_variable": {{"cement": 300, "water": 180, "age": 28}}}}

示例5（目标导向 - 百分比）：
用户问："如果我想强度提升10%，应该如何调整配合比？"
回复：{{"analysis_type": "intervention", "target_improvement": 10, "specified_variables": []}}

示例6（目标导向 - 绝对值）：
用户问："现在我想强度达到45，水泥和粉煤灰应该怎么调？"
回复：{{"analysis_type": "intervention", "target_value": 45, "specified_variables": ["cement", "fly_ash"]}}

示例7（目标导向 - 指定变量）：
用户问："如何通过调整水和减水剂使强度达到50 MPa？"
回复：{{"analysis_type": "intervention", "target_value": 50, "specified_variables": ["water", "superplasticizer"]}}

【关键】运算类型映射：
- "增加"/"添加"/"加" → "add"
- "减少"/"降低"/"减" → "subtract"
- "乘以"/"翻倍" → "multiply"
- "除以"/"减半" → "divide"
"""
    
    response = llm.invoke(prompt)
    
    # 解析 LLM 响应
    import json
    try:
        # 提取JSON内容
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        parsed = json.loads(content)
        
        print(f"\n📋 分析类型: {parsed['analysis_type']}")
        print(f"🎯 目标变量: {parsed['target_variable']}")
        print(f"💡 推理: {parsed['reasoning']}")
        
        # 优先使用API传入的参数，否则使用LLM解析的结果
        final_target_value = api_target_value if api_target_value is not None else parsed.get('target_value')
        final_specified_variables = api_specified_variables if api_specified_variables else parsed.get('specified_variables', [])
        
        if parsed.get('target_improvement'):
            print(f"🎯 目标提升: {parsed['target_improvement']}%")
        if final_target_value:
            print(f"🎯 目标值: {final_target_value}")
        if final_specified_variables:
            print(f"🔧 指定调整变量: {', '.join(final_specified_variables)}")
        
        return {
            "analysis_type": parsed['analysis_type'],
            "target_variable": parsed['target_variable'],
            "routing_reasoning": parsed['reasoning'],
            "intervention_params": parsed.get('extracted_info', {}),
            "target_improvement": parsed.get('target_improvement'),
            "target_value": final_target_value,
            "specified_variables": final_specified_variables
        }
        
    except Exception as e:
        print(f"⚠️ 路由解析失败: {e}")
        # 默认使用干预分析，并使用API传入的参数（如果有）
        return {
            "analysis_type": "intervention",
            "target_variable": "concrete_compressive_strength",
            "routing_reasoning": "使用默认配置",
            "intervention_params": {},
            "target_value": api_target_value,
            "specified_variables": api_specified_variables if api_specified_variables else []
        }


def causal_analyst_agent(state: CausalAnalysisState) -> dict:
    """
    Causal Analyst Agent：执行因果分析
    
    职责：
    1. 根据 Router 的指示调用对应的因果分析工具
    2. 执行计算并返回定量结果
    """
    print("\n📊 Causal Analyst Agent 正在执行因果分析...")
    
    analysis_type = state['analysis_type']
    target_variable = state['target_variable']
    
    try:
        if analysis_type == 'attribution':
            print("执行归因分析...")
            # 默认对比前300条和后300条数据（真实数据共1030条）
            result = attribution_analysis_tool.invoke({
                "target_variable": target_variable,
                "old_period_start": 0,
                "old_period_end": 300,
                "new_period_start": 700,
                "new_period_end": 1000
            })
            
        elif analysis_type == 'intervention':
            print("执行干预分析...")
            result = intervention_analysis_tool.invoke({
                "target_variable": target_variable,
                "step_size": 1.0
            })
            
        elif analysis_type == 'counterfactual':
            print("执行反事实分析...")
            params = state.get('intervention_params', {})
            
            # 从 Router 提取的参数中获取干预信息
            intervention_variable = params.get('intervention_variable')
            intervention_value = params.get('intervention_value')
            original_value = params.get('original_value')
            operation = params.get('operation')  # 新增：数学运算
            operand = params.get('operand')  # 新增：操作数
            interventions_list = params.get('interventions')  # 新增：多变量运算列表
            
            # 获取基准配比
            def get_base_config():
                if state.get('observed_config') is not None:
                    return state['observed_config']
                elif state.get('reference_sample_index') is not None:
                    idx = state['reference_sample_index']
                    df = _causal_model_instance.df
                    # 验证索引是否在范围内
                    if idx < 0 or idx >= len(df):
                        print(f"  ⚠️  参考索引 {idx} 超出范围 [0, {len(df)-1}]，使用默认中位数样本")
                        median_idx = (df['concrete_compressive_strength'] - df['concrete_compressive_strength'].median()).abs().idxmin()
                        return df.iloc[median_idx].to_dict()
                    return df.iloc[idx].to_dict()
                else:
                    # 使用数据集中位数样本
                    df = _causal_model_instance.df
                    median_idx = (df['concrete_compressive_strength'] - df['concrete_compressive_strength'].median()).abs().idxmin()
                    return df.iloc[median_idx].to_dict()
            
            base_config = get_base_config()
            
            # 构建干预字典
            interventions = {}
            
            # 情况1：多变量数学运算（最常见的复杂情况）
            if interventions_list is not None and isinstance(interventions_list, list):
                print(f"\n  🧮 使用数学计算工具处理多变量运算:")
                for item in interventions_list:
                    var = item.get('variable')
                    op = item.get('operation')
                    val = item.get('operand')
                    
                    if var and op and val is not None:
                        base_val = float(base_config.get(var, 0))
                        # 使用数学计算工具
                        new_val = math_calculator_tool.invoke({
                            'variable': var,
                            'base_value': base_val,
                            'operation': op,
                            'operand': val
                        })
                        interventions[var] = new_val
            
            # 情况2：单变量数学运算
            elif isinstance(intervention_variable, str) and operation is not None and operand is not None:
                print(f"\n  🧮 使用数学计算工具处理单变量运算:")
                base_value = float(base_config.get(intervention_variable, 0))
                new_value = math_calculator_tool.invoke({
                    'variable': intervention_variable,
                    'base_value': base_value,
                    'operation': operation,
                    'operand': operand
                })
                interventions[intervention_variable] = new_value
                original_value = base_value
            
            # 情况3：单个变量的绝对值干预
            elif isinstance(intervention_variable, str) and intervention_value is not None:
                interventions[intervention_variable] = intervention_value
                print(f"  📊 绝对值干预: {intervention_variable} = {intervention_value}")
            
            # 情况4：多变量绝对值干预（字典形式）
            elif isinstance(intervention_variable, dict):
                interventions = intervention_variable
                print(f"  📊 多变量绝对值干预: {interventions}")
            elif isinstance(intervention_value, dict):
                interventions = intervention_value
                print(f"  📊 多变量绝对值干预: {interventions}")
            
            # 情况5：没有提取到干预信息，使用默认
            if not interventions:
                interventions = {'water': 150}  # 默认降低用水量
                print(f"  ⚠️  未提取到干预信息，使用默认干预: {interventions}")
            
            # 选择观测数据的来源
            # 优先级: observed_config > reference_sample_index > 自动匹配 > 默认样本
            
            if state.get('observed_config') is not None:
                # 用户直接输入了观测配比
                print(f"  ✅ 使用用户输入的观测配比")
                
                # 如果observed_config中缺少concrete_compressive_strength，先用因果模型预测
                observed_config_full = state['observed_config'].copy()
                if 'concrete_compressive_strength' not in observed_config_full:
                    print(f"  🔮 预测基准强度...")
                    from dowhy import gcm
                    
                    # 使用因果模型预测基准强度
                    intervention_funcs = {
                        'cement': lambda x: observed_config_full.get('cement', 280),
                        'blast_furnace_slag': lambda x: observed_config_full.get('blast_furnace_slag', 0),
                        'fly_ash': lambda x: observed_config_full.get('fly_ash', 0),
                        'water': lambda x: observed_config_full.get('water', 180),
                        'superplasticizer': lambda x: observed_config_full.get('superplasticizer', 0),
                        'coarse_aggregate': lambda x: observed_config_full.get('coarse_aggregate', 1000),
                        'fine_aggregate': lambda x: observed_config_full.get('fine_aggregate', 800),
                        'age': lambda x: observed_config_full.get('age', 28)
                    }
                    samples = gcm.interventional_samples(
                        _causal_model_instance.causal_model,
                        intervention_funcs,
                        num_samples_to_draw=100
                    )
                    predicted_strength = float(samples['concrete_compressive_strength'].mean())
                    observed_config_full['concrete_compressive_strength'] = predicted_strength
                    print(f"  ✓ 基准强度: {predicted_strength:.2f} MPa")
                
                observed_data_df = pd.DataFrame([observed_config_full])
                
                # 直接使用counterfactual_analysis
                result_dict = _causal_model_instance.counterfactual_analysis(
                    observed_data=observed_data_df,
                    interventions=interventions,
                    target=target_variable,
                    num_samples=1000
                )
                
                # 构建结果
                result = {
                    "type": "counterfactual",
                    "sample_index": "用户输入",
                    "target": target_variable,
                    "interventions": [{
                        "variable": var,
                        "original_value": float(observed_config_full.get(var, 0)),
                        "new_value": float(new_val)
                    } for var, new_val in interventions.items()],
                    "observed_mean": float(result_dict['observed_mean']),
                    "counterfactual_mean": float(result_dict['counterfactual_mean']),
                    "causal_effect": float(result_dict['causal_effect'])
                }
                
            else:
                # 使用样本索引
                df = _causal_model_instance.df
                if state.get('reference_sample_index') is not None:
                    sample_index = state['reference_sample_index']
                    # 验证索引是否在范围内
                    if sample_index < 0 or sample_index >= len(df):
                        print(f"  ⚠️  参考索引 {sample_index} 超出范围 [0, {len(df)-1}]，使用默认样本")
                        sample_index = min(100, len(df) - 1)  # 使用默认样本，确保不超出范围
                        
                        # 显示默认样本的完整信息
                        sample = df.iloc[sample_index]
                        print(f"\n  📋 默认基准样本详情:")
                        print(f"    • 水泥: {sample['cement']:.1f} kg/m³")
                        print(f"    • 高炉矿渣: {sample['blast_furnace_slag']:.1f} kg/m³")
                        print(f"    • 粉煤灰: {sample['fly_ash']:.1f} kg/m³")
                        print(f"    • 水: {sample['water']:.1f} kg/m³")
                        print(f"    • 减水剂: {sample['superplasticizer']:.1f} kg/m³")
                        print(f"    • 粗骨料: {sample['coarse_aggregate']:.1f} kg/m³")
                        print(f"    • 细骨料: {sample['fine_aggregate']:.1f} kg/m³")
                        print(f"    • 龄期: {sample['age']:.0f} 天")
                        print(f"    • 原始强度: {sample['concrete_compressive_strength']:.2f} MPa")
                        
                        # 保存基准样本信息到state
                        state['base_sample_info'] = build_sample_info_dict(sample, source="默认样本（索引{}）".format(sample_index))
                    else:
                        print(f"  使用用户选择的参考批次: 索引 {sample_index}")
                        
                        # 显示参考样本的完整信息
                        sample = df.iloc[sample_index]
                        print(f"\n  📋 参考样本详情:")
                        print(f"    • 水泥: {sample['cement']:.1f} kg/m³")
                        print(f"    • 高炉矿渣: {sample['blast_furnace_slag']:.1f} kg/m³")
                        print(f"    • 粉煤灰: {sample['fly_ash']:.1f} kg/m³")
                        print(f"    • 水: {sample['water']:.1f} kg/m³")
                        print(f"    • 减水剂: {sample['superplasticizer']:.1f} kg/m³")
                        print(f"    • 粗骨料: {sample['coarse_aggregate']:.1f} kg/m³")
                        print(f"    • 细骨料: {sample['fine_aggregate']:.1f} kg/m³")
                        print(f"    • 龄期: {sample['age']:.0f} 天")
                        print(f"    • 原始强度: {sample['concrete_compressive_strength']:.2f} MPa")
                        
                        # 保存基准样本信息到state
                        state['base_sample_info'] = build_sample_info_dict(sample, source="用户指定样本（索引{}）".format(sample_index))
                # 如果提取到了原始值，尝试找到接近该值的样本
                elif original_value is not None and _causal_model_instance is not None:
                    # 使用第一个干预变量找到最接近的样本
                    first_var = list(interventions.keys())[0]
                    if first_var in df.columns:
                        closest_idx = (df[first_var] - original_value).abs().idxmin()
                        sample_index = int(closest_idx)
                        print(f"  找到最接近原始值 {original_value} 的样本: 索引 {sample_index}")
                        
                        # 显示找到的样本的完整信息
                        sample = df.iloc[sample_index]
                        print(f"\n  📋 基准样本详情:")
                        print(f"    • 水泥: {sample['cement']:.1f} kg/m³")
                        print(f"    • 高炉矿渣: {sample['blast_furnace_slag']:.1f} kg/m³")
                        print(f"    • 粉煤灰: {sample['fly_ash']:.1f} kg/m³")
                        print(f"    • 水: {sample['water']:.1f} kg/m³")
                        print(f"    • 减水剂: {sample['superplasticizer']:.1f} kg/m³")
                        print(f"    • 粗骨料: {sample['coarse_aggregate']:.1f} kg/m³")
                        print(f"    • 细骨料: {sample['fine_aggregate']:.1f} kg/m³")
                        print(f"    • 龄期: {sample['age']:.0f} 天")
                        print(f"    • 原始强度: {sample['concrete_compressive_strength']:.2f} MPa")
                        
                        # 保存基准样本信息到state
                        state['base_sample_info'] = build_sample_info_dict(sample, source="自动匹配样本（索引{}）".format(sample_index))
                else:
                    sample_index = min(100, len(df) - 1)  # 默认样本，确保不超出范围
                    print(f"  使用默认样本索引: {sample_index}")
                    
                    # 获取并显示默认样本的完整信息
                    sample = df.iloc[sample_index]
                    print(f"\n  📋 默认基准样本详情:")
                    print(f"    • 水泥: {sample['cement']:.1f} kg/m³")
                    print(f"    • 高炉矿渣: {sample['blast_furnace_slag']:.1f} kg/m³")
                    print(f"    • 粉煤灰: {sample['fly_ash']:.1f} kg/m³")
                    print(f"    • 水: {sample['water']:.1f} kg/m³")
                    print(f"    • 减水剂: {sample['superplasticizer']:.1f} kg/m³")
                    print(f"    • 粗骨料: {sample['coarse_aggregate']:.1f} kg/m³")
                    print(f"    • 细骨料: {sample['fine_aggregate']:.1f} kg/m³")
                    print(f"    • 龄期: {sample['age']:.0f} 天")
                    print(f"    • 原始强度: {sample['concrete_compressive_strength']:.2f} MPa")
                    
                    # 保存基准样本信息到state
                    state['base_sample_info'] = build_sample_info_dict(sample, source="默认样本（索引{})".format(sample_index))
                
                print(f"\n  🔧 干预变量: {', '.join(interventions.keys())}")
                print(f"  🎯 干预值: {interventions}")
                
                result = counterfactual_analysis_tool.invoke({
                    "sample_index": sample_index,
                    "interventions": interventions,
                    "target_variable": target_variable
                })
            
        else:
            result = {"error": f"未知的分析类型: {analysis_type}"}
        
        if "error" in result:
            print(f"❌ 分析失败: {result['error']}")
            return {
                "causal_results": result,
                "analysis_summary": f"分析失败: {result['error']}",
                "error": result['error']
            }
        
        # 生成简要摘要
        summary = _generate_analysis_summary(result)
        print(f"\n✓ 分析完成")
        print(f"📝 摘要: {summary[:200]}...")
        
        return {
            "causal_results": result,
            "analysis_summary": summary
        }
        
    except Exception as e:
        print(f"❌ 分析异常: {e}")
        return {
            "causal_results": {"error": str(e)},
            "analysis_summary": f"分析异常: {e}",
            "error": str(e)
        }


def optimizer_agent(state: CausalAnalysisState) -> dict:
    """
    Optimizer Agent：根据因果分析结果，生成优化配比并预测强度
    
    职责：
    1. 根据干预分析结果，找出最有效的调整变量
    2. 基于用户目标（如"提升10%"），计算具体配比
    3. 使用因果模型预测新配比的强度
    4. 验证是否达到目标
    """
    print("\n🔧 Optimizer Agent 正在生成优化配比...")
    
    # 只对干预分析和反事实分析需要优化配比
    analysis_type = state['analysis_type']
    
    if analysis_type not in ['intervention', 'counterfactual']:
        print("  跳过优化（仅干预和反事实分析需要）")
        return {
            "optimized_config": None,
            "predicted_strength": None,
            "optimization_summary": ""
        }
    
    try:
        from dowhy import gcm
        import numpy as np
        
        # 获取基准配比
        if state.get('observed_config'):
            base_config = state['observed_config'].copy()
            print(f"  基准配比：用户输入")
            
            # 如果用户输入的配比中没有强度，先预测一个
            if 'concrete_compressive_strength' not in base_config:
                print(f"  🔮 预测基准强度...")
                intervention_funcs = {
                    'cement': lambda x: base_config.get('cement', 280),
                    'blast_furnace_slag': lambda x: base_config.get('blast_furnace_slag', 0),
                    'fly_ash': lambda x: base_config.get('fly_ash', 0),
                    'water': lambda x: base_config.get('water', 180),
                    'superplasticizer': lambda x: base_config.get('superplasticizer', 0),
                    'coarse_aggregate': lambda x: base_config.get('coarse_aggregate', 1000),
                    'fine_aggregate': lambda x: base_config.get('fine_aggregate', 800),
                    'age': lambda x: base_config.get('age', 28)
                }
                samples = gcm.interventional_samples(
                    _causal_model_instance.causal_model,
                    intervention_funcs,
                    num_samples_to_draw=100
                )
                predicted_strength = float(samples['concrete_compressive_strength'].mean())
                base_config['concrete_compressive_strength'] = predicted_strength
                print(f"  ✓ 基准强度: {predicted_strength:.2f} MPa")
                
        elif state.get('reference_sample_index') is not None:
            idx = state['reference_sample_index']
            df = _causal_model_instance.df
            # 验证索引是否在范围内
            if idx < 0 or idx >= len(df):
                print(f"  ⚠️  参考索引 {idx} 超出范围 [0, {len(df)-1}]，使用默认中等强度样本")
                df_28d = df[df['age'] == 28]
                if len(df_28d) > 0:
                    median_idx = (df_28d['concrete_compressive_strength'] - df_28d['concrete_compressive_strength'].median()).abs().idxmin()
                    base_config = df.loc[median_idx].to_dict()
                else:
                    median_idx = (df['concrete_compressive_strength'] - df['concrete_compressive_strength'].median()).abs().idxmin()
                    base_config = df.loc[median_idx].to_dict()
                print(f"  基准配比：中等强度样本")
                
                # 显示默认样本的完整信息
                print(f"\n  📋 默认基准样本详情:")
                print(f"    • 水泥: {base_config['cement']:.1f} kg/m³")
                print(f"    • 高炉矿渣: {base_config['blast_furnace_slag']:.1f} kg/m³")
                print(f"    • 粉煤灰: {base_config['fly_ash']:.1f} kg/m³")
                print(f"    • 水: {base_config['water']:.1f} kg/m³")
                print(f"    • 减水剂: {base_config['superplasticizer']:.1f} kg/m³")
                print(f"    • 粗骨料: {base_config['coarse_aggregate']:.1f} kg/m³")
                print(f"    • 细骨料: {base_config['fine_aggregate']:.1f} kg/m³")
                print(f"    • 龄期: {base_config['age']:.0f} 天")
                print(f"    • 原始强度: {base_config['concrete_compressive_strength']:.2f} MPa")
                
                # 保存基准样本信息到state
                state['base_sample_info'] = build_sample_info_dict(pd.Series(base_config), source="默认样本（中等强度）")
            else:
                base_config = df.iloc[idx].to_dict()
                print(f"  基准配比：参考批次#{idx}")
                
                # 显示参考样本的完整信息
                print(f"\n  📋 参考样本详情:")
                print(f"    • 水泥: {base_config['cement']:.1f} kg/m³")
                print(f"    • 高炉矿渣: {base_config['blast_furnace_slag']:.1f} kg/m³")
                print(f"    • 粉煤灰: {base_config['fly_ash']:.1f} kg/m³")
                print(f"    • 水: {base_config['water']:.1f} kg/m³")
                print(f"    • 减水剂: {base_config['superplasticizer']:.1f} kg/m³")
                print(f"    • 粗骨料: {base_config['coarse_aggregate']:.1f} kg/m³")
                print(f"    • 细骨料: {base_config['fine_aggregate']:.1f} kg/m³")
                print(f"    • 龄期: {base_config['age']:.0f} 天")
                print(f"    • 原始强度: {base_config['concrete_compressive_strength']:.2f} MPa")
                
                # 保存基准样本信息到state
                state['base_sample_info'] = build_sample_info_dict(pd.Series(base_config), source="用户指定样本（索引{}）".format(idx))
        else:
            # 没有提供基准配比或参考索引
            # 检查是否有明确的目标值或目标提升
            target_value = state.get('target_value')
            target_improvement = state.get('target_improvement')
            
            # 如果没有明确目标，说明是纯探索性问题，不进行优化
            if target_value is None and target_improvement is None:
                print(f"  ℹ️  未提供基准配比或参考索引，且无明确优化目标")
                print(f"  → 这是一个探索性问题，只返回因素分析结果")
                print(f"  （如需具体优化配比，请提供基准配比或参考索引）")
                return {
                    "optimized_config": None,
                    "predicted_strength": None,
                    "optimization_summary": ""
                }
            
            # 如果有明确目标，使用默认中等强度样本作为基准
            df = _causal_model_instance.df
            df_28d = df[df['age'] == 28]
            if len(df_28d) > 0:
                median_idx = (df_28d['concrete_compressive_strength'] - df_28d['concrete_compressive_strength'].median()).abs().idxmin()
                base_config = df.loc[median_idx].to_dict()
            else:
                median_idx = (df['concrete_compressive_strength'] - df['concrete_compressive_strength'].median()).abs().idxmin()
                base_config = df.loc[median_idx].to_dict()
            
            print(f"  基准配比：默认中等强度样本（因提供了明确目标）")
            print(f"\n  📋 默认基准样本详情:")
            print(f"    • 水泥: {base_config['cement']:.1f} kg/m³")
            print(f"    • 高炉矿渣: {base_config['blast_furnace_slag']:.1f} kg/m³")
            print(f"    • 粉煤灰: {base_config['fly_ash']:.1f} kg/m³")
            print(f"    • 水: {base_config['water']:.1f} kg/m³")
            print(f"    • 减水剂: {base_config['superplasticizer']:.1f} kg/m³")
            print(f"    • 粗骨料: {base_config['coarse_aggregate']:.1f} kg/m³")
            print(f"    • 细骨料: {base_config['fine_aggregate']:.1f} kg/m³")
            print(f"    • 龄期: {base_config['age']:.0f} 天")
            print(f"    • 原始强度: {base_config['concrete_compressive_strength']:.2f} MPa")
            
            # 保存基准样本信息到state
            state['base_sample_info'] = build_sample_info_dict(pd.Series(base_config), source="默认样本（中等强度）")
        
        # 提取当前强度和目标提升
        base_strength = base_config.get('concrete_compressive_strength', 35.0)
        # target_improvement 已在前面获取，这里直接使用 state
        target_improvement = state.get('target_improvement')  # 百分比，如10表示提升10%
        
        # 从干预分析结果获取最有效的变量
        causal_results = state.get('causal_results', {})
        
        # 生成优化配比
        optimized_config = base_config.copy()
        
        if analysis_type == 'intervention' and 'interventions' in causal_results:
            # 基于干预分析结果优化
            interventions = causal_results['interventions']
            
            # 检查用户是否指定了要调整的变量
            specified_vars = state.get('specified_variables', [])
            
            if specified_vars:
                # 用户指定了变量，只使用这些变量
                print(f"\n  🔧 使用用户指定的变量: {', '.join(specified_vars)}")
                top_interventions = [
                    i for i in interventions 
                    if i['variable'] in specified_vars
                ]
                
                # 按效应大小排序
                top_interventions = sorted(top_interventions, 
                                          key=lambda x: abs(x['causal_effect']), 
                                          reverse=True)
                
                if not top_interventions:
                    print(f"  ⚠️  指定的变量未在干预分析结果中找到，将使用Top 3")
                    # 回退到Top 3
                    significant_interventions = [
                        i for i in interventions 
                        if i['confidence_interval'][0] * i['confidence_interval'][1] > 0
                    ]
                    top_interventions = sorted(significant_interventions, 
                                              key=lambda x: abs(x['causal_effect']), 
                                              reverse=True)[:3]
                    
                    if not top_interventions:
                        top_interventions = sorted(interventions, 
                                                  key=lambda x: abs(x['causal_effect']), 
                                                  reverse=True)[:3]
            else:
                # 用户未指定变量，自动选择Top 3最有效的变量
                significant_interventions = [
                    i for i in interventions 
                    if i['confidence_interval'][0] * i['confidence_interval'][1] > 0  # 同号说明显著
                ]
                top_interventions = sorted(significant_interventions, 
                                          key=lambda x: abs(x['causal_effect']), 
                                          reverse=True)[:3]
                
                if not top_interventions:
                    # 如果没有显著变量，使用绝对效应最大的前3个
                    top_interventions = sorted(interventions, 
                                              key=lambda x: abs(x['causal_effect']), 
                                              reverse=True)[:3]
            
            print(f"\n  Top {len(top_interventions)} 有效变量:")
            for interv in top_interventions:
                print(f"    • {interv['variable']}: 效应={interv['causal_effect']:+.4f}")
            
            # 如果用户指定了目标值或目标提升，使用精确优化
            target_value = state.get('target_value')
            
            if target_value is not None:
                # 用户指定了绝对目标值
                print(f"\n  🎯 目标：强度达到 {target_value} MPa")
                print(f"  使用迭代优化算法寻找最优配比...")
                target_strength = float(target_value)
            elif target_improvement is not None and target_improvement != 0:
                # 用户指定了相对提升百分比
                print(f"\n  🎯 目标：提升 {target_improvement}%")
                print(f"  使用迭代优化算法寻找最优配比...")
                target_strength = base_strength * (1 + target_improvement / 100.0)
            else:
                target_strength = None
            
            if target_strength is not None:
                
                # 定义预测函数
                def predict_strength(config):
                    """给定配比，预测强度"""
                    intervention_funcs = {
                        'cement': lambda x: config.get('cement', 280),
                        'blast_furnace_slag': lambda x: config.get('blast_furnace_slag', 0),
                        'fly_ash': lambda x: config.get('fly_ash', 0),
                        'water': lambda x: config.get('water', 180),
                        'superplasticizer': lambda x: config.get('superplasticizer', 0),
                        'coarse_aggregate': lambda x: config.get('coarse_aggregate', 1000),
                        'fine_aggregate': lambda x: config.get('fine_aggregate', 800),
                        'age': lambda x: config.get('age', 28)
                    }
                    samples = gcm.interventional_samples(
                        _causal_model_instance.causal_model,
                        intervention_funcs,
                        num_samples_to_draw=100
                    )
                    return float(samples['concrete_compressive_strength'].mean())
                
                # 使用二分搜索找到合适的调整比例
                best_scale = 1.0
                best_config = base_config.copy()
                best_strength = base_strength
                best_diff = abs(base_strength - target_strength)
                
                # 二分搜索范围
                low_scale = 0.0
                high_scale = 0.5  # 最多调整50%
                
                max_iterations = 8
                tolerance = target_strength * 0.02  # 允许2%的误差
                
                for iteration in range(max_iterations):
                    mid_scale = (low_scale + high_scale) / 2.0
                    
                    # 应用调整
                    test_config = base_config.copy()
                    for interv in top_interventions:
                        var = interv['variable']
                        effect = interv['causal_effect']
                        if var in test_config:
                            current_val = base_config[var]
                            # 正效应增加，负效应减少
                            if effect > 0:
                                test_config[var] = current_val * (1 + mid_scale)
                            else:
                                test_config[var] = current_val * (1 - mid_scale)
                    
                    # 预测强度
                    pred_strength = predict_strength(test_config)
                    diff = pred_strength - target_strength
                    
                    print(f"    迭代 {iteration+1}: scale={mid_scale:.3f}, 预测={pred_strength:.2f} MPa, 差距={diff:+.2f} MPa")
                    
                    # 更新最优解
                    if abs(diff) < best_diff:
                        best_diff = abs(diff)
                        best_scale = mid_scale
                        best_config = test_config.copy()
                        best_strength = pred_strength
                    
                    # 检查是否达到目标
                    if abs(diff) < tolerance:
                        print(f"    ✓ 已达到目标（误差 < {tolerance:.2f} MPa）")
                        break
                    
                    # 调整搜索范围
                    if diff < 0:
                        # 预测强度不够，需要更大的调整
                        low_scale = mid_scale
                    else:
                        # 预测强度过高，减小调整
                        high_scale = mid_scale
                
                optimized_config = best_config
                predicted_strength = best_strength
                
                print(f"\n  ✓ 最优调整比例: {best_scale:.1%}")
                print(f"  ✓ 配比调整详情:")
                for interv in top_interventions:
                    var = interv['variable']
                    if var in optimized_config:
                        old_val = base_config[var]
                        new_val = optimized_config[var]
                        if old_val != 0:
                            change_pct = ((new_val - old_val) / old_val) * 100
                            print(f"    • {var}: {old_val:.1f} → {new_val:.1f} ({change_pct:+.1f}%)")
                        else:
                            change = new_val - old_val
                            print(f"    • {var}: {old_val:.1f} → {new_val:.1f} ({change:+.1f} kg/m³)")
            
            else:
                # 没有指定目标 - 检查用户意图
                user_query = state.get('user_query', '').lower()
                is_pure_prediction = any(keyword in user_query for keyword in ['预测', '预报', '强度是多少', '多少mpa', '能达到'])
                
                # 如果是纯预测查询（没有"优化"、"提升"、"改进"等词），则只预测不优化
                if is_pure_prediction and not any(keyword in user_query for keyword in ['优化', '提升', '改进', '调整', '增加', '降低']):
                    print(f"\n  ℹ️  检测到纯预测查询，返回当前配比的预测强度")
                    print(f"  （如需优化配比，请在查询中明确指出目标或调整需求）")
                    
                    # 直接返回基准强度预测，不进行优化
                    optimized_config = base_config.copy()
                    predicted_strength = base_strength
                else:
                    # 用户想要优化但没有指定具体目标，使用默认10%调整
                    print(f"\n  使用默认调整策略（每个变量±10%）")
                    for interv in top_interventions:
                        var = interv['variable']
                        effect = interv['causal_effect']
                        
                        if var in optimized_config:
                            current_val = optimized_config[var]
                            
                            if effect > 0:
                                new_val = current_val * 1.1
                                print(f"    • {var}: {current_val:.1f} → {new_val:.1f} (↑10%, 效应: +{effect:.3f})")
                            else:
                                new_val = current_val * 0.9
                                print(f"    • {var}: {current_val:.1f} → {new_val:.1f} (↓10%, 效应: {effect:.3f})")
                            
                            optimized_config[var] = new_val
                    
                    # 预测优化后的强度
                    intervention_funcs = {
                        'cement': lambda x: optimized_config.get('cement', 280),
                        'blast_furnace_slag': lambda x: optimized_config.get('blast_furnace_slag', 0),
                        'fly_ash': lambda x: optimized_config.get('fly_ash', 0),
                        'water': lambda x: optimized_config.get('water', 180),
                        'superplasticizer': lambda x: optimized_config.get('superplasticizer', 0),
                        'coarse_aggregate': lambda x: optimized_config.get('coarse_aggregate', 1000),
                        'fine_aggregate': lambda x: optimized_config.get('fine_aggregate', 800),
                        'age': lambda x: optimized_config.get('age', 28)
                    }
                    
                    samples = gcm.interventional_samples(
                        _causal_model_instance.causal_model,
                        intervention_funcs,
                        num_samples_to_draw=100
                    )
                    
                    predicted_strength = float(samples['concrete_compressive_strength'].mean())
        
        else:
            # 反事实分析：应用干预值到配比
            if analysis_type == 'counterfactual' and 'interventions' in causal_results:
                # 从反事实分析结果中提取干预值
                interventions_list = causal_results.get('interventions', [])
                print(f"\n  📊 应用反事实干预:")
                for interv in interventions_list:
                    var = interv['variable']
                    old_val = interv['original_value']
                    new_val = interv['new_value']
                    if var in optimized_config:
                        optimized_config[var] = new_val
                        print(f"    • {var}: {old_val:.1f} → {new_val:.1f}")
            
            # 使用干预后的配比预测强度
            intervention_funcs = {
                'cement': lambda x: optimized_config.get('cement', 280),
                'blast_furnace_slag': lambda x: optimized_config.get('blast_furnace_slag', 0),
                'fly_ash': lambda x: optimized_config.get('fly_ash', 0),
                'water': lambda x: optimized_config.get('water', 180),
                'superplasticizer': lambda x: optimized_config.get('superplasticizer', 0),
                'coarse_aggregate': lambda x: optimized_config.get('coarse_aggregate', 1000),
                'fine_aggregate': lambda x: optimized_config.get('fine_aggregate', 800),
                'age': lambda x: optimized_config.get('age', 28)
            }
            
            samples = gcm.interventional_samples(
                _causal_model_instance.causal_model,
                intervention_funcs,
                num_samples_to_draw=100
            )
            
            predicted_strength = float(samples['concrete_compressive_strength'].mean())
        
        strength_improvement = ((predicted_strength - base_strength) / base_strength) * 100 if base_strength != 0 else 0
        
        print(f"\n  ✓ 基准强度: {base_strength:.2f} MPa")
        print(f"  ✓ 预测强度: {predicted_strength:.2f} MPa")
        print(f"  ✓ 实际提升: {strength_improvement:+.1f}%")
        if target_improvement:
            error = abs(strength_improvement - target_improvement)
            print(f"  ✓ 目标提升: {target_improvement:+.1f}%")
            print(f"  ✓ 误差: {error:.2f}个百分点")
        
        # 生成优化摘要
        optimization_summary = f"""
优化配比方案：
  基准强度: {base_strength:.2f} MPa
  优化强度: {predicted_strength:.2f} MPa
  实际提升: {strength_improvement:+.1f}%
{"  目标提升: " + f"{target_improvement:+.1f}%" if target_improvement else ""}

建议配比：
  • 水泥: {optimized_config.get('cement', 0):.1f} kg/m³
  • 高炉矿渣: {optimized_config.get('blast_furnace_slag', 0):.1f} kg/m³
  • 粉煤灰: {optimized_config.get('fly_ash', 0):.1f} kg/m³
  • 水: {optimized_config.get('water', 0):.1f} kg/m³
  • 高效减水剂: {optimized_config.get('superplasticizer', 0):.1f} kg/m³
  • 粗骨料: {optimized_config.get('coarse_aggregate', 0):.1f} kg/m³
  • 细骨料: {optimized_config.get('fine_aggregate', 0):.1f} kg/m³
  • 龄期: {optimized_config.get('age', 28):.0f} 天
"""
        
        return {
            "optimized_config": optimized_config,
            "predicted_strength": predicted_strength,
            "optimization_summary": optimization_summary
        }
        
    except Exception as e:
        print(f"  ⚠️  优化失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            "optimized_config": None,
            "predicted_strength": None,
            "optimization_summary": f"优化失败: {e}"
        }


def advisor_agent(state: CausalAnalysisState) -> dict:
    """
    Advisor Agent：解读因果分析结果，生成决策建议
    
    职责：
    1. 理解因果分析的数值结果
    2. 生成通俗易懂的解释
    3. 提供可操作的工艺优化建议
    """
    print("\n💡 Advisor Agent 正在生成决策建议...")
    
    # 检查是否有错误
    if state.get('error'):
        return {
            "recommendations": f"分析过程出现错误，无法生成建议。错误信息：{state['error']}"
        }
    
    # 使用 LLM 生成建议
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0.3,
        openai_api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE
    )
    
    import json
    
    # 准备优化配比信息
    optimization_info = ""
    if state.get('optimized_config') and state.get('predicted_strength'):
        optimization_info = f"""

优化配比方案：
{state.get('optimization_summary', '')}"""
    
    prompt = f"""你是一个混凝土配合比优化的专家顾问。请基于因果分析结果，生成实用的决策建议。

应用场景：高性能混凝土配合比设计与强度优化
数据来源：UCI Machine Learning Repository (Yeh 1998, 1030个真实样本)

用户问题：{state['user_query']}

分析类型：{state['analysis_type']}
分析摘要：{state['analysis_summary']}

详细结果：
{json.dumps(state['causal_results'], indent=2, ensure_ascii=False)}
{optimization_info}

关键变量说明（基于UCI真实数据集，9个原始变量）：
- concrete_compressive_strength: 抗压强度（MPa）**【目标变量】** - 2.3-82.6，均值35.8
- cement: 水泥（kg/m³）**【关键材料】** - 102-540，均值281
- blast_furnace_slag: 高炉矿渣（kg/m³）- 0-359，均值74，改善密实度和耐久性
- fly_ash: 粉煤灰（kg/m³）- 0-200，均值54，火山灰反应，长期强度
- water: 水（kg/m³）**【Abrams定律：水越多强度越低】** - 122-247，均值182
- superplasticizer: 高效减水剂（kg/m³）- 0-32，均值6.2，与水负相关（r=-0.66）
- coarse_aggregate: 粗骨料（kg/m³）- 801-1145，均值973
- fine_aggregate: 细骨料（kg/m³）- 594-993，均值774
- age: 龄期（天）**【时间效应】** - 1-365，均值46

水胶比计算公式：water / (cement + blast_furnace_slag + fly_ash)
推荐水胶比范围：0.35-0.50（越低强度越高）

请提供：
1. **核心发现**：用1-2句话总结最重要的发现
2. **具体建议**：提供3-5条可操作的改进措施（按优先级排序）
   - 明确指出应该调整哪个参数
   - 说明调整方向和幅度
   - 预期效果
3. **实施方案**：建议的实施顺序和注意事项
4. **风险提示**：可能的副作用或需要监控的指标

请用专业但通俗的语言，确保建议具有可操作性。
"""
    
    response = llm.invoke(prompt)
    recommendations = response.content
    
    return {
        "recommendations": recommendations
    }


def _generate_analysis_summary(result: dict) -> str:
    """生成分析结果的简要摘要"""
    analysis_type = result.get('type', 'unknown')
    
    if analysis_type == 'attribution':
        top_factors = result.get('top_factors', [])
        if top_factors:
            top_3 = top_factors[:3]
            summary = f"归因分析完成。主要影响因素：" + "、".join([
                f"{f['variable']}(贡献{f['contribution']:.4f})" 
                for f in top_3
            ])
        else:
            summary = "归因分析完成，但未发现显著影响因素。"
            
    elif analysis_type == 'intervention':
        interventions = result.get('interventions', [])
        if interventions:
            top_3 = sorted(interventions, key=lambda x: abs(x['causal_effect']), reverse=True)[:3]
            summary = f"干预分析完成。最有效的干预措施：" + "、".join([
                f"{i['variable']}(效应{i['causal_effect']:.4f})"
                for i in top_3
            ])
        else:
            summary = "干预分析完成，但未发现有效的干预措施。"
            
    elif analysis_type == 'counterfactual':
        effect = result.get('causal_effect', 0)
        interventions_list = result.get('interventions', [])
        
        if len(interventions_list) == 1:
            # 单变量干预
            interv = interventions_list[0]
            summary = f"反事实分析完成。如果{interv['variable']}从" \
                      f"{interv['original_value']:.4f}改为{interv['new_value']:.4f}，" \
                      f"{result['target']}预期变化{effect:.4f}。"
        else:
            # 多变量干预
            interv_desc = "、".join([
                f"{i['variable']}={i['new_value']:.2f}" 
                for i in interventions_list
            ])
            summary = f"反事实分析完成。如果干预{interv_desc}，" \
                      f"{result['target']}预期变化{effect:.4f}。"
    else:
        summary = "分析完成。"
    
    return summary


# ============================================================================
# 第四步：构建 LangGraph Workflow
# ============================================================================

def create_causal_agent_graph():
    """创建因果分析智能体工作流"""
    
    # 创建状态图
    workflow = StateGraph(CausalAnalysisState)
    
    # 添加四个智能体节点
    workflow.add_node("router", router_agent)
    workflow.add_node("analyst", causal_analyst_agent)
    workflow.add_node("optimizer", optimizer_agent)  # 新增优化器节点
    workflow.add_node("advisor", advisor_agent)
    
    # 定义流程：START → Router → Analyst → Optimizer → Advisor → END
    workflow.add_edge(START, "router")
    workflow.add_edge("router", "analyst")
    workflow.add_edge("analyst", "optimizer")  # 分析后优化
    workflow.add_edge("optimizer", "advisor")  # 优化后建议
    workflow.add_edge("advisor", END)
    
    # 编译工作流
    app = workflow.compile()
    
    return app


# ============================================================================
# 导出接口
# ============================================================================

__all__ = [
    'CausalAnalysisState',
    'initialize_causal_model',
    'create_causal_agent_graph',
    'router_agent',
    'causal_analyst_agent',
    'optimizer_agent',
    'advisor_agent',
    'math_calculator_tool'
]

