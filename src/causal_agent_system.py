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
    reference_sample_index: int        # 参考批次索引（用于反事实分析）
    
    # Router 输出
    analysis_type: str                 # 'attribution' | 'intervention' | 'counterfactual'
    target_variable: str               # 目标变量
    intervention_params: dict          # 干预参数
    routing_reasoning: str             # 路由推理过程
    
    # Causal Analyst 输出
    causal_results: dict               # 因果分析数值结果
    analysis_summary: str              # 分析摘要
    
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
    """
    print("\n" + "="*80)
    print("🔍 Router Agent 正在分析您的问题...")
    print("="*80)
    
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
    "extracted_info": {{
        // 如果是反事实分析：
        // 单变量干预：
        "intervention_variable": "变量名",
        "original_value": 原始数值,
        "intervention_value": 新数值
        // 或者多变量干预：
        "intervention_variable": {{"water": 150, "cement": 300}}
    }}
}}

示例1（单变量）：
用户问："如果水用量从200降到150，强度会怎样？"
回复：{{"intervention_variable": "water", "original_value": 200, "intervention_value": 150}}

示例2（多变量）：
用户问："如果水泥300、水180、龄期28天，强度是多少？"
回复：{{"intervention_variable": {{"cement": 300, "water": 180, "age": 28}}}}
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
        
        return {
            "analysis_type": parsed['analysis_type'],
            "target_variable": parsed['target_variable'],
            "routing_reasoning": parsed['reasoning'],
            "intervention_params": parsed.get('extracted_info', {})
        }
        
    except Exception as e:
        print(f"⚠️ 路由解析失败: {e}")
        # 默认使用干预分析
        return {
            "analysis_type": "intervention",
            "target_variable": "concrete_compressive_strength",
            "routing_reasoning": "使用默认配置",
            "intervention_params": {}
        }


def causal_analyst_agent(state: CausalAnalysisState) -> dict:
    """
    Causal Analyst Agent：执行因果分析
    
    职责：
    1. 根据 Router 的指示调用对应的因果分析工具
    2. 执行计算并返回定量结果
    """
    print("\n" + "="*80)
    print("📊 Causal Analyst Agent 正在执行因果分析...")
    print("="*80)
    
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
            # 支持单个或多个变量的干预
            intervention_variable = params.get('intervention_variable')
            intervention_value = params.get('intervention_value')
            original_value = params.get('original_value')
            
            # 构建干预字典
            interventions = {}
            
            # 情况1：提取到单个变量的干预
            if isinstance(intervention_variable, str) and intervention_value is not None:
                interventions[intervention_variable] = intervention_value
            
            # 情况2：提取到多个变量的干预（Router可能提取为字典）
            elif isinstance(intervention_variable, dict):
                interventions = intervention_variable
            elif isinstance(intervention_value, dict):
                interventions = intervention_value
            
            # 情况3：没有提取到干预信息，使用默认
            if not interventions:
                interventions = {'water': 150}  # 默认降低用水量
                print(f"  ⚠️  未提取到干预信息，使用默认干预: {interventions}")
            
            # 选择合适的样本进行分析
            # 优先使用用户选择的参考批次
            if state.get('reference_sample_index') is not None:
                sample_index = state['reference_sample_index']
                print(f"  使用用户选择的参考批次: 索引 {sample_index}")
            # 如果提取到了原始值，尝试找到接近该值的样本
            elif original_value is not None and _causal_model_instance is not None:
                df = _causal_model_instance.df
                # 使用第一个干预变量找到最接近的样本
                first_var = list(interventions.keys())[0]
                if first_var in df.columns:
                    closest_idx = (df[first_var] - original_value).abs().idxmin()
                    sample_index = int(closest_idx)
                    print(f"  找到最接近原始值 {original_value} 的样本: 索引 {sample_index}")
            else:
                sample_index = 100  # 默认样本
                print(f"  使用默认样本索引: {sample_index}")
            
            print(f"  干预变量: {', '.join(interventions.keys())}")
            print(f"  干预值: {interventions}")
            
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


def advisor_agent(state: CausalAnalysisState) -> dict:
    """
    Advisor Agent：解读因果分析结果，生成决策建议
    
    职责：
    1. 理解因果分析的数值结果
    2. 生成通俗易懂的解释
    3. 提供可操作的工艺优化建议
    """
    print("\n" + "="*80)
    print("💡 Advisor Agent 正在生成决策建议...")
    print("="*80)
    
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
    
    prompt = f"""你是一个混凝土配合比优化的专家顾问。请基于因果分析结果，生成实用的决策建议。

应用场景：高性能混凝土配合比设计与强度优化
数据来源：UCI Machine Learning Repository (Yeh 1998, 1030个真实样本)

用户问题：{state['user_query']}

分析类型：{state['analysis_type']}
分析摘要：{state['analysis_summary']}

详细结果：
{json.dumps(state['causal_results'], indent=2, ensure_ascii=False)}

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
    
    print("\n" + "="*80)
    print("📋 决策建议")
    print("="*80)
    print(recommendations)
    
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
    
    # 添加三个智能体节点
    workflow.add_node("router", router_agent)
    workflow.add_node("analyst", causal_analyst_agent)
    workflow.add_node("advisor", advisor_agent)
    
    # 定义流程：START → Router → Analyst → Advisor → END
    workflow.add_edge(START, "router")
    workflow.add_edge("router", "analyst")
    workflow.add_edge("analyst", "advisor")
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
    'advisor_agent'
]

