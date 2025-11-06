"""
因果分析智能体系统 - FastAPI 后端服务
提供RESTful API接口，支持归因分析、干预分析、反事实分析
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
from dotenv import load_dotenv
import json
import asyncio
from io import StringIO

warnings.filterwarnings('ignore')

# 加载环境变量
load_dotenv()

# 检查 API Key
if not os.getenv('OPENAI_API_KEY'):
    raise RuntimeError("未找到 OPENAI_API_KEY，请检查 .env 文件")

from src.causal_agent_system import (
    initialize_causal_model,
    create_causal_agent_graph
)

# ============================================================================
# 初始化
# ============================================================================

print("🚀 初始化因果分析智能体系统...")
print("-" * 80)

# 初始化因果模型
print("📦 加载因果模型...")
try:
    causal_model = initialize_causal_model()
    print("✓ 因果模型加载完成\n")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    print("请先运行 train_causal_model.py 训练模型")
    sys.exit(1)

# 创建智能体工作流
print("🏗️  构建智能体工作流...")
agent_graph = create_causal_agent_graph()
print("✓ 工作流构建完成\n")

# 加载真实数据（用于提供参考批次选择）
print("📊 加载真实混凝土数据（UCI数据集）...")
df = pd.read_csv('data/real/concrete_compressive_strength.csv')
df.columns = df.columns.str.strip()  # 清理列名
print(f"✓ 数据加载完成：{len(df)} 条记录，{len(df.columns)} 个变量\n")

print("="*80)
print("✅ 系统初始化完成，准备提供服务")
print("="*80)
print()

# ============================================================================
# FastAPI 应用
# ============================================================================

app = FastAPI(
    title="因果分析智能体系统",
    description="混凝土配合比因果分析API - 支持归因分析、干预分析、反事实分析",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件目录
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ============================================================================
# 数据模型
# ============================================================================

class ObservedConfig(BaseModel):
    """观测配比数据（用于反事实分析）"""
    cement: float
    blast_furnace_slag: float = 0
    fly_ash: float = 0
    water: float
    superplasticizer: float = 0
    coarse_aggregate: float
    fine_aggregate: float
    age: int = 28


class QueryRequest(BaseModel):
    """查询请求"""
    query: str = Field(..., description="用户自然语言查询")
    reference_sample_index: Optional[int] = Field(None, description="参考批次索引（反事实分析可选）")
    observed_config: Optional[ObservedConfig] = Field(None, description="观测配比数据（反事实分析可选，优先于reference_sample_index）")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "如果水用量从200降到150，强度会提升多少？",
                "observed_config": {
                    "cement": 380,
                    "blast_furnace_slag": 100,
                    "fly_ash": 50,
                    "water": 200,
                    "superplasticizer": 8,
                    "coarse_aggregate": 1000,
                    "fine_aggregate": 800,
                    "age": 28
                }
            }
        }


class AnalysisResponse(BaseModel):
    """分析响应"""
    success: bool
    analysis_type: str
    target_variable: str
    routing_reasoning: str
    causal_results: Dict
    analysis_summary: str
    optimized_config: Optional[Dict] = None  # 优化后的配比
    predicted_strength: Optional[float] = None  # 预测强度
    optimization_summary: Optional[str] = None  # 优化摘要
    recommendations: str
    error: Optional[str] = None


class SampleInfo(BaseModel):
    """样本信息"""
    index: int
    cement: float
    blast_furnace_slag: float
    fly_ash: float
    water: float
    superplasticizer: float
    coarse_aggregate: float
    fine_aggregate: float
    age: int
    concrete_compressive_strength: float
    category: str  # 'low', 'medium', 'high', 'target'


class SamplesResponse(BaseModel):
    """样本列表响应"""
    samples: List[SampleInfo]
    total_count: int


class PredictRequest(BaseModel):
    """强度预测请求"""
    cement: float = Field(..., description="水泥 (kg/m³)", ge=100, le=600)
    blast_furnace_slag: float = Field(..., description="高炉矿渣 (kg/m³)", ge=0, le=400)
    fly_ash: float = Field(..., description="粉煤灰 (kg/m³)", ge=0, le=250)
    water: float = Field(..., description="水 (kg/m³)", ge=100, le=300)
    superplasticizer: float = Field(..., description="高效减水剂 (kg/m³)", ge=0, le=40)
    coarse_aggregate: float = Field(..., description="粗骨料 (kg/m³)", ge=700, le=1200)
    fine_aggregate: float = Field(..., description="细骨料 (kg/m³)", ge=500, le=1100)
    age: int = Field(..., description="龄期 (天)", ge=1, le=365)
    
    class Config:
        json_schema_extra = {
            "example": {
                "cement": 280,
                "blast_furnace_slag": 100,
                "fly_ash": 50,
                "water": 180,
                "superplasticizer": 8,
                "coarse_aggregate": 1000,
                "fine_aggregate": 800,
                "age": 28
            }
        }


class PredictResponse(BaseModel):
    """强度预测响应"""
    success: bool
    predicted_strength: float
    water_binder_ratio: float
    total_binder: float
    sand_ratio: float
    confidence_interval: Optional[Dict[str, float]] = None
    interpretation: str
    similar_samples: List[Dict] = []
    feature_weights: Optional[Dict[str, Dict]] = None  # 特征权重信息
    error: Optional[str] = None


class OptimizeRequest(BaseModel):
    """智能优化请求"""
    base_config: ObservedConfig = Field(..., description="基准配比")
    target_strength: float = Field(..., description="目标强度 (MPa)", ge=20, le=80)
    adjust_factors: List[str] = Field(..., description="要调整的因素列表")
    
    class Config:
        json_schema_extra = {
            "example": {
                "base_config": {
                    "cement": 300,
                    "blast_furnace_slag": 0,
                    "fly_ash": 0,
                    "water": 185,
                    "superplasticizer": 3,
                    "coarse_aggregate": 1050,
                    "fine_aggregate": 850,
                    "age": 28
                },
                "target_strength": 45,
                "adjust_factors": ["cement", "fly_ash"]
            }
        }


class OptimizeResponse(BaseModel):
    """智能优化响应"""
    success: bool
    base_config: Dict = Field(..., description="基准配比")
    base_strength: float = Field(..., description="基准强度 (MPa)")
    optimized_config: Dict = Field(..., description="优化后的配比")
    predicted_strength: float = Field(..., description="预测强度 (MPa)")
    improvement_percent: float = Field(..., description="强度提升百分比")
    adjustments: List[Dict] = Field(..., description="调整详情")
    recommendations: str = Field(..., description="建议")
    error: Optional[str] = None


# ============================================================================
# API 端点
# ============================================================================

@app.get("/")
async def root():
    """根路径 - 返回Web界面"""
    static_file = Path(__file__).parent / "static" / "index.html"
    if static_file.exists():
        return FileResponse(static_file)
    else:
        return {
            "message": "因果分析智能体系统 API",
            "version": "1.0.0",
            "endpoints": {
                "health": "/health",
                "samples": "/api/samples",
                "analyze": "/api/analyze",
                "docs": "/docs"
            }
        }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": causal_model is not None,
        "graph_nodes": causal_model.causal_graph.number_of_nodes() if causal_model else 0,
        "graph_edges": causal_model.causal_graph.number_of_edges() if causal_model else 0,
        "data_samples": len(df)
    }


@app.get("/api/samples", response_model=SamplesResponse)
async def get_reference_samples():
    """
    获取参考批次列表（基于真实UCI数据集，28天龄期样本）
    
    返回4种典型样本：低强度、中等强度、高强度、接近图片配合比
    """
    try:
        samples = []
        
        # 只选择28天龄期的样本
        df_28d = df[df['age'] == 28]
        
        # 1. 低强度样本
        low_strength_sample = df_28d.nsmallest(1, 'concrete_compressive_strength').iloc[0]
        samples.append(SampleInfo(
            index=int(low_strength_sample.name),
            cement=float(low_strength_sample['cement']),
            blast_furnace_slag=float(low_strength_sample['blast_furnace_slag']),
            fly_ash=float(low_strength_sample['fly_ash']),
            water=float(low_strength_sample['water']),
            superplasticizer=float(low_strength_sample['superplasticizer']),
            coarse_aggregate=float(low_strength_sample['coarse_aggregate']),
            fine_aggregate=float(low_strength_sample['fine_aggregate']),
            age=int(low_strength_sample['age']),
            concrete_compressive_strength=float(low_strength_sample['concrete_compressive_strength']),
            category='low'
        ))
        
        # 2. 中等强度样本
        median_strength = df_28d['concrete_compressive_strength'].median()
        medium_strength_sample = df_28d.iloc[(df_28d['concrete_compressive_strength'] - median_strength).abs().argmin()]
        samples.append(SampleInfo(
            index=int(medium_strength_sample.name),
            cement=float(medium_strength_sample['cement']),
            blast_furnace_slag=float(medium_strength_sample['blast_furnace_slag']),
            fly_ash=float(medium_strength_sample['fly_ash']),
            water=float(medium_strength_sample['water']),
            superplasticizer=float(medium_strength_sample['superplasticizer']),
            coarse_aggregate=float(medium_strength_sample['coarse_aggregate']),
            fine_aggregate=float(medium_strength_sample['fine_aggregate']),
            age=int(medium_strength_sample['age']),
            concrete_compressive_strength=float(medium_strength_sample['concrete_compressive_strength']),
            category='medium'
        ))
        
        # 3. 高强度样本
        high_strength_sample = df_28d.nlargest(1, 'concrete_compressive_strength').iloc[0]
        samples.append(SampleInfo(
            index=int(high_strength_sample.name),
            cement=float(high_strength_sample['cement']),
            blast_furnace_slag=float(high_strength_sample['blast_furnace_slag']),
            fly_ash=float(high_strength_sample['fly_ash']),
            water=float(high_strength_sample['water']),
            superplasticizer=float(high_strength_sample['superplasticizer']),
            coarse_aggregate=float(high_strength_sample['coarse_aggregate']),
            fine_aggregate=float(high_strength_sample['fine_aggregate']),
            age=int(high_strength_sample['age']),
            concrete_compressive_strength=float(high_strength_sample['concrete_compressive_strength']),
            category='high'
        ))
        
        # 4. 接近图片配合比的样本（水胶比≈0.43）
        # 计算28天样本的水胶比
        df_28d_copy = df_28d.copy()
        df_28d_copy['calc_wb'] = df_28d_copy['water'] / (df_28d_copy['cement'] + df_28d_copy['blast_furnace_slag'] + df_28d_copy['fly_ash'])
        target_sample_idx = (df_28d_copy['calc_wb'] - 0.43).abs().idxmin()
        target_sample = df.loc[target_sample_idx]
        samples.append(SampleInfo(
            index=int(target_sample_idx),
            cement=float(target_sample['cement']),
            blast_furnace_slag=float(target_sample['blast_furnace_slag']),
            fly_ash=float(target_sample['fly_ash']),
            water=float(target_sample['water']),
            superplasticizer=float(target_sample['superplasticizer']),
            coarse_aggregate=float(target_sample['coarse_aggregate']),
            fine_aggregate=float(target_sample['fine_aggregate']),
            age=int(target_sample['age']),
            concrete_compressive_strength=float(target_sample['concrete_compressive_strength']),
            category='target'
        ))
        
        return SamplesResponse(
            samples=samples,
            total_count=len(df)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取样本失败: {str(e)}")


class OutputCapture:
    """捕获标准输出的辅助类"""
    def __init__(self):
        self.output = []
        self.original_stdout = None
        
    def write(self, text):
        if text.strip():
            self.output.append(text)
        if self.original_stdout:
            self.original_stdout.write(text)
            
    def flush(self):
        if self.original_stdout:
            self.original_stdout.flush()
    
    def get_output(self):
        return ''.join(self.output)


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_query(request: QueryRequest):
    """
    执行因果分析（传统方式，返回完整结果）
    
    - **query**: 用户自然语言查询
    - **reference_sample_index**: 参考批次索引（可选，反事实分析建议提供）
    
    返回分析结果和决策建议
    """
    try:
        print(f"\n📥 收到查询: {request.query}")
        if request.reference_sample_index is not None:
            print(f"📍 参考批次: #{request.reference_sample_index}")
        
        # 构建状态
        state_input = {
            "user_query": request.query
        }
        
        # 如果提供了观测配比数据，添加到状态中（优先）
        if request.observed_config is not None:
            state_input["observed_config"] = {
                "cement": request.observed_config.cement,
                "blast_furnace_slag": request.observed_config.blast_furnace_slag,
                "fly_ash": request.observed_config.fly_ash,
                "water": request.observed_config.water,
                "superplasticizer": request.observed_config.superplasticizer,
                "coarse_aggregate": request.observed_config.coarse_aggregate,
                "fine_aggregate": request.observed_config.fine_aggregate,
                "age": request.observed_config.age
            }
            print(f"📋 使用用户输入的观测配比")
        # 否则，如果提供了参考批次，添加到状态中
        elif request.reference_sample_index is not None:
            state_input["reference_sample_index"] = request.reference_sample_index
            print(f"📍 使用参考批次索引: {request.reference_sample_index}")
        
        # 执行分析
        result = agent_graph.invoke(state_input)
        
        # 构建响应
        response = AnalysisResponse(
            success=True,
            analysis_type=result.get('analysis_type', 'unknown'),
            target_variable=result.get('target_variable', ''),
            routing_reasoning=result.get('routing_reasoning', ''),
            causal_results=result.get('causal_results', {}),
            analysis_summary=result.get('analysis_summary', ''),
            optimized_config=result.get('optimized_config'),
            predicted_strength=result.get('predicted_strength'),
            optimization_summary=result.get('optimization_summary'),
            recommendations=result.get('recommendations', ''),
            error=result.get('error')
        )
        
        print(f"\n✅ 分析完成: {response.analysis_type}\n")
        
        return response
        
    except Exception as e:
        print(f"\n❌ 分析失败: {str(e)}\n")
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")


@app.post("/api/analyze_stream")
async def analyze_query_stream(request: QueryRequest):
    """
    执行因果分析（流式响应，实时推送进度）
    
    使用Server-Sent Events (SSE)推送分析过程
    """
    async def event_generator():
        try:
            # 捕获输出
            output_capture = OutputCapture()
            original_stdout = sys.stdout
            sys.stdout = output_capture
            output_capture.original_stdout = original_stdout
            
            # 发送开始消息
            yield f"data: {json.dumps({'type': 'start', 'message': '开始分析...'}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.1)
            
            # 构建状态
            state_input = {"user_query": request.query}
            
            if request.observed_config is not None:
                state_input["observed_config"] = {
                    "cement": request.observed_config.cement,
                    "blast_furnace_slag": request.observed_config.blast_furnace_slag,
                    "fly_ash": request.observed_config.fly_ash,
                    "water": request.observed_config.water,
                    "superplasticizer": request.observed_config.superplasticizer,
                    "coarse_aggregate": request.observed_config.coarse_aggregate,
                    "fine_aggregate": request.observed_config.fine_aggregate,
                    "age": request.observed_config.age
                }
                yield f"data: {json.dumps({'type': 'progress', 'message': '📋 使用用户输入的观测配比'}, ensure_ascii=False)}\n\n"
            elif request.reference_sample_index is not None:
                state_input["reference_sample_index"] = request.reference_sample_index
                yield f"data: {json.dumps({'type': 'progress', 'message': f'📍 使用参考批次 #{request.reference_sample_index}'}, ensure_ascii=False)}\n\n"
            
            await asyncio.sleep(0.1)
            
            # 执行分析（在单独的线程中）
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(agent_graph.invoke, state_input)
                
                # 定期检查输出
                last_output_len = 0
                while not future.done():
                    current_output = output_capture.get_output()
                    if len(current_output) > last_output_len:
                        new_content = current_output[last_output_len:]
                        # 按行发送
                        for line in new_content.split('\n'):
                            if line.strip():
                                yield f"data: {json.dumps({'type': 'progress', 'message': line}, ensure_ascii=False)}\n\n"
                        last_output_len = len(current_output)
                    await asyncio.sleep(0.2)
                
                # 获取结果
                result = future.result()
            
            # 恢复stdout
            sys.stdout = original_stdout
            
            # 发送最后的输出
            final_output = output_capture.get_output()
            if len(final_output) > last_output_len:
                new_content = final_output[last_output_len:]
                for line in new_content.split('\n'):
                    if line.strip():
                        yield f"data: {json.dumps({'type': 'progress', 'message': line}, ensure_ascii=False)}\n\n"
            
            # 构建响应
            response_data = {
                "success": True,
                "analysis_type": result.get('analysis_type', 'unknown'),
                "target_variable": result.get('target_variable', ''),
                "routing_reasoning": result.get('routing_reasoning', ''),
                "causal_results": result.get('causal_results', {}),
                "analysis_summary": result.get('analysis_summary', ''),
                "optimized_config": result.get('optimized_config'),
                "predicted_strength": result.get('predicted_strength'),
                "optimization_summary": result.get('optimization_summary'),
                "recommendations": result.get('recommendations', ''),
                "error": result.get('error')
            }
            
            # 发送完整结果
            yield f"data: {json.dumps({'type': 'result', 'data': response_data}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'end', 'message': '分析完成'}, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            sys.stdout = original_stdout
            error_msg = f"分析失败: {str(e)}"
            yield f"data: {json.dumps({'type': 'error', 'message': error_msg}, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/api/variables")
async def get_variables():
    """
    获取因果图中的所有可用变量（真实UCI数据集，9个原始变量）
    
    返回变量列表及其说明
    """
    variables = {
        "materials": {
            "cement": "水泥 (102-540 kg/m³, 均值281) ⭐⭐⭐关键材料",
            "blast_furnace_slag": "高炉矿渣 (0-359 kg/m³, 均值74) - 提高密实度",
            "fly_ash": "粉煤灰 (0-200 kg/m³, 均值54) - 长期强度",
            "water": "水 (122-247 kg/m³, 均值182) ⭐⭐⭐Abrams定律",
            "superplasticizer": "高效减水剂 (0-32 kg/m³, 均值6.2) - 与水负相关",
            "coarse_aggregate": "粗骨料 (801-1145 kg/m³, 均值973)",
            "fine_aggregate": "细骨料 (594-993 kg/m³, 均值774)"
        },
        "process": {
            "age": "龄期 (1-365天, 均值46天) ⭐⭐⭐时间效应"
        },
        "target": {
            "concrete_compressive_strength": "抗压强度 (2.3-82.6 MPa, 均值35.8) 🎯目标变量"
        },
        "important_notes": {
            "water_cement_relation": "Abrams定律：水越多，强度越低（负相关）",
            "water_sp_correlation": "水与减水剂负相关（r=-0.66）",
            "scm_synergy": "矿渣和粉煤灰有协同效应",
            "age_effect": "早期（7d）水泥主导，长期（28d+）掺合料贡献增加"
        }
    }
    
    return {
        "variables": variables,
        "total_nodes": causal_model.causal_graph.number_of_nodes(),
        "total_edges": causal_model.causal_graph.number_of_edges(),
        "data_source": "UCI Machine Learning Repository (Yeh 1998)"
    }


@app.get("/api/graph")
async def get_causal_graph():
    """
    获取因果图结构
    
    返回节点和边的列表
    """
    if causal_model is None or causal_model.causal_graph is None:
        raise HTTPException(status_code=500, detail="因果图未初始化")
    
    graph = causal_model.causal_graph
    
    return {
        "nodes": list(graph.nodes()),
        "edges": [{"source": u, "target": v} for u, v in graph.edges()],
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges()
    }


@app.post("/api/predict", response_model=PredictResponse)
async def predict_strength(request: PredictRequest):
    """
    预测混凝土抗压强度
    
    根据用户输入的配合比参数，使用因果模型预测28天抗压强度
    
    - **cement**: 水泥用量 (kg/m³)
    - **blast_furnace_slag**: 高炉矿渣 (kg/m³)
    - **fly_ash**: 粉煤灰 (kg/m³)
    - **water**: 水用量 (kg/m³)
    - **superplasticizer**: 高效减水剂 (kg/m³)
    - **coarse_aggregate**: 粗骨料 (kg/m³)
    - **fine_aggregate**: 细骨料 (kg/m³)
    - **age**: 龄期 (天)
    
    返回预测强度及相关分析
    """
    try:
        print(f"\n{'='*80}")
        print(f"🔮 收到强度预测请求")
        print(f"  • 水泥: {request.cement} kg/m³")
        print(f"  • 水: {request.water} kg/m³")
        print(f"  • 龄期: {request.age} 天")
        print(f"{'='*80}\n")
        
        # 构建输入数据（仅使用9个原始字段）
        input_data = pd.DataFrame([{
            'cement': request.cement,
            'blast_furnace_slag': request.blast_furnace_slag,
            'fly_ash': request.fly_ash,
            'water': request.water,
            'superplasticizer': request.superplasticizer,
            'coarse_aggregate': request.coarse_aggregate,
            'fine_aggregate': request.fine_aggregate,
            'age': request.age
        }])
        
        # 使用因果模型预测（通过干预分析）
        from dowhy import gcm
        
        # 使用DoWhy的interventional_samples进行预测
        # 先创建干预函数字典
        intervention_funcs = {
            'cement': lambda x: request.cement,
            'blast_furnace_slag': lambda x: request.blast_furnace_slag,
            'fly_ash': lambda x: request.fly_ash,
            'water': lambda x: request.water,
            'superplasticizer': lambda x: request.superplasticizer,
            'coarse_aggregate': lambda x: request.coarse_aggregate,
            'fine_aggregate': lambda x: request.fine_aggregate,
            'age': lambda x: request.age
        }
        
        # 使用interventional_samples进行预测
        samples = gcm.interventional_samples(
            causal_model.causal_model,
            intervention_funcs,
            num_samples_to_draw=100
        )
        
        # 获取预测值和置信区间
        predicted_strength = float(samples['concrete_compressive_strength'].mean())
        std_strength = float(samples['concrete_compressive_strength'].std())
        
        confidence_interval = {
            'lower': float(predicted_strength - 1.96 * std_strength),
            'upper': float(predicted_strength + 1.96 * std_strength)
        }
        
        # 生成解释（仅使用原始字段）
        interpretation = f"""
根据您输入的配合比参数，预测结果如下：

📊 输入配合比（9个原始字段）：
  • 水泥 (Cement): {request.cement:.1f} kg/m³
  • 高炉矿渣 (Blast Furnace Slag): {request.blast_furnace_slag:.1f} kg/m³
  • 粉煤灰 (Fly Ash): {request.fly_ash:.1f} kg/m³
  • 水 (Water): {request.water:.1f} kg/m³
  • 高效减水剂 (Superplasticizer): {request.superplasticizer:.1f} kg/m³
  • 粗骨料 (Coarse Aggregate): {request.coarse_aggregate:.1f} kg/m³
  • 细骨料 (Fine Aggregate): {request.fine_aggregate:.1f} kg/m³
  • 龄期 (Age): {request.age} 天

🎯 预测抗压强度: {predicted_strength:.2f} MPa
📊 95%置信区间: [{confidence_interval['lower']:.2f}, {confidence_interval['upper']:.2f}] MPa

💡 工程评估：
"""
        
        # 根据强度等级给出建议
        if predicted_strength >= 50:
            interpretation += "  • 高强度混凝土，适用于高层建筑、桥梁等重要结构\n"
        elif predicted_strength >= 30:
            interpretation += "  • 常规强度混凝土，适用于一般民用建筑\n"
        else:
            interpretation += "  • 强度偏低，建议优化配合比\n"
        
        # 找相似样本（基于欧氏距离，使用原始字段）
        df_age_filtered = df[df['age'] == request.age]
        if len(df_age_filtered) > 0:
            df_age_filtered = df_age_filtered.copy()
            
            # 计算归一化的欧氏距离（只使用主要材料）
            df_age_filtered['distance'] = (
                ((df_age_filtered['cement'] - request.cement) / 500) ** 2 +
                ((df_age_filtered['water'] - request.water) / 200) ** 2 +
                ((df_age_filtered['blast_furnace_slag'] - request.blast_furnace_slag) / 300) ** 2
            ) ** 0.5
            
            # 找最相似的3个样本
            similar = df_age_filtered.nsmallest(3, 'distance')
            similar_samples = []
            for idx, row in similar.iterrows():
                similar_samples.append({
                    'cement': float(row['cement']),
                    'water': float(row['water']),
                    'blast_furnace_slag': float(row['blast_furnace_slag']),
                    'actual_strength': float(row['concrete_compressive_strength']),
                    'age': int(row['age'])
                })
        else:
            similar_samples = []
        
        # 计算特征权重（使用干预分析）
        print("  计算特征权重...")
        try:
            # 使用小步长进行干预分析来估算权重
            weights_df = causal_model.intervention_analysis(
                target='concrete_compressive_strength',
                step_size=10,  # 每个变量增加10个单位
                non_interveneable_nodes=[],
                num_samples=500,  # 减少采样数以提高速度
                num_bootstrap_resamples=10
            )
            
            # 转换为权重百分比
            total_abs_effect = weights_df['Causal_Effect'].abs().sum()
            feature_weights = {}
            
            # 变量中文名映射
            var_names = {
                'cement': '水泥用量',
                'blast_furnace_slag': '高炉矿渣',
                'fly_ash': '粉煤灰',
                'water': '水用量',
                'superplasticizer': '高效减水剂',
                'coarse_aggregate': '粗骨料',
                'fine_aggregate': '细骨料',
                'age': '养护成熟度'
            }
            
            for idx, row in weights_df.iterrows():
                var = row['Variable']
                effect = row['Causal_Effect']
                weight_pct = abs(effect) / total_abs_effect * 100 if total_abs_effect > 0 else 0
                
                # 根据权重给出质量评分（简化版）
                if weight_pct > 30:
                    score = 85
                elif weight_pct > 20:
                    score = 75
                elif weight_pct > 10:
                    score = 90
                elif weight_pct > 5:
                    score = 85
                else:
                    score = 80
                
                feature_weights[var] = {
                    'name': var_names.get(var, var),
                    'weight_pct': float(weight_pct),
                    'causal_effect': float(effect),
                    'score': score,
                    'direction': '正向' if effect > 0 else '负向'
                }
            
            print(f"  ✓ 权重计算完成")
        except Exception as e:
            print(f"  ⚠️  权重计算失败: {e}")
            feature_weights = None
        
        response = PredictResponse(
            success=True,
            predicted_strength=predicted_strength,
            water_binder_ratio=0.0,  # 不使用衍生指标
            total_binder=0.0,  # 不使用衍生指标
            sand_ratio=0.0,  # 不使用衍生指标
            confidence_interval=confidence_interval,
            interpretation=interpretation,
            similar_samples=similar_samples,
            feature_weights=feature_weights,
            error=None
        )
        
        print(f"\n✅ 预测完成: {predicted_strength:.2f} MPa\n")
        
        return response
        
    except Exception as e:
        print(f"\n❌ 预测失败: {str(e)}\n")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@app.post("/api/optimize", response_model=OptimizeResponse)
async def optimize_config(request: OptimizeRequest):
    """
    智能优化混凝土配合比
    
    根据基准配比、目标强度和可调整因素，自动优化配合比以达到目标强度
    
    - **base_config**: 基准配比
    - **target_strength**: 目标强度 (MPa)
    - **adjust_factors**: 可调整的因素列表（如 ["cement", "fly_ash"]）
    
    返回优化后的配比方案
    """
    try:
        from dowhy import gcm
        
        print(f"\n{'='*80}")
        print(f"🎯 收到智能优化请求")
        print(f"  • 目标强度: {request.target_strength} MPa")
        print(f"  • 调整因素: {', '.join(request.adjust_factors)}")
        print(f"{'='*80}\n")
        
        # 1. 预测基准强度
        base_config_dict = {
            'cement': request.base_config.cement,
            'blast_furnace_slag': request.base_config.blast_furnace_slag,
            'fly_ash': request.base_config.fly_ash,
            'water': request.base_config.water,
            'superplasticizer': request.base_config.superplasticizer,
            'coarse_aggregate': request.base_config.coarse_aggregate,
            'fine_aggregate': request.base_config.fine_aggregate,
            'age': request.base_config.age
        }
        
        print("📊 步骤1：预测基准强度...")
        base_intervention_funcs = {k: (lambda v: lambda x: v)(v) for k, v in base_config_dict.items()}
        base_samples = gcm.interventional_samples(
            causal_model.causal_model,
            base_intervention_funcs,
            num_samples_to_draw=100
        )
        base_strength = float(base_samples['concrete_compressive_strength'].mean())
        print(f"  ✓ 基准强度: {base_strength:.2f} MPa\n")
        
        # 2. 执行干预分析，获取各因素的因果效应
        print("📊 步骤2：分析各因素的因果效应...")
        weights_df = causal_model.intervention_analysis(
            target='concrete_compressive_strength',
            step_size=1.0,
            num_samples=5000,
            num_bootstrap_resamples=20
        )
        
        # 筛选用户指定的因素
        selected_factors = weights_df[weights_df['Variable'].isin(request.adjust_factors)].copy()
        selected_factors = selected_factors.sort_values('Causal_Effect', key=abs, ascending=False)
        
        print(f"  选中因素效应:")
        for _, row in selected_factors.iterrows():
            print(f"    • {row['Variable']}: {row['Causal_Effect']:+.4f}")
        print()
        
        # 3. 使用二分搜索优化配比
        print("📊 步骤3：使用迭代优化算法寻找最优配比...")
        
        def predict_strength_for_config(config):
            """给定配比，预测强度"""
            intervention_funcs = {k: (lambda v: lambda x: v)(v) for k, v in config.items()}
            samples = gcm.interventional_samples(
                causal_model.causal_model,
                intervention_funcs,
                num_samples_to_draw=100
            )
            return float(samples['concrete_compressive_strength'].mean())
        
        # 二分搜索参数
        low_scale = 0.0
        high_scale = 0.5  # 最多调整50%
        best_config = base_config_dict.copy()
        best_strength = base_strength
        best_diff = abs(base_strength - request.target_strength)
        
        max_iterations = 10
        tolerance = request.target_strength * 0.02  # 2%误差
        
        for iteration in range(max_iterations):
            mid_scale = (low_scale + high_scale) / 2.0
            
            # 应用调整
            test_config = base_config_dict.copy()
            for _, row in selected_factors.iterrows():
                var = row['Variable']
                effect = row['Causal_Effect']
                if var in test_config:
                    current_val = base_config_dict[var]
                    # 正效应增加，负效应减少
                    if effect > 0:
                        test_config[var] = current_val * (1 + mid_scale)
                    else:
                        test_config[var] = current_val * (1 - mid_scale)
            
            # 预测强度
            pred_strength = predict_strength_for_config(test_config)
            diff = pred_strength - request.target_strength
            
            print(f"  迭代 {iteration+1}: scale={mid_scale:.3f}, 预测={pred_strength:.2f} MPa, 差距={diff:+.2f} MPa")
            
            # 更新最优解
            if abs(diff) < best_diff:
                best_diff = abs(diff)
                best_config = test_config.copy()
                best_strength = pred_strength
            
            # 检查是否达到目标
            if abs(diff) < tolerance:
                print(f"  ✓ 已达到目标（误差 < {tolerance:.2f} MPa）\n")
                break
            
            # 调整搜索范围
            if diff < 0:
                low_scale = mid_scale
            else:
                high_scale = mid_scale
        
        print(f"  ✓ 优化完成\n")
        
        # 4. 生成调整详情
        adjustments = []
        var_names_cn = {
            'cement': '水泥',
            'blast_furnace_slag': '高炉矿渣',
            'fly_ash': '粉煤灰',
            'water': '水',
            'superplasticizer': '高效减水剂',
            'coarse_aggregate': '粗骨料',
            'fine_aggregate': '细骨料',
            'age': '龄期'
        }
        
        for var in request.adjust_factors:
            if var in base_config_dict and var in best_config:
                old_val = base_config_dict[var]
                new_val = best_config[var]
                change = new_val - old_val
                change_pct = (change / old_val * 100) if old_val != 0 else 0
                
                adjustments.append({
                    'variable': var,
                    'name': var_names_cn.get(var, var),
                    'old_value': round(old_val, 2),
                    'new_value': round(new_val, 2),
                    'change': round(change, 2),
                    'change_percent': round(change_pct, 2)
                })
        
        # 5. 生成建议
        improvement_pct = ((best_strength - base_strength) / base_strength * 100) if base_strength != 0 else 0
        
        recommendations = f"""
🎯 优化方案摘要

基准强度：{base_strength:.2f} MPa
优化强度：{best_strength:.2f} MPa
实际提升：{improvement_pct:+.1f}%
目标强度：{request.target_strength:.2f} MPa
误差：{abs(best_strength - request.target_strength):.2f} MPa

📝 配比调整建议：
"""
        
        for adj in adjustments:
            recommendations += f"\n• {adj['name']}: {adj['old_value']:.1f} → {adj['new_value']:.1f} kg/m³ ({adj['change_percent']:+.1f}%)"
        
        recommendations += f"""

💡 实施建议：
1. 建议按照优化后的配比进行试配
2. 关注施工和易性的变化
3. 必要时微调减水剂用量
4. 建议至少制作3组试块验证强度
"""
        
        response = OptimizeResponse(
            success=True,
            base_config=base_config_dict,
            base_strength=round(base_strength, 2),
            optimized_config={k: round(v, 2) for k, v in best_config.items()},
            predicted_strength=round(best_strength, 2),
            improvement_percent=round(improvement_pct, 2),
            adjustments=adjustments,
            recommendations=recommendations,
            error=None
        )
        
        print(f"✅ 优化完成: {base_strength:.2f} → {best_strength:.2f} MPa ({improvement_pct:+.1f}%)\n")
        
        return response
        
    except Exception as e:
        print(f"\n❌ 优化失败: {str(e)}\n")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"优化失败: {str(e)}")


# ============================================================================
# 启动服务
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*80)
    print("🌐 启动 FastAPI 服务器")
    print("="*80)
    print()
    print("API 文档:")
    print("  • Swagger UI: http://localhost:8000/docs")
    print("  • ReDoc: http://localhost:8000/redoc")
    print()
    print("主要端点:")
    print("  • POST /api/analyze - 执行因果分析")
    print("  • GET  /api/samples - 获取参考批次")
    print("  • GET  /api/variables - 获取可用变量")
    print("  • GET  /api/graph - 获取因果图结构")
    print()
    print("="*80)
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

