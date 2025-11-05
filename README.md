# 混凝土强度因果分析系统 API 文档

基于因果推断的混凝土配合比智能分析系统，提供强度预测、因果分析、权重可视化等功能。

## 📋 目录

- [快速开始](#快速开始)
- [API端点总览](#api端点总览)
- [详细接口文档](#详细接口文档)
  - [1. 健康检查](#1-健康检查)
  - [2. 强度预测](#2-强度预测)
  - [3. 因果分析](#3-因果分析)
  - [3.2. 因果分析（流式响应）](#32-因果分析流式响应)
  - [4. 参考批次](#4-参考批次)
  - [5. 变量信息](#5-变量信息)
  - [6. 因果图结构](#6-因果图结构)
- [数据模型](#数据模型)
- [错误处理](#错误处理)
- [使用示例](#使用示例)

---

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置环境变量

创建 `.env` 文件：

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-5-mini
```

### 启动服务器

```bash
python3 api_server.py
```

服务器将在 `http://localhost:8000` 启动。

### 访问方式

- **Web界面**: http://localhost:8000
- **API文档**: http://localhost:8000/docs (Swagger UI)
- **备用文档**: http://localhost:8000/redoc (ReDoc)

---

## 📡 API端点总览

| 方法 | 端点 | 描述 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/api/predict` | 预测混凝土强度 |
| POST | `/api/analyze` | 因果分析（智能问答，完整响应） |
| POST | `/api/analyze_stream` | 🔥 **因果分析（流式响应，实时进度）** |
| GET | `/api/samples` | 获取参考批次 |
| GET | `/api/variables` | 获取变量信息 |
| GET | `/api/graph` | 获取因果图结构 |

---

## 📖 详细接口文档

### 1. 健康检查

检查服务状态和模型加载情况。

**请求**

```http
GET /health
```

**响应**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "graph_nodes": 9,
  "graph_edges": 9,
  "data_samples": 1030
}
```

---

### 2. 强度预测

根据配合比参数预测混凝土抗压强度。

**请求**

```http
POST /api/predict
Content-Type: application/json
```

**请求体**

```json
{
  "cement": 380,
  "blast_furnace_slag": 100,
  "fly_ash": 50,
  "water": 170,
  "superplasticizer": 8,
  "coarse_aggregate": 1000,
  "fine_aggregate": 800,
  "age": 28
}
```

**参数说明**

| 参数 | 类型 | 范围 | 单位 | 说明 |
|------|------|------|------|------|
| `cement` | float | 100-600 | kg/m³ | 水泥用量 |
| `blast_furnace_slag` | float | 0-400 | kg/m³ | 高炉矿渣 |
| `fly_ash` | float | 0-250 | kg/m³ | 粉煤灰 |
| `water` | float | 100-300 | kg/m³ | 水用量 |
| `superplasticizer` | float | 0-40 | kg/m³ | 高效减水剂 |
| `coarse_aggregate` | float | 700-1200 | kg/m³ | 粗骨料 |
| `fine_aggregate` | float | 500-1100 | kg/m³ | 细骨料 |
| `age` | int | 1-365 | 天 | 养护龄期 |

**响应**

```json
{
  "success": true,
  "predicted_strength": 52.35,
  "water_binder_ratio": 0.0,
  "total_binder": 0.0,
  "sand_ratio": 0.0,
  "confidence_interval": {
    "lower": 48.23,
    "upper": 56.47
  },
  "interpretation": "根据您输入的配合比参数，预测结果如下：...",
  "similar_samples": [
    {
      "cement": 375.0,
      "water": 168.0,
      "blast_furnace_slag": 95.0,
      "actual_strength": 51.2,
      "age": 28
    }
  ],
  "feature_weights": {
    "cement": {
      "name": "水泥用量",
      "weight_pct": 35.2,
      "causal_effect": 0.85,
      "score": 85,
      "direction": "正向"
    },
    "water": {
      "name": "水用量",
      "weight_pct": 28.5,
      "causal_effect": -0.72,
      "score": 75,
      "direction": "负向"
    }
  },
  "error": null
}
```

**响应字段说明**

- `predicted_strength`: 预测的抗压强度 (MPa)
- `confidence_interval`: 95%置信区间
- `interpretation`: 结果解释和工程建议
- `similar_samples`: 相似历史样本（最多3个）
- `feature_weights`: 各因素的影响权重分析
  - `weight_pct`: 相对重要性百分比
  - `causal_effect`: 因果效应值（增加10单位对强度的影响）
  - `direction`: 正向（提高强度）或负向（降低强度）

---

### 3. 因果分析

使用自然语言提问，智能体自动判断分析类型并给出结果。**支持精确目标控制**（如"提升10%"）和用户自定义配比输入。

**请求**

```http
POST /api/analyze
Content-Type: application/json
```

**请求体示例1：基于参考批次**

```json
{
  "query": "如果我想强度提升10%，应该如何调整配合比？",
  "reference_sample_index": 100
}
```

**请求体示例2：用户自定义配比**

```json
{
  "query": "如果水用量从200降到150 kg/m³，强度会提升多少？",
  "observed_config": {
    "cement": 164.8,
    "blast_furnace_slag": 190.0,
    "fly_ash": 148.0,
    "water": 200.0,
    "superplasticizer": 19.0,
    "coarse_aggregate": 838.0,
    "fine_aggregate": 741.0,
    "age": 30
  }
}
```

**参数说明**

- `query` (必填): 自然语言问题
- `reference_sample_index` (可选): 参考批次索引，用于基准配比
- `observed_config` (可选): 用户输入的观测配比，优先于 `reference_sample_index`

**支持的问题类型**

| 分析类型 | 示例问题 | 特点 |
|---------|---------|------|
| 归因分析 | "为什么强度下降了？" | 找出根本原因 |
| 干预分析 | "如何提高混凝土强度？" | 评估优化措施 |
| 反事实分析（绝对值） | "如果水用量降到150，强度会怎样？" | 模拟假设场景（指定具体值） |
| 🔥 **反事实分析（数学运算）** | "水泥增加50 kg/m³，强度会怎样？"<br>"添加矿渣100，减少水泥50，强度会怎样？" | **支持加减乘除运算，智能处理多变量** |
| 🎯 **目标导向优化** | "如果我想强度提升10%，应该如何调整配合比？" | **精确控制目标，自动生成最优配比** |

**🔥 数学运算支持（新功能）**

系统现在支持智能数学运算，可以处理复杂的多变量调整：

| 运算类型 | 示例表达 | 自动识别为 |
|---------|---------|-----------|
| **加法** | "增加50"、"添加100"、"加30" | `add` |
| **减法** | "减少50"、"降低30"、"减20" | `subtract` |
| **乘法** | "乘以2"、"翻倍" | `multiply` |
| **除法** | "除以2"、"减半" | `divide` |

**示例**：
- ✅ "水泥增加50 kg/m³" → `cement = 原值 + 50`
- ✅ "水减少30 kg/m³" → `water = 原值 - 30`
- ✅ "添加矿渣100，减少水泥50" → `slag = 原值 + 100; cement = 原值 - 50`
- ✅ "龄期翻倍" → `age = 原值 × 2`

**响应示例1：目标导向优化（新功能）**

```json
{
  "success": true,
  "analysis_type": "intervention",
  "target_variable": "concrete_compressive_strength",
  "routing_reasoning": "用户要求强度提升10%，这是目标导向的干预优化场景...",
  "causal_results": {
    "interventions": [
      {
        "variable": "cement",
        "causal_effect": 0.1809,
        "confidence_interval": [0.175, 0.187]
      },
      {
        "variable": "water",
        "causal_effect": -0.1661,
        "confidence_interval": [-0.172, -0.160]
      }
    ]
  },
  "analysis_summary": "干预分析完成。最有效的干预措施：cement(效应0.1809)、water(效应-0.1661)...",
  "optimized_config": {
    "cement": 178.2,
    "blast_furnace_slag": 0.0,
    "fly_ash": 0.0,
    "water": 161.1,
    "superplasticizer": 0.0,
    "coarse_aggregate": 1119.0,
    "fine_aggregate": 789.0,
    "age": 30.8,
    "concrete_compressive_strength": 37.14
  },
  "predicted_strength": 37.14,
  "optimization_summary": "优化配比方案：\n  基准强度: 33.76 MPa\n  优化强度: 37.14 MPa\n  实际提升: +10.0%\n  目标提升: +10.0%",
  "recommendations": "建议采取以下措施：\n1. 水泥增加至178.2 kg/m³（+10%）\n2. 水用量降至161.1 kg/m³（-10%）\n3. ...",
  "error": null
}
```

**响应示例2：传统干预分析**

```json
{
  "success": true,
  "analysis_type": "intervention",
  "target_variable": "concrete_compressive_strength",
  "routing_reasoning": "用户询问优化措施，这是典型的干预分析场景...",
  "causal_results": {
    "interventions": [
      {
        "variable": "water",
        "causal_effect": -0.25,
        "confidence_interval": [-0.27, -0.23]
      }
    ]
  },
  "analysis_summary": "根据因果分析，降低水用量可显著提高强度...",
  "recommendations": "建议采取以下措施：\n1. 优化水胶比至0.40-0.45\n2. ...",
  "error": null
}
```

**响应字段说明**

- `analysis_type`: 分析类型（attribution/intervention/counterfactual）
- `target_variable`: 目标变量名称
- `routing_reasoning`: Router Agent的推理过程
- `causal_results`: 因果分析的数值结果
- `analysis_summary`: 分析结果摘要
- `recommendations`: LLM生成的决策建议
- **`optimized_config`** (新增): 优化后的配比方案（目标导向优化时返回）
- **`predicted_strength`** (新增): 优化配比的预测强度（目标导向优化时返回）
- **`optimization_summary`** (新增): 优化摘要，包含目标达成情况（目标导向优化时返回）

**分析类型说明**

- `attribution`: 归因分析 - 找出变化的根本原因
- `intervention`: 干预分析 - 评估优化措施的效果（**支持精确目标控制**）
- `counterfactual`: 反事实分析 - 回答"如果...会怎样"

---

### 3.2. 因果分析（流式响应）🔥 **新功能**

使用Server-Sent Events (SSE)实时推送分析进度，提供更好的用户体验。

**请求**

```http
POST /api/analyze_stream
Content-Type: application/json
```

**请求体**（与 `/api/analyze` 相同）

```json
{
  "query": "如果我想强度提升10%，应该如何调整配合比？",
  "reference_sample_index": 100,
  "observed_config": {
    "cement": 280,
    "water": 180,
    "age": 28
  }
}
```

**响应格式**（Server-Sent Events）

流式响应会实时推送以下事件：

```
data: {"type": "start", "message": "开始分析..."}

data: {"type": "progress", "message": "🔍 Router Agent 正在分析您的问题..."}

data: {"type": "progress", "message": "📋 分析类型: intervention"}

data: {"type": "progress", "message": "📊 Causal Analyst Agent 正在执行因果分析..."}

data: {"type": "progress", "message": "执行干预分析..."}

data: {"type": "progress", "message": "🔧 Optimizer Agent 正在生成优化配比..."}

data: {"type": "progress", "message": "💡 Advisor Agent 正在生成决策建议..."}

data: {"type": "result", "data": { ... 完整分析结果 ... }}

data: {"type": "end", "message": "分析完成"}
```

**事件类型**

- `start`: 开始分析
- `progress`: 进度消息（Agent执行状态、中间结果）
- `result`: 完整的分析结果（与 `/api/analyze` 响应格式相同）
- `end`: 分析完成
- `error`: 错误消息

**优势**

✅ 实时反馈：用户可以看到Agent的执行过程
✅ 更好的体验：长时间分析不会感觉"卡住"
✅ 调试友好：清晰展示每个步骤的输出

**前端使用示例**

```javascript
const response = await fetch('http://localhost:8000/api/analyze_stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: "...", reference_sample_index: 100 })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    
    const chunk = decoder.decode(value);
    const lines = chunk.split('\n\n');
    
    for (const line of lines) {
        if (line.startsWith('data: ')) {
            const data = JSON.parse(line.substring(6));
            
            if (data.type === 'progress') {
                console.log(data.message);  // 显示进度
            } else if (data.type === 'result') {
                console.log('分析完成:', data.data);
            }
        }
    }
}
```

---

### 4. 参考批次

获取典型的参考批次样本（28天龄期）。

**请求**

```http
GET /api/samples
```

**响应**

```json
{
  "samples": [
    {
      "index": 789,
      "cement": 238.0,
      "blast_furnace_slag": 0.0,
      "fly_ash": 0.0,
      "water": 185.0,
      "superplasticizer": 0.0,
      "coarse_aggregate": 1118.8,
      "fine_aggregate": 789.3,
      "age": 28,
      "concrete_compressive_strength": 17.54,
      "category": "low"
    },
    {
      "category": "medium",
      "concrete_compressive_strength": 35.2
    },
    {
      "category": "high",
      "concrete_compressive_strength": 82.6
    },
    {
      "category": "target",
      "concrete_compressive_strength": 52.3
    }
  ],
  "total_count": 1030
}
```

**样本类型**

- `low`: 低强度样本（< 20 MPa）
- `medium`: 中等强度样本（≈ 35 MPa）
- `high`: 高强度样本（> 60 MPa）
- `target`: 接近目标配合比的样本

---

### 5. 变量信息

获取因果图中所有变量的详细信息。

**请求**

```http
GET /api/variables
```

**响应**

```json
{
  "variables": {
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
  },
  "total_nodes": 9,
  "total_edges": 9,
  "data_source": "UCI Machine Learning Repository (Yeh 1998)"
}
```

---

### 6. 因果图结构

获取因果图的节点和边信息。

**请求**

```http
GET /api/graph
```

**响应**

```json
{
  "nodes": [
    "cement",
    "blast_furnace_slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
    "age",
    "concrete_compressive_strength"
  ],
  "edges": [
    {"source": "cement", "target": "concrete_compressive_strength"},
    {"source": "water", "target": "concrete_compressive_strength"},
    {"source": "age", "target": "concrete_compressive_strength"}
  ],
  "num_nodes": 9,
  "num_edges": 9
}
```

---

## 📊 数据模型

### PredictRequest

```typescript
{
  cement: number;              // 100-600 kg/m³
  blast_furnace_slag: number;  // 0-400 kg/m³
  fly_ash: number;             // 0-250 kg/m³
  water: number;               // 100-300 kg/m³
  superplasticizer: number;    // 0-40 kg/m³
  coarse_aggregate: number;    // 700-1200 kg/m³
  fine_aggregate: number;      // 500-1100 kg/m³
  age: number;                 // 1-365 天
}
```

### PredictResponse

```typescript
{
  success: boolean;
  predicted_strength: number;
  confidence_interval: {
    lower: number;
    upper: number;
  };
  interpretation: string;
  similar_samples: Array<{
    cement: number;
    water: number;
    blast_furnace_slag: number;
    actual_strength: number;
    age: number;
  }>;
  feature_weights: {
    [variable: string]: {
      name: string;
      weight_pct: number;
      causal_effect: number;
      score: number;
      direction: "正向" | "负向";
    };
  };
  error: string | null;
}
```

### QueryRequest

```typescript
{
  query: string;                    // 必填：自然语言查询
  reference_sample_index?: number;  // 可选：参考批次索引
  observed_config?: {               // 可选：用户输入的观测配比（优先级高于reference_sample_index）
    cement: number;
    blast_furnace_slag: number;
    fly_ash: number;
    water: number;
    superplasticizer: number;
    coarse_aggregate: number;
    fine_aggregate: number;
    age: number;
  };
}
```

### AnalysisResponse

```typescript
{
  success: boolean;
  analysis_type: "attribution" | "intervention" | "counterfactual";
  target_variable: string;
  routing_reasoning: string;
  causal_results: object;
  analysis_summary: string;
  recommendations: string;
  // ⭐ 新增字段（目标导向优化时返回）
  optimized_config?: {              // 优化后的配比方案
    cement: number;
    blast_furnace_slag: number;
    fly_ash: number;
    water: number;
    superplasticizer: number;
    coarse_aggregate: number;
    fine_aggregate: number;
    age: number;
    concrete_compressive_strength: number;
  };
  predicted_strength?: number;       // 优化配比的预测强度
  optimization_summary?: string;     // 优化摘要（包含目标提升vs实际提升）
  error: string | null;
}
```

---

## ⚠️ 错误处理

### 标准错误响应

```json
{
  "detail": "错误描述信息"
}
```

### 常见错误码

| 状态码 | 说明 |
|--------|------|
| 400 | 请求参数错误 |
| 404 | 资源不存在 |
| 500 | 服务器内部错误 |

### 参数验证错误示例

```json
{
  "detail": [
    {
      "loc": ["body", "cement"],
      "msg": "ensure this value is greater than or equal to 100",
      "type": "value_error.number.not_ge"
    }
  ]
}
```

---

## 💻 使用示例

### Python

```python
import requests

# 1. 强度预测
response = requests.post(
    "http://localhost:8000/api/predict",
    json={
        "cement": 380,
        "blast_furnace_slag": 100,
        "fly_ash": 50,
        "water": 170,
        "superplasticizer": 8,
        "coarse_aggregate": 1000,
        "fine_aggregate": 800,
        "age": 28
    }
)
result = response.json()
print(f"预测强度: {result['predicted_strength']:.2f} MPa")

# 2. 因果分析 - 传统方式
response = requests.post(
    "http://localhost:8000/api/analyze",
    json={
        "query": "如何提高混凝土强度？"
    }
)
result = response.json()
print(f"分析类型: {result['analysis_type']}")
print(f"建议: {result['recommendations']}")

# 3. 🎯 目标导向优化（新功能）
response = requests.post(
    "http://localhost:8000/api/analyze",
    json={
        "query": "如果我想强度提升10%，应该如何调整配合比？",
        "reference_sample_index": 100
    }
)
result = response.json()
print(f"目标提升: 10%")
print(f"预测强度: {result['predicted_strength']:.2f} MPa")
print(f"优化配比: {result['optimized_config']}")

# 4. 基于用户配比的反事实分析（新功能）
response = requests.post(
    "http://localhost:8000/api/analyze",
    json={
        "query": "如果水用量从200降到150，强度会提升多少？",
        "observed_config": {
            "cement": 164.8,
            "blast_furnace_slag": 190.0,
            "fly_ash": 148.0,
            "water": 200.0,
            "superplasticizer": 19.0,
            "coarse_aggregate": 838.0,
            "fine_aggregate": 741.0,
            "age": 30
        }
    }
)
result = response.json()
print(f"因果效应: {result['causal_results']['causal_effect']:.2f} MPa")

# 5. 🔥 数学运算支持（新功能）
response = requests.post(
    "http://localhost:8000/api/analyze",
    json={
        "query": "添加矿渣100 kg/m³，减少水泥50 kg/m³，强度会怎样？",
        "reference_sample_index": 830
    }
)
result = response.json()
print(f"多变量运算效果: {result['analysis_summary']}")
print(f"优化配比: {result['optimized_config']}")

# 6. 🔥 流式响应（新功能）
import json
response = requests.post(
    "http://localhost:8000/api/analyze_stream",
    json={
        "query": "如果我想强度提升10%，应该如何调整配合比？",
        "reference_sample_index": 100
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        line_str = line.decode('utf-8')
        if line_str.startswith('data: '):
            event = json.loads(line_str[6:])
            if event['type'] == 'progress':
                print(f"📡 {event['message']}")
            elif event['type'] == 'result':
                final_result = event['data']
                print(f"✅ 分析完成: {final_result['predicted_strength']:.2f} MPa")
```

### JavaScript (Fetch API)

```javascript
// 1. 强度预测
const predictStrength = async () => {
  const response = await fetch('http://localhost:8000/api/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      cement: 380,
      blast_furnace_slag: 100,
      fly_ash: 50,
      water: 170,
      superplasticizer: 8,
      coarse_aggregate: 1000,
      fine_aggregate: 800,
      age: 28
    })
  });
  
  const data = await response.json();
  console.log(`预测强度: ${data.predicted_strength.toFixed(2)} MPa`);
  return data;
};

// 2. 因果分析
const analyzeQuery = async (query) => {
  const response = await fetch('http://localhost:8000/api/analyze', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ query })
  });
  
  const data = await response.json();
  console.log('分析结果:', data.analysis_summary);
  return data;
};

// 3. 🎯 目标导向优化（新功能）
const optimizeWithTarget = async (targetImprovement) => {
  const response = await fetch('http://localhost:8000/api/analyze', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query: `如果我想强度提升${targetImprovement}%，应该如何调整配合比？`,
      reference_sample_index: 100
    })
  });
  
  const data = await response.json();
  console.log(`目标提升: ${targetImprovement}%`);
  console.log(`预测强度: ${data.predicted_strength.toFixed(2)} MPa`);
  console.log('优化配比:', data.optimized_config);
  return data;
};
```

### cURL

```bash
# 1. 健康检查
curl http://localhost:8000/health

# 2. 强度预测
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "cement": 380,
    "blast_furnace_slag": 100,
    "fly_ash": 50,
    "water": 170,
    "superplasticizer": 8,
    "coarse_aggregate": 1000,
    "fine_aggregate": 800,
    "age": 28
  }'

# 3. 因果分析 - 传统方式
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "query": "如何提高混凝土强度？"
  }'

# 4. 🎯 目标导向优化（新功能）
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "query": "如果我想强度提升10%，应该如何调整配合比？",
    "reference_sample_index": 100
  }'

# 5. 基于用户配比的反事实分析（新功能）
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "query": "如果水用量从200降到150，强度会提升多少？",
    "observed_config": {
      "cement": 164.8,
      "blast_furnace_slag": 190.0,
      "fly_ash": 148.0,
      "water": 200.0,
      "superplasticizer": 19.0,
      "coarse_aggregate": 838.0,
      "fine_aggregate": 741.0,
      "age": 30
    }
  }'

# 6. 🔥 数学运算支持（新功能）
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "query": "添加矿渣100 kg/m³，减少水泥50 kg/m³，强度会怎样？",
    "reference_sample_index": 830
  }'

# 7. 🔥 流式响应（新功能）
curl -X POST http://localhost:8000/api/analyze_stream \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "query": "如果我想强度提升10%，应该如何调整配合比？",
    "reference_sample_index": 100
  }'

# 8. 获取参考批次
curl http://localhost:8000/api/samples
```

---

## 🔬 技术说明

### 预测方法

本系统使用**因果干预采样 (Causal Interventional Sampling)** 方法进行预测：

1. **因果图结构**: 9个节点（8个输入+1个输出），基于真实物理因果关系
2. **干预操作**: 使用 do-operator 固定输入参数
3. **采样预测**: 从因果模型采样100次，计算均值和置信区间
4. **优势**: 
   - 可解释性强（明确因果路径）
   - 自动量化不确定性
   - 支持反事实推理

### 🎯 目标导向优化算法（新功能）

实现**精确目标控制**的智能优化：

1. **目标提取**: Router Agent从用户查询中提取目标提升百分比（如"提升10%"）
2. **因果效应分析**: Causal Analyst计算各变量的因果效应（每单位变化对强度的影响）
3. **二分搜索优化**: Optimizer Agent使用二分搜索算法寻找最优调整比例
   - **搜索范围**: 0% ~ 50%
   - **迭代次数**: 最多8次
   - **精度控制**: 目标强度的±2%误差
4. **调整策略**: 
   - 正效应变量（cement, age）→ 增加
   - 负效应变量（water）→ 减少
5. **结果验证**: 使用因果模型预测优化配比的强度，确保达到目标

**示例**：
```
用户要求: 提升10%
迭代1: scale=0.250 → 预测44.9% ❌ 过高
迭代2: scale=0.125 → 预测13.2% ✓ 接近
迭代3: scale=0.062 → 预测10.1% ✅ 达标
```

### 模型性能

基于UCI真实数据集验证：

| 指标 | 数值 | 评价 |
|------|------|------|
| R² | 0.9901 | 优秀 |
| MAE | 1.28 MPa | 高精度 |
| MAPE | 3.76% | 误差小 |

### 数据来源

- **数据集**: UCI Machine Learning Repository
- **作者**: Yeh (1998)
- **样本数**: 1030条
- **变量数**: 9个（8个输入 + 1个输出）

---

## ✨ 新功能亮点

### 🔥 智能数学计算工具（v2.1新增）

**问题**: 传统方法只支持绝对值干预（如"水用量150"），无法处理相对变化（如"增加50"）和复杂的多变量运算。

**解决方案**: 
- **Math Calculator Tool**: 专门的数学运算工具，支持加减乘除四则运算
- **智能识别**: Router Agent自动识别"增加"、"减少"、"翻倍"等自然语言表达
- **多变量协同**: 一次处理多个变量的复杂运算（如"添加矿渣100，减少水泥50"）

**支持的运算**:
| 自然语言 | 运算类型 | 处理方式 |
|---------|---------|---------|
| "增加50"、"添加100" | `add` | `新值 = 原值 + 操作数` |
| "减少30"、"降低20" | `subtract` | `新值 = 原值 - 操作数` |
| "翻倍"、"乘以2" | `multiply` | `新值 = 原值 × 操作数` |
| "减半"、"除以2" | `divide` | `新值 = 原值 ÷ 操作数` |

**效果对比**:
```
传统方法: "添加矿渣100，减少水泥50" → ❌ 理解为绝对值，矿渣=100，水泥=-50
新方法:   "添加矿渣100，减少水泥50" → ✅ 矿渣 = 190 + 100 = 290，水泥 = 162 - 50 = 112
```

### 🔥 流式响应（v2.1新增）

**问题**: 传统API在长时间分析时，用户无法看到进度，体验不佳，容易误认为"卡住"。

**解决方案**: 
- **Server-Sent Events**: 使用SSE协议实时推送Agent执行状态
- **可视化进度**: 前端显示"📡 实时分析进度"区域，展示每个步骤
- **更好体验**: 用户可看到Router、Analyst、Optimizer、Advisor各Agent的执行情况

**推送内容**:
```
📡 🔍 Router Agent 正在分析您的问题...
📡 📋 分析类型: intervention
📡 📊 Causal Analyst Agent 正在执行因果分析...
📡 执行干预分析...
📡 🔧 Optimizer Agent 正在生成优化配比...
📡   迭代 1: scale=0.250, 预测=52.91 MPa
📡   迭代 2: scale=0.125, 预测=48.81 MPa
📡 💡 Advisor Agent 正在生成决策建议...
✅ 分析完成
```

### 🎯 精确目标控制（v2.0）

**问题**: 传统方法简单地对Top变量调整10%，导致累积效应过大，无法达到用户的精确目标。

**解决方案**: 
- **智能提取目标**: 从"提升10%"、"增加5%"等自然语言中提取精确百分比
- **二分搜索优化**: 迭代调整变量比例，直到预测强度达到目标（误差≤2%）
- **多变量协同**: 同时优化Top 3有效变量，考虑变量间的协同效应

**效果对比**:
```
传统方法: 目标10% → 实际44.9% ❌ (误差+34.9%)
新方法:   目标10% → 实际10.1% ✅ (误差+0.1%)
```

### 📝 用户自定义配比输入（v2.0）

**功能**: 用户可以直接输入任意配比进行反事实分析，无需选择预设的参考批次。

**优势**:
- ✅ 灵活性更高（支持任意配比组合）
- ✅ 自动预测基准强度（系统自动补全缺失的强度值）
- ✅ 实时分析（无需等待数据库查询）

### 📊 完整参考批次信息（v2.0）

**显示内容**: 每个参考批次卡片显示完整的8个配比参数
- 水泥、高炉矿渣、粉煤灰
- 水、高效减水剂
- 粗骨料、细骨料
- 龄期 + 强度

**布局优化**: 2列网格布局，信息密度提升50%

---

## 📚 相关资源

- **Web界面**: http://localhost:8000
- **Swagger文档**: http://localhost:8000/docs
- **ReDoc文档**: http://localhost:8000/redoc
- **源代码**: `api_server.py` | `src/causal_agent_system.py`
- **测试脚本**: `test_optimizer.py`

---

## 📝 版本更新日志

### v2.1.0 (2025-11-05) 🔥

**重大更新：数学计算工具 + 流式响应**

**新增功能**:
- 🔥 **Math Calculator Tool**: 智能数学运算支持（加减乘除），自动处理多变量复杂调整
  - ✅ "水泥增加50" → `add` 操作
  - ✅ "添加矿渣100，减少水泥50" → 多变量协同运算
  - ✅ "龄期翻倍" → `multiply` 操作
- 🔥 **流式响应API** (`/api/analyze_stream`): 使用Server-Sent Events实时推送Agent执行进度
  - ✅ 实时反馈：用户可看到每个Agent的执行状态
  - ✅ 更好体验：长时间分析不会"卡住"
  - ✅ 调试友好：清晰展示每步输出

**技术改进**:
- 🔧 新增 `math_calculator_tool`: 专门处理变量的加减乘除运算
- 🔧 Router Agent增强：智能识别运算类型（add/subtract/multiply/divide）并提取操作数
- 🔧 Causal Analyst Agent优化：集成数学计算工具，支持单变量和多变量运算
- 🔧 UI优化：移除冗余的分隔线，简化输出，决策建议不再重复显示

**API变更**:
- 新增 `POST /api/analyze_stream` 端点（流式响应）
- Router支持提取 `operation` 和 `operand` 字段（数学运算参数）
- Router支持提取 `interventions` 列表（多变量运算）

**用户体验提升**:
- 前端实时显示分析进度（📡 实时分析进度区域）
- 支持更自然的问题表达（"增加"、"减少"、"翻倍"等口语化表达）
- 自动识别并执行复杂的多变量数学运算

---

### v2.0.0 (2025-11-05) 🎯

**重大更新：精确目标控制优化**

**新增功能**:
- ✨ **目标导向优化**: 支持精确控制强度提升目标（如"提升10%"），使用二分搜索算法自动生成最优配比
- ✨ **用户自定义配比输入**: 用户可直接输入任意配比进行反事实分析，系统自动预测基准强度
- ✨ **完整参考批次显示**: 参考批次卡片显示全部8个配比参数，2列网格布局

**技术改进**:
- 🔧 Router Agent增强：智能提取目标提升百分比
- 🔧 Optimizer Agent重写：二分搜索算法实现精确优化（8次迭代，误差≤2%）
- 🔧 Causal Analyst Agent优化：支持用户输入配比的反事实分析

**API变更**:
- `POST /api/analyze` 新增 `observed_config` 字段（用户自定义配比）
- 响应新增 `optimized_config`、`predicted_strength`、`optimization_summary` 字段

**性能指标**:
- 目标精确度：±2% 误差范围内
- 优化速度：8次迭代内收敛
- 适用范围：5%-50%强度提升

---

### v1.0.0 (2025-11-04)

**初始版本发布**

- 基础强度预测功能
- 三种因果分析（归因、干预、反事实）
- 特征权重可视化
- Web交互界面
- RESTful API

---

## 📄 许可证

MIT License

---

## 👥 联系方式

如有问题或建议，请联系开发团队。

