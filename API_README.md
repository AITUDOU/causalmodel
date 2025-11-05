# 混凝土强度因果分析系统 API 文档

基于因果推断的混凝土配合比智能分析系统，提供强度预测、因果分析、权重可视化等功能。

## 📋 目录

- [快速开始](#快速开始)
- [API端点总览](#api端点总览)
- [详细接口文档](#详细接口文档)
  - [1. 健康检查](#1-健康检查)
  - [2. 强度预测](#2-强度预测)
  - [3. 因果分析](#3-因果分析)
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
| POST | `/api/analyze` | 因果分析（智能问答） |
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

使用自然语言提问，智能体自动判断分析类型并给出结果。

**请求**

```http
POST /api/analyze
Content-Type: application/json
```

**请求体**

```json
{
  "query": "如果水用量从200降到150 kg/m³，强度会提升多少？",
  "reference_sample_index": 100
}
```

**参数说明**

- `query` (必填): 自然语言问题
- `reference_sample_index` (可选): 参考批次索引，反事实分析时需要

**支持的问题类型**

| 分析类型 | 示例问题 |
|---------|---------|
| 归因分析 | "为什么强度下降了？" |
| 干预分析 | "如何提高混凝土强度？" |
| 反事实分析 | "如果水胶比降到0.43，强度会提升多少？" |

**响应**

```json
{
  "success": true,
  "analysis_type": "intervention",
  "target_variable": "concrete_compressive_strength",
  "routing_reasoning": "用户询问优化措施，这是典型的干预分析场景...",
  "causal_results": {
    "top_factors": [
      {
        "variable": "water",
        "effect": -0.25,
        "confidence": [0.91, 0.95]
      }
    ]
  },
  "analysis_summary": "根据因果分析，降低水用量可显著提高强度...",
  "recommendations": "建议采取以下措施：\n1. 优化水胶比至0.40-0.45\n2. ...",
  "error": null
}
```

**分析类型说明**

- `attribution`: 归因分析 - 找出变化的根本原因
- `intervention`: 干预分析 - 评估优化措施的效果
- `counterfactual`: 反事实分析 - 回答"如果...会怎样"

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
  query: string;
  reference_sample_index?: number;
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

# 2. 因果分析
response = requests.post(
    "http://localhost:8000/api/analyze",
    json={
        "query": "如何提高混凝土强度？"
    }
)
result = response.json()
print(f"分析类型: {result['analysis_type']}")
print(f"建议: {result['recommendations']}")
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

# 3. 因果分析
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "query": "如何提高混凝土强度？"
  }'

# 4. 获取参考批次
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

## 📚 相关资源

- **Web界面**: http://localhost:8000
- **Swagger文档**: http://localhost:8000/docs
- **ReDoc文档**: http://localhost:8000/redoc
- **源代码**: `api_server.py`

---

## 📄 许可证

MIT License

---

## 👥 联系方式

如有问题或建议，请联系开发团队。

