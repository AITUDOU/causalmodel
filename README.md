# 混凝土强度因果分析系统 API 文档

基于因果推断的混凝土配合比智能分析系统，使用自然语言即可完成因果分析。

## 📋 目录

- [快速开始](#快速开始)
- [核心API](#核心api)
  - [因果分析 - 完整响应](#因果分析---完整响应)
  - [因果分析 - 流式响应](#因果分析---流式响应-推荐)
  - [使用场景对比](#使用场景对比)
- [其他API](#其他api)
- [常见问题](#常见问题)

---

## 🚀 快速开始

### 1. 启动服务器

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量（创建.env文件）
OPENAI_API_KEY=your_api_key

# 启动服务
python3 api_server.py
```

服务器地址：http://localhost:8000

### 2. 快速测试

访问 http://localhost:8000/test 打开API测试工具，点击"加载示例"即可快速体验。

---

## 核心API

系统提供两种因果分析API：**完整响应**和**流式响应**。推荐使用流式响应，可实时查看分析进度。

### 因果分析 - 完整响应

等待分析完成后返回完整结果。适合后台批处理场景。

**端点**: `POST /api/analyze`

#### 📋 请求参数

```typescript
{
  query: string;                    // 必填：自然语言问题
  observed_config?: {               // 可选：自定义配比
    cement, water, age, ...         // 8个配比参数
  };
  reference_sample_index?: number;  // 可选：参考批次（默认样本）
  adjust_factors?: string[];        // 可选：指定调整的变量
  target_strength?: number;         // 可选：目标强度 (MPa)
}
```

#### 🎯 使用示例

**示例1：纯预测**（只想知道强度，不需要优化建议）

```json
{
  "query": "预测一下强度",
  "observed_config": {
    "cement": 300,
    "water": 185,
    "age": 28,
    ...
  }
}
```

**示例2：探索性问题**（想知道哪些因素影响强度）

```json
{
  "query": "如何提高强度？"
}
```
→ 系统会分析所有因素的因果效应，但**不会**生成具体配比（因为没有提供基准）

**示例3：优化配比**（想得到具体的优化方案）

```json
{
  "query": "如何达到45 MPa？",
  "observed_config": {
    "cement": 300,
    "water": 185,
    "age": 28,
    ...
  },
  "adjust_factors": ["cement", "fly_ash"],  // 只调整这两个变量
  "target_strength": 45                      // 目标强度
}
```
→ 系统会生成具体的优化配比方案

**示例4：反事实分析**

```json
{
  "query": "水泥增加50，强度会怎样？",
  "reference_sample_index": 100
}
```

#### 📤 响应格式

```json
{
  "success": true,
  "analysis_type": "intervention | attribution | counterfactual",
  "causal_results": { ... },           // 因果分析数值结果
  "analysis_summary": "...",           // 分析摘要
  "recommendations": "...",            // LLM生成的建议
  "optimized_config": { ... },         // 优化配比（有明确目标时返回）
  "predicted_strength": 45.0,          // 预测强度（有优化时返回）
  "base_sample_info": { ... },         // 基准样本信息（使用默认样本时返回）
  "error": null
}
```

---

### 因果分析 - 流式响应 ✨ **推荐**

实时推送分析进度，推荐用于前端界面。

**端点**: `POST /api/analyze_stream`

#### 📋 请求参数

与 `/api/analyze` 完全相同。

#### 📡 响应格式 (Server-Sent Events)

流式返回多个事件：

```
data: {"type": "start", "message": "开始分析..."}

data: {"type": "progress", "message": "🔍 Router Agent 正在分析..."}

data: {"type": "progress", "message": "📊 执行因果分析..."}

data: {"type": "progress", "message": "🔧 生成优化配比..."}

data: {"type": "result", "data": { 完整结果 }}

data: {"type": "end", "message": "分析完成"}
```

#### 🔧 前端使用

```javascript
const response = await fetch('http://localhost:8000/api/analyze_stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: "如何达到45 MPa？",
    observed_config: {...}
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { value, done } = await reader.read();
  if (done) break;
  
  const chunk = decoder.decode(value);
  for (const line of chunk.split('\n\n')) {
    if (line.startsWith('data: ')) {
      const event = JSON.parse(line.slice(6));
      
      if (event.type === 'progress') {
        console.log(event.message);  // 显示进度
      } else if (event.type === 'result') {
        console.log('完成:', event.data);
      }
    }
  }
}
```

---

### 使用场景对比

| 场景 | query | observed_config | adjust_factors | target_strength | 结果 |
|-----|-------|----------------|----------------|----------------|-----|
| **纯预测** | "预测强度" | ✅ | ❌ | ❌ | 只返回预测值 |
| **探索分析** | "如何提高强度？" | ❌ | ❌ | ❌ | 返回因素分析，无具体配比 |
| **精确优化** | "如何达到45 MPa？" | ✅ | ✅ | ✅ | 返回优化配比方案 |
| **反事实** | "水泥增加50" | ❌ | ❌ | ❌ | 使用默认样本分析效果 |

## 其他API

系统还提供以下辅助API：

### 1. 强度预测
- **端点**: `POST /api/predict`
- **用途**: 直接预测给定配比的强度（无需自然语言）
- **文档**: http://localhost:8000/docs

### 2. 配比优化（GUI专用）
- **端点**: `POST /api/optimize`
- **用途**: 结构化参数优化，适合GUI操作
- **说明**: 如果用API，推荐使用`/api/analyze`（更灵活）

### 3. 参考批次
- **端点**: `GET /api/samples`
- **用途**: 获取预设的参考配比样本

### 4. 变量信息
- **端点**: `GET /api/variables`
- **用途**: 查询8个配比参数的范围和说明

### 5. 因果图结构
- **端点**: `GET /api/graph`
- **用途**: 获取因果图的节点和边

**详细文档**: 访问 http://localhost:8000/docs 查看完整API文档（Swagger UI）

---

## 常见问题

### 1. 什么情况下使用`/api/analyze` vs `/api/analyze_stream`？

- **批处理/后台任务**: 使用 `/api/analyze`
- **前端界面/需要实时反馈**: 使用 `/api/analyze_stream` ✨

### 2. 为什么没有返回`optimized_config`？

系统只在以下情况生成优化配比：
1. 提供了基准配比（`observed_config`或`reference_sample_index`）
2. 有明确的目标（`target_strength`或查询中提到目标强度/提升比例）

**探索性问题**（如"如何提高强度？"）只返回因素分析，不生成具体配比。

### 3. `adjust_factors`的作用是什么？

指定系统**只能调整**这些变量进行优化。例如：

```json
{
  "query": "如何达到45 MPa？",
  "observed_config": {...},
  "adjust_factors": ["cement", "fly_ash"]
}
```

→ 系统只会调整水泥和粉煤灰，其他参数保持不变。

### 4. 什么是`base_sample_info`？

当您没有提供`observed_config`或`reference_sample_index`时，系统会自动使用默认样本作为基准。`base_sample_info`会告诉您使用的是哪个样本的完整配比信息。

### 5. 分析耗时多久？

- **纯预测/反事实分析**: 5-15秒
- **干预分析**: 10-30秒（因果效应计算）
- **优化配比**: 20-60秒（迭代优化）

建议设置超时时间≥180秒（3分钟）。

### 6. 如何测试API？

访问 http://localhost:8000/test 使用可视化测试工具，支持：
- 一键加载示例
- 实时流式进度显示
- 多选框选择调整变量

### 7. 支持哪些自然语言表达？

**预测类**:
- "预测强度"、"这个配比强度是多少"

**探索类**:
- "如何提高强度？"、"哪些因素影响强度？"

**优化类**:
- "如何达到45 MPa？"、"提升10%应该怎么调？"

**反事实类**:
- "水泥增加50"、"添加矿渣100，减少水泥50"

系统会自动识别您的意图！

---

## 版本历史

### v2.3.0 (2025-11-09) - 当前版本
- ✅ API参数精确控制（`adjust_factors`, `target_strength`）
- ✅ 智能意图识别（纯预测 vs 优化）
- ✅ API测试工具（`/test`）
- ✅ 基准样本信息返回（`base_sample_info`）
- 🐛 修复除零错误、参数优先级问题

### v2.2.0 (2025-11-06)
- GUI驱动的配比优化API（`/api/optimize`）

### v2.1.0 (2025-11-05)
- 流式响应（`/api/analyze_stream`）
- 数学运算支持（"增加50"、"翻倍"等）

### v2.0.0 (2025-11-05)
- 精确目标控制（二分搜索优化）
- 用户自定义配比输入

### v1.0.0 (2025-11-04)
- 初始版本发布

---

## 🔗 资源链接

- **主界面**: http://localhost:8000
- **API测试工具**: http://localhost:8000/test
- **API文档**: http://localhost:8000/docs
- **源代码**: 
  - `api_server.py` - 后端服务
  - `src/causal_agent_system.py` - 智能体系统
  - `test_analyze_api.html` - 测试工具

---

## 📊 配比参数范围

| 参数 | 中文名 | 范围 | 单位 |
|------|--------|------|------|
| `cement` | 水泥 | 100-600 | kg/m³ |
| `blast_furnace_slag` | 高炉矿渣 | 0-400 | kg/m³ |
| `fly_ash` | 粉煤灰 | 0-250 | kg/m³ |
| `water` | 水 | 100-300 | kg/m³ |
| `superplasticizer` | 减水剂 | 0-40 | kg/m³ |
| `coarse_aggregate` | 粗骨料 | 700-1200 | kg/m³ |
| `fine_aggregate` | 细骨料 | 500-1100 | kg/m³ |
| `age` | 龄期 | 1-365 | 天 |

---

## 💻 快速示例

### Python

```python
import requests

# 1. 探索性分析
response = requests.post("http://localhost:8000/api/analyze", 
    json={"query": "如何提高强度？"})

# 2. 精确优化
response = requests.post("http://localhost:8000/api/analyze",
    json={
        "query": "如何达到45 MPa？",
        "observed_config": {
            "cement": 300, "water": 185, "age": 28, ...
        },
        "adjust_factors": ["cement", "fly_ash"],
        "target_strength": 45
    })

# 3. 流式响应（推荐）
response = requests.post("http://localhost:8000/api/analyze_stream",
    json={"query": "提升10%应该怎么调？", "reference_sample_index": 100},
    stream=True)

for line in response.iter_lines():
    if line and line.startswith(b'data: '):
        event = json.loads(line[6:])
        if event['type'] == 'progress':
            print(event['message'])
```

### cURL

```bash
# 探索性分析
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "如何提高强度？"}'

# 精确优化
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "query": "如何达到45 MPa？",
    "observed_config": {"cement": 300, "water": 185, "age": 28, ...},
    "adjust_factors": ["cement", "fly_ash"],
    "target_strength": 45
  }'

# 流式响应（加 -N 参数）
curl -N -X POST http://localhost:8000/api/analyze_stream \
  -H "Content-Type: application/json" \
  -d '{"query": "提升10%", "reference_sample_index": 100}'
```

---

## 🔬 技术架构

### 核心技术

- **因果推断**: 基于DoWhy和GCM，支持干预、归因、反事实分析
- **智能体系统**: LangGraph多Agent协作（Router → Analyst → Optimizer → Advisor）
- **优化算法**: 二分搜索 + 因果效应分析，精度±2%
- **数据来源**: UCI真实数据集（1030样本，R²=0.99）

### 模型性能

| 指标 | 数值 |
|------|------|
| R² | 0.9901 |
| MAE | 1.28 MPa |
| MAPE | 3.76% |

---

## ✨ 新功能亮点

### 🎯 GUI驱动的智能配比优化（v2.2新增）

**问题**: 传统的自然语言分析方式对于GUI操作不够友好，用户需要构造复杂的问句，且系统会自动选择要调整的变量，无法精确控制。

**解决方案**: 
- **专用API端点** (`/api/optimize`): 接收结构化参数，直接返回优化结果
- **三步骤交互流程**: 基准配比 → 预测强度 → 选择因素 → 优化结果
- **用户精确控制**: 用户通过复选框明确指定哪些因素可以调整
- **动态范围调整**: 基于基准强度自动设置目标强度的合理范围

**工作流程**:
```
1. 用户输入基准配比（或选择预设）→ 系统预测基准强度（如30.5 MPa）
2. 用户滑动选择目标强度（滑块范围自动设为30-60 MPa）
3. 用户勾选允许调整的因素（如：☑️ 水泥、☑️ 粉煤灰）
4. 点击"开始智能优化" → 系统只调整选中的因素，达到目标强度
```

**技术特点**:
- **双重优化路径**: 
  - 自然语言路径：`/api/analyze` - 适合智能问答
  - GUI驱动路径：`/api/optimize` - 适合界面操作
- **精确因素控制**: 只调整用户指定的变量（如只调整水泥+粉煤灰）
- **结构化响应**: 返回调整详情、完整配比、工程建议
- **快速响应**: 10-20秒（跳过自然语言理解）

**效果对比**:
```
传统方式 (/api/analyze):
- 输入: "我想强度达到45 MPa，水泥和粉煤灰应该怎么调？"
- 问题: 需要构造复杂问句，系统可能自动选择Top 3变量（不一定是用户想要的）
- 耗时: 15-30秒（含LLM处理）

新方式 (/api/optimize):
- 输入: {base_config: {...}, target_strength: 45, adjust_factors: ["cement", "fly_ash"]}
- 优势: 结构化参数，精确指定调整因素，返回完整调整详情
- 耗时: 10-20秒（无LLM，纯因果推断）
```

**前端交互**:
- ✅ 步骤1：预设配比按钮（C30/C40/C50/低水胶比）+ 手动输入
- ✅ 步骤2：目标强度滑块（带实时数值显示）
- ✅ 步骤3：8个因素的复选框（带emoji图标）
- ✅ 结果展示：基准vs优化对比表格 + 完整配比卡片

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

- **Web主界面**: http://localhost:8000
- **API测试工具**: http://localhost:8000/test 🔥 **v2.3新增**
- **Swagger文档**: http://localhost:8000/docs
- **ReDoc文档**: http://localhost:8000/redoc
- **源代码**: 
  - 后端服务：`api_server.py`
  - 智能体系统：`src/causal_agent_system.py`
  - 测试工具：`test_analyze_api.html`
- **测试脚本**: `test_optimizer.py`

---

## 📝 版本更新日志

### v2.3.0 (2025-11-09) 🎯

**重大更新：API精确控制 + 智能意图识别 + 测试工具**

**新增功能**:
- 🎯 **API参数精确控制**: `/api/analyze` 新增 `adjust_factors` 和 `target_strength` 参数
  - ✅ **优先级机制**: API参数优先于LLM解析结果
  - ✅ **精确控制**: 明确指定哪些变量可调整、目标强度是多少
  - ✅ **最佳实践**: 结合自然语言+结构化参数，灵活性与精确性兼得
- 🧪 **API测试工具**: 新增 `/test` 端点，提供可视化测试界面
  - ✅ **流式分析按钮**: 实时查看Agent执行进度（推荐使用）
  - ✅ **三标签页设计**: 基础参数、观测配比、高级选项
  - ✅ **多选框控制**: 可视化选择要调整的变量
  - ✅ **一键示例**: 快速加载完整测试用例
- 🤖 **智能意图识别**: 自动区分纯预测vs优化需求
  - ✅ **纯预测关键词**: "预测"、"预报"、"强度是多少"、"能达到"
  - ✅ **优化关键词**: "优化"、"提升"、"改进"、"调整"、"增加"、"降低"
  - ✅ **智能判断**: "预测一下强度" → 只返回预测值，不进行优化
  - ✅ **避免过度优化**: 用户没要求时不自动优化配比

**Bug修复**:
- 🐛 修复API参数被LLM解析结果覆盖的问题
- 🐛 修复纯预测查询被误判为优化的问题
- 🐛 修复变量值为0时除以零错误（如 fly_ash=0）
- 🐛 修复流式响应中字符串拼接错误

**用户体验优化**:
- ⏱️ 默认超时时间：60秒 → 180秒（3分钟）
- 📊 流式进度显示：实时展示Router、Analyst、Optimizer、Advisor执行状态
- 💡 超时错误提示：详细说明和解决建议
- 🎨 变量值为0时显示绝对变化量（如 "+45.3 kg/m³"）而非百分比

**技术改进**:
- 🔧 Router Agent参数优先级：`api_specified_variables` > `parsed_specified_variables`
- 🔧 Router Agent参数优先级：`api_target_value` > `parsed_target_value`
- 🔧 Optimizer Agent意图识别：检测纯预测关键词，避免不必要的优化
- 🔧 Optimizer Agent除零保护：安全处理变量值为0的情况
- 🔧 测试工具实现：完整的HTML+JavaScript前端，支持SSE流式响应

**API变更**:
- 新增 `GET /test` 端点（API测试工具）
- `QueryRequest` 新增字段：
  - `adjust_factors`: 可选，要调整的变量列表
  - `target_strength`: 可选，目标强度值
- Router Agent增强：优先使用API传入的参数

**使用场景对比**:

**场景1：纯预测** ✅
```json
{
  "query": "预测一下强度",
  "observed_config": {...}
}
→ 只返回当前配比的预测强度（不进行优化）
```

**场景2：自然语言优化** 
```json
{
  "query": "如何提升强度到45 MPa？",
  "observed_config": {...}
}
→ LLM自动选择Top 3变量进行优化
```

**场景3：精确控制优化** 🎯 **推荐**
```json
{
  "query": "如何提升强度到45 MPa？",
  "observed_config": {...},
  "adjust_factors": ["cement", "fly_ash"],
  "target_strength": 45
}
→ 只调整指定的变量，达到指定目标
```

**性能指标**:
- 响应时间：10-30秒（视分析类型而定）
- 流式延迟：< 1秒（实时推送进度）
- 参数优先级：100%生效
- 意图识别准确率：> 95%

---

### v2.2.0 (2025-11-06) 🎯

**重大更新：GUI驱动的智能配比优化**

**新增功能**:
- 🎯 **智能配比优化API** (`/api/optimize`): 专为GUI界面设计的直接优化端点
  - ✅ 三步骤工作流：设置基准配比 → 预测基准强度 → 选择调整因素 → 获得优化方案
  - ✅ 用户精确控制：只调整用户勾选的因素（如水泥+粉煤灰）
  - ✅ 动态目标范围：基于基准强度智能调整目标强度滑块范围
  - ✅ 结构化响应：返回完整的调整详情、优化配比、工程建议
- 🎨 **前端UI重设计**: 因果分析页面全新交互体验
  - ✅ 步骤1：预设配比/手动输入 → 预测基准强度
  - ✅ 步骤2：滑块选择目标强度（范围自动适配）
  - ✅ 步骤3：多选框勾选要调整的因素
  - ✅ 结果展示：基准vs优化对比、调整详情表格

**技术改进**:
- 🔧 新增 `OptimizeRequest` 和 `OptimizeResponse` Pydantic模型
- 🔧 二分搜索算法优化：只调整用户指定的因素，最多10次迭代
- 🔧 因果分析系统增强：支持 `specified_variables` 和 `target_value`
- 🔧 Router Agent改进：识别用户指定的调整变量和目标强度

**API变更**:
- 新增 `POST /api/optimize` 端点（GUI驱动优化）
- `CausalAnalysisState` 新增 `specified_variables` 和 `target_value` 字段
- Optimizer Agent优先使用用户指定的变量进行优化

**用户体验提升**:
- 更直观：可视化的三步骤引导流程
- 更快速：10-20秒完成优化（无需自然语言处理）
- 更精确：用户完全控制哪些因素可以调整
- 更实用：显示完整的调整详情和工程建议

**性能指标**:
- 响应时间：10-20秒
- 精度控制：目标强度±2%误差
- 支持因素：8个配比参数任意组合

---

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

