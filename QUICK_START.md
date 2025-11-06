# 🚀 快速启动指南 - 智能优化功能

## 启动服务

```bash
cd /Users/superkang/Desktop/causalmodel
python api_server.py
```

服务启动后访问：**http://localhost:8000**

## 使用步骤

### 1️⃣ 设置基准配比
- 点击"因果分析"标签页
- 选择预设配比（如"C30 常规配合比"）或手动输入
- 点击 **"🔮 预测基准强度"** 按钮
- 等待3-5秒，系统显示当前配比的预测强度

### 2️⃣ 设置目标强度
- 在步骤2中，滑动选择您希望达到的强度
- 滑块范围会根据基准强度自动调整

### 3️⃣ 选择调整因素
- 勾选您允许系统调整的配比参数
- 建议选择2-3个因素（如：水泥 + 粉煤灰）

### 4️⃣ 开始优化
- 点击 **"🚀 开始智能优化"** 按钮
- 等待10-20秒，系统自动计算最优配比
- 查看优化结果和建议

## 示例场景

### 场景1：提升C30到C40
```
基准配比：C30预设
基准强度：30.5 MPa
目标强度：40 MPa
调整因素：水泥 + 减水剂
预期结果：约10-15秒得到优化方案
```

### 场景2：达到高强混凝土
```
基准配比：C40预设
基准强度：41.2 MPa
目标强度：55 MPa
调整因素：水泥 + 粉煤灰 + 水
预期结果：约15-20秒得到优化方案
```

## API测试

### 使用curl测试
```bash
curl -X POST "http://localhost:8000/api/optimize" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

## 故障排除

### 问题1：预测失败
- 检查模型是否已训练：`ls models/causal_model.pkl`
- 如果没有，运行：`python train_real_model.py`

### 问题2：优化速度慢
- 正常情况：10-20秒
- 如果超过30秒，检查网络连接和OpenAI API

### 问题3：结果不合理
- 检查目标强度是否过高或过低
- 尝试增加可调整的因素数量
- 确保基准配比本身合理

## 注意事项

1. **基准强度预测**：必须先预测基准强度才能进行优化
2. **合理目标**：目标强度建议在基准强度+5到+20 MPa之间
3. **因素选择**：建议选择2-3个因素，太少可能无法达到目标，太多可能导致过度调整
4. **工程验证**：优化结果仅供参考，实际应用需要试配验证

## 技术支持

- 查看详细文档：`OPTIMIZATION_UPDATE_SUMMARY.md`
- API文档：http://localhost:8000/docs
- 问题反馈：GitHub Issues

---

**祝您使用愉快！** 🎉

