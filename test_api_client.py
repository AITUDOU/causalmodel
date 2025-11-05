"""
API 测试客户端
用于测试 FastAPI 后端服务
"""

import requests
import json
from typing import Optional

# API 基础URL
BASE_URL = "http://localhost:8000"


def test_health():
    """测试健康检查"""
    print("\n" + "="*80)
    print("🏥 测试健康检查")
    print("="*80)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")


def test_get_samples():
    """测试获取参考批次"""
    print("\n" + "="*80)
    print("📦 获取参考批次列表")
    print("="*80)
    
    response = requests.get(f"{BASE_URL}/api/samples")
    data = response.json()
    
    print(f"\n共有 {data['total_count']} 条数据")
    print(f"推荐的参考批次：\n")
    
    for sample in data['samples']:
        category_names = {
            'low': '低强度批次',
            'medium': '中等强度批次',
            'high': '高强度批次',
            'target': '图片配合比'
        }
        print(f"【{category_names.get(sample['category'], '未知')}】")
        print(f"  索引: {sample['index']}")
        print(f"  水胶比: {sample['water_binder_ratio']:.3f}")
        print(f"  水泥: {sample['cement_content']:.0f} kg/m³")
        print(f"  砂率: {sample['sand_rate']:.3f}")
        print(f"  28天强度: {sample['strength_28d_mpa']:.1f} MPa\n")
    
    return data['samples']


def test_analyze(query: str, reference_sample_index: Optional[int] = None):
    """测试因果分析"""
    print("\n" + "="*80)
    print(f"🔍 执行分析查询")
    print("="*80)
    print(f"查询: {query}")
    if reference_sample_index is not None:
        print(f"参考批次: #{reference_sample_index}")
    print()
    
    payload = {
        "query": query,
        "reference_sample_index": reference_sample_index
    }
    
    response = requests.post(f"{BASE_URL}/api/analyze", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        
        print("="*80)
        print("📊 分析结果")
        print("="*80)
        print(f"\n✅ 成功: {data['success']}")
        print(f"🎯 分析类型: {data['analysis_type']}")
        print(f"📈 目标变量: {data['target_variable']}")
        print(f"\n💡 推理过程:\n{data['routing_reasoning']}")
        print(f"\n📝 分析摘要:\n{data['analysis_summary']}")
        print(f"\n💡 决策建议:\n{data['recommendations']}")
        
        return data
    else:
        print(f"❌ 请求失败: {response.status_code}")
        print(f"错误信息: {response.text}")
        return None


def test_get_variables():
    """测试获取变量列表"""
    print("\n" + "="*80)
    print("📋 获取可用变量列表")
    print("="*80)
    
    response = requests.get(f"{BASE_URL}/api/variables")
    data = response.json()
    
    print(f"\n因果图信息:")
    print(f"  节点数: {data['total_nodes']}")
    print(f"  边数: {data['total_edges']}")
    
    print(f"\n可用变量分类:\n")
    
    for category, variables in data['variables'].items():
        category_names = {
            'root_nodes': '根节点（材料选择）',
            'controllable_params': '可控参数（配合比）',
            'quality_indicators': '质量指标',
            'target_variables': '目标变量'
        }
        print(f"【{category_names.get(category, category)}】")
        for var, desc in variables.items():
            print(f"  • {var}: {desc}")
        print()


def main():
    """主测试流程"""
    print("="*80)
    print("🧪 因果分析智能体系统 - API 测试")
    print("="*80)
    
    try:
        # 1. 健康检查
        test_health()
        
        # 2. 获取变量列表
        test_get_variables()
        
        # 3. 获取参考批次
        samples = test_get_samples()
        
        # 4. 测试归因分析（不需要参考批次）
        print("\n" + "🔍"*40)
        print("测试场景 1：归因分析")
        print("🔍"*40)
        test_analyze("为什么28天强度下降了？")
        
        input("\n按回车继续下一个测试...")
        
        # 5. 测试干预分析（不需要参考批次）
        print("\n" + "🔧"*40)
        print("测试场景 2：干预分析")
        print("🔧"*40)
        test_analyze("如何提高28天强度？")
        
        input("\n按回车继续下一个测试...")
        
        # 6. 测试反事实分析（使用低强度样本）
        print("\n" + "🔮"*40)
        print("测试场景 3：反事实分析")
        print("🔮"*40)
        low_sample = samples[0]  # 使用低强度样本
        test_analyze(
            f"如果水胶比从{low_sample['water_binder_ratio']:.2f}降到0.40，28天强度会提升多少？",
            reference_sample_index=low_sample['index']
        )
        
        print("\n" + "="*80)
        print("✅ 所有测试完成！")
        print("="*80)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ 无法连接到服务器")
        print("请先启动 API 服务器: python3 api_server.py")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

