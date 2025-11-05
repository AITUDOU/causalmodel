"""
完整的API接口测试脚本
测试所有6个API端点
"""

import requests
import json
import time

API_BASE = 'http://localhost:8000'

def print_separator(title):
    """打印分隔符"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def test_health():
    """测试1: 健康检查"""
    print_separator("测试 1/6: GET /health - 健康检查")
    
    try:
        response = requests.get(f'{API_BASE}/health', timeout=5)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ 健康检查通过")
            print(f"  • 服务状态: {data['status']}")
            print(f"  • 模型已加载: {data['model_loaded']}")
            print(f"  • 因果图节点数: {data['graph_nodes']}")
            print(f"  • 因果图边数: {data['graph_edges']}")
            print(f"  • 数据样本数: {data['data_samples']}")
            return True
        else:
            print(f"❌ 测试失败: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return False


def test_variables():
    """测试2: 获取变量信息"""
    print_separator("测试 2/6: GET /api/variables - 获取变量信息")
    
    try:
        response = requests.get(f'{API_BASE}/api/variables', timeout=5)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ 变量信息获取成功")
            print(f"  • 节点数: {data['total_nodes']}")
            print(f"  • 边数: {data['total_edges']}")
            print(f"  • 数据来源: {data['data_source']}")
            print("\n  材料变量:")
            for var, desc in list(data['variables']['materials'].items())[:3]:
                print(f"    - {var}: {desc}")
            return True
        else:
            print(f"❌ 测试失败: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return False


def test_graph():
    """测试3: 获取因果图结构"""
    print_separator("测试 3/6: GET /api/graph - 获取因果图结构")
    
    try:
        response = requests.get(f'{API_BASE}/api/graph', timeout=5)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ 因果图结构获取成功")
            print(f"  • 节点数: {data['num_nodes']}")
            print(f"  • 边数: {data['num_edges']}")
            print(f"\n  节点列表: {', '.join(data['nodes'][:5])}...")
            print(f"\n  边示例:")
            for edge in data['edges'][:3]:
                print(f"    {edge['source']} → {edge['target']}")
            return True
        else:
            print(f"❌ 测试失败: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return False


def test_samples():
    """测试4: 获取参考批次"""
    print_separator("测试 4/6: GET /api/samples - 获取参考批次")
    
    try:
        response = requests.get(f'{API_BASE}/api/samples', timeout=10)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ 参考批次获取成功")
            print(f"  • 样本数: {len(data['samples'])}")
            print(f"  • 总记录数: {data['total_count']}")
            print("\n  样本详情:")
            for sample in data['samples']:
                print(f"    [{sample['category']}] "
                      f"水泥:{sample['cement']:.0f}, "
                      f"水:{sample['water']:.0f}, "
                      f"强度:{sample['concrete_compressive_strength']:.1f} MPa")
            return True
        else:
            print(f"❌ 测试失败: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return False


def test_predict():
    """测试5: 强度预测"""
    print_separator("测试 5/6: POST /api/predict - 强度预测")
    
    # 测试数据：C40配合比
    test_params = {
        "cement": 380,
        "blast_furnace_slag": 100,
        "fly_ash": 50,
        "water": 170,
        "superplasticizer": 8,
        "coarse_aggregate": 1000,
        "fine_aggregate": 800,
        "age": 28
    }
    
    print("输入参数:")
    for key, value in test_params.items():
        print(f"  • {key}: {value}")
    
    try:
        response = requests.post(
            f'{API_BASE}/api/predict',
            json=test_params,
            timeout=60  # 预测可能需要较长时间
        )
        print(f"\n状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ 预测成功")
            print(f"\n  🎯 预测强度: {data['predicted_strength']:.2f} MPa")
            print(f"  📊 置信区间: [{data['confidence_interval']['lower']:.2f}, "
                  f"{data['confidence_interval']['upper']:.2f}] MPa")
            
            # 显示权重信息
            if data.get('feature_weights'):
                print("\n  📊 影响权重 (Top 5):")
                weights = sorted(
                    data['feature_weights'].items(),
                    key=lambda x: x[1]['weight_pct'],
                    reverse=True
                )[:5]
                for var, info in weights:
                    direction = "↑" if info['causal_effect'] > 0 else "↓"
                    print(f"    {info['name']}: {info['weight_pct']:.1f}% "
                          f"{direction} (效应: {info['causal_effect']:+.2f})")
            
            # 显示相似样本
            if data.get('similar_samples'):
                print(f"\n  📌 找到 {len(data['similar_samples'])} 个相似样本")
            
            return True
        else:
            print(f"❌ 测试失败: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return False


def test_analyze():
    """测试6: 因果分析（智能问答）"""
    print_separator("测试 6/6: POST /api/analyze - 因果分析")
    
    # 测试不同类型的问题
    test_queries = [
        {
            "name": "干预分析",
            "query": "如何提高混凝土强度？",
            "expected_type": "intervention"
        },
        {
            "name": "归因分析",
            "query": "为什么强度下降了？",
            "expected_type": "attribution"
        }
    ]
    
    results = []
    
    for test in test_queries:
        print(f"📝 测试问题: {test['query']}")
        print(f"   期望类型: {test['expected_type']}\n")
        
        try:
            response = requests.post(
                f'{API_BASE}/api/analyze',
                json={"query": test['query']},
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ 分析成功")
                print(f"    • 分析类型: {data['analysis_type']}")
                print(f"    • 目标变量: {data['target_variable']}")
                print(f"    • 摘要: {data['analysis_summary'][:80]}...")
                results.append(True)
            else:
                print(f"  ❌ 测试失败: {response.status_code}")
                results.append(False)
                
        except Exception as e:
            print(f"  ❌ 请求失败: {e}")
            results.append(False)
        
        print()
    
    return all(results)


def main():
    """主测试函数"""
    print("\n" + "="*80)
    print("🚀 API接口完整性测试")
    print("="*80)
    print(f"\nAPI服务器: {API_BASE}")
    print("测试范围: 6个主要端点")
    print()
    
    # 等待服务器启动
    print("⏳ 等待服务器启动...")
    time.sleep(3)
    print("✓ 开始测试\n")
    
    # 执行测试
    test_results = {}
    
    test_results['health'] = test_health()
    time.sleep(1)
    
    test_results['variables'] = test_variables()
    time.sleep(1)
    
    test_results['graph'] = test_graph()
    time.sleep(1)
    
    test_results['samples'] = test_samples()
    time.sleep(1)
    
    test_results['predict'] = test_predict()
    time.sleep(1)
    
    test_results['analyze'] = test_analyze()
    
    # 汇总结果
    print_separator("📊 测试结果汇总")
    
    total = len(test_results)
    passed = sum(test_results.values())
    
    print(f"总测试数: {total}")
    print(f"通过: {passed}")
    print(f"失败: {total - passed}")
    print(f"通过率: {passed/total*100:.1f}%\n")
    
    print("详细结果:")
    for endpoint, result in test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {endpoint:15s} {status}")
    
    print("\n" + "="*80)
    if passed == total:
        print("🎉 所有测试通过！API服务运行正常")
    else:
        print("⚠️  部分测试失败，请检查日志")
    print("="*80 + "\n")
    
    return passed == total


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
        exit(130)
    except Exception as e:
        print(f"\n\n❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

