"""
测试预测准确性
从真实数据集中随机抽取样本，使用API预测强度，然后与实际值对比
"""

import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# API配置
API_BASE = 'http://localhost:8000'

def test_prediction_accuracy(num_samples=50, age_filter=28):
    """
    测试预测准确性
    
    Parameters:
    -----------
    num_samples : int
        测试样本数量
    age_filter : int
        龄期过滤（默认只测试28天龄期样本）
    """
    
    print(f"{'='*80}")
    print(f"🔬 测试预测准确性")
    print(f"{'='*80}\n")
    
    # 1. 加载真实数据
    print("📊 加载真实数据...")
    df = pd.read_csv('data/real/concrete_compressive_strength.csv')
    df.columns = df.columns.str.strip()
    
    # 过滤指定龄期
    df_filtered = df[df['age'] == age_filter]
    print(f"✓ 数据加载完成：{len(df_filtered)} 条 {age_filter}天龄期样本\n")
    
    # 2. 随机抽样
    print(f"🎲 随机抽取 {num_samples} 个样本...")
    if len(df_filtered) < num_samples:
        num_samples = len(df_filtered)
        print(f"⚠️  样本数量不足，调整为 {num_samples} 个")
    
    test_samples = df_filtered.sample(n=num_samples, random_state=42)
    print(f"✓ 抽样完成\n")
    
    # 3. 批量预测
    print(f"🔮 开始批量预测...\n")
    predictions = []
    actuals = []
    errors = []
    
    for idx, (_, row) in enumerate(test_samples.iterrows(), 1):
        # 构建请求参数
        params = {
            'cement': float(row['cement']),
            'blast_furnace_slag': float(row['blast_furnace_slag']),
            'fly_ash': float(row['fly_ash']),
            'water': float(row['water']),
            'superplasticizer': float(row['superplasticizer']),
            'coarse_aggregate': float(row['coarse_aggregate']),
            'fine_aggregate': float(row['fine_aggregate']),
            'age': int(row['age'])
        }
        
        actual_strength = float(row['concrete_compressive_strength'])
        
        try:
            # 调用预测API
            response = requests.post(
                f'{API_BASE}/api/predict',
                json=params,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                predicted_strength = data['predicted_strength']
                error = predicted_strength - actual_strength
                
                predictions.append(predicted_strength)
                actuals.append(actual_strength)
                errors.append(error)
                
                # 显示进度
                print(f"  [{idx:2d}/{num_samples}] 实际: {actual_strength:6.2f} MPa | "
                      f"预测: {predicted_strength:6.2f} MPa | "
                      f"误差: {error:+6.2f} MPa")
            else:
                print(f"  [{idx:2d}/{num_samples}] ❌ API错误: {response.status_code}")
                
        except Exception as e:
            print(f"  [{idx:2d}/{num_samples}] ❌ 请求失败: {str(e)}")
    
    print(f"\n✓ 预测完成\n")
    
    # 4. 计算评估指标
    print(f"{'='*80}")
    print(f"📈 评估指标")
    print(f"{'='*80}\n")
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    errors = np.array(errors)
    
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)
    mape = np.mean(np.abs(errors / actuals)) * 100
    
    print(f"  • 平均绝对误差 (MAE):     {mae:.2f} MPa")
    print(f"  • 均方根误差 (RMSE):      {rmse:.2f} MPa")
    print(f"  • 决定系数 (R²):          {r2:.4f}")
    print(f"  • 平均绝对百分比误差:     {mape:.2f}%")
    print(f"  • 最大误差:               {np.max(np.abs(errors)):.2f} MPa")
    print(f"  • 最小误差:               {np.min(np.abs(errors)):.2f} MPa")
    print()
    
    # 5. 可视化结果
    print(f"📊 生成可视化图表...\n")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 图1: 预测值 vs 实际值散点图
    ax1 = axes[0, 0]
    ax1.scatter(actuals, predictions, alpha=0.6, s=80)
    ax1.plot([actuals.min(), actuals.max()], 
             [actuals.min(), actuals.max()], 
             'r--', lw=2, label='完美预测线')
    ax1.set_xlabel('实际强度 (MPa)', fontsize=12)
    ax1.set_ylabel('预测强度 (MPa)', fontsize=12)
    ax1.set_title(f'预测值 vs 实际值\nR² = {r2:.4f}', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图2: 误差分布直方图
    ax2 = axes[0, 1]
    ax2.hist(errors, bins=20, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=2, label='零误差线')
    ax2.set_xlabel('预测误差 (MPa)', fontsize=12)
    ax2.set_ylabel('频数', fontsize=12)
    ax2.set_title(f'误差分布\nMAE = {mae:.2f} MPa', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 图3: 残差图
    ax3 = axes[1, 0]
    ax3.scatter(actuals, errors, alpha=0.6, s=80)
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('实际强度 (MPa)', fontsize=12)
    ax3.set_ylabel('残差 (预测 - 实际)', fontsize=12)
    ax3.set_title(f'残差图\nRMSE = {rmse:.2f} MPa', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 图4: 相对误差分布
    ax4 = axes[1, 1]
    relative_errors = (errors / actuals) * 100
    ax4.boxplot(relative_errors, vert=True)
    ax4.set_ylabel('相对误差 (%)', fontsize=12)
    ax4.set_title(f'相对误差箱线图\nMAPE = {mape:.2f}%', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=0, color='r', linestyle='--', linewidth=2)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = 'results/figures/prediction_accuracy_test.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 图表已保存: {output_path}\n")
    
    plt.show()
    
    # 6. 生成详细报告
    print(f"{'='*80}")
    print(f"📝 详细测试报告")
    print(f"{'='*80}\n")
    
    # 按强度等级分析
    def classify_strength(strength):
        if strength < 20:
            return '低强度 (<20 MPa)'
        elif strength < 40:
            return '中低强度 (20-40 MPa)'
        elif strength < 60:
            return '中高强度 (40-60 MPa)'
        else:
            return '高强度 (≥60 MPa)'
    
    results_df = pd.DataFrame({
        '实际强度': actuals,
        '预测强度': predictions,
        '绝对误差': np.abs(errors),
        '相对误差(%)': relative_errors,
        '强度等级': [classify_strength(s) for s in actuals]
    })
    
    print("按强度等级分析：\n")
    for grade in results_df['强度等级'].unique():
        grade_data = results_df[results_df['强度等级'] == grade]
        print(f"  {grade}:")
        print(f"    样本数: {len(grade_data)}")
        print(f"    平均绝对误差: {grade_data['绝对误差'].mean():.2f} MPa")
        print(f"    平均相对误差: {grade_data['相对误差(%)'].mean():.2f}%")
        print()
    
    # 保存结果到CSV
    results_path = 'results/reports/prediction_accuracy_results.csv'
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"✓ 详细结果已保存: {results_path}\n")
    
    # 7. 结论
    print(f"{'='*80}")
    print(f"✅ 测试完成")
    print(f"{'='*80}\n")
    
    if r2 > 0.8:
        print("🎉 模型性能优秀！R² > 0.8")
    elif r2 > 0.6:
        print("👍 模型性能良好，R² > 0.6")
    else:
        print("⚠️  模型性能有待提升，建议优化")
    
    if mae < 5:
        print("🎯 预测精度高，MAE < 5 MPa")
    elif mae < 10:
        print("✓ 预测精度可接受，MAE < 10 MPa")
    else:
        print("⚠️  预测误差较大，建议检查模型")
    
    print()
    
    return results_df


if __name__ == "__main__":
    # 测试配置
    NUM_SAMPLES = 50  # 测试样本数量
    AGE_FILTER = 28   # 只测试28天龄期样本
    
    print("\n" + "="*80)
    print("🚀 混凝土强度预测模型准确性测试")
    print("="*80)
    print()
    print("测试说明：")
    print("  1. 从真实UCI数据集中随机抽取样本")
    print("  2. 使用因果模型预测API进行预测")
    print("  3. 计算预测误差和评估指标")
    print("  4. 生成可视化图表和详细报告")
    print()
    print("请确保API服务器正在运行 (http://localhost:8000)")
    print()
    
    try:
        # 执行测试
        results = test_prediction_accuracy(
            num_samples=NUM_SAMPLES,
            age_filter=AGE_FILTER
        )
        
        print("="*80)
        print("测试成功完成！")
        print("="*80)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ 错误：无法连接到API服务器")
        print("   请先运行: python3 api_server.py")
        print()
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

