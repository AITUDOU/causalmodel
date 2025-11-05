"""
快速预测准确性测试（简化版）
不调用API，直接使用因果模型进行预测
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import sys
import os

# 添加src路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dowhy import gcm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

def quick_test_accuracy(num_samples=20, age_filter=28):
    """
    快速测试预测准确性（不通过API）
    
    Parameters:
    -----------
    num_samples : int
        测试样本数量
    age_filter : int
        龄期过滤
    """
    
    print(f"{'='*80}")
    print(f"🔬 快速预测准确性测试")
    print(f"{'='*80}\n")
    
    # 1. 加载模型
    print("📦 加载因果模型...")
    model_path = 'models/causal_model.pkl'
    with open(model_path, 'rb') as f:
        causal_model = pickle.load(f)
    print(f"✓ 模型加载完成\n")
    
    # 2. 加载数据
    print("📊 加载真实数据...")
    df = pd.read_csv('data/real/concrete_compressive_strength.csv')
    df.columns = df.columns.str.strip()
    
    # 过滤龄期
    df_filtered = df[df['age'] == age_filter]
    print(f"✓ 数据加载完成：{len(df_filtered)} 条 {age_filter}天龄期样本\n")
    
    # 3. 随机抽样
    print(f"🎲 随机抽取 {num_samples} 个样本...")
    test_samples = df_filtered.sample(n=num_samples, random_state=42)
    print(f"✓ 抽样完成\n")
    
    # 4. 批量预测
    print(f"🔮 开始批量预测...\n")
    predictions = []
    actuals = []
    
    for idx, (_, row) in enumerate(test_samples.iterrows(), 1):
        try:
            # 准备输入数据
            input_vars = {
                'cement': lambda x, v=row['cement']: v,
                'blast_furnace_slag': lambda x, v=row['blast_furnace_slag']: v,
                'fly_ash': lambda x, v=row['fly_ash']: v,
                'water': lambda x, v=row['water']: v,
                'superplasticizer': lambda x, v=row['superplasticizer']: v,
                'coarse_aggregate': lambda x, v=row['coarse_aggregate']: v,
                'fine_aggregate': lambda x, v=row['fine_aggregate']: v,
                'age': lambda x, v=row['age']: v
            }
            
            # 使用因果模型预测
            samples = gcm.interventional_samples(
                causal_model.causal_model,
                input_vars,
                num_samples_to_draw=100
            )
            
            predicted_strength = float(samples['concrete_compressive_strength'].mean())
            actual_strength = float(row['concrete_compressive_strength'])
            
            predictions.append(predicted_strength)
            actuals.append(actual_strength)
            
            error = predicted_strength - actual_strength
            
            # 显示进度
            print(f"  [{idx:2d}/{num_samples}] 实际: {actual_strength:6.2f} MPa | "
                  f"预测: {predicted_strength:6.2f} MPa | "
                  f"误差: {error:+6.2f} MPa")
                  
        except Exception as e:
            print(f"  [{idx:2d}/{num_samples}] ❌ 预测失败: {str(e)}")
    
    print(f"\n✓ 预测完成\n")
    
    # 5. 计算评估指标
    print(f"{'='*80}")
    print(f"📈 评估指标")
    print(f"{'='*80}\n")
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    errors = predictions - actuals
    
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)
    mape = np.mean(np.abs(errors / actuals)) * 100
    
    print(f"  • 样本数量:               {len(predictions)}")
    print(f"  • 平均绝对误差 (MAE):     {mae:.2f} MPa")
    print(f"  • 均方根误差 (RMSE):      {rmse:.2f} MPa")
    print(f"  • 决定系数 (R²):          {r2:.4f}")
    print(f"  • 平均绝对百分比误差:     {mape:.2f}%")
    print(f"  • 最大正误差:             +{np.max(errors):.2f} MPa")
    print(f"  • 最大负误差:             {np.min(errors):.2f} MPa")
    print()
    
    # 6. 可视化
    print(f"📊 生成可视化图表...\n")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 图1: 预测值 vs 实际值
    ax1 = axes[0, 0]
    ax1.scatter(actuals, predictions, alpha=0.6, s=100, edgecolors='black', linewidth=1)
    
    # 绘制完美预测线
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='完美预测线')
    
    # 绘制±10% 误差带
    margin = (max_val - min_val) * 0.1
    ax1.fill_between([min_val, max_val], 
                      [min_val - margin, max_val - margin],
                      [min_val + margin, max_val + margin],
                      alpha=0.2, color='green', label='±10% 误差带')
    
    ax1.set_xlabel('实际强度 (MPa)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('预测强度 (MPa)', fontsize=12, fontweight='bold')
    ax1.set_title(f'预测值 vs 实际值\nR² = {r2:.4f}', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 图2: 误差分布
    ax2 = axes[0, 1]
    n, bins, patches = ax2.hist(errors, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='零误差线')
    ax2.axvline(x=errors.mean(), color='green', linestyle='--', linewidth=2, label=f'平均误差: {errors.mean():.2f}')
    ax2.set_xlabel('预测误差 (MPa)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('频数', fontsize=12, fontweight='bold')
    ax2.set_title(f'误差分布\nMAE = {mae:.2f} MPa', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 图3: 残差图
    ax3 = axes[1, 0]
    ax3.scatter(actuals, errors, alpha=0.6, s=100, edgecolors='black', linewidth=1)
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax3.axhline(y=mae, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    ax3.axhline(y=-mae, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    ax3.set_xlabel('实际强度 (MPa)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('残差 (预测 - 实际, MPa)', fontsize=12, fontweight='bold')
    ax3.set_title(f'残差分析\nRMSE = {rmse:.2f} MPa', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 图4: 相对误差箱线图
    ax4 = axes[1, 1]
    relative_errors = (errors / actuals) * 100
    bp = ax4.boxplot(relative_errors, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_edgecolor('black')
    bp['boxes'][0].set_linewidth(2)
    
    ax4.set_ylabel('相对误差 (%)', fontsize=12, fontweight='bold')
    ax4.set_title(f'相对误差分布\nMAPE = {mape:.2f}%', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax4.set_xticklabels(['所有样本'])
    
    plt.tight_layout()
    
    # 保存图表
    output_path = 'results/figures/quick_prediction_test.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 图表已保存: {output_path}\n")
    
    plt.show()
    
    # 7. 详细结果
    print(f"{'='*80}")
    print(f"📊 详细分析")
    print(f"{'='*80}\n")
    
    # 分析误差在不同强度区间的表现
    results_df = pd.DataFrame({
        '实际强度': actuals,
        '预测强度': predictions,
        '绝对误差': np.abs(errors),
        '相对误差(%)': relative_errors
    })
    
    # 按强度等级分组
    results_df['强度等级'] = pd.cut(results_df['实际强度'], 
                                     bins=[0, 20, 40, 60, 100],
                                     labels=['低强度(<20)', '中低强度(20-40)', 
                                            '中高强度(40-60)', '高强度(≥60)'])
    
    print("按强度等级分析：\n")
    for grade in results_df['强度等级'].cat.categories:
        grade_data = results_df[results_df['强度等级'] == grade]
        if len(grade_data) > 0:
            print(f"  {grade}:")
            print(f"    样本数: {len(grade_data)}")
            print(f"    平均绝对误差: {grade_data['绝对误差'].mean():.2f} MPa")
            print(f"    平均相对误差: {grade_data['相对误差(%)'].mean():.2f}%")
            print(f"    最大误差: {grade_data['绝对误差'].max():.2f} MPa")
            print()
    
    # 保存结果
    results_path = 'results/reports/quick_prediction_test_results.csv'
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"✓ 详细结果已保存: {results_path}\n")
    
    # 8. 结论
    print(f"{'='*80}")
    print(f"✅ 测试完成")
    print(f"{'='*80}\n")
    
    if r2 > 0.8:
        print("🎉 模型性能优秀！R² > 0.8")
    elif r2 > 0.6:
        print("👍 模型性能良好，R² > 0.6")
    elif r2 > 0.4:
        print("⚠️  模型性能一般，R² > 0.4")
    else:
        print("❌ 模型性能较差，需要优化")
    
    if mae < 5:
        print("🎯 预测精度高，MAE < 5 MPa")
    elif mae < 10:
        print("✓ 预测精度可接受，MAE < 10 MPa")
    else:
        print("⚠️  预测误差较大，MAE ≥ 10 MPa")
    
    if mape < 15:
        print(f"✓ 相对误差小，MAPE < 15%")
    elif mape < 25:
        print(f"⚠️  相对误差中等，MAPE < 25%")
    else:
        print(f"❌ 相对误差较大，MAPE ≥ 25%")
    
    print()
    
    return results_df


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 快速预测准确性测试（不通过API）")
    print("="*80)
    print()
    print("测试说明：")
    print("  • 直接使用因果模型进行预测（速度更快）")
    print("  • 从真实UCI数据集中随机抽取样本")
    print("  • 计算预测误差和评估指标")
    print("  • 生成可视化图表")
    print()
    
    try:
        # 执行测试
        results = quick_test_accuracy(
            num_samples=20,  # 减少样本数量以提高速度
            age_filter=28
        )
        
        print("="*80)
        print("测试成功完成！")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

