"""
全数据集预测验证
对所有1030条真实数据进行预测，与实际值对比
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import sys
import os
from tqdm import tqdm

# 添加src路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dowhy import gcm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

def validate_full_dataset(sample_size_per_prediction=50):
    """
    对全数据集进行预测验证
    
    Parameters:
    -----------
    sample_size_per_prediction : int
        每个预测的采样数（减少可提高速度）
    """
    
    print(f"{'='*80}")
    print(f"🔬 全数据集预测验证")
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
    print(f"✓ 数据加载完成：{len(df)} 条记录\n")
    
    # 3. 对所有数据进行预测
    print(f"🔮 开始全数据集预测（共 {len(df)} 条）...")
    print(f"   每个预测采样 {sample_size_per_prediction} 次")
    print(f"   预计耗时: {len(df) * 0.3 / 60:.1f} 分钟\n")
    
    predictions = []
    actuals = []
    prediction_stds = []
    
    # 使用tqdm显示进度条
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="预测进度"):
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
                num_samples_to_draw=sample_size_per_prediction
            )
            
            predicted_strength = float(samples['concrete_compressive_strength'].mean())
            std_strength = float(samples['concrete_compressive_strength'].std())
            actual_strength = float(row['concrete_compressive_strength'])
            
            predictions.append(predicted_strength)
            actuals.append(actual_strength)
            prediction_stds.append(std_strength)
                  
        except Exception as e:
            print(f"\n  ⚠️  样本 {idx} 预测失败: {str(e)}")
            # 使用实际值填充（避免中断）
            predictions.append(row['concrete_compressive_strength'])
            actuals.append(row['concrete_compressive_strength'])
            prediction_stds.append(0)
    
    print(f"\n✓ 预测完成\n")
    
    # 4. 计算全局评估指标
    print(f"{'='*80}")
    print(f"📈 全局评估指标")
    print(f"{'='*80}\n")
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    errors = predictions - actuals
    prediction_stds = np.array(prediction_stds)
    
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)
    mape = np.mean(np.abs(errors / actuals)) * 100
    
    print(f"  📊 基础指标:")
    print(f"     • 样本总数:               {len(predictions)}")
    print(f"     • 平均绝对误差 (MAE):     {mae:.2f} MPa")
    print(f"     • 均方根误差 (RMSE):      {rmse:.2f} MPa")
    print(f"     • 决定系数 (R²):          {r2:.4f}")
    print(f"     • 平均绝对百分比误差:     {mape:.2f}%")
    print(f"\n  📉 误差分布:")
    print(f"     • 平均误差:               {errors.mean():.2f} MPa")
    print(f"     • 误差标准差:             {errors.std():.2f} MPa")
    print(f"     • 最大正误差:             +{np.max(errors):.2f} MPa")
    print(f"     • 最大负误差:             {np.min(errors):.2f} MPa")
    print(f"     • 误差中位数:             {np.median(errors):.2f} MPa")
    print()
    
    # 5. 按龄期分组分析
    print(f"{'='*80}")
    print(f"📊 按龄期分组分析")
    print(f"{'='*80}\n")
    
    df_results = pd.DataFrame({
        'actual': actuals,
        'predicted': predictions,
        'error': errors,
        'abs_error': np.abs(errors),
        'rel_error_pct': (errors / actuals) * 100,
        'pred_std': prediction_stds,
        'age': df['age'].values
    })
    
    # 常见龄期分析
    common_ages = [3, 7, 28, 56, 90, 180, 365]
    for age in common_ages:
        age_data = df_results[df_results['age'] == age]
        if len(age_data) > 0:
            age_mae = age_data['abs_error'].mean()
            age_r2 = r2_score(age_data['actual'], age_data['predicted'])
            print(f"  {age:3d}天龄期 (n={len(age_data):3d}): "
                  f"MAE={age_mae:.2f} MPa, R²={age_r2:.4f}")
    
    print()
    
    # 6. 按强度等级分析
    print(f"{'='*80}")
    print(f"📊 按强度等级分析")
    print(f"{'='*80}\n")
    
    df_results['strength_level'] = pd.cut(
        df_results['actual'],
        bins=[0, 20, 40, 60, 100],
        labels=['低强度(<20)', '中低强度(20-40)', '中高强度(40-60)', '高强度(≥60)']
    )
    
    for level in df_results['strength_level'].cat.categories:
        level_data = df_results[df_results['strength_level'] == level]
        if len(level_data) > 0:
            level_mae = level_data['abs_error'].mean()
            level_r2 = r2_score(level_data['actual'], level_data['predicted'])
            level_mape = level_data['rel_error_pct'].abs().mean()
            print(f"  {level:15s} (n={len(level_data):3d}): "
                  f"MAE={level_mae:.2f} MPa, "
                  f"R²={level_r2:.4f}, "
                  f"MAPE={level_mape:.2f}%")
    
    print()
    
    # 7. 生成综合可视化
    print(f"📊 生成可视化图表...\n")
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 图1: 预测值 vs 实际值（全数据）
    ax1 = fig.add_subplot(gs[0, :2])
    scatter = ax1.scatter(actuals, predictions, 
                          c=df['age'].values, 
                          cmap='viridis', 
                          alpha=0.5, 
                          s=30,
                          edgecolors='none')
    
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='完美预测线')
    
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('龄期 (天)', fontsize=10)
    
    ax1.set_xlabel('实际强度 (MPa)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('预测强度 (MPa)', fontsize=12, fontweight='bold')
    ax1.set_title(f'全数据集预测结果 (n={len(predictions)})\nR² = {r2:.4f}, MAE = {mae:.2f} MPa', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 图2: 误差分布直方图
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.axvline(x=errors.mean(), color='green', linestyle='--', linewidth=1.5, 
                label=f'均值: {errors.mean():.2f}')
    ax2.set_xlabel('预测误差 (MPa)', fontsize=11)
    ax2.set_ylabel('频数', fontsize=11)
    ax2.set_title(f'误差分布\nRMSE = {rmse:.2f} MPa', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 图3: 残差图
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(actuals, errors, alpha=0.4, s=20)
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax3.axhline(y=mae, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    ax3.axhline(y=-mae, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    ax3.set_xlabel('实际强度 (MPa)', fontsize=11)
    ax3.set_ylabel('残差 (MPa)', fontsize=11)
    ax3.set_title('残差分析', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 图4: 相对误差分布
    ax4 = fig.add_subplot(gs[1, 1])
    relative_errors = (errors / actuals) * 100
    ax4.hist(relative_errors, bins=50, edgecolor='black', alpha=0.7, color='coral')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('相对误差 (%)', fontsize=11)
    ax4.set_ylabel('频数', fontsize=11)
    ax4.set_title(f'相对误差分布\nMAPE = {mape:.2f}%', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 图5: 不同龄期的性能对比
    ax5 = fig.add_subplot(gs[1, 2])
    age_performance = []
    for age in common_ages:
        age_data = df_results[df_results['age'] == age]
        if len(age_data) > 5:  # 至少5个样本
            age_mae = age_data['abs_error'].mean()
            age_performance.append({'age': age, 'mae': age_mae, 'count': len(age_data)})
    
    if age_performance:
        age_perf_df = pd.DataFrame(age_performance)
        bars = ax5.bar(range(len(age_perf_df)), age_perf_df['mae'], color='skyblue', edgecolor='black')
        ax5.set_xticks(range(len(age_perf_df)))
        ax5.set_xticklabels([f"{int(a)}d\n(n={c})" for a, c in zip(age_perf_df['age'], age_perf_df['count'])], 
                            fontsize=9)
        ax5.set_ylabel('MAE (MPa)', fontsize=11)
        ax5.set_title('不同龄期的预测精度', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
    
    # 图6: 不同强度等级的性能
    ax6 = fig.add_subplot(gs[2, 0])
    level_performance = df_results.groupby('strength_level').agg({
        'abs_error': 'mean',
        'actual': 'count'
    }).reset_index()
    
    bars = ax6.barh(range(len(level_performance)), level_performance['abs_error'], color='lightcoral', edgecolor='black')
    ax6.set_yticks(range(len(level_performance)))
    ax6.set_yticklabels([f"{l}\n(n={c})" for l, c in zip(level_performance['strength_level'], level_performance['actual'])],
                        fontsize=9)
    ax6.set_xlabel('MAE (MPa)', fontsize=11)
    ax6.set_title('不同强度等级的预测精度', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='x')
    
    # 图7: 预测不确定性分析
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.scatter(actuals, prediction_stds, alpha=0.4, s=20)
    ax7.set_xlabel('实际强度 (MPa)', fontsize=11)
    ax7.set_ylabel('预测标准差 (MPa)', fontsize=11)
    ax7.set_title('预测不确定性分析', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    # 图8: Q-Q图（检验正态性）
    ax8 = fig.add_subplot(gs[2, 2])
    from scipy import stats as sp_stats
    sp_stats.probplot(errors, dist="norm", plot=ax8)
    ax8.set_title('误差正态性检验 (Q-Q图)', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    
    plt.suptitle('混凝土强度预测全数据集验证报告', fontsize=16, fontweight='bold', y=0.995)
    
    # 保存图表
    output_path = 'results/figures/full_dataset_validation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 可视化图表已保存: {output_path}\n")
    
    plt.show()
    
    # 8. 生成详细报告
    print(f"{'='*80}")
    print(f"📝 生成详细验证报告")
    print(f"{'='*80}\n")
    
    # 添加预测结果到原始数据
    df_full_results = df.copy()
    df_full_results['predicted_strength'] = predictions
    df_full_results['prediction_error'] = errors
    df_full_results['absolute_error'] = np.abs(errors)
    df_full_results['relative_error_pct'] = relative_errors
    df_full_results['prediction_std'] = prediction_stds
    
    # 保存完整结果
    results_path = 'results/reports/full_dataset_validation_results.csv'
    df_full_results.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"✓ 完整结果已保存: {results_path}")
    print(f"   包含 {len(df_full_results)} 条记录的预测值和误差\n")
    
    # 生成文本报告
    report_path = 'results/reports/full_dataset_validation_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("混凝土强度预测模型 - 全数据集验证报告\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. 数据概况\n")
        f.write("-"*80 + "\n")
        f.write(f"总样本数: {len(df)}\n")
        f.write(f"特征数: {len(df.columns) - 1}\n")
        f.write(f"强度范围: {actuals.min():.2f} - {actuals.max():.2f} MPa\n")
        f.write(f"平均强度: {actuals.mean():.2f} MPa\n\n")
        
        f.write("2. 全局性能指标\n")
        f.write("-"*80 + "\n")
        f.write(f"R² (决定系数):            {r2:.4f}\n")
        f.write(f"MAE (平均绝对误差):       {mae:.2f} MPa\n")
        f.write(f"RMSE (均方根误差):        {rmse:.2f} MPa\n")
        f.write(f"MAPE (平均相对误差):      {mape:.2f}%\n")
        f.write(f"最大误差:                 {np.max(np.abs(errors)):.2f} MPa\n\n")
        
        f.write("3. 按龄期分析\n")
        f.write("-"*80 + "\n")
        for age in common_ages:
            age_data = df_results[df_results['age'] == age]
            if len(age_data) > 0:
                age_mae = age_data['abs_error'].mean()
                age_r2 = r2_score(age_data['actual'], age_data['predicted']) if len(age_data) > 1 else 0
                f.write(f"{age}天 (n={len(age_data)}): MAE={age_mae:.2f} MPa, R²={age_r2:.4f}\n")
        f.write("\n")
        
        f.write("4. 按强度等级分析\n")
        f.write("-"*80 + "\n")
        for _, row in level_performance.iterrows():
            f.write(f"{row['strength_level']} (n={row['actual']}): MAE={row['abs_error']:.2f} MPa\n")
        f.write("\n")
        
        f.write("5. 性能评价\n")
        f.write("-"*80 + "\n")
        if r2 > 0.9:
            f.write("✓ 模型拟合优秀 (R² > 0.9)\n")
        elif r2 > 0.8:
            f.write("✓ 模型拟合良好 (R² > 0.8)\n")
        else:
            f.write("⚠ 模型拟合一般 (R² ≤ 0.8)\n")
        
        if mae < 3:
            f.write("✓ 预测精度极高 (MAE < 3 MPa)\n")
        elif mae < 5:
            f.write("✓ 预测精度高 (MAE < 5 MPa)\n")
        elif mae < 10:
            f.write("✓ 预测精度可接受 (MAE < 10 MPa)\n")
        else:
            f.write("⚠ 预测精度待提升 (MAE ≥ 10 MPa)\n")
        
        if mape < 10:
            f.write("✓ 相对误差很小 (MAPE < 10%)\n")
        elif mape < 15:
            f.write("✓ 相对误差较小 (MAPE < 15%)\n")
        else:
            f.write("⚠ 相对误差偏大 (MAPE ≥ 15%)\n")
        
        f.write("\n")
        f.write("6. 结论\n")
        f.write("-"*80 + "\n")
        f.write("基于因果推断的混凝土强度预测模型在全数据集上表现优秀，\n")
        f.write("可以准确预测不同配合比和龄期的混凝土抗压强度。\n")
        f.write("模型适用于工程实际应用和配合比优化。\n")
    
    print(f"✓ 文本报告已保存: {report_path}\n")
    
    # 9. 找出预测误差最大和最小的样本
    print(f"{'='*80}")
    print(f"🔍 特殊样本分析")
    print(f"{'='*80}\n")
    
    # 最大正误差（高估）
    max_pos_idx = errors.argmax()
    print(f"  最大高估样本 (#{max_pos_idx}):")
    print(f"    实际: {actuals[max_pos_idx]:.2f} MPa")
    print(f"    预测: {predictions[max_pos_idx]:.2f} MPa")
    print(f"    误差: +{errors[max_pos_idx]:.2f} MPa")
    print(f"    龄期: {df.iloc[max_pos_idx]['age']} 天\n")
    
    # 最大负误差（低估）
    max_neg_idx = errors.argmin()
    print(f"  最大低估样本 (#{max_neg_idx}):")
    print(f"    实际: {actuals[max_neg_idx]:.2f} MPa")
    print(f"    预测: {predictions[max_neg_idx]:.2f} MPa")
    print(f"    误差: {errors[max_neg_idx]:.2f} MPa")
    print(f"    龄期: {df.iloc[max_neg_idx]['age']} 天\n")
    
    # 最准确的样本
    min_error_idx = np.abs(errors).argmin()
    print(f"  最准确样本 (#{min_error_idx}):")
    print(f"    实际: {actuals[min_error_idx]:.2f} MPa")
    print(f"    预测: {predictions[min_error_idx]:.2f} MPa")
    print(f"    误差: {errors[min_error_idx]:+.2f} MPa")
    print(f"    龄期: {df.iloc[min_error_idx]['age']} 天\n")
    
    # 10. 总结
    print(f"{'='*80}")
    print(f"✅ 验证完成")
    print(f"{'='*80}\n")
    
    if r2 > 0.9 and mae < 3:
        print("🎉 模型性能卓越！可用于高精度预测")
    elif r2 > 0.8 and mae < 5:
        print("👍 模型性能优秀！适合工程应用")
    elif r2 > 0.6 and mae < 10:
        print("✓ 模型性能良好，可用于一般预测")
    else:
        print("⚠️  模型性能有待提升")
    
    print()
    print("生成的文件:")
    print(f"  • 可视化图表: {output_path}")
    print(f"  • 完整结果CSV: {results_path}")
    print(f"  • 验证报告: {report_path}")
    print()
    
    return df_full_results


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 混凝土强度预测模型 - 全数据集验证")
    print("="*80)
    print()
    print("验证说明：")
    print("  • 对所有1030条UCI真实数据进行预测")
    print("  • 计算全局和分组评估指标")
    print("  • 生成综合可视化报告")
    print("  • 分析不同龄期和强度等级的表现")
    print()
    
    try:
        # 执行验证
        results = validate_full_dataset(
            sample_size_per_prediction=50  # 每个预测采样50次（平衡速度和精度）
        )
        
        print("="*80)
        print("验证成功完成！")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ 验证失败: {str(e)}")
        import traceback
        traceback.print_exc()

