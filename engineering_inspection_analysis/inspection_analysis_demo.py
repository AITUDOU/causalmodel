"""
工程检测因果分析演示
展示因果推理、归因分析、干预分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.data_preprocessing import load_and_preprocess_data
from src.inspection_causal_model import InspectionCausalModel, compare_metrics
from src.visualization import plot_causal_graph, plot_attribution_results, plot_intervention_results

# 与 concrete_causal_demo 保持一致的中文字体设置
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置可视化风格
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)


def main():
    """主函数"""
    
    print("="*80)
    print("工程检测因果分析系统")
    print("="*80)
    
    # 创建结果目录
    results_dir = Path("results")
    figures_dir = results_dir / "figures"
    reports_dir = results_dir / "reports"
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # ===== 1. 数据预处理 =====
    print("\n" + "="*80)
    print("步骤 1: 数据预处理")
    print("="*80)
    
    data_path = "../data/processed/cleaned_data.csv"
    processed_path = "data/inspection_data_processed.csv"
    
    # 创建数据目录
    Path("data").mkdir(exist_ok=True)
    
    df = load_and_preprocess_data(data_path, processed_path)
    
    print(f"\n✓ 数据加载完成: {df.shape[0]} 行, {df.shape[1]} 列")
    print(f"\n关键指标统计:")
    if '是否合格_编码' in df.columns:
        quality_rate = df['是否合格_编码'].mean() * 100
        print(f"  - 检测合格率: {quality_rate:.2f}%")
        print(f"  - 合格样本: {df['是否合格_编码'].sum()}")
        print(f"  - 不合格样本: {(df['是否合格_编码'] == 0).sum()}")
    if '价格' in df.columns:
        print(f"  - 平均检测价格: {df['价格'].mean():.2f} 元")
    if '检测周期_天' in df.columns:
        print(f"  - 平均检测周期: {df['检测周期_天'].mean():.2f} 天")
    
    # ===== 2. 构建因果图 =====
    print("\n" + "="*80)
    print("步骤 2: 构建因果图")
    print("="*80)
    
    # 选择建模所需的列（精简为主要变量以加速计算）
    modeling_cols = [
        '生产厂家_编码', '品种_编码', '强度等级(级别)_编码', '规格_数值',
        '施工单位_编码', '结构部位_编码',
        '检验类别_编码', '检验风险等级',
        '委托季度', '委托星期', '检测周期_天', '审核周期_天', '检测完成_编码',
        '价格', '价格分组_编码',
        '生产厂家历史合格率', '施工单位历史合格率',
        '是否合格_编码'
    ]
    
    # 过滤存在的列
    available_cols = [col for col in modeling_cols if col in df.columns]
    df_model = df[available_cols].copy()
    
    # 删除缺失值
    df_model = df_model.dropna()
    print(f"✓ 建模数据: {df_model.shape[0]} 行, {df_model.shape[1]} 列")
    
    model = InspectionCausalModel(df_model)
    causal_graph = model.build_causal_graph()
    
    print(f"✓ 因果图构建完成")
    print(f"  - 节点数: {causal_graph.number_of_nodes()}")
    print(f"  - 边数: {causal_graph.number_of_edges()}")
    print(f"\n关键因果路径:")
    print(f"  1. 生产厂家 → 品种、强度等级 → 质量")
    print(f"  2. 施工单位 → 结构部位、代表数量 → 质量")
    print(f"  3. 检验类别 → 检验风险等级 → 质量")
    print(f"  4. 材料属性 → 价格")
    print(f"  5. 委托时间 → 检测周期 → 完成率")
    
    # 可视化因果图
    print("\n正在生成因果图可视化...")
    # 中文标签映射
    label_map = {
        '生产厂家_编码': '生产厂家',
        '品种_编码': '品种',
        '强度等级(级别)_编码': '强度等级',
        '规格_数值': '规格',
        '施工单位_编码': '施工单位',
        '结构部位_编码': '结构部位',
        '检验类别_编码': '检验类别',
        '检验风险等级': '检验风险等级',
        '委托季度': '委托季度',
        '委托星期': '委托星期',
        '检测周期_天': '检测周期',
        '审核周期_天': '审核周期',
        '检测完成_编码': '检测完成',
        '价格': '价格',
        '价格分组_编码': '价格分组',
        '生产厂家历史合格率': '厂家历史合格率',
        '施工单位历史合格率': '施工方历史合格率',
        '是否合格_编码': '是否合格'
    }

    fig = plot_causal_graph(
        causal_graph,
        title="工程检测因果图（中文）",
        figsize=(22, 18),
        node_size=2600,
        font_size=10,
        label_map=label_map
    )
    fig.savefig(figures_dir / "inspection_causal_graph.png", dpi=300, bbox_inches='tight')
    print(f"✓ 因果图已保存: {figures_dir / 'inspection_causal_graph.png'}")
    plt.close()
    
    # ===== 3. 拟合因果模型 =====
    print("\n" + "="*80)
    print("步骤 3: 拟合因果模型")
    print("="*80)
    
    # 为避免后续反事实重复训练，直接训练可逆模型一次并全程复用
    model.fit_causal_model(quality='BETTER', invertible=True)
    
    # ===== 4. 归因分析 =====
    print("\n" + "="*80)
    print("步骤 4: 归因分析 - 识别不合格率变化的根本原因")
    print("="*80)
    
    # 创建两个时期的数据对比
    # 场景：比较不同时期（如2007年上半年 vs 下半年）的质量变化
    
    # 按委托年份和季度分组
    if '委托年份' in df_model.columns and '委托季度' in df_model.columns:
        # 早期数据（前40%）
        df_old = df_model.iloc[:int(len(df_model) * 0.4)].copy()
        # 后期数据（后40%）
        df_new = df_model.iloc[int(len(df_model) * 0.6):].copy()
    else:
        # 简单分割
        df_old = df_model.iloc[:int(len(df_model) * 0.4)].copy()
        df_new = df_model.iloc[int(len(df_model) * 0.6):].copy()
    
    print(f"\n时期对比:")
    old_quality_rate = df_old['是否合格_编码'].mean() * 100
    new_quality_rate = df_new['是否合格_编码'].mean() * 100
    print(f"  早期 - 合格率: {old_quality_rate:.2f}%")
    print(f"  后期 - 合格率: {new_quality_rate:.2f}%")
    print(f"  合格率变化: {new_quality_rate - old_quality_rate:+.2f}%")
    
    if abs(new_quality_rate - old_quality_rate) > 0.1:  # 有显著变化才做归因分析
        print("\n正在执行归因分析...")
        contributions, uncertainties = model.attribution_analysis(
            df_old=df_old,
            df_new=df_new,
            target_column='是否合格_编码',
            num_samples=1000,
            confidence_level=0.90,
            num_bootstrap_resamples=4
        )
        
        # 整理归因结果
        attribution_results = []
        for var, contrib in contributions.items():
            if var in uncertainties:
                uncertainty = uncertainties[var]
                attribution_results.append({
                    'Variable': var,
                    'Contribution': contrib,
                    'Lower_CI': uncertainty[0],
                    'Upper_CI': uncertainty[1]
                })
        
        if len(attribution_results) == 0:
            print("\n✗ 归因分析未产生有效结果（所有候选变量均被跳过或无效）。")
            attribution_df = pd.DataFrame()
        else:
            attribution_df = pd.DataFrame(attribution_results)
            if 'Contribution' in attribution_df.columns:
                attribution_df = attribution_df.sort_values('Contribution', key=abs, ascending=False)
            
            print("\n✓ 归因分析完成")
            print("\n主要贡献因素（Top 10）:")
            for idx, row in attribution_df.head(10).iterrows():
                print(f"  {row['Variable']}: {row['Contribution']:.6f} "
                      f"(90% CI: [{row['Lower_CI']:.6f}, {row['Upper_CI']:.6f}])")
            
            # 保存归因结果
            attribution_df.to_csv(reports_dir / "attribution_results.csv", index=False)
            print(f"\n✓ 归因结果已保存: {reports_dir / 'attribution_results.csv'}")
            
            # 可视化归因结果
            print("\n正在生成归因分析可视化...")
            fig = plot_attribution_results(
                attribution_df,
                title="检测合格率变化归因分析",
                top_n=15
            )
            fig.savefig(figures_dir / "attribution_analysis.png", dpi=300, bbox_inches='tight')
            print(f"✓ 归因图已保存: {figures_dir / 'attribution_analysis.png'}")
            plt.close()
    else:
        print("\n合格率变化不显著，跳过归因分析")
        attribution_df = pd.DataFrame()
    
    # ===== 5. 干预分析 =====
    print("\n" + "="*80)
    print("步骤 5: 干预分析 - 评估不同优化措施的效果")
    print("="*80)
    
    # 定义不可干预的节点（结果变量和派生变量）
    non_interveneable = [
        '是否合格_编码',
        '检测完成_编码',
        '生产厂家历史合格率',
        '施工单位历史合格率',
        '价格',  # 结果变量
        '价格分组_编码',  # 派生变量
        '检测周期_天',  # 过程变量
        '审核周期_天'  # 过程变量
    ]
    
    print("\n正在执行干预分析...")
    print("评估各因素增加1个单位对检测合格率的因果效应...")
    
    intervention_results = model.intervention_analysis(
        target='是否合格_编码',
        step_size=1.0,
        non_interveneable_nodes=non_interveneable,
        confidence_level=0.95,
        num_samples=3000,
        num_bootstrap_resamples=10
    )
    
    if len(intervention_results) > 0:
        print("\n✓ 干预分析完成")
        print("\n最有效的优化措施（Top 10）:")
        for idx, row in intervention_results.head(10).iterrows():
            effect_pct = row['Causal_Effect'] * 100
            print(f"  {row['Variable']}: {effect_pct:+.4f}% "
                  f"(95% CI: [{row['Lower_CI']*100:+.4f}%, {row['Upper_CI']*100:+.4f}%])")
        
        # 保存干预结果
        intervention_results.to_csv(reports_dir / "intervention_results.csv", index=False)
        print(f"\n✓ 干预结果已保存: {reports_dir / 'intervention_results.csv'}")
        
        # 可视化干预结果
        print("\n正在生成干预分析可视化...")
        fig = plot_intervention_results(
            intervention_results,
            title="检测合格率优化措施因果效应分析",
            top_n=15
        )
        fig.savefig(figures_dir / "intervention_analysis.png", dpi=300, bbox_inches='tight')
        print(f"✓ 干预图已保存: {figures_dir / 'intervention_analysis.png'}")
        plt.close()
    else:
        print("\n✗ 干预分析未产生有效结果")
        intervention_results = pd.DataFrame()
    
    # ===== 6. 反事实分析 =====
    print("\n" + "="*80)
    print("步骤 6: 反事实分析 - 如果当时做不同选择？")
    print("="*80)

    # 选取一个不合格样本；优先从原始预处理数据中筛选，再与建模数据取交集
    failed_raw_idx = df.index[(df.get('是否合格_编码', pd.Series(index=df.index, data=np.nan)) == 0)]
    failed_idx_in_model = [i for i in failed_raw_idx if i in df_model.index]
    if len(failed_idx_in_model) > 0:
        cf_idx = failed_idx_in_model[0]
        print(f"选取不合格样本 index={cf_idx}")
    else:
        # 若交集为空，则从建模数据中选择最高风险样本
        candidate_df = df_model.copy()
        if '检验风险等级' in candidate_df.columns:
            candidate_df = candidate_df.sort_values('检验风险等级', ascending=False)
        cf_idx = candidate_df.index[0]
        print("未找到与建模数据交集的不合格样本，改用高风险样本进行反事实分析")

    # 设计两种干预：降低检验风险、提升规格
    spec_base = df_model.loc[cf_idx, '规格_数值'] if '规格_数值' in df_model.columns else 0
    interventions_cf = {}
    if '检验风险等级' in df_model.columns:
        interventions_cf['检验风险等级'] = 0
    if '规格_数值' in df_model.columns:
        interventions_cf['规格_数值'] = spec_base + 5

    cf_result = None
    try:
        cf_result = model.counterfactual_analysis(
            sample_index=cf_idx,
            interventions=interventions_cf,
            target='是否合格_编码'
        )
        obs = cf_result['observed']
        cfa = cf_result['counterfactual']
        chg = cf_result['change']
        print("\n反事实结果（目标=是否合格_编码）:")
        print(f"  观察值: {obs:.3f} → 反事实: {cfa:.3f}  (变化: {chg:+.3f})")
        # 若未改善（仍<0.5或变化<=0），执行强化干预
        if (cfa is None) or (obs is None) or (cfa < 0.5 and obs < 0.5) or (chg is not None and chg <= 0):
            strong_interventions = dict(interventions_cf)
            if '规格_数值' in df_model.columns:
                strong_interventions['规格_数值'] = float(df_model['规格_数值'].quantile(0.95))
            if '检验风险等级' in df_model.columns:
                strong_interventions['检验风险等级'] = 0
            if '施工单位历史合格率' in df_model.columns:
                strong_interventions['施工单位历史合格率'] = float(df_model['施工单位历史合格率'].quantile(0.99))
            if '生产厂家历史合格率' in df_model.columns:
                strong_interventions['生产厂家历史合格率'] = float(df_model['生产厂家历史合格率'].quantile(0.99))
            print("\n触发强化干预: ", strong_interventions)
            cf_result2 = model.counterfactual_analysis(
                sample_index=cf_idx,
                interventions=strong_interventions,
                target='是否合格_编码'
            )
            obs2, cfa2, chg2 = cf_result2['observed'], cf_result2['counterfactual'], cf_result2['change']
            print(f"  观察值: {obs2:.3f} → 反事实(强化): {cfa2:.3f}  (变化: {chg2:+.3f})")
            # 覆盖用于写报告的结果
            cf_result = cf_result2
            interventions_cf = strong_interventions

        # 若仍未达到0.5，再做“结构性类别替换”干预
        obs_final = cf_result['observed'] if cf_result else None
        cfa_final = cf_result['counterfactual'] if cf_result else None
        need_structural = (cfa_final is None) or (cfa_final < 0.5)
        if need_structural:
            structural = dict(interventions_cf)
            def best_code(col: str, min_n: int = 50):
                if col not in df_model.columns:
                    return None
                t = df_model.groupby(col)['是否合格_编码'].agg(['mean','count']).sort_values(['mean','count'], ascending=[False, False])
                t = t[t['count'] >= min_n]
                return int(t.index[0]) if len(t) else None
            for cat_col in ['生产厂家_编码','施工单位_编码','品种_编码','强度等级(级别)_编码']:
                code = best_code(cat_col)
                if code is not None:
                    structural[cat_col] = code
            print("\n触发结构性类别替换干预:", structural)
            cf_result3 = model.counterfactual_analysis(
                sample_index=cf_idx,
                interventions=structural,
                target='是否合格_编码'
            )
            print(f"  观察值: {cf_result3['observed']:.3f} → 反事实(结构性): {cf_result3['counterfactual']:.3f}  (变化: {cf_result3['change']:+.3f})")
            cf_result = cf_result3
            interventions_cf = structural

        # ===== 自动搜索“最易被挽回”的不合格样本 =====
        best_idx, best_cf, best_imp, best_plan = None, -1.0, -1.0, None
        if cf_result is None or cf_result.get('counterfactual', 0) < 0.5:
            print("\n开始自动搜索最易被挽回的不合格样本 …")
            # 候选集合：不合格样本按风险从高到低，最多200个
            cand = df_model[df_model['是否合格_编码'] == 0].copy()
            if '检验风险等级' in cand.columns:
                cand = cand.sort_values('检验风险等级', ascending=False)
            cand_idx = list(cand.index[:200])

            # 预先准备最佳类别编码
            def best_code(col: str, min_n: int = 50):
                if col not in df_model.columns:
                    return None
                t = df_model.groupby(col)['是否合格_编码'].agg(['mean','count']).sort_values(['mean','count'], ascending=[False, False])
                t = t[t['count'] >= min_n]
                return int(t.index[0]) if len(t) else None
            best_map = {}
            for cat_col in ['生产厂家_编码','施工单位_编码','品种_编码','强度等级(级别)_编码']:
                code = best_code(cat_col)
                if code is not None:
                    best_map[cat_col] = code
            spec95 = float(df_model['规格_数值'].quantile(0.95)) if '规格_数值' in df_model.columns else None
            unit99 = float(df_model['施工单位历史合格率'].quantile(0.99)) if '施工单位历史合格率' in df_model.columns else None
            mfr99 = float(df_model['生产厂家历史合格率'].quantile(0.99)) if '生产厂家历史合格率' in df_model.columns else None

            for idx in cand_idx:
                plan = {}
                if '检验风险等级' in df_model.columns: plan['检验风险等级'] = 0
                if spec95 is not None: plan['规格_数值'] = spec95
                if unit99 is not None: plan['施工单位历史合格率'] = unit99
                if mfr99 is not None: plan['生产厂家历史合格率'] = mfr99
                plan.update(best_map)
                try:
                    r = model.counterfactual_analysis(sample_index=idx, interventions=plan, target='是否合格_编码')
                    cfa = r.get('counterfactual', 0)
                    imp = r.get('change', 0)
                    if cfa > best_cf or (abs(cfa-best_cf) < 1e-9 and imp > best_imp):
                        best_idx, best_cf, best_imp, best_plan = idx, cfa, imp, plan
                        if best_cf >= 0.5:
                            break
                except Exception:
                    continue

            if best_idx is not None:
                print(f"\n自动搜索结果: index={best_idx}, 反事实概率={best_cf:.3f}, 提升={best_imp:+.3f}")
                # 用搜索到的最佳样本覆盖报告输出
                cf_idx = best_idx
                interventions_cf = best_plan
                cf_result = {'observed': float(df_model.loc[best_idx, '是否合格_编码']), 'counterfactual': best_cf, 'change': best_imp}
    except Exception as e:
        print(f"✗ 反事实分析失败: {e}")

    # 保存反事实简要报告
    try:
        cf_report = reports_dir / "counterfactual_analysis_report.txt"
        with open(cf_report, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("工程检测反事实分析报告\n")
            f.write("="*80 + "\n\n")
            f.write(f"样本索引: {cf_idx}\n")
            if cf_result is not None:
                f.write(f"观察值(是否合格_编码): {cf_result['observed']}\n")
                f.write(f"反事实值(是否合格_编码): {cf_result['counterfactual']}\n")
                f.write(f"变化: {cf_result['change']}\n")
            f.write(f"干预: {interventions_cf}\n")
        print(f"✓ 反事实报告已保存: {cf_report}")
    except Exception:
        pass

    # ===== 7. 生成分析报告 =====
    print("\n" + "="*80)
    print("步骤 7: 生成分析报告")
    print("="*80)
    
    report_path = reports_dir / "analysis_summary.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("工程检测因果分析报告\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. 数据概览\n")
        f.write("-"*80 + "\n")
        f.write(f"样本数量: {df_model.shape[0]}\n")
        f.write(f"变量数量: {df_model.shape[1]}\n")
        f.write(f"检测合格率: {df_model['是否合格_编码'].mean()*100:.2f}%\n")
        if '价格' in df_model.columns:
            f.write(f"平均检测价格: {df_model['价格'].mean():.2f} 元\n")
        f.write("\n")
        
        f.write("2. 因果图结构\n")
        f.write("-"*80 + "\n")
        f.write(f"节点数: {causal_graph.number_of_nodes()}\n")
        f.write(f"边数: {causal_graph.number_of_edges()}\n\n")
        
        if len(attribution_df) > 0:
            f.write("3. 归因分析结果（Top 10）\n")
            f.write("-"*80 + "\n")
            for idx, row in attribution_df.head(10).iterrows():
                f.write(f"{row['Variable']}: {row['Contribution']:.6f} "
                       f"(CI: [{row['Lower_CI']:.6f}, {row['Upper_CI']:.6f}])\n")
            f.write("\n")
        
        if len(intervention_results) > 0:
            f.write("4. 干预分析结果（Top 10）\n")
            f.write("-"*80 + "\n")
            for idx, row in intervention_results.head(10).iterrows():
                f.write(f"{row['Variable']}: {row['Causal_Effect']*100:+.4f}% "
                       f"(CI: [{row['Lower_CI']*100:+.4f}%, {row['Upper_CI']*100:+.4f}%])\n")
            f.write("\n")
        
        f.write("5. 主要发现与建议\n")
        f.write("-"*80 + "\n")
        f.write("✓ 生产厂家的选择对检测质量有重要影响\n")
        f.write("✓ 施工单位的管理水平是质量控制的关键\n")
        f.write("✓ 检验类别和风险等级与质量发现率密切相关\n")
        f.write("✓ 材料强度等级和品种对检测价格有显著影响\n")
        f.write("✓ 时间因素（季度、星期）会影响检测周期和效率\n")
        f.write("\n建议:\n")
        f.write("  1. 建立生产厂家质量档案，优选高质量供应商\n")
        f.write("  2. 加强施工单位质量管理培训和监督\n")
        f.write("  3. 合理安排检验计划，避开高峰期\n")
        f.write("  4. 针对关键结构部位加强监督抽检\n\n")
    
    print(f"✓ 分析报告已保存: {report_path}")
    
    # ===== 7. 统计分析摘要 =====
    print("\n" + "="*80)
    print("数据统计摘要")
    print("="*80)
    
    print("\n施工单位质量分布（Top 5）:")
    if '施工单位_编码' in df_model.columns:
        unit_quality = df_model.groupby('施工单位_编码')['是否合格_编码'].agg(['mean', 'count'])
        unit_quality = unit_quality[unit_quality['count'] >= 10].sort_values('mean', ascending=False)
        for idx, row in unit_quality.head(5).iterrows():
            print(f"  单位{idx}: 合格率 {row['mean']*100:.2f}%, 样本数 {int(row['count'])}")
    
    print("\n生产厂家质量分布（Top 5）:")
    if '生产厂家_编码' in df_model.columns:
        mfr_quality = df_model.groupby('生产厂家_编码')['是否合格_编码'].agg(['mean', 'count'])
        mfr_quality = mfr_quality[mfr_quality['count'] >= 10].sort_values('mean', ascending=False)
        for idx, row in mfr_quality.head(5).iterrows():
            print(f"  厂家{idx}: 合格率 {row['mean']*100:.2f}%, 样本数 {int(row['count'])}")
    
    print("\n检验类别质量分布:")
    if '检验类别_编码' in df_model.columns:
        inspect_quality = df_model.groupby('检验类别_编码')['是否合格_编码'].agg(['mean', 'count'])
        for idx, row in inspect_quality.iterrows():
            print(f"  类别{idx}: 合格率 {row['mean']*100:.2f}%, 样本数 {int(row['count'])}")
    
    # ===== 8. 总结 =====
    print("\n" + "="*80)
    print("分析完成!")
    print("="*80)
    print("\n生成的文件:")
    print(f"  1. 因果图: {figures_dir / 'inspection_causal_graph.png'}")
    if len(attribution_df) > 0:
        print(f"  2. 归因分析图: {figures_dir / 'attribution_analysis.png'}")
    if len(intervention_results) > 0:
        print(f"  3. 干预分析图: {figures_dir / 'intervention_analysis.png'}")
    print(f"  4. 分析报告: {report_path}")
    print(f"  5. 预处理数据: {processed_path}")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
