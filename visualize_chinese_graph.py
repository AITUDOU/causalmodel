"""
生成中文标注的混凝土集料因果图
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
from pathlib import Path

from src.causal_model import ConcreteAggregateCausalModel

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 英文到中文的映射
CHINESE_LABELS = {
    # 集料特性
    'aggregate_type': '集料类型',
    'origin': '产地',
    'production_process': '生产工艺',
    'material_density': '材质密度',
    'particle_shape_index': '粒形指数',
    'crushing_value_pct': '压碎值',
    'flaky_particle_pct': '针片状含量',
    'mud_content_pct': '含泥量',
    'clay_lump_pct': '泥块含量',
    
    # 环境因素
    'ambient_temp_c': '环境温度',
    'ambient_humidity_pct': '环境湿度',
    
    # 砂石比例
    'coarse_aggregate_pct': '碎石比例',
    'sand_pct': '砂比例',
    'sand_rate': '砂率',
    
    # 配合比
    'water_binder_ratio': '水胶比',
    'cement_content': '水泥用量',
    'fly_ash_content': '粉煤灰用量',
    'slag_powder_content': '矿渣粉用量',
    'total_binder': '胶凝材料总量',
    'mix_ratio': '掺配比例',
    'water_content': '水用量',
    'admixture_content': '外加剂用量',
    'mix_design_accuracy_pct': '配合比准确性',
    
    # 结果变量
    'concrete_strength_mpa': '混凝土强度',
    'quality_pass': '质量合格',
    'quality_pass_rate_pct': '质量合格率',
    'pavement_performance_score': '路面性能',
    'structural_strength_mpa': '结构强度'
}


def plot_chinese_causal_graph():
    """绘制中文标注的因果图"""
    
    print("="*80)
    print("生成中文标注的混凝土集料因果图")
    print("="*80)
    
    # 加载数据并构建因果图
    data_path = "data/synthetic/concrete_aggregate_data.csv"
    df = pd.read_csv(data_path)
    
    model = ConcreteAggregateCausalModel(df)
    causal_graph = model.build_causal_graph()
    
    print(f"✓ 因果图加载完成: {causal_graph.number_of_nodes()} 个节点, {causal_graph.number_of_edges()} 条边")
    
    # 创建中文标签映射
    chinese_labels = {node: CHINESE_LABELS.get(node, node) for node in causal_graph.nodes()}
    
    # 创建带中文标签的新图（用于显示）
    display_graph = nx.DiGraph()
    for u, v in causal_graph.edges():
        u_cn = chinese_labels[u]
        v_cn = chinese_labels[v]
        display_graph.add_edge(u_cn, v_cn)
    
    # 设置图形大小
    fig, ax = plt.subplots(figsize=(24, 18))
    
    # 使用层次化布局
    try:
        # 尝试使用 graphviz 的 dot 布局
        pos = nx.nx_agraph.graphviz_layout(display_graph, prog='dot')
    except:
        # 如果 graphviz 不可用，使用 spring 布局
        print("⚠ Graphviz 不可用，使用 spring 布局")
        pos = nx.spring_layout(display_graph, k=3, iterations=50, seed=42)
    
    # 节点分类着色
    node_colors = []
    for node in display_graph.nodes():
        if node in ['集料类型', '产地', '生产工艺', '环境温度', '环境湿度']:
            node_colors.append('#FFB6C1')  # 粉红色 - 外生变量
        elif node in ['材质密度', '粒形指数', '压碎值', '针片状含量', '含泥量', '泥块含量']:
            node_colors.append('#87CEEB')  # 天蓝色 - 集料质量指标
        elif node in ['水胶比', '水泥用量', '粉煤灰用量', '矿渣粉用量', '胶凝材料总量', 
                     '掺配比例', '水用量', '外加剂用量', '配合比准确性']:
            node_colors.append('#98FB98')  # 淡绿色 - 配合比系统
        elif node in ['混凝土强度', '质量合格', '质量合格率', '路面性能', '结构强度']:
            node_colors.append('#FFD700')  # 金色 - 结果变量
        else:
            node_colors.append('#D3D3D3')  # 灰色 - 其他
    
    # 绘制边
    nx.draw_networkx_edges(
        display_graph, pos,
        edge_color='#666666',
        width=1.5,
        alpha=0.6,
        arrows=True,
        arrowsize=15,
        arrowstyle='->',
        connectionstyle='arc3,rad=0.1',
        ax=ax
    )
    
    # 绘制节点
    nx.draw_networkx_nodes(
        display_graph, pos,
        node_color=node_colors,
        node_size=2500,
        alpha=0.9,
        edgecolors='black',
        linewidths=2,
        ax=ax
    )
    
    # 绘制标签
    nx.draw_networkx_labels(
        display_graph, pos,
        font_size=9,
        font_family='sans-serif',
        font_weight='bold',
        ax=ax
    )
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FFB6C1', edgecolor='black', label='外生变量（根节点）'),
        Patch(facecolor='#87CEEB', edgecolor='black', label='集料质量指标'),
        Patch(facecolor='#98FB98', edgecolor='black', label='配合比系统'),
        Patch(facecolor='#FFD700', edgecolor='black', label='结果变量')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12, 
             framealpha=0.9, edgecolor='black')
    
    ax.set_title('混凝土集料因果关系图（中文标注）', 
                fontsize=20, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "concrete_causal_graph_chinese.png"
    
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ 中文因果图已保存: {output_path}")
    
    plt.close()
    
    # 打印关键因果路径（中文）
    print("\n" + "="*80)
    print("主要因果路径:")
    print("="*80)
    
    key_paths = [
        ['集料类型', '材质密度', '混凝土强度'],
        ['产地', '材质密度', '混凝土强度'],
        ['生产工艺', '含泥量', '混凝土强度'],
        ['含泥量', '泥块含量', '混凝土强度'],
        ['水胶比', '粉煤灰用量', '胶凝材料总量', '掺配比例', '配合比准确性', '混凝土强度'],
        ['材质密度', '粒形指数', '混凝土强度'],
        ['材质密度', '压碎值', '混凝土强度'],
        ['混凝土强度', '质量合格率'],
        ['混凝土强度', '结构强度'],
        ['粒形指数', '路面性能']
    ]
    
    for i, path in enumerate(key_paths, 1):
        print(f"{i:2d}. {' → '.join(path)}")
    
    print("\n" + "="*80)
    print("节点分类统计:")
    print("="*80)
    print(f"外生变量（根节点）: 5 个")
    print(f"集料质量指标: 6 个")
    print(f"配合比系统: 9 个")
    print(f"结果变量: 5 个")
    print(f"其他中间变量: {causal_graph.number_of_nodes() - 25} 个")
    print(f"总计: {causal_graph.number_of_nodes()} 个节点, {causal_graph.number_of_edges()} 条边")
    print("="*80)
    
    return display_graph


if __name__ == "__main__":
    graph = plot_chinese_causal_graph()

