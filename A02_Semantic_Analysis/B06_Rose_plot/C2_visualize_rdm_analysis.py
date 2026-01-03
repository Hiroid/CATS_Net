import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os


def load_rdm_from_csv(csv_path, name):
    """
    从CSV文件加载RDM矩阵
    
    Parameters:
    csv_path: str, CSV文件路径
    name: str, RDM名称
    
    Returns:
    rdm: numpy array, RDM矩阵
    """
    print(f"加载 {name} RDM: {csv_path}")
    rdm = pd.read_csv(csv_path, header=None).values
    print(f"  形状: {rdm.shape}")
    print(f"  范围: [{rdm.min():.4f}, {rdm.max():.4f}]")
    return rdm


def rdm_to_percentile(rdm):
    """
    将RDM矩阵的值转换为百分位数(0-100)
    
    Parameters:
    rdm: numpy array, RDM矩阵
    
    Returns:
    rdm_percentile: numpy array, 百分位数表示的RDM矩阵
    """
    # 展平矩阵
    rdm_flat = rdm.flatten()
    
    # 计算每个值的百分位数
    rdm_percentile = np.zeros_like(rdm)
    for i in range(rdm.shape[0]):
        for j in range(rdm.shape[1]):
            # 计算小于等于当前值的元素比例
            percentile = (rdm_flat <= rdm[i, j]).sum() / len(rdm_flat) * 100
            rdm_percentile[i, j] = percentile
    
    return rdm_percentile


def plot_rdm_matrices(rdm_dict, save_path="RDM_visualization.png"):
    """
    绘制RDM矩阵热图（参照提供的样式）
    
    Parameters:
    rdm_dict: dict, 包含RDM矩阵的字典 {name: rdm_array}
    save_path: str, 保存路径
    """
    print("\n开始绘制RDM矩阵...")
    
    # 转换所有RDM为百分位数
    rdm_percentile_dict = {}
    for name, rdm in rdm_dict.items():
        rdm_percentile_dict[name] = rdm_to_percentile(rdm)
        print(f"{name} 转换为百分位数: [{rdm_percentile_dict[name].min():.2f}, {rdm_percentile_dict[name].max():.2f}]")
    
    n_rdms = len(rdm_percentile_dict)
    
    # 创建图形，调整subplot间距以容纳共享的颜色条
    fig = plt.figure(figsize=(6*n_rdms + 1, 5))
    
    # 创建子图
    axes = []
    for idx in range(n_rdms):
        ax = fig.add_subplot(1, n_rdms, idx + 1)
        axes.append(ax)
    
    # 定义颜色映射（蓝色到橙色/棕色）
    colors = ['#08519c', '#3182bd', '#6baed6', '#9ecae1', '#c6dbef', 
              '#fee6ce', '#fdbe85', '#fd8d3c', '#e6550d', '#a63603']
    cmap = LinearSegmentedColormap.from_list('custom_blue_orange', colors, N=256)
    
    # 标题映射
    title_map = {
        'CATS': 'Averaged CATS RDM',
        'THINGS': 'THINGS RDM',
        'Binder65': 'Binder65 RDM'
    }
    
    # 绘制每个RDM矩阵
    for idx, (name, rdm_percentile) in enumerate(rdm_percentile_dict.items()):
        ax = axes[idx]
        
        # 绘制热图
        im = ax.imshow(rdm_percentile, cmap=cmap, aspect='equal', 
                      interpolation='nearest', vmin=0, vmax=100)
        
        # 设置标题
        title = title_map.get(name, f'{name} RDM')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=10)
        
        # 移除刻度
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 移除边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
    
    # 添加共享的颜色条（在最右侧）
    # 调整子图位置以留出空间给颜色条
    plt.subplots_adjust(right=0.92, wspace=0.1)
    
    # 创建颜色条的轴
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Percentile (%)', rotation=270, labelpad=25, fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    
    # 保存图形
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ RDM可视化已保存: {save_path}")
    plt.show()
    plt.close()


def main():
    """
    主函数
    """
    print("=" * 60)
    print("RDM矩阵可视化")
    print("=" * 60)
    
    # 设置输入/输出目录
    input_dir = "./RDM_analyses"
    output_dir = "./RDM_analyses"
    
    if not os.path.exists(input_dir):
        print(f"错误: {input_dir} 文件夹不存在")
        return
    
    # 定义要读取的RDM文件（按照图片中的顺序）
    rdm_files = {
        'CATS': 'CATS_RDM.csv',
        'THINGS': 'THINGS_RDM.csv',
        'Binder65': 'Binder65_RDM.csv'
    }
    
    # 加载RDM矩阵
    print("\n加载RDM矩阵...")
    rdm_dict = {}
    for name, filename in rdm_files.items():
        filepath = os.path.join(input_dir, filename)
        if os.path.exists(filepath):
            rdm_dict[name] = load_rdm_from_csv(filepath, name)
        else:
            print(f"警告: {filepath} 不存在，跳过")
    
    if len(rdm_dict) == 0:
        print("错误: 没有找到任何RDM文件")
        return
    
    print(f"\n成功加载 {len(rdm_dict)} 个RDM矩阵")
    
    # 绘制RDM矩阵
    viz_path = os.path.join(output_dir, "RDM_visualization.png")
    plot_rdm_matrices(rdm_dict, viz_path)
    
    print("\n" + "=" * 60)
    print("可视化完成！")
    print("=" * 60)
    print(f"输出文件: {viz_path}")


if __name__ == "__main__":
    main()
