import numpy as np
from scipy.spatial.distance import pdist, squareform
import os
import pandas as pd
import json


def compute_rdm_pearson(vectors):
    """
    Compute RDM matrix based on Pearson correlation distance
    
    Parameters:
    vectors: numpy array, shape (n_items, n_features)
    
    Returns:
    rdm: numpy array, shape (n_items, n_items) - RDM matrix
    """
    print(f"Computing RDM matrix, input shape: {vectors.shape}")
    
    # Use scipy's pdist to compute Pearson correlation distance
    # 'correlation' distance = 1 - Pearson correlation coefficient
    distances = pdist(vectors, metric='correlation')
    
    # Convert to square matrix
    rdm = squareform(distances)
    
    print(f"RDM matrix shape: {rdm.shape}")
    return rdm


def compute_cats_rdm_averaged(cats_vectors):
    """
    为CATS向量计算平均的RDM矩阵
    遍历第一维，每次使用第二维的数值，然后跨第一维进行平均
    
    Parameters:
    cats_vectors: numpy array, shape (30, 20, 334)
    
    Returns:
    rdm_averaged: numpy array, shape (334, 334) - 平均后的RDM矩阵
    """
    print(f"CATS向量形状: {cats_vectors.shape}")
    
    if len(cats_vectors.shape) != 3:
        raise ValueError(f"CATS向量应该是3维的，当前形状: {cats_vectors.shape}")
    
    n_first_dim, n_second_dim, n_features = cats_vectors.shape
    print(f"第一维: {n_first_dim}, 第二维: {n_second_dim}, 特征维: {n_features}")
    
    # 存储所有RDM矩阵
    rdm_matrices = []
    
    # 遍历第一维（30）
    for i in range(n_first_dim):
        # 获取当前第一维索引下的数据，形状为 (20, 334)
        current_vectors = cats_vectors[i, :, :]  # shape: (20, 334)
        
        # 转置以获得 (334, 20) 的形状，然后计算334个特征之间的RDM
        features_vectors = current_vectors.T  # shape: (334, 20)
        
        # 计算当前的RDM矩阵 (334, 334)
        rdm_current = compute_rdm_pearson(features_vectors)
        rdm_matrices.append(rdm_current)
        
        print(f"完成第 {i+1}/{n_first_dim} 个RDM计算")
    
    # 将所有RDM矩阵堆叠并计算平均值
    rdm_stack = np.stack(rdm_matrices, axis=0)  # shape: (30, 334, 334)
    rdm_averaged = np.mean(rdm_stack, axis=0)  # shape: (334, 334)
    
    print(f"平均RDM矩阵形状: {rdm_averaged.shape}")
    return rdm_averaged, rdm_stack


def load_vectors_from_csv(csv_path, vector_name):
    """
    从CSV文件中加载向量数据
    
    Parameters:
    csv_path: str, CSV文件路径
    vector_name: str, 向量名称（用于日志输出）
    
    Returns:
    vectors: numpy array, 向量矩阵
    metadata: dict, 包含wordnet_id, things_index, imagenet_index的信息
    """
    print(f"从CSV文件加载 {vector_name} 向量: {csv_path}")
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    print(f"CSV文件形状: {df.shape}")
    
    # 提取元数据列
    metadata = {
        'wordnet_id': df['wordnet_id'].values if 'wordnet_id' in df.columns else None,
        'things_index': df['things_index'].values if 'things_index' in df.columns else None,
        'imagenet_index': df['imagenet_index'].values if 'imagenet_index' in df.columns else None
    }
    
    # 提取向量数据（排除元数据列）
    metadata_columns = ['wordnet_id', 'things_index', 'imagenet_index']
    vector_columns = [col for col in df.columns if col not in metadata_columns]
    
    vectors = df[vector_columns].values
    print(f"提取的向量形状: {vectors.shape}")
    
    return vectors, metadata


def reshape_cats_vectors(vectors_2d, n_trails=30):
    """
    将CATS向量从2D CSV格式重塑为3D格式
    
    Parameters:
    vectors_2d: numpy array, shape (n_samples, n_trails * vector_dim)
    n_trails: int, trail的数量（默认30）
    
    Returns:
    vectors_3d: numpy array, shape (n_trails, vector_dim, n_samples)
    """
    n_samples, total_dims = vectors_2d.shape
    vector_dim = total_dims // n_trails
    
    print(f"重塑CATS向量: ({n_samples}, {total_dims}) -> ({n_trails}, {vector_dim}, {n_samples})")
    
    # 重塑为 (n_samples, n_trails, vector_dim)
    vectors_reshaped = vectors_2d.reshape(n_samples, n_trails, vector_dim)
    
    # 转置为 (n_trails, vector_dim, n_samples)
    vectors_3d = vectors_reshaped.transpose(1, 2, 0)
    
    print(f"重塑后的形状: {vectors_3d.shape}")
    return vectors_3d


def main():
    """
    主函数：从CSV文件计算RDM矩阵并保存
    """
    print("=" * 60)
    print("RDM矩阵计算脚本")
    print("=" * 60)
    
    # Create output directory
    output_dir = "./RDM_analyses"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # Check SharedVec folder
    shared_vec_path = "./SharedVec"
    if not os.path.exists(shared_vec_path):
        print(f"错误: {shared_vec_path} 文件夹不存在")
        return
    
    # 读取三个CSV文件
    files_to_process = {
        'CATS': 'CATS_vectors.csv',
        'THINGS': 'THINGS_vectors.csv', 
        'Binder65': 'Binder65_vectors.csv'
    }
    
    results = {}
    cats_rdm_stack = None
    metadata_info = {}
    
    for name, filename in files_to_process.items():
        filepath = os.path.join(shared_vec_path, filename)
        
        if not os.path.exists(filepath):
            print(f"警告: {filepath} 文件不存在，跳过")
            continue
        
        print("\n" + "=" * 60)
        print(f"处理 {name} 向量")
        print("=" * 60)
        
        try:
            # 从CSV文件读取向量
            vectors, metadata = load_vectors_from_csv(filepath, name)
            metadata_info[name] = metadata
            
            # 根据不同类型计算RDM
            if name == 'CATS':
                # CATS特殊处理：从CSV读取的是2D，需要重塑为3D
                vectors_3d = reshape_cats_vectors(vectors, n_trails=30)
                
                # 计算平均RDM和所有30个模型的RDM堆叠
                rdm_averaged, cats_rdm_stack = compute_cats_rdm_averaged(vectors_3d)
                results[name] = rdm_averaged
                
            else:
                # THINGS和Binder65向量：二维数据，直接计算
                if len(vectors.shape) == 2:
                    n_samples = vectors.shape[0]
                    print(f"样本数量: {n_samples}")
                    rdm = compute_rdm_pearson(vectors)
                    results[name] = rdm
                else:
                    print(f"警告: {name} 向量的形状不是2维: {vectors.shape}")
                    continue
            
            # 显示统计信息
            if name in results:
                rdm = results[name]
                print(f"\nRDM统计信息:")
                print(f"  形状: {rdm.shape}")
                print(f"  最小值: {rdm.min():.4f}")
                print(f"  最大值: {rdm.max():.4f}")
                print(f"  平均值: {rdm.mean():.4f}")
                print(f"  对角线元素平均值 (应接近0): {np.diag(rdm).mean():.6f}")
            
        except Exception as e:
            print(f"处理 {name} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存结果
    print("\n" + "=" * 60)
    print("保存RDM矩阵")
    print("=" * 60)
    
    # 1. 保存CATS平均RDM、THINGS RDM、Binder65 RDM为CSV格式
    for name, rdm in results.items():
        rdm_df = pd.DataFrame(rdm)
        output_csv = os.path.join(output_dir, f"{name}_RDM.csv")
        rdm_df.to_csv(output_csv, index=False, header=False)
        print(f"✓ {name} RDM 已保存为CSV: {output_csv}")
    
    # 2. 保存CATS 30个模型的RDM堆叠为NPY格式
    if cats_rdm_stack is not None:
        cats_stack_path = os.path.join(output_dir, "CATS_RDM_stack_30models.npy")
        np.save(cats_stack_path, cats_rdm_stack)
        print(f"✓ CATS 30个模型的RDM堆叠已保存为NPY: {cats_stack_path}")
        print(f"  形状: {cats_rdm_stack.shape}")
    
    # 3. 保存元数据信息（可选）
    if metadata_info:
        metadata_path = os.path.join(output_dir, "rdm_metadata.json")
        metadata_serializable = {}
        for key, meta in metadata_info.items():
            metadata_serializable[key] = {
                'wordnet_id': meta['wordnet_id'].tolist() if meta['wordnet_id'] is not None else None,
                'things_index': meta['things_index'].tolist() if meta['things_index'] is not None else None,
                'imagenet_index': meta['imagenet_index'].tolist() if meta['imagenet_index'] is not None else None
            }
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_serializable, f, indent=2, ensure_ascii=False)
        print(f"✓ 元数据信息已保存: {metadata_path}")
    
    print("\n" + "=" * 60)
    print("RDM计算完成！")
    print("=" * 60)
    print(f"输出目录: {output_dir}")
    print(f"生成文件:")
    print(f"  - CATS_RDM.csv (平均RDM)")
    print(f"  - THINGS_RDM.csv")
    print(f"  - Binder65_RDM.csv")
    print(f"  - CATS_RDM_stack_30models.npy (30个模型的RDM堆叠)")
    print(f"  - rdm_metadata.json (元数据)")


if __name__ == "__main__":
    main()
