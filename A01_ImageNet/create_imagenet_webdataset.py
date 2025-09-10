import os
import argparse
import glob
from tqdm import tqdm
import webdataset as wds

def get_unified_class_mapping(src_path):
    """
    创建统一的类别映射，确保train和val使用相同的类别索引
    
    Args:
        src_path (str): ImageNet源目录路径
    
    Returns:
        dict: 类名到索引的映射字典
    """
    # 获取所有可能的类目录
    train_dirs = glob.glob(os.path.join(src_path, 'train', '*'))
    val_dirs = glob.glob(os.path.join(src_path, 'val', '*'))
    
    # 提取类名
    train_classes = set(os.path.basename(d) for d in train_dirs if os.path.isdir(d))
    val_classes = set(os.path.basename(d) for d in val_dirs if os.path.isdir(d))
    
    # 合并所有类别并排序，确保一致性
    all_classes = sorted(list(train_classes | val_classes))
    
    print(f"发现 {len(all_classes)} 个类别")
    print(f"Train类别数: {len(train_classes)}")
    print(f"Val类别数: {len(val_classes)}")
    
    # 检查是否有不匹配的类别
    train_only = train_classes - val_classes
    val_only = val_classes - train_classes
    
    if train_only:
        print(f"警告: Train独有的类别: {train_only}")
    if val_only:
        print(f"警告: Val独有的类别: {val_only}")
    
    # 创建统一映射
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(all_classes)}
    
    return class_to_idx

def create_webdataset(src_path, dst_path, split, shard_size_gb, class_to_idx):
    """
    使用统一的类别映射转换ImageFolder格式数据集为WebDataset格式

    Args:
        src_path (str): ImageNet源目录路径
        dst_path (str): WebDataset输出目录路径
        split (str): 'train' 或 'val'
        shard_size_gb (int): 每个分片的大小(GB)
        class_to_idx (dict): 统一的类别映射
    """
    # 确保输出目录存在
    split_dst_path = os.path.join(dst_path, split)
    os.makedirs(split_dst_path, exist_ok=True)

    # 查找所有类目录
    class_dirs = sorted(glob.glob(os.path.join(src_path, split, '*')))
    
    # 过滤出实际存在于映射中的类目录
    valid_class_dirs = []
    for class_dir in class_dirs:
        if os.path.isdir(class_dir):
            class_name = os.path.basename(class_dir)
            if class_name in class_to_idx:
                valid_class_dirs.append(class_dir)
            else:
                print(f"警告: 类别 {class_name} 不在统一映射中，跳过")

    # 输出分片文件模式
    pattern = os.path.join(split_dst_path, f"imagenet-{split}-%06d.tar")

    # 设置每个分片的最大大小
    maxsize = shard_size_gb * 1e9  # 转换为字节

    with wds.ShardWriter(pattern, maxsize=maxsize) as sink:
        # 遍历每个类目录
        for class_dir in tqdm(valid_class_dirs, desc=f"处理 {split} 数据集"):
            class_name = os.path.basename(class_dir)
            class_idx = class_to_idx[class_name]

            # 查找类目录中的所有图像文件
            image_files = glob.glob(os.path.join(class_dir, '*.JPEG'))
            
            if not image_files:
                print(f"警告: 类别 {class_name} 中没有找到JPEG文件")
                continue

            # 遍历每个图像文件
            for image_path in image_files:
                try:
                    # 读取图像数据
                    with open(image_path, "rb") as stream:
                        image_bytes = stream.read()

                    # 样本键为文件名(不含扩展名)
                    sample_key = os.path.splitext(os.path.basename(image_path))[0]

                    # 创建样本字典
                    sample = {
                        "__key__": sample_key,
                        "jpg": image_bytes,
                        "cls": class_idx,
                    }

                    # 写入分片
                    sink.write(sample)
                    
                except Exception as e:
                    print(f"处理文件 {image_path} 时出错: {e}")
                    continue

    print(f"{split} 数据集转换完成")

def main():
    parser = argparse.ArgumentParser(description="将ImageNet转换为WebDataset格式(修复版本)")
    parser.add_argument("--src", default="/data0/share/datasets/ImageNet", help="ImageNet源目录")
    parser.add_argument("--dst", default="/data0/share/datasets/ImageNet-WebDataset", help="WebDataset输出目录")
    parser.add_argument("--shard_size", type=int, default=2, help="分片大小(GB)")

    args = parser.parse_args()

    print("开始分析类别映射...")
    class_to_idx = get_unified_class_mapping(args.src)
    
    print(f"\n类别映射创建完成，共 {len(class_to_idx)} 个类别")
    print(f"标签范围: 0 到 {len(class_to_idx)-1}")
    
    # 保存类别映射到文件
    mapping_file = os.path.join(args.dst, "class_mapping.txt")
    os.makedirs(args.dst, exist_ok=True)
    with open(mapping_file, 'w') as f:
        for class_name, idx in sorted(class_to_idx.items(), key=lambda x: x[1]):
            f.write(f"{idx}\t{class_name}\n")
    print(f"类别映射已保存到: {mapping_file}")

    print("\n开始转换 'train' 数据集...")
    create_webdataset(args.src, args.dst, 'train', args.shard_size, class_to_idx)

    print("\n开始转换 'val' 数据集...")
    create_webdataset(args.src, args.dst, 'val', args.shard_size, class_to_idx)

    print("\n转换完成!")
    print(f"类别映射文件: {mapping_file}")
    print(f"输出目录: {args.dst}")

if __name__ == '__main__':
    main()