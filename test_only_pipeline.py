#!/usr/bin/env python3
"""
仅测试管道脚本 - 用于已训练完成的模型
对指定的模型目录进行测试，类似 compare_all_epochs.py 的功能
"""

import os
import sys
import subprocess
import argparse
import time
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description='仅测试管道 - 测试已训练完成的模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 测试指定模型目录
  python test_only_pipeline.py -m data/models/jaad/Transformer_depth/12Nov2025-07h36m46s
  
  # 自动查找最新的模型目录进行测试
  python test_only_pipeline.py --latest
  
  # 测试指定数据集类型的最新模型
  python test_only_pipeline.py --latest --dataset jaad
        '''
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-m', '--model-dir', 
                      help='要测试的模型目录路径')
    group.add_argument('--latest', action='store_true',
                      help='自动查找并测试最新的模型目录')
    
    parser.add_argument('--dataset', 
                       choices=['jaad', 'pie'],
                       help='指定数据集类型（与--latest一起使用）')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🧪 测试管道启动")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 确定要测试的模型目录
    if args.latest:
        print("\n🔍 查找最新模型目录...")
        model_dir = find_latest_model_dir(args.dataset)
        if not model_dir:
            print("❌ 未找到模型目录")
            sys.exit(1)
        print(f"✅ 找到最新模型: {model_dir}")
    else:
        model_dir = args.model_dir
        if not os.path.exists(model_dir):
            print(f"❌ 错误: 模型目录不存在: {model_dir}")
            sys.exit(1)
        print(f"\n📁 模型目录: {model_dir}")
    
    # 检查模型目录是否有效
    if not os.path.exists(os.path.join(model_dir, 'configs.yaml')):
        print(f"❌ 错误: 模型目录中缺少 configs.yaml 文件")
        sys.exit(1)
    
    # 运行测试
    print(f"\n🧪 开始测试模型...\n")
    test_cmd = [sys.executable, "compare_all_epochs.py", "-d", model_dir]
    
    start_time = time.time()
    test_result = subprocess.run(test_cmd)
    end_time = time.time()
    
    if test_result.returncode != 0:
        print(f"\n❌ 测试失败，退出码: {test_result.returncode}")
        sys.exit(1)
    
    print(f"\n✅ 测试完成 (耗时: {(end_time - start_time) / 60:.1f} 分钟)")
    
    # 完成
    print("\n" + "=" * 80)
    print("🎉 测试管道完成!")
    print("=" * 80)
    print(f"模型目录: {model_dir}")
    print(f"测试耗时: {(end_time - start_time) / 60:.1f} 分钟")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


def find_latest_model_dir(dataset=None):
    """
    找到最新创建的模型目录
    
    Args:
        dataset: 指定数据集类型 ('jaad' 或 'pie')，None表示所有数据集
    """
    base_path = "data/models"
    
    if not os.path.exists(base_path):
        return None
    
    # 如果指定了数据集，限定搜索路径
    if dataset:
        search_paths = [os.path.join(base_path, dataset)]
    else:
        search_paths = [base_path]
    
    model_dirs = []
    for search_path in search_paths:
        if not os.path.exists(search_path):
            continue
            
        for root, dirs, files in os.walk(search_path):
            for dir_name in dirs:
                full_path = os.path.join(root, dir_name)
                try:
                    # 检查是否包含 configs.yaml（表示这是一个有效的模型目录）
                    if os.path.exists(os.path.join(full_path, 'configs.yaml')):
                        # 检查是否有模型文件
                        has_model = (
                            os.path.exists(os.path.join(full_path, 'model.h5')) or
                            os.path.exists(os.path.join(full_path, 'best.h5')) or
                            os.path.exists(os.path.join(full_path, 'epochs'))
                        )
                        if has_model:
                            model_dirs.append(full_path)
                except (OSError, PermissionError):
                    continue
    
    if not model_dirs:
        return None
    
    # 按修改时间排序，返回最新的
    model_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return model_dirs[0]


if __name__ == '__main__':
    main()
