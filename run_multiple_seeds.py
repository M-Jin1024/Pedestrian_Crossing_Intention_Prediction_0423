#!/usr/bin/env python
"""
运行多个不同seed值的训练和测试脚本
使用方法: python run_multiple_seeds.py -c config_files/my/my_jaad.yaml --seeds 42 123 456
"""

import subprocess
import sys
import argparse
import yaml
import os
import time
from datetime import datetime


def find_latest_model_dir(base_path="data/models"):
    """找到最新创建的模型目录"""
    if not os.path.exists(base_path):
        return None
    
    model_dirs = []
    for root, dirs, files in os.walk(base_path):
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


def run_with_seed(config_file, seed_value, run_index, total_runs, skip_test=False):
    """
    使用指定的seed值运行训练和测试
    
    Args:
        config_file: 配置文件路径
        seed_value: seed值
        run_index: 当前运行索引（从1开始）
        total_runs: 总运行次数
        skip_test: 是否跳过测试
    """
    print(f"\n{'='*80}")
    print(f"🚀 开始运行 [{run_index}/{total_runs}] - Seed: {seed_value}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # 读取原始配置文件
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # 更新seed值
    if 'exp_opts' not in config_data:
        config_data['exp_opts'] = {}
    config_data['exp_opts']['seed'] = seed_value
    
    # 创建临时配置文件
    temp_config_file = config_file.replace('.yaml', f'_seed_{seed_value}_temp.yaml')
    with open(temp_config_file, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
    
    print(f"📄 临时配置文件已创建: {temp_config_file}")
    print(f"🎯 使用 Seed = {seed_value} 开始训练...\n")
    
    try:
        # ========== 1. 运行训练 ==========
        train_start_time = time.time()
        result = subprocess.run([
            sys.executable,
            'train_test.py',
            '-c', temp_config_file
        ])
        train_end_time = time.time()
        
        # 清理临时配置文件
        if os.path.exists(temp_config_file):
            os.remove(temp_config_file)
            print(f"\n🗑️  临时配置文件已删除: {temp_config_file}")
        
        if result.returncode != 0:
            print(f"\n❌ Seed {seed_value} 的训练失败，返回码: {result.returncode}")
            return False, None
        
        print(f"\n✅ Seed {seed_value} 的训练完成 (耗时: {(train_end_time - train_start_time) / 60:.1f} 分钟)")
        
        # ========== 2. 运行测试 ==========
        if not skip_test:
            print(f"\n🔍 查找最新模型目录...")
            model_dir = find_latest_model_dir()
            
            if not model_dir:
                print("⚠️  未找到模型目录，跳过测试")
                return True, None
            
            print(f"📁 找到模型目录: {model_dir}")
            print(f"\n🧪 开始测试模型...\n")
            
            test_start_time = time.time()
            test_result = subprocess.run([
                sys.executable,
                'compare_all_epochs.py',
                '-d', model_dir
            ])
            test_end_time = time.time()
            
            if test_result.returncode != 0:
                print(f"\n⚠️  Seed {seed_value} 的测试失败，返回码: {test_result.returncode}")
                print(f"   但训练成功，继续下一个seed")
                return True, model_dir
            
            print(f"\n✅ Seed {seed_value} 的测试完成 (耗时: {(test_end_time - test_start_time) / 60:.1f} 分钟)")
            
            total_time = (test_end_time - train_start_time) / 60
            print(f"\n{'='*80}")
            print(f"✓ Seed {seed_value} 的训练和测试全部完成 (总耗时: {total_time:.1f} 分钟)")
            print(f"{'='*80}")
            
            return True, model_dir
        else:
            return True, None
            
    except KeyboardInterrupt:
        print(f"\n⚠️  用户中断了运行")
        # 清理临时配置文件
        if os.path.exists(temp_config_file):
            os.remove(temp_config_file)
        raise
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        # 清理临时配置文件
        if os.path.exists(temp_config_file):
            os.remove(temp_config_file)
        return False, None


def main():
    parser = argparse.ArgumentParser(
        description='使用多个不同的seed值依次运行训练和测试',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 使用默认的seed值列表并进行训练和测试
  python run_multiple_seeds.py -c config_files/my/my_jaad.yaml
  
  # 指定自定义的seed值列表
  python run_multiple_seeds.py -c config_files/my/my_jaad.yaml --seeds 42 100 200
  
  # 只训练，不测试
  python run_multiple_seeds.py -c config_files/my/my_jaad.yaml --skip-test
  
  # 失败后继续运行其他seed
  python run_multiple_seeds.py -c config_files/my/my_jaad.yaml --continue-on-error
        '''
    )
    parser.add_argument('-c', '--config', required=True, 
                       help='配置文件路径')
    parser.add_argument('--seeds', type=int, nargs='+', 
                       default=[42, 43, 44, 45, 46],
                       help='要使用的seed值列表 (默认: 42, 43, 44, 45, 46)')
    parser.add_argument('--skip-test', action='store_true',
                       help='跳过测试，只进行训练')
    parser.add_argument('--continue-on-error', action='store_true',
                       help='如果某个seed运行失败，继续运行后续的seed')
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"❌ 错误: 配置文件不存在: {args.config}")
        sys.exit(1)
    
    seeds = args.seeds
    total_runs = len(seeds)
    
    print(f"\n{'='*80}")
    print(f"🎯 多Seed训练{'和测试' if not args.skip_test else ''}脚本")
    print(f"{'='*80}")
    print(f"配置文件: {args.config}")
    print(f"Seed列表: {seeds}")
    print(f"总运行次数: {total_runs}")
    print(f"运行模式: {'仅训练' if args.skip_test else '训练+测试'}")
    print(f"失败后继续: {'是' if args.continue_on_error else '否'}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    success_count = 0
    failed_seeds = []
    model_dirs = []
    
    try:
        for idx, seed in enumerate(seeds, start=1):
            success, model_dir = run_with_seed(args.config, seed, idx, total_runs, args.skip_test)
            
            if success:
                success_count += 1
                if model_dir:
                    model_dirs.append((seed, model_dir))
            else:
                failed_seeds.append(seed)
                if not args.continue_on_error:
                    print(f"\n⚠️  由于 seed {seed} 运行失败，停止后续运行")
                    break
    
    except KeyboardInterrupt:
        print(f"\n\n⚠️  用户中断了脚本执行")
    
    # 打印总结
    print(f"\n\n{'='*80}")
    print(f"📊 运行总结")
    print(f"{'='*80}")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"成功运行: {success_count}/{total_runs}")
    
    if failed_seeds:
        print(f"失败的seeds: {failed_seeds}")
    else:
        print(f"✓ 所有seed都运行成功!")
    
    if model_dirs:
        print(f"\n已生成的模型目录:")
        for seed, model_dir in model_dirs:
            print(f"  Seed {seed}: {model_dir}")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
