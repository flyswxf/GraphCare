#!/usr/bin/env python3
"""
内存优化版本的GraphCare训练脚本
解决CUDA内存不足问题
"""

import os
import torch
import gc
from graphcare import main
import sys

def optimize_memory():
    """设置内存优化参数"""
    # 设置PyTorch CUDA内存分配策略
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # 设置内存增长策略
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def run_with_memory_optimization():
    """使用内存优化参数运行GraphCare"""
    # 应用内存优化
    optimize_memory()
    
    # 修改命令行参数以减少内存使用
    original_argv = sys.argv.copy()
    
    # 设置更小的batch_size和其他内存友好的参数
    memory_optimized_args = [
        'run_memory_optimized.py',
        '--batch_size', '16',  # 从64减少到16
        '--hidden_dim', '64',  # 从128减少到64
        '--num_layers', '2',   # 从3减少到2
        '--epochs', '50',      # 减少训练轮数用于测试
    ]
    
    # 保留其他默认参数
    sys.argv = memory_optimized_args
    
    try:
        print("开始使用内存优化参数训练...")
        print(f"优化参数: batch_size=16, hidden_dim=64, num_layers=2")
        print(f"CUDA内存配置: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
        
        # 运行主程序
        main()
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"仍然遇到内存不足错误: {e}")
        print("建议进一步减少batch_size或使用CPU训练")
        
        # 尝试更激进的内存优化
        print("尝试更小的参数...")
        sys.argv = [
            'run_memory_optimized.py',
            '--batch_size', '8',   # 进一步减少
            '--hidden_dim', '32',  # 进一步减少
            '--num_layers', '1',   # 最小层数
            '--epochs', '10',      # 测试用的小轮数
        ]
        
        # 清理内存后重试
        torch.cuda.empty_cache()
        gc.collect()
        
        try:
            main()
        except Exception as e2:
            print(f"极限优化后仍失败: {e2}")
            print("建议使用CPU训练或升级GPU")
    
    except Exception as e:
        print(f"其他错误: {e}")
    
    finally:
        # 恢复原始参数
        sys.argv = original_argv
        
        # 清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == '__main__':
    run_with_memory_optimization()