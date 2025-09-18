#!/usr/bin/env python3
"""
GPU显存监控脚本
用于检查当前GPU显存使用状态，包括总显存、已用显存、可用显存等信息
"""

import os
import sys
import subprocess
import json
from typing import Dict, List, Optional
import time
import argparse


def check_nvidia_smi() -> bool:
    """检查nvidia-smi是否可用"""
    try:
        subprocess.run(['nvidia-smi'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_gpu_memory_info() -> List[Dict]:
    """获取GPU显存信息"""
    if not check_nvidia_smi():
        print("错误: nvidia-smi不可用，请确保安装了NVIDIA驱动")
        return []
    
    try:
        # 使用nvidia-smi获取GPU信息
        cmd = [
            'nvidia-smi',
            '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu',
            '--format=csv,noheader,nounits'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        
        gpu_info = []
        for line in lines:
            if line.strip():
                parts = [part.strip() for part in line.split(',')]
                if len(parts) >= 7:
                    gpu_info.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'memory_total_mb': int(parts[2]),
                        'memory_used_mb': int(parts[3]),
                        'memory_free_mb': int(parts[4]),
                        'gpu_utilization': int(parts[5]),
                        'temperature': int(parts[6]) if parts[6] != '[Not Supported]' else None
                    })
        
        return gpu_info
        
    except subprocess.CalledProcessError as e:
        print(f"执行nvidia-smi时出错: {e}")
        return []
    except Exception as e:
        print(f"解析GPU信息时出错: {e}")
        return []


def get_pytorch_memory_info() -> Optional[Dict]:
    """获取PyTorch显存使用信息"""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        
        device_count = torch.cuda.device_count()
        pytorch_info = {}
        
        for i in range(device_count):
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**2  # MB
            cached = torch.cuda.memory_reserved(i) / 1024**2  # MB
            max_allocated = torch.cuda.max_memory_allocated(i) / 1024**2  # MB
            max_cached = torch.cuda.max_memory_reserved(i) / 1024**2  # MB
            
            pytorch_info[f'gpu_{i}'] = {
                'allocated_mb': allocated,
                'cached_mb': cached,
                'max_allocated_mb': max_allocated,
                'max_cached_mb': max_cached
            }
        
        return pytorch_info
        
    except ImportError:
        return None
    except Exception as e:
        print(f"获取PyTorch显存信息时出错: {e}")
        return None


def format_memory_size(mb: float) -> str:
    """格式化显存大小显示"""
    if mb >= 1024:
        return f"{mb/1024:.1f} GB"
    else:
        return f"{mb:.0f} MB"


def print_gpu_status(gpu_info: List[Dict], pytorch_info: Optional[Dict] = None):
    """打印GPU状态信息"""
    if not gpu_info:
        print("未检测到GPU或无法获取GPU信息")
        return
    
    print("=" * 80)
    print("GPU 显存状态监控")
    print("=" * 80)
    
    for gpu in gpu_info:
        idx = gpu['index']
        name = gpu['name']
        total_mb = gpu['memory_total_mb']
        used_mb = gpu['memory_used_mb']
        free_mb = gpu['memory_free_mb']
        utilization = gpu['gpu_utilization']
        temp = gpu['temperature']
        
        usage_percent = (used_mb / total_mb) * 100 if total_mb > 0 else 0
        
        print(f"\nGPU {idx}: {name}")
        print(f"  总显存:     {format_memory_size(total_mb)}")
        print(f"  已用显存:   {format_memory_size(used_mb)} ({usage_percent:.1f}%)")
        print(f"  可用显存:   {format_memory_size(free_mb)}")
        print(f"  GPU利用率:  {utilization}%")
        if temp is not None:
            print(f"  温度:       {temp}°C")
        
        # 显示PyTorch显存信息
        if pytorch_info and f'gpu_{idx}' in pytorch_info:
            pt_info = pytorch_info[f'gpu_{idx}']
            print(f"  PyTorch分配: {format_memory_size(pt_info['allocated_mb'])}")
            print(f"  PyTorch缓存: {format_memory_size(pt_info['cached_mb'])}")
            print(f"  PyTorch峰值: {format_memory_size(pt_info['max_allocated_mb'])}")
        
        # 显存使用状态条
        bar_length = 50
        used_bars = int((used_mb / total_mb) * bar_length)
        free_bars = bar_length - used_bars
        bar = "█" * used_bars + "░" * free_bars
        print(f"  显存使用:   [{bar}] {usage_percent:.1f}%")


def monitor_continuously(interval: int = 2):
    """连续监控显存状态"""
    print("开始连续监控GPU显存状态 (按Ctrl+C停止)")
    print(f"刷新间隔: {interval}秒")
    
    try:
        while True:
            # 清屏
            os.system('clear' if os.name == 'posix' else 'cls')
            
            gpu_info = get_gpu_memory_info()
            pytorch_info = get_pytorch_memory_info()
            
            print_gpu_status(gpu_info, pytorch_info)
            print(f"\n刷新时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("按Ctrl+C停止监控")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n监控已停止")


def save_to_json(filename: str):
    """将GPU信息保存到JSON文件"""
    gpu_info = get_gpu_memory_info()
    pytorch_info = get_pytorch_memory_info()
    
    data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'gpu_info': gpu_info,
        'pytorch_info': pytorch_info
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"GPU信息已保存到: {filename}")


def main():
    parser = argparse.ArgumentParser(description='GPU显存监控工具')
    parser.add_argument('--monitor', '-m', action='store_true', 
                       help='连续监控模式')
    parser.add_argument('--interval', '-i', type=int, default=2,
                       help='监控刷新间隔(秒), 默认2秒')
    parser.add_argument('--save', '-s', type=str,
                       help='保存GPU信息到JSON文件')
    parser.add_argument('--pytorch', '-p', action='store_true',
                       help='显示PyTorch显存信息')
    
    args = parser.parse_args()
    
    if args.save:
        save_to_json(args.save)
    elif args.monitor:
        monitor_continuously(args.interval)
    else:
        gpu_info = get_gpu_memory_info()
        pytorch_info = get_pytorch_memory_info() if args.pytorch else None
        print_gpu_status(gpu_info, pytorch_info)


if __name__ == "__main__":
    main()