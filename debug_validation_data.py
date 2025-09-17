#!/usr/bin/env python3
"""
调试脚本：保存验证数据的类型和值信息
用于分析runSparseModel.py第477-486行的类型错误问题
"""

import numpy as np
import pickle
import json
import os
from datetime import datetime


def save_validation_debug_info(y_true_val, y_prob_val, epoch=None, task=None, mode=None):
    """
    保存验证数据的详细信息到samples文件夹
    
    Args:
        y_true_val: 真实标签
        y_prob_val: 预测概率
        epoch: 当前epoch（可选）
        task: 任务名称（可选）
        mode: 模式（binary/multilabel/multiclass）
    """
    
    # 创建时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建保存目录
    save_dir = "samples"
    os.makedirs(save_dir, exist_ok=True)
    
    # 收集详细的类型和形状信息
    debug_info = {
        "timestamp": timestamp,
        "epoch": epoch,
        "task": task,
        "mode": mode,
        "y_true_val": {
            "type": str(type(y_true_val)),
            "dtype": str(y_true_val.dtype) if hasattr(y_true_val, 'dtype') else "N/A",
            "shape": y_true_val.shape if hasattr(y_true_val, 'shape') else "N/A",
            "size": y_true_val.size if hasattr(y_true_val, 'size') else "N/A",
            "ndim": y_true_val.ndim if hasattr(y_true_val, 'ndim') else "N/A",
            "min_val": float(np.min(y_true_val)) if hasattr(y_true_val, 'min') else "N/A",
            "max_val": float(np.max(y_true_val)) if hasattr(y_true_val, 'max') else "N/A",
            "unique_values": np.unique(y_true_val).tolist() if hasattr(y_true_val, 'unique') else "N/A",
            "first_10_values": y_true_val.flatten()[:10].tolist() if hasattr(y_true_val, 'flatten') else "N/A"
        },
        "y_prob_val": {
            "type": str(type(y_prob_val)),
            "dtype": str(y_prob_val.dtype) if hasattr(y_prob_val, 'dtype') else "N/A",
            "shape": y_prob_val.shape if hasattr(y_prob_val, 'shape') else "N/A",
            "size": y_prob_val.size if hasattr(y_prob_val, 'size') else "N/A",
            "ndim": y_prob_val.ndim if hasattr(y_prob_val, 'ndim') else "N/A",
            "min_val": float(np.min(y_prob_val)) if hasattr(y_prob_val, 'min') else "N/A",
            "max_val": float(np.max(y_prob_val)) if hasattr(y_prob_val, 'max') else "N/A",
            "first_10_values": y_prob_val.flatten()[:10].tolist() if hasattr(y_prob_val, 'flatten') else "N/A"
        }
    }
    
    # 保存JSON格式的调试信息
    json_filename = f"{save_dir}/debug_validation_{timestamp}.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(debug_info, f, indent=2, ensure_ascii=False)
    
    # 保存原始数据（pickle格式）
    pickle_filename = f"{save_dir}/validation_data_{timestamp}.pkl"
    with open(pickle_filename, 'wb') as f:
        pickle.dump({
            'y_true_val': y_true_val,
            'y_prob_val': y_prob_val,
            'epoch': epoch,
            'task': task,
            'mode': mode,
            'debug_info': debug_info
        }, f)
    
    # 打印调试信息
    print(f"\n=== 验证数据调试信息 (Epoch {epoch}) ===")
    print(f"任务: {task}, 模式: {mode}")
    print(f"y_true_val:")
    print(f"  类型: {debug_info['y_true_val']['type']}")
    print(f"  数据类型: {debug_info['y_true_val']['dtype']}")
    print(f"  形状: {debug_info['y_true_val']['shape']}")
    print(f"  维度: {debug_info['y_true_val']['ndim']}")
    print(f"  取值范围: [{debug_info['y_true_val']['min_val']}, {debug_info['y_true_val']['max_val']}]")
    print(f"  唯一值: {debug_info['y_true_val']['unique_values']}")
    
    print(f"y_prob_val:")
    print(f"  类型: {debug_info['y_prob_val']['type']}")
    print(f"  数据类型: {debug_info['y_prob_val']['dtype']}")
    print(f"  形状: {debug_info['y_prob_val']['shape']}")
    print(f"  维度: {debug_info['y_prob_val']['ndim']}")
    print(f"  取值范围: [{debug_info['y_prob_val']['min_val']}, {debug_info['y_prob_val']['max_val']}]")
    
    print(f"\n调试信息已保存到:")
    print(f"  JSON: {json_filename}")
    print(f"  PKL: {pickle_filename}")
    print("=" * 50)
    
    return debug_info


def analyze_sklearn_compatibility(y_true_val, y_prob_val, mode):
    """
    分析数据与sklearn函数的兼容性
    """
    print(f"\n=== sklearn兼容性分析 ===")
    
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        
        # 检查数据维度兼容性
        if mode == "binary":
            print("二分类模式检查:")
            print(f"  y_true形状: {y_true_val.shape}")
            print(f"  y_prob形状: {y_prob_val.shape}")
            
            # 检查是否需要reshape
            if y_true_val.ndim > 1:
                print(f"  警告: y_true_val有{y_true_val.ndim}维，可能需要flatten")
            if y_prob_val.ndim > 1:
                print(f"  警告: y_prob_val有{y_prob_val.ndim}维，可能需要flatten")
                
            # 尝试计算指标
            try:
                if y_true_val.ndim == y_prob_val.ndim == 1:
                    pr_auc = average_precision_score(y_true_val, y_prob_val)
                    roc_auc = roc_auc_score(y_true_val, y_prob_val)
                    print(f"  成功计算: PR-AUC={pr_auc:.4f}, ROC-AUC={roc_auc:.4f}")
                else:
                    print("  需要reshape数据才能计算指标")
            except Exception as e:
                print(f"  计算指标时出错: {e}")
                
        else:
            print(f"多分类/多标签模式 ({mode}):")
            print(f"  y_true形状: {y_true_val.shape}")
            print(f"  y_prob形状: {y_prob_val.shape}")
            
            # 多分类情况下的特殊检查
            if mode in ("multilabel", "multiclass"):
                try:
                    # 尝试使用multi_class参数
                    if y_prob_val.ndim == 2 and y_prob_val.shape[1] > 2:
                        roc_auc = roc_auc_score(y_true_val, y_prob_val, multi_class='ovr', average='macro')
                        print(f"  多分类ROC-AUC计算成功: {roc_auc:.4f}")
                    else:
                        print("  多分类数据格式可能不正确")
                except Exception as e:
                    print(f"  多分类指标计算出错: {e}")
                    
    except ImportError:
        print("无法导入sklearn，跳过兼容性检查")
    
    print("=" * 30)


if __name__ == "__main__":
    print("调试脚本已准备就绪，可以通过以下方式调用:")
    print("save_validation_debug_info(y_true_val, y_prob_val, epoch, task, mode)")