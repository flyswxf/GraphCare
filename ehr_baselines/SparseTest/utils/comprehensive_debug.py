"""
Comprehensive debugging utilities for diagnosing model performance issues
"""

import os
import json
import numpy as np
import torch
# import torch.nn.functional as F
# from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
# import matplotlib.pyplot as plt
# import seaborn as sns
from datetime import datetime

# 缓存每次进程内已创建的调试目录，避免同一次运行中重复创建
_DEBUG_DIR_CACHE = {}

def _get_model_tag(model) -> str:
    """根据模型类名生成简洁标签: 'sparsemodel' 或 'graphcare'"""
    name = model.__class__.__name__.lower() if hasattr(model, '__class__') else str(model).lower()
    return 'sparsemodel' if 'sparse' in name else 'graphcare'

def _ensure_unique_subdir(base_dir: str, subdir_name: str) -> str:
    """在 base_dir 下为 subdir_name 选择不重复的目录名，如果存在则追加数字后缀。并创建该目录。"""
    os.makedirs(base_dir, exist_ok=True)
    candidate = os.path.join(base_dir, subdir_name)
    if not os.path.exists(candidate):
        os.makedirs(candidate, exist_ok=False)
        return candidate

    idx = 1
    while True:
        candidate_i = os.path.join(base_dir, f"{subdir_name}_{idx}")
        if not os.path.exists(candidate_i):
            os.makedirs(candidate_i, exist_ok=False)
            return candidate_i
        idx += 1

class ComprehensiveDebugger:
    def __init__(self, save_dir="ehr_baselines/SparseTest/debug_analysis_sparse_drug"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.debug_data = defaultdict(list)
        
    def save_model_internal_states(self, model, epoch, phase="val"):
        """保存模型内部状态数据"""
        states = {}
        
        # 1. 参数统计
        param_stats = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_stats[name] = {
                    'mean': float(param.data.mean()),
                    'std': float(param.data.std()),
                    'min': float(param.data.min()),
                    'max': float(param.data.max()),
                    'grad_norm': float(param.grad.norm()) if param.grad is not None else 0.0
                }
        states['parameter_stats'] = param_stats
        
        # 2. 稀疏化相关信息
        if hasattr(model, 'use_sparsification') and model.use_sparsification:
            states['sparsification'] = {
                'ratio': model.sparsification_ratio,
                'l1_lambda': model.l1_lambda,
                'connectivity_lambda': model.connectivity_lambda
            }
            
            # 如果模型有稀疏化掩码，保存其统计信息
            if hasattr(model, 'last_sparsification_mask'):
                mask = model.last_sparsification_mask
                if mask is not None:
                    states['sparsification']['mask_stats'] = {
                        'kept_edges_ratio': float(mask.float().mean()),
                        'total_edges': int(mask.numel()),
                        'kept_edges': int(mask.sum())
                    }
        
        # 保存到文件
        save_path = os.path.join(self.save_dir, f"model_states_epoch{epoch}_{phase}.json")
        with open(save_path, 'w') as f:
            json.dump(states, f, indent=2)
            
        return states
    
    def save_prediction_analysis(self, y_true, y_prob, y_pred, epoch, phase="val", mode="multilabel"):
        """保存预测分析数据"""
        analysis = {}
        
        # 1. 基本统计
        analysis['basic_stats'] = {
            'num_samples': len(y_true),
            'num_classes': y_true.shape[1] if len(y_true.shape) > 1 else 1,
            'positive_ratio': float(y_true.mean()) if mode == "binary" else float(y_true.sum() / y_true.size)
        }
        
        # 2. 置信度分布
        prob_stats = {
            'mean': float(y_prob.mean()),
            'std': float(y_prob.std()),
            'min': float(y_prob.min()),
            'max': float(y_prob.max()),
            'percentiles': {
                '25': float(np.percentile(y_prob, 25)),
                '50': float(np.percentile(y_prob, 50)),
                '75': float(np.percentile(y_prob, 75)),
                '90': float(np.percentile(y_prob, 90)),
                '95': float(np.percentile(y_prob, 95))
            }
        }
        analysis['probability_stats'] = prob_stats
        
        # 3. 错误分析
        if mode == "multilabel":
            # 多标签分类的错误分析
            errors = []
            for i in range(len(y_true)):
                true_labels = np.where(y_true[i] == 1)[0].tolist()
                pred_labels = np.where(y_pred[i] == 1)[0].tolist()
                
                if set(true_labels) != set(pred_labels):
                    errors.append({
                        'sample_idx': i,
                        'true_labels': true_labels,
                        'pred_labels': pred_labels,
                        'missed_labels': list(set(true_labels) - set(pred_labels)),
                        'false_positive_labels': list(set(pred_labels) - set(true_labels)),
                        'confidence_scores': y_prob[i].tolist()
                    })
            
            analysis['error_samples'] = errors[:50]  # 保存前50个错误样本
            analysis['error_stats'] = {
                'total_errors': len(errors),
                'error_rate': len(errors) / len(y_true)
            }
        
        # 4. 类别级别分析
        if mode == "multilabel":
            class_stats = {}
            for class_idx in range(y_true.shape[1]):
                true_class = y_true[:, class_idx]
                prob_class = y_prob[:, class_idx]
                pred_class = y_pred[:, class_idx]
                
                class_stats[f'class_{class_idx}'] = {
                    'support': int(true_class.sum()),
                    'predicted_positive': int(pred_class.sum()),
                    'avg_confidence': float(prob_class.mean()),
                    'true_positive_confidence': float(prob_class[true_class == 1].mean()) if true_class.sum() > 0 else 0.0,
                    'false_positive_confidence': float(prob_class[(pred_class == 1) & (true_class == 0)].mean()) if ((pred_class == 1) & (true_class == 0)).sum() > 0 else 0.0
                }
            
            analysis['class_level_stats'] = class_stats
        
        # 保存到文件
        save_path = os.path.join(self.save_dir, f"prediction_analysis_epoch{epoch}_{phase}.json")
        with open(save_path, 'w') as f:
            json.dump(analysis, f, indent=2)
            
        return analysis
    
    def save_training_dynamics(self, train_loss, sparse_loss, epoch, batch_losses=None):
        """保存训练动态信息"""
        dynamics = {
            'epoch': epoch,
            'train_loss': float(train_loss),
            'sparse_loss': float(sparse_loss),
            'total_loss': float(train_loss + sparse_loss),
            'loss_ratio': float(sparse_loss / train_loss) if train_loss > 0 else 0.0,
            'timestamp': datetime.now().isoformat()
        }
        
        if batch_losses is not None:
            dynamics['batch_statistics'] = {
                'num_batches': len(batch_losses),
                'loss_std': float(np.std(batch_losses)),
                'loss_range': float(np.max(batch_losses) - np.min(batch_losses)),
                'loss_trend': 'increasing' if batch_losses[-1] > batch_losses[0] else 'decreasing'
            }
        
        # 累积保存训练动态
        dynamics_file = os.path.join(self.save_dir, "training_dynamics.jsonl")
        with open(dynamics_file, 'a') as f:
            f.write(json.dumps(dynamics) + '\n')
            
        return dynamics
    
    def save_graph_analysis(self, edge_index, edge_weights=None, sparsification_mask=None, epoch=None):
        """保存图结构分析数据"""
        analysis = {}
        
        # 1. 基本图统计
        num_edges = edge_index.shape[1]
        num_nodes = max(edge_index.max().item() + 1, edge_index.shape[1])
        
        analysis['basic_graph_stats'] = {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'avg_degree': float(2 * num_edges / num_nodes),
            'density': float(2 * num_edges / (num_nodes * (num_nodes - 1))) if num_nodes > 1 else 0.0
        }
        
        # 2. 边权重分析
        if edge_weights is not None:
            weight_stats = {
                'mean': float(edge_weights.mean()),
                'std': float(edge_weights.std()),
                'min': float(edge_weights.min()),
                'max': float(edge_weights.max()),
                'zero_weights': int((edge_weights == 0).sum()),
                'negative_weights': int((edge_weights < 0).sum())
            }
            analysis['edge_weight_stats'] = weight_stats
        
        # 3. 稀疏化分析
        if sparsification_mask is not None:
            kept_edges = sparsification_mask.sum().item()
            removed_edges = num_edges - kept_edges
            
            sparsification_stats = {
                'kept_edges': kept_edges,
                'removed_edges': removed_edges,
                'actual_sparsification_ratio': float(removed_edges / num_edges),
                'connectivity_preserved': kept_edges > 0
            }
            analysis['sparsification_stats'] = sparsification_stats
        
        # 保存到文件
        suffix = f"_epoch{epoch}" if epoch is not None else ""
        save_path = os.path.join(self.save_dir, f"graph_analysis{suffix}.json")
        with open(save_path, 'w') as f:
            json.dump(analysis, f, indent=2)
            
        return analysis
    
    def save_attention_analysis(self, attention_weights, epoch, phase="val"):
        """保存注意力权重分析"""
        if attention_weights is None:
            return None
            
        analysis = {}
        
        # 注意力权重统计
        attn_stats = {
            'mean': float(attention_weights.mean()),
            'std': float(attention_weights.std()),
            'min': float(attention_weights.min()),
            'max': float(attention_weights.max()),
            'entropy': float(-torch.sum(attention_weights * torch.log(attention_weights + 1e-8)).item()),
            'sparsity': float((attention_weights < 0.01).float().mean())
        }
        analysis['attention_stats'] = attn_stats
        
        # 保存到文件
        save_path = os.path.join(self.save_dir, f"attention_analysis_epoch{epoch}_{phase}.json")
        with open(save_path, 'w') as f:
            json.dump(analysis, f, indent=2)
            
        return analysis
    
    def generate_summary_report(self):
        """生成综合分析报告"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'debug_files': os.listdir(self.save_dir),
            'analysis_summary': {}
        }
        
        # 分析训练动态
        dynamics_file = os.path.join(self.save_dir, "training_dynamics.jsonl")
        if os.path.exists(dynamics_file):
            with open(dynamics_file, 'r') as f:
                dynamics = [json.loads(line) for line in f]
            
            if dynamics:
                report['analysis_summary']['training'] = {
                    'total_epochs': len(dynamics),
                    'final_train_loss': dynamics[-1]['train_loss'],
                    'final_sparse_loss': dynamics[-1]['sparse_loss'],
                    'loss_trend': 'improving' if dynamics[-1]['train_loss'] < dynamics[0]['train_loss'] else 'worsening'
                }
        
        # 保存报告
        report_path = os.path.join(self.save_dir, "debug_summary_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        return report

def save_comprehensive_debug_info(model, y_true, y_prob, y_pred, epoch, phase="val", 
                                mode="multilabel", edge_index=None, attention_weights=None,
                                train_loss=None, sparse_loss=None, task=None,
                                save_dir_base="ehr_baselines/SparseTest/debug_analysis"):
    """
    综合调试信息保存函数
    
    Args:
        model: 训练的模型
        y_true: 真实标签
        y_prob: 预测概率
        y_pred: 预测标签
        epoch: 当前epoch
        phase: 训练阶段 ("train", "val", "test")
        mode: 任务模式 ("binary", "multiclass", "multilabel")
        edge_index: 图的边索引
        attention_weights: 注意力权重
        train_loss: 训练损失
        sparse_loss: 稀疏化损失
        task: 当前任务名称（如 "readmission", "drugrec", "procedure" 等）
        save_dir_base: 基础保存目录（默认 ehr_baselines/SparseTest/debug_analysis）
    """
    # 计算目标保存目录：基础目录 / (模型标签_任务)
    model_tag = _get_model_tag(model)
    task_tag = (task or 'unknown').lower()
    cache_key = f"{model_tag}_{task_tag}"

    final_dir = _DEBUG_DIR_CACHE.get(cache_key)
    if final_dir is None:
        subdir = f"{model_tag}_{task_tag}"
        final_dir = _ensure_unique_subdir(save_dir_base, subdir)
        _DEBUG_DIR_CACHE[cache_key] = final_dir

    debugger = ComprehensiveDebugger(final_dir)
    
    # 保存各类调试信息
    results = {}
    
    # 1. 模型内部状态
    results['model_states'] = debugger.save_model_internal_states(model, epoch, phase)
    
    # 2. 预测分析
    results['prediction_analysis'] = debugger.save_prediction_analysis(
        y_true, y_prob, y_pred, epoch, phase, mode
    )
    
    # 3. 训练动态（仅在训练阶段）
    if phase == "train" and train_loss is not None:
        results['training_dynamics'] = debugger.save_training_dynamics(
            train_loss, sparse_loss or 0.0, epoch
        )
    
    # 4. 图结构分析
    if edge_index is not None:
        sparsification_mask = getattr(model, 'last_sparsification_mask', None)
        results['graph_analysis'] = debugger.save_graph_analysis(
            edge_index, sparsification_mask=sparsification_mask, epoch=epoch
        )
    
    # 5. 注意力分析
    if attention_weights is not None:
        results['attention_analysis'] = debugger.save_attention_analysis(
            attention_weights, epoch, phase
        )
    
    # 6. 生成综合报告
    if epoch % 2 == 0 or phase == "test":  # 每2个epoch或测试时生成报告
        results['summary_report'] = debugger.generate_summary_report()
    
    print(f"[DEBUG] Comprehensive debug info saved for epoch {epoch}, phase {phase}")
    print(f"[DEBUG] Saved to: {final_dir}")
    
    return results