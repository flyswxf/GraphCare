"""
GraphCare with Soft Sparsification for Mortality Prediction
"""
import sys
import os
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/r/root/workspace/GraphCare')

# 导入调试脚本
from debug_validation_data import save_validation_debug_info, analyze_sklearn_compatibility
from utils.comprehensive_debug import save_comprehensive_debug_info

import argparse
from graphcare import load_everything, get_mode_and_out_channels_and_loss_func, get_dataloader
from graphcare import label_ehr_nodes, get_rel_emb, label_k_hop_nodes
from SparseModel import SparseGraphCare
from graphcare_ import split_by_patient
import torch
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, jaccard_score,cohen_kappa_score
import wandb
from logger import get_logger
import torch.nn as nn
import re
from graphcare import get_subgraph
import json
from tqdm import tqdm
import csv
import os


# ===== Paths constants (feedback files) =====
# Make it easy to adjust later without touching code logic
FEEDBACK_CLUSTER_INDEX_FILE = os.path.join(
    os.path.dirname(__file__), 'utils', 'feedback', 'result', 'clusterIndex.txt'
)

# ===== Helper: FocalLoss and multilabel decision strategy =====
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: (N, C), targets: (N, C)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal = (self.alpha * (1 - pt) ** self.gamma * bce)
        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        return focal

def compute_pos_weight(loader, num_classes, device):
    pos_counts = torch.zeros(num_classes, device=device)
    neg_counts = torch.zeros(num_classes, device=device)
    for batch in loader:
        y = batch.label
        pos_counts += y.sum(dim=0)
        neg_counts += (y.shape[0] - y.sum(dim=0))
    # Avoid div-by-zero
    pos_counts = torch.clamp(pos_counts, min=1.0)
    neg_counts = torch.clamp(neg_counts, min=1.0)
    return neg_counts / pos_counts

def multilabel_decision(y_prob, strategy='threshold', threshold=0.5, topk=10, per_class_thresholds=None):
    # y_prob: np.ndarray of shape (N, C)
    if strategy == 'threshold':
        if per_class_thresholds is not None and len(per_class_thresholds) == y_prob.shape[1]:
            thr = np.array(per_class_thresholds)
            return (y_prob >= thr[None, :]).astype(int)
        return (y_prob >= float(threshold)).astype(int)
    elif strategy == 'topk':
        N, C = y_prob.shape
        y_bin = np.zeros_like(y_prob, dtype=int)
        k = int(topk)
        k = max(1, min(C, k))
        top_idx = np.argpartition(-y_prob, kth=k-1, axis=1)[:, :k]
        rows = np.arange(N)[:, None]
        y_bin[rows, top_idx] = 1
        return y_bin
    elif strategy == 'hybrid':
        # threshold first, if none selected in a row, fallback to topk=1
        y_bin = multilabel_decision(y_prob, strategy='threshold', threshold=threshold, per_class_thresholds=per_class_thresholds)
        row_sum = y_bin.sum(axis=1)
        fallback_rows = np.where(row_sum == 0)[0]
        if len(fallback_rows) > 0:
            k = max(1, min(y_prob.shape[1], int(topk)))
            top_idx = np.argpartition(-y_prob[fallback_rows], kth=k-1, axis=1)[:, :k]
            rows = fallback_rows[:, None]
            y_bin[rows, top_idx] = 1
        return y_bin
    else:
        # default threshold
        return (y_prob >= float(threshold)).astype(int)

# ===== Helper: search per-class thresholds to optimize F1 =====
def find_best_per_class_thresholds(y_true: np.ndarray, y_prob: np.ndarray, grid_size: int = 200):
    """
    For multilabel outputs, find a threshold per class that maximizes binary F1 per class on provided data.
    y_true: (N, C) binary labels
    y_prob: (N, C) probabilities
    Returns list[float] of length C
    """
    N, C = y_prob.shape
    thresholds = []
    for c in range(C):
        yt = y_true[:, c].astype(int)
        yp = y_prob[:, c].astype(float)

        # If class has no positive labels, fall back to 0.5 to avoid degenerate F1
        if np.sum(yt) == 0:
            thresholds.append(0.5)
            continue

        # Candidate thresholds from quantiles of yp to keep compute bounded
        uniq = np.unique(yp)
        if uniq.shape[0] > grid_size:
            cand = np.quantile(yp, np.linspace(0.01, 0.99, grid_size))
        else:
            cand = uniq

        best_t = 0.5
        best_f1 = -1.0
        # Evaluate single-class F1 across candidates
        for t in cand:
            yhat = (yp >= float(t)).astype(int)
            f1c = f1_score(yt, yhat, average='binary', zero_division=0)
            if f1c > best_f1:
                best_f1 = f1c
                best_t = float(t)
        thresholds.append(best_t)
    return thresholds

def _resolve_thresholds_out_path(dataset: str, task: str, Heart: bool, arg_path: str):
    """Decide where to save per-class thresholds JSON."""
    if arg_path is not None and len(str(arg_path)) > 0:
        return str(arg_path)
    default_dir = os.path.join(os.path.dirname(__file__), 'result')
    os.makedirs(default_dir, exist_ok=True)
    fname = f'per_class_thresholds_{dataset}_{task}{"_Heart" if Heart else ""}.json'
    return os.path.join(default_dir, fname)

# CLI arguments
parser = argparse.ArgumentParser(description="Sparse GraphCare runner")
parser.add_argument('--dataset', type=str, default='mimic3', choices=['mimic3', 'mimic4'], help='Dataset to use')
parser.add_argument('--task', type=str, default='drugrec', choices=['readmission', 'mortality', 'lenofstay', 'drugrec', 'procedure'], help='Task to run')
parser.add_argument('--Heart', action='store_true', help='Enable Heart dataset')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')

# Inference mode args
parser.add_argument('--infer', action='store_true', help='Enable single-sample inference mode')
parser.add_argument('--feedback', action='store_true', help='Enable feedback mode (infer only): apply add/remove cluster indices to EHR nodes')
# patient_id或sample_index任选其一
parser.add_argument('--patient_id', type=str, default=None, help='Patient ID for single-sample inference')
parser.add_argument('--sample_index', type=int, default=None, help='Sample index for single-sample inference (0-based)')
parser.add_argument('--weights_path', type=str, default=None, help='Path to model weights file; defaults to ./data/weights/saved_weights_{dataset}_{task}_sparse.pkl')
parser.add_argument('--out', type=str, default=None, help='Optional JSON path to save inference result')

# Decision strategy for multilabel tasks
parser.add_argument('--decision_strategy', type=str, default='hybrid', choices=['threshold', 'topk', 'hybrid'], help='Decision policy for multilabel predictions')
parser.add_argument('--threshold', type=float, default=0.5, help='Global threshold for multilabel prediction')
parser.add_argument('--per_class_thresholds', type=str, default=None, help='JSON file path containing per-class thresholds list')
parser.add_argument('--topk', type=int, default=20, help='Top-K per sample for multilabel prediction')
# Sparsification controls
parser.add_argument('--use_sparsification', action='store_true', help='Enable sparsification (soft edge weighting + top-k mask)')
parser.add_argument('--sparsification_ratio', type=float, default=0.1, help='Fraction of edges to keep (Top-K)')
parser.add_argument('--l1_lambda', type=float, default=1e-4, help='L1 sparsification regularization strength')
parser.add_argument('--connectivity_lambda', type=float, default=1e-3, help='Connectivity preservation strength')
# Loss options
parser.add_argument('--use_focal', action='store_true', help='Use FocalLoss for multilabel')
parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma')
parser.add_argument('--focal_alpha', type=float, default=0.25, help='Focal loss alpha (pos class weight)')

args = parser.parse_args()
# 推理模式下的参数校验
if args.infer:
    if args.sample_index is None and args.patient_id is None:
        parser.error("Inference mode requires either --sample_index or --patient_id")
    if args.weights_path is None:
        parser.error("Inference mode requires --weights_path to load model weights")
# 启动推理模式的代码示例
# python -u ehr_baselines/SparseTest/runSparseModel.py --dataset mimic3 --task drugrec --infer --sample_index 50 --weights_path ./data/weights/saved_weights_mimic3_drugrec_sparse.pkl --out ./ehr_baselines/SparseTest/result/inference_result.json
# 启动heart数据集的代码示例
# python -u ehr_baselines/SparseTest/runSparseModel.py --dataset mimic3 --task drugrec --Heart 
# 启动heart数据集的推理模式代码示例
# python -u ehr_baselines/SparseTest/runSparseModel.py --dataset mimic3 --task drugrec --Heart --infer --patient_id 21 --weights_path ./data/weights/saved_weights_mimic3_drugrec_sparse_Heart.pkl --out ./ehr_baselines/SparseTest/result/inference_result.json

# Configuration
dataset = args.dataset
task = args.task
Heart = args.Heart
batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
print(f"Dataset: {dataset}, Task: {task}, Heart: {Heart}")

# 当处于推理模式时禁用wandb
os.environ["WANDB_MODE"] = "offline" if args.infer else "online"
wandb_config = {
    "dataset": dataset,
    "task": task,
    "Heart": Heart,
    "batch_size": batch_size,
    "epochs": epochs,
    "lr": lr,
    # sparsification params
    "sparsification_ratio": float(args.sparsification_ratio),
    "l1_lambda": float(args.l1_lambda),
    "connectivity_lambda": float(args.connectivity_lambda),
    # attention mechanism - 本次实验使用beta注意力机制
    "use_beta_attention": True,  # 启用beta注意力机制进行图神经网络的注意力计算
    "attention_type": "beta",    # 注意力类型标识
}
# 初始化wandb项目
run = wandb.init(project=f"{task}", config=wandb_config,
                 notes="稀疏化GraphCare模型实验" + (", 包含心脏问题的病人数据集" if Heart else ""))
exp_name = f"{dataset}_{task}_sparse_bs{batch_size}_ep{epochs}_lr{lr}_{'Heart' if Heart else 'NoHeart'}"
# 初始化日志记录器
logger = get_logger(exp_name)

# Load GraphCare data and graph
try:
    sample_dataset, graph, ent2id, rel2id, ent_emb, rel_emb, \
    map_cluster, map_cluster_inv, map_cluster_rel, map_cluster_rel_inv, \
    ccscm_id2clus, ccsproc_id2clus, atc3_id2clus = load_everything(
        dataset, task, inferMode=args.infer, patient_id=args.patient_id, index=args.sample_index, Heart=Heart
    )
    
    print(f"Loaded {len(sample_dataset)} samples")
    print(f"Graph nodes: {graph.number_of_nodes()}, edges: {graph.number_of_edges()}")
    
    # # Heart augmentation: append cardiac flag as extra channel for drugrec
    # if Heart and task == 'drugrec':
    #     cardiac_map = {}
    #     csv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'dataPrepare', 'match_stats', 'cardiac_condition_flags.csv')
    #     if os.path.exists(csv_path):
    #         try:
    #             with open(csv_path, 'r', encoding='utf-8') as f:
    #                 reader = csv.DictReader(f)
    #                 for row in reader:
    #                     try:
    #                         pid = int(row.get('patient_id'))
    #                         flag = int(row.get('cardiac'))
    #                         cardiac_map[pid] = flag
    #                     except Exception:
    #                         pass
    #         except Exception as e:
    #             print(f"[HEART] Failed reading cardiac flags: {e}")
    #     else:
    #         print(f"[HEART] Cardiac flags CSV not found at {csv_path}; skipping augmentation")
    #     if cardiac_map:
    #         for p in sample_dataset:
    #             pid = int(p.get('patient_id', -1))
    #             flag = float(cardiac_map.get(pid, 0))
    #             # 增加一个额外的通道来表示心脏问题
    #             if isinstance(p.get('drugs_ind'), torch.Tensor):
    #                 p['drugs_ind'] = torch.cat([p['drugs_ind'].float(), torch.tensor([flag], dtype=torch.float32)], dim=0)
    #             else:
    #                 arr = np.array(p.get('drugs_ind'), dtype=float)
    #                 p['drugs_ind'] = torch.tensor(np.append(arr, flag), dtype=torch.float32)
    #         print(f"[HEART] Appended cardiac flag to drugs_ind for {len(sample_dataset)} samples")

except Exception as e:
    print(f"Error loading data: {e}")
    print("Please ensure GraphCare data files are available at the expected paths")
    sys.exit(1)

G_tg = from_networkx(graph)
# 保持 G_tg 在 CPU 上，避免在 Dataset.__getitem__ 内部进行子图提取时出现“indices/device”不匹配错误；
# 后续每个 batch 的 Data 会在训练/评估循环里被 .to(device) 移动到 GPU。

# 获取模型的损失函数和输出通道数
mode, out_channels, loss_function = get_mode_and_out_channels_and_loss_func(task, sample_dataset, Heart)
print(f"Task mode: {mode}, Output channels: {out_channels}")
print(f"Epochs: {epochs}")

max_nodes = G_tg.num_nodes  # keep consistent with visit_padded_node last dim (built from G_tg)

# Optionally load feedback cluster indices
feedback_add_clusters = None
feedback_remove_clusters = None
if args.feedback and args.infer:
    try:
        if os.path.exists(FEEDBACK_CLUSTER_INDEX_FILE):
            with open(FEEDBACK_CLUSTER_INDEX_FILE, 'r', encoding='utf-8') as f:
                fb = json.load(f)
            feedback_add_clusters = fb.get('add') or []
            feedback_remove_clusters = fb.get('remove') or []
            print(f"[FEEDBACK] Loaded add={len(feedback_add_clusters)} remove={len(feedback_remove_clusters)} cluster indices")
        else:
            print(f"[FEEDBACK] Cluster index file not found at {FEEDBACK_CLUSTER_INDEX_FILE}; proceeding without feedback")
    except Exception as e:
        print(f"[FEEDBACK] Failed to load feedback clusters: {e}; proceeding without feedback")

sample_dataset = label_ehr_nodes(
    task, sample_dataset, max_nodes, ccscm_id2clus, ccsproc_id2clus, atc3_id2clus,
    feedback_add_clusters=feedback_add_clusters,
    feedback_remove_clusters=feedback_remove_clusters
)

# Label k-hop subgraph nodes
sample_dataset = label_k_hop_nodes(G_tg, sample_dataset, k=1)

if not args.infer:
    # Split dataset
    train_dataset, val_dataset, test_dataset = split_by_patient(sample_dataset, [0.8, 0.1, 0.1], seed=528)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Create data loaders
    train_loader, val_loader, test_loader = get_dataloader(
        G_tg, train_dataset, val_dataset, test_dataset, task, batch_size
    )

    # Configure loss for multilabel with class imbalance handling
    if mode == "multilabel":
        if args.use_focal:
            loss_function = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
            print(f"[INFO] Using FocalLoss(alpha={args.focal_alpha}, gamma={args.focal_gamma}) for multilabel task")
        else:
            # Compute pos_weight from training data
            pos_weight = compute_pos_weight(train_loader, out_channels, device)
            loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            print(f"[INFO] Using BCEWithLogitsLoss with pos_weight (shape={tuple(pos_weight.shape)}) for multilabel task")

# Initialize model with sparsification
num_nodes = max_nodes
# Determine max_visit from dataset to match visit_padded_node
max_visit = sample_dataset[0]['visit_padded_node'].shape[0] if 'visit_padded_node' in sample_dataset[0] else 64

# Prepare embeddings so their sizes match graph dims to avoid matmul mismatch
# Use G_tg.x (num_nodes x emb_dim) as node embeddings to align with ehr_nodes length
node_emb_tensor = G_tg.x if hasattr(G_tg, 'x') and G_tg.x is not None else torch.FloatTensor(ent_emb)
# Use relation embeddings from clustered rel mapping (consistent with edges)
rel_emb_tensor = get_rel_emb(map_cluster_rel)

# Infer dimensions from tensors
embedding_dim = int(node_emb_tensor.shape[1])
num_rels = int(rel_emb_tensor.shape[0])

# 初始化SparseGraphCare模型 - 配置使用beta注意力机制
model = SparseGraphCare(
    num_nodes=num_nodes,
    num_rels=num_rels,
    max_visit=max_visit,
    embedding_dim=embedding_dim,
    hidden_dim=128,
    out_channels=out_channels,
    layers=3,
    dropout=0.5,
    decay_rate=0.01,
    node_emb=node_emb_tensor,
    rel_emb=rel_emb_tensor,
    freeze=False,
    patient_mode="joint",
    use_alpha=False,
    use_beta=True,              # 启用beta注意力机制
    use_edge_attn=True,
    self_attn=0.,
    gnn="BAT",
    attn_init=None,
    drop_rate=0.,
    # Sparsification parameters
    use_sparsification=bool(args.use_sparsification),
    sparsification_ratio=float(args.sparsification_ratio),
    l1_lambda=float(args.l1_lambda),
    connectivity_lambda=float(args.connectivity_lambda),
).to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

# Training function
def train_one_epoch():
    model.train()
    total_loss = 0
    total_sparse_loss = 0
    
    # 创建进度条
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    
    for i, batch_data in pbar:
        batch_data = batch_data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        node_ids = batch_data.y
        rel_ids = batch_data.relation
        edge_index = batch_data.edge_index
        batch = batch_data.batch
        
        # 使用实际 batch 大小进行重排，避免最后一个 batch 大小变化导致错位
        curr_bs = int(batch.max().item() + 1)
        visits_per_patient = int(batch_data.visit_padded_node.shape[0] // curr_bs)
        
        # Reshape tensors for GraphCare format
        visit_node = batch_data.visit_padded_node.reshape(
            curr_bs, visits_per_patient, batch_data.visit_padded_node.shape[1]
        ).float()
        ehr_nodes = batch_data.ehr_nodes.reshape(
            curr_bs, -1
        ).float()
        
        # Model forward
        if model.use_sparsification:
            logits, sparse_loss = model(
                node_ids=node_ids,
                rel_ids=rel_ids,
                edge_index=edge_index,
                batch=batch,
                visit_node=visit_node,
                ehr_nodes=ehr_nodes,
                in_drop=True
            )
        else:
            logits = model(
                node_ids=node_ids,
                rel_ids=rel_ids,
                edge_index=edge_index,
                batch=batch,
                visit_node=visit_node,
                ehr_nodes=ehr_nodes,
                in_drop=True
            )
            sparse_loss = 0
        
        # Compute prediction loss
        labels = batch_data.label.reshape(curr_bs, -1).float()
        pred_loss = loss_function(logits, labels)
        
        # Total loss
        total_loss_batch = pred_loss + sparse_loss
        total_loss_batch.backward()
        optimizer.step()
        
        total_loss += pred_loss.item()
        total_sparse_loss += sparse_loss.item() if torch.is_tensor(sparse_loss) else sparse_loss
        
        # 更新进度条描述，显示当前损失
        current_avg_loss = total_loss / (i + 1)
        current_avg_sparse_loss = total_sparse_loss / (i + 1)
        pbar.set_description(f'Loss: {current_avg_loss:.4f}, Sparse Loss: {current_avg_sparse_loss:.6f}')
    
    return total_loss / len(train_loader), total_sparse_loss / len(train_loader)

# Evaluation function
def evaluate(loader:DataLoader):
    model.eval()
    y_true_all = []
    y_prob_all = []
    
    with torch.no_grad():
        # 创建验证进度条
        eval_pbar = tqdm(loader, desc='Evaluating')
        
        for batch_data in eval_pbar:
            batch_data = batch_data.to(device)
            
            node_ids = batch_data.y
            rel_ids = batch_data.relation
            edge_index = batch_data.edge_index
            batch = batch_data.batch
            
            # 使用实际 batch 大小进行重排
            curr_bs = int(batch.max().item() + 1)
            visits_per_patient = int(batch_data.visit_padded_node.shape[0] // curr_bs)
            
            visit_node = batch_data.visit_padded_node.reshape(
                curr_bs, visits_per_patient, batch_data.visit_padded_node.shape[1]
            ).float()
            ehr_nodes = batch_data.ehr_nodes.reshape(
                curr_bs, -1
            ).float()
            
            out = model(
                node_ids=node_ids,
                rel_ids=rel_ids,
                edge_index=edge_index,
                batch=batch,
                visit_node=visit_node,
                ehr_nodes=ehr_nodes,
                in_drop=False
            )
            
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out
            
            if mode == "multiclass":
                y_prob = F.softmax(logits, dim=-1)
            else:
                y_prob = torch.sigmoid(logits)
            
            labels = batch_data.label.reshape(curr_bs, -1)
            
            y_true_all.append(labels.cpu().numpy())
            y_prob_all.append(y_prob.cpu().numpy())
    
    # 将 y_true/y_prob 展平为 1D，用于计算整体 AUC/PRAUC
    # y_true_all = np.concatenate(y_true_all, axis=0).reshape(-1)
    # y_prob_all = np.concatenate(y_prob_all, axis=0).reshape(-1)
    y_true_all = np.concatenate(y_true_all, axis=0)
    y_prob_all = np.concatenate(y_prob_all, axis=0)
    
    return y_true_all, y_prob_all

# ===== Inference mode: single-sample forward with strict weight loading =====
if args.infer:
    print("Inference mode enabled")
    # 添加Heart数据集的权重路径判断
    if Heart:
        weights_path = args.weights_path or f'./data/weights/saved_weights_{dataset}_{task}_sparse_Heart.pkl'
    else:
        weights_path = args.weights_path or f'./data/weights/saved_weights_{dataset}_{task}_sparse.pkl'
    if not os.path.exists(weights_path):
        print(f"[ERROR] Weights not found: {weights_path}. Please train the model first or specify --weights_path.")
        # 确保wandb结束（如果意外初始化）
        try:
            wandb.finish()
        except Exception:
            pass
        sys.exit(1)

    # Load weights strictly
    try:
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"[INFO] Loaded weights from {weights_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load weights: {e}")
        try:
            wandb.finish()
        except Exception:
            pass
        sys.exit(1)

    # # Create a single-sample dataset and DataLoader for proper batch handling
    from graphcare import Dataset
    from torch_geometric.loader import DataLoader
    
    # # Create a dataset with just the target sample
    # single_sample_dataset = [sample_dataset[idx]]
    inference_dataset = Dataset(G=G_tg, dataset=sample_dataset, task=task)
    inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False)

    y_true_all, y_prob_all = evaluate(loader=inference_loader)

    # 使用统一的决策逻辑与multilabel输出
    # y_true_all, y_prob_all: numpy arrays with shape (1, C) for batch_size=1
    # 读取每类阈值（如提供）
    per_class_thr = None
    # 优先使用用户提供路径，否则尝试默认保存路径
    try:
        thr_path_infer = None
        if args.per_class_thresholds is not None and os.path.exists(args.per_class_thresholds):
            thr_path_infer = args.per_class_thresholds
        else:
            thr_path_infer = _resolve_thresholds_out_path(dataset, task, Heart, args.per_class_thresholds)
        if thr_path_infer and os.path.exists(thr_path_infer):
            with open(thr_path_infer, 'r', encoding='utf-8') as f:
                per_class_thr = json.load(f)
    except Exception:
        per_class_thr = None

    # Heart+drugrec 情况下：分离最后一位（心源性休克概率），并从输出维度中移除
    cardiogenic_prob = None
    if Heart and task == 'drugrec' and mode == "multilabel":
        C_full = y_prob_all.shape[1]
        if C_full >= 1:
            c_idx = C_full - 1
            try:
                cardiogenic_prob = float(y_prob_all[0][c_idx])
            except Exception:
                cardiogenic_prob = float('nan')
            # 去掉最后一位
            y_prob_all = y_prob_all[:, :c_idx]
            y_true_all = y_true_all[:, :c_idx]
            # 若提供了每类阈值，截断为一致长度
            if per_class_thr is not None and isinstance(per_class_thr, list) and len(per_class_thr) == C_full:
                per_class_thr = per_class_thr[:c_idx]

    # 计算预测标签（使用可能被截断后的概率）
    if mode == "multilabel":
        y_pred_all = multilabel_decision(
            y_prob_all,
            strategy=args.decision_strategy,
            threshold=args.threshold,
            topk=args.topk,
            per_class_thresholds=per_class_thr
        )
    elif mode == "binary":
        y_pred_all = (y_prob_all >= float(args.threshold)).astype(int)
    else:
        y_pred_all = np.argmax(y_prob_all, axis=-1)

    # 组装输出结果（仅单样本；y_prob/y_true/y_pred 若为 Heart+drugrec 已去除最后一位）
    pid_val = sample_dataset[0].get('patient_id', None)
    result = {
        "patient_id": None if pid_val is None else str(pid_val),
        "sample_index": None if not args.sample_index else int(args.sample_index),
        "mode": mode,
        "decision_strategy": args.decision_strategy if mode == "multilabel" else None,
        "threshold": float(args.threshold) if mode in ("multilabel", "binary") else None,
        "topk": int(args.topk) if mode == "multilabel" else None,
        "per_class_thresholds": per_class_thr,
        "y_true": y_true_all[0].tolist(),
        "y_prob": y_prob_all[0].tolist(),
        "y_pred": y_pred_all[0].tolist() if mode == "multilabel" else int(y_pred_all[0]) if mode == "binary" else int(y_pred_all[0])
    }

    # 单独输出心源性休克概率（浮点），但不参与 topk 与其它输出的维度
    if cardiogenic_prob is not None:
        result["cardiogenic_shock"] = float(cardiogenic_prob)

    # 为drugrec/procedure任务提供top-k索引与分数（按概率降序；Heart+drugrec时已排除最后一位）
    if task in ("drugrec", "procedure") and mode == "multilabel":
        C = y_prob_all.shape[1]
        k = max(1, min(C, int(args.topk) if args.topk is not None else 10))
        probs_row = y_prob_all[0]
        top_idx = np.argsort(-probs_row)[:k]
        top_scores = probs_row[top_idx]
        result.update({
            "topk_indices": top_idx.tolist(),
            "topk_scores": top_scores.tolist(),
        })

    print("[INFER] Single-sample inference done.")
    # 打印摘要信息（避免输出大数组）
    summary = {
        k: (v if k not in ["y_prob", "y_pred", "y_true"] else f"shape={np.array(v).shape}")
        for k, v in result.items()
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2)
    )

    # Save to file if requested
    if args.out:
        out_path = args.out
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False)
        print(f"[INFER] Result saved to {out_path}")

    try:
        wandb.finish()
    except Exception:
        pass
    sys.exit(0)


# Update wandb config with model params (training/validation only)
wandb.config.update({
    "embedding_dim": embedding_dim,
    "hidden_dim": 128,
    "layers": 3,
    "dropout": 0.5,
    "decay_rate": 0.03,
    "gnn": "BAT",
    "patient_mode": "joint",
    "num_nodes": num_nodes,
    "num_rels": num_rels,
    "max_visit": max_visit,
    "use_sparsification": bool(args.use_sparsification),
    "sparsification_ratio": float(args.sparsification_ratio),
    "l1_lambda": float(args.l1_lambda),
    "connectivity_lambda": float(args.connectivity_lambda),
}, allow_val_change=True)
print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


# Training loop with comprehensive WandB logging
print("Starting training...")
best_val_auc = 0
early_stop_indicator = 0
early_stop = 5
best_val_f1_opt = -1.0

for epoch in range(1, epochs + 1):
    # Train
    train_loss, sparse_loss = train_one_epoch()
    
    # Validate
    y_true_val, y_prob_val = evaluate(val_loader)
    
    # 保存验证数据用于调试
    # save_validation_debug_info(y_true_val, y_prob_val, epoch, "val")
    
    # 保存综合调试信息
    try:
        if mode == "multilabel":
            per_class_thr = None
            if args.per_class_thresholds is not None and os.path.exists(args.per_class_thresholds):
                with open(args.per_class_thresholds, 'r', encoding='utf-8') as f:
                    per_class_thr = json.load(f)
            y_pred_val = multilabel_decision(
                y_prob_val,
                strategy=args.decision_strategy,
                threshold=args.threshold,
                topk=args.topk,
                per_class_thresholds=per_class_thr
            )
        elif mode == "binary":
            y_pred_val = (y_prob_val >= float(args.threshold)).astype(int)
        else:
            y_pred_val = np.argmax(y_prob_val, axis=-1)
        save_comprehensive_debug_info(
            model=model,
            y_true=y_true_val,
            y_prob=y_prob_val,
            y_pred=y_pred_val,
            epoch=epoch,
            phase="val",
            mode=mode,
            task=task,
            edge_index=G_tg.edge_index,
            train_loss=train_loss,
            sparse_loss=sparse_loss
        )
    except Exception as debug_e:
        print(f"综合调试信息保存失败: {debug_e}")
    
    # 计算验证指标
    
    # Calculate comprehensive validation metrics (following graphcare.py but using probabilities for AUC/PR-AUC)
    if mode == "binary":
        y_pred_val = (y_prob_val >= float(args.threshold)).astype(int)
        
        val_pr_auc = average_precision_score(y_true_val, y_prob_val)
        val_roc_auc = roc_auc_score(y_true_val, y_prob_val)
        val_jaccard = jaccard_score(y_true_val, y_pred_val, average="macro", zero_division=1)
        val_acc = accuracy_score(y_true_val, y_pred_val)
        val_f1 = f1_score(y_true_val, y_pred_val, average="macro", zero_division=1)
        val_precision = precision_score(y_true_val, y_pred_val, average="macro", zero_division=1)
        val_recall = recall_score(y_true_val, y_pred_val, average="macro", zero_division=1)
    elif mode == "multiclass":
        y_pred_val = np.argmax(y_prob_val, axis=-1)
        y_true_val = np.argmax(y_true_val, axis=-1)

        val_pr_auc = 0
        val_roc_auc = roc_auc_score(y_true_val, y_prob_val, multi_class="ovr", average="weighted")
        val_jaccard = cohen_kappa_score(y_true_val, y_pred_val)
        val_acc = accuracy_score(y_true_val, y_pred_val)
        val_f1 = f1_score(y_true_val, y_pred_val, average="weighted")
        val_precision = 0
        val_recall = 0
    elif mode == "multilabel":
        # 当 Heart+drugrec 时，原始指标需去掉最后一维（心源性休克）
        calc_y_true = y_true_val
        calc_y_prob = y_prob_val
        per_class_thr = None
        # 若用户提供了阈值文件，尝试读取（仅用于对比）
        if args.per_class_thresholds is not None and os.path.exists(args.per_class_thresholds):
            try:
                with open(args.per_class_thresholds, 'r', encoding='utf-8') as f:
                    per_class_thr = json.load(f)
            except Exception:
                per_class_thr = None
        if Heart and task == 'drugrec':
            C_full = y_prob_val.shape[1]
            if C_full >= 1:
                c_idx = C_full - 1
                calc_y_true = y_true_val[:, :c_idx]
                calc_y_prob = y_prob_val[:, :c_idx]
                if per_class_thr is not None and isinstance(per_class_thr, list) and len(per_class_thr) == C_full:
                    per_class_thr = per_class_thr[:c_idx]

        # 基于当前验证集概率，搜索每类F1最优阈值并用于计算验证指标
        per_class_thr_opt = find_best_per_class_thresholds(calc_y_true, calc_y_prob)

        # 使用搜索到的每类阈值进行决策
        y_pred_val = multilabel_decision(
            calc_y_prob,
            strategy='threshold',
            threshold=args.threshold,
            topk=args.topk,
            per_class_thresholds=per_class_thr_opt
        )
        val_pr_auc = average_precision_score(calc_y_true, calc_y_prob, average="samples")
        val_roc_auc = roc_auc_score(calc_y_true, calc_y_prob, average="samples")
        val_jaccard = jaccard_score(calc_y_true, y_pred_val, average="samples", zero_division=1)
        val_acc = accuracy_score(calc_y_true, y_pred_val)
        val_f1 = f1_score(calc_y_true, y_pred_val, average="samples", zero_division=1)
        val_precision = precision_score(calc_y_true, y_pred_val, average="samples", zero_division=1)
        val_recall = recall_score(calc_y_true, y_pred_val, average="samples", zero_division=1)

        # 保存阈值文件（在F1优化后且更优时）
        try:
            if val_f1 > best_val_f1_opt:
                best_val_f1_opt = val_f1
                thr_out_path = _resolve_thresholds_out_path(dataset, task, Heart, args.per_class_thresholds)
                # Heart+drugrec 情况下，阈值文件仅保存截断后的类别阈值
                os.makedirs(os.path.dirname(thr_out_path), exist_ok=True)
                with open(thr_out_path, 'w', encoding='utf-8') as f:
                    json.dump([float(t) for t in per_class_thr_opt], f)
                log_msg_thr = f"Saved per-class thresholds to: {thr_out_path} (len={len(per_class_thr_opt)}) with Val F1={val_f1:.4f}"
                print(log_msg_thr)
                logger.info(log_msg_thr)
        except Exception as _thr_e:
            print(f"保存阈值文件失败: {_thr_e}")

        # Extra cardiac metrics when Heart and drugrec
        if Heart and task == 'drugrec':
            c_idx = y_prob_val.shape[1] - 1
            y_true_c = y_true_val[:, c_idx]
            y_prob_c = y_prob_val[:, c_idx]
            thr_c = float(args.threshold)
            if per_class_thr is not None and len(per_class_thr) == y_prob_val.shape[1]:
                try:
                    thr_c = float(per_class_thr[c_idx])
                except Exception:
                    pass
            y_pred_c = (y_prob_c >= thr_c).astype(int)
            try:
                val_roc_auc_c = roc_auc_score(y_true_c, y_prob_c)
            except Exception:
                val_roc_auc_c = float('nan')
            try:
                val_pr_auc_c = average_precision_score(y_true_c, y_prob_c)
            except Exception:
                val_pr_auc_c = float('nan')
            val_acc_c = accuracy_score(y_true_c, y_pred_c)
            val_f1_c = f1_score(y_true_c, y_pred_c, zero_division=1)
            val_precision_c = precision_score(y_true_c, y_pred_c, zero_division=1)
            val_recall_c = recall_score(y_true_c, y_pred_c, zero_division=1)
            val_jaccard_c = jaccard_score(y_true_c, y_pred_c, zero_division=1)
        
    
    # Model saving and early stopping
    if val_roc_auc >= best_val_auc:
        # Create weights directory if it doesn't exist
        os.makedirs('./data/weights', exist_ok=True)
        if Heart:
            model_path = f'./data/weights/saved_weights_{dataset}_{task}_sparse_Heart.pkl'
        else:
            model_path = f'./data/weights/saved_weights_{dataset}_{task}_sparse.pkl'
        torch.save(model.state_dict(), model_path)
        print(f"  New best model saved! ROC-AUC: {val_roc_auc:.4f}")
        
        # Log model as WandB Artifact
        artifact = wandb.Artifact(f"{dataset}_{task}_sparse_model", type="model", metadata={"val_roc_auc": float(val_roc_auc), "epoch": epoch})
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
        
        best_val_auc = val_roc_auc
        early_stop_indicator = 0
    else:
        early_stop_indicator += 1
        if early_stop_indicator >= early_stop:
            print(f"Early stopping triggered after {epoch} epochs")
            break
    
    # WandB logging with all metrics
    wandb_metrics = {
        "train/loss": train_loss,
        "train/sparse_loss": sparse_loss,
        "val/pr_auc": val_pr_auc,
        "val/roc_auc": val_roc_auc,
        "val/acc": val_acc,
        "val/f1": val_f1,
        "val/precision": val_precision,
        "val/recall": val_recall,
        "val/jaccard": val_jaccard,
        "epoch": epoch
    }
    wandb.log(wandb_metrics)
    
    # Console and logger output
    log_msg = f'Epoch: {epoch}, Training loss: {train_loss:.4f}, Sparse loss: {sparse_loss:.6f}, Val PRAUC: {val_pr_auc:.4f}, Val ROC_AUC: {val_roc_auc:.4f}, Val acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val precision: {val_precision:.4f}, Val recall: {val_recall:.4f}, Val jaccard: {val_jaccard:.4f}'
    print(log_msg)
    logger.info(log_msg)
    log_msg_c = f'Val Heart ROC_AUC: {val_roc_auc_c:.4f}, Val Heart PRAUC: {val_pr_auc_c:.4f}, Val Heart acc: {val_acc_c:.4f}, Val Heart F1: {val_f1_c:.4f}, Val Heart precision: {val_precision_c:.4f}, Val Heart recall: {val_recall_c:.4f}, Val Heart jaccard: {val_jaccard_c:.4f}'
    print(log_msg_c)
    logger.info(log_msg_c)
    

# Final evaluation on test set
print("\nFinal evaluation on test set...")
y_true_test, y_prob_test = evaluate(test_loader)

# 保存测试集的综合调试信息
try:
    y_pred_test = (y_prob_test >= 0.5).astype(int) if mode == "multilabel" else np.argmax(y_prob_test, axis=-1)
    save_comprehensive_debug_info(
        model=model,
        y_true=y_true_test,
        y_prob=y_prob_test,
        y_pred=y_pred_test,
        epoch=epochs,  # 使用最终epoch
        phase="test",
        mode=mode,
        task=task,
        edge_index=G_tg.edge_index
    )
except Exception as debug_e:
    print(f"测试集调试信息保存失败: {debug_e}")


if mode == "binary":
    y_pred_test = (y_prob_test >= 0.5).astype(int)
    test_pr_auc = average_precision_score(y_true_test, y_prob_test)
    test_roc_auc = roc_auc_score(y_true_test, y_prob_test)
    test_jaccard = jaccard_score(y_true_test, y_pred_test, average="macro", zero_division=1)
    test_acc = accuracy_score(y_true_test, y_pred_test)
    test_f1 = f1_score(y_true_test, y_pred_test, average="macro", zero_division=1)
    test_precision = precision_score(y_true_test, y_pred_test, average="macro", zero_division=1)
    test_recall = recall_score(y_true_test, y_pred_test, average="macro", zero_division=1)
elif mode == "multiclass":
    y_pred_test = np.argmax(y_prob_test, axis=-1)
    y_true_test = np.argmax(y_true_test, axis=-1)

    test_pr_auc = 0
    test_roc_auc = roc_auc_score(y_true_test, y_prob_test, multi_class="ovr", average="weighted")
    test_jaccard = cohen_kappa_score(y_true_test, y_pred_test)
    test_acc = accuracy_score(y_true_test, y_pred_test)
    test_f1 = f1_score(y_true_test, y_pred_test, average="weighted")
    test_precision = 0
    test_recall = 0
elif mode == "multilabel":
    # Heart+drugrec：测试集原始指标需去掉最后一维（心源性休克）
    calc_y_true = y_true_test
    calc_y_prob = y_prob_test
    if Heart and task == 'drugrec':
        C_full = y_prob_test.shape[1]
        if C_full >= 1:
            c_idx = C_full - 1
            calc_y_true = y_true_test[:, :c_idx]
            calc_y_prob = y_prob_test[:, :c_idx]

    # 加载并使用保存的每类阈值；若不存在则回退到0.5
    test_thr = None
    thr_out_path = _resolve_thresholds_out_path(dataset, task, Heart, args.per_class_thresholds)
    if os.path.exists(thr_out_path):
        try:
            with open(thr_out_path, 'r', encoding='utf-8') as f:
                test_thr = json.load(f)
            if isinstance(test_thr, list) and len(test_thr) == calc_y_prob.shape[1]:
                y_pred_test = multilabel_decision(
                    calc_y_prob,
                    strategy='threshold',
                    threshold=0.5,
                    topk=args.topk,
                    per_class_thresholds=test_thr
                )
            else:
                y_pred_test = (calc_y_prob >= 0.5).astype(int)
        except Exception:
            y_pred_test = (calc_y_prob >= 0.5).astype(int)
    else:
        y_pred_test = (calc_y_prob >= 0.5).astype(int)
    test_pr_auc = average_precision_score(calc_y_true, calc_y_prob, average="samples")
    test_roc_auc = roc_auc_score(calc_y_true, calc_y_prob, average="samples")
    test_jaccard = jaccard_score(calc_y_true, y_pred_test, average="samples", zero_division=1)
    test_acc = accuracy_score(calc_y_true, y_pred_test)
    test_f1 = f1_score(calc_y_true, y_pred_test, average="samples", zero_division=1)
    test_precision = precision_score(calc_y_true, y_pred_test, average="samples", zero_division=1)
    test_recall = recall_score(calc_y_true, y_pred_test, average="samples", zero_division=1)
    
print(f"Test ROC-AUC: {test_roc_auc:.4f}")
print(f"Test PR-AUC: {test_pr_auc:.4f}")

# Log test metrics to WandB and set summary
wandb.log({
    "test/pr_auc": test_pr_auc,
    "test/roc_auc": test_roc_auc,
    "test/acc": test_acc,
    "test/f1": test_f1,
    "test/precision": test_precision,
    "test/recall": test_recall,
    "test/jaccard": test_jaccard,
})
wandb.run.summary["best/val_roc_auc"] = best_val_auc

# Print sparsification statistics if enabled
if model.use_sparsification:
    print(f"\nSparsification enabled:")
    print(f"  Target sparsification ratio: {model.sparsification_ratio}")
    print(f"  L1 regularization: {model.l1_lambda}")
    print(f"  Connectivity preservation: {model.connectivity_lambda}")
    logger.info(f"Sparsification enabled:")
    logger.info(f"  Target sparsification ratio: {model.sparsification_ratio}")
    logger.info(f"  L1 regularization: {model.l1_lambda}")
    logger.info(f"  Connectivity preservation: {model.connectivity_lambda}")

wandb.finish()
