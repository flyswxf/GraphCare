"""
GraphCare with Soft Sparsification for Mortality Prediction
"""
import sys
import os
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/r/root/workspace/GraphCare')

import argparse
from graphcare import load_everything, get_mode_and_out_channels_and_loss_func, get_dataloader
from graphcare import label_ehr_nodes, get_rel_emb, label_k_hop_nodes, prepare_procedure_indices
from SparseModel import SparseGraphCare
from graphcare_ import split_by_patient
import torch
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, jaccard_score
import wandb
from logger import get_logger
import torch.nn as nn
import re
from graphcare import get_subgraph
import json
from tqdm import tqdm

# CLI arguments
parser = argparse.ArgumentParser(description="Sparse GraphCare runner")
parser.add_argument('--dataset', type=str, default='mimic3', choices=['mimic3', 'mimic4'], help='Dataset to use')
parser.add_argument('--task', type=str, default='readmission', choices=['readmission', 'mortality', 'lenofstay', 'drugrec', 'procedure'], help='Task to run')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
# Inference mode args
parser.add_argument('--infer', action='store_true', help='Enable single-sample inference mode')
# patient_id或sample_index任选其一
parser.add_argument('--patient_id', type=str, default=None, help='Patient ID for single-sample inference')
parser.add_argument('--sample_index', type=int, default=None, help='Sample index for single-sample inference (0-based)')
parser.add_argument('--weights_path', type=str, default=None, help='Path to model weights file; defaults to ./data/weights/saved_weights_{dataset}_{task}_sparse.pkl')
parser.add_argument('--out', type=str, default=None, help='Optional JSON path to save inference result')
args = parser.parse_args()
# 推理模式下的参数校验
if args.infer:
    if args.sample_index is None and args.patient_id is None:
        parser.error("Inference mode requires either --sample_index or --patient_id")
    if args.weights_path is None:
        parser.error("Inference mode requires --weights_path to load model weights")
# 启动推理模式的代码示例
# python -u ehr_baselines/SparseTest/runSparseModel.py --dataset mimic3 --task readmission --infer --sample_index 50 --weights_path ./data/weights/saved_weights_mimic3_readmission_sparse.pkl --out ./inference_result.json 

# Configuration
dataset = args.dataset
task = args.task
batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
print(f"Dataset: {dataset}, Task: {task}")

# Initialize logging and WandB (following graphcare.py style)
# 当处于推理模式时禁用wandb
os.environ["WANDB_MODE"] = "offline" if args.infer else "online"
wandb_config = {
    "dataset": dataset,
    "task": task,
    "batch_size": batch_size,
    "epochs": epochs,
    "lr": lr,
    # sparsification params
    "sparsification_ratio": 0.1,
    "l1_lambda": 1e-4,
    "connectivity_lambda": 1e-3,
    # attention mechanism - 本次实验使用beta注意力机制
    "use_beta_attention": True,  # 启用beta注意力机制进行图神经网络的注意力计算
    "attention_type": "beta",    # 注意力类型标识
}
# 初始化wandb项目 - 
run = wandb.init(project="GraphCareSparseTest", config=wandb_config,
                 notes="使用beta注意力机制的GraphCare稀疏化模型实验")
exp_name = f"{dataset}_{task}_sparse_bs{batch_size}_ep{epochs}_lr{lr}"
# 初始化日志记录器
logger = get_logger(exp_name)

# Load GraphCare data and graph
try:
    sample_dataset, graph, ent2id, rel2id, ent_emb, rel_emb, \
    map_cluster, map_cluster_inv, map_cluster_rel, map_cluster_rel_inv, \
    ccscm_id2clus, ccsproc_id2clus, atc3_id2clus = load_everything(
        dataset, task, inferMode=args.infer, patient_id=args.patient_id, index=args.sample_index
    )

    # For procedure task, create multilabel indices similar to drugs_ind
    if task == "procedure":
        sample_dataset = prepare_procedure_indices(sample_dataset)
    
    print(f"Loaded {len(sample_dataset)} samples")
    print(f"Graph nodes: {graph.number_of_nodes()}, edges: {graph.number_of_edges()}")
    
except Exception as e:
    print(f"Error loading data: {e}")
    print("Please ensure GraphCare data files are available at the expected paths")
    sys.exit(1)

# Convert networkx graph to PyTorch Geometric format
G_tg = from_networkx(graph)
# 保持 G_tg 在 CPU 上，避免在 Dataset.__getitem__ 内部进行子图提取时出现“indices/device”不匹配错误；
# 后续每个 batch 的 Data 会在训练/评估循环里被 .to(device) 移动到 GPU。
# G_tg = G_tg.to(device)

# Get task configuration
mode, out_channels, loss_function = get_mode_and_out_channels_and_loss_func(task, sample_dataset)
print(f"Task mode: {mode}, Output channels: {out_channels}")

# Label EHR nodes with patient data
max_nodes = G_tg.num_nodes  # keep consistent with visit_padded_node last dim (built from G_tg)
sample_dataset = label_ehr_nodes(task, sample_dataset, max_nodes, ccscm_id2clus, ccsproc_id2clus, atc3_id2clus)

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
    use_beta=True,              # 启用beta注意力机制 - 关键配置
    use_edge_attn=True,
    self_attn=0.,
    gnn="BAT",                  # 使用BAT (Beta Attention Transformer) GNN架构
    attn_init=None,
    drop_rate=0.,
    # Sparsification parameters
    use_sparsification=True,
    sparsification_ratio=0.1,    # Keep top 10% of edges
    l1_lambda=1e-4,             # L1 regularization strength
    connectivity_lambda=1e-3,    # Connectivity preservation strength
).to(device)

# ===== Inference mode: single-sample forward with strict weight loading =====
if args.infer:
    print("Inference mode enabled")
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
    
    model.eval()
    with torch.no_grad():
        # Get the batched data from DataLoader
        for batch_data in inference_loader:
            batch_data = batch_data.to(device)
            
            node_ids = batch_data.y
            rel_ids = batch_data.relation
            edge_index = batch_data.edge_index
            batch = batch_data.batch
            
            # Extract visit and ehr node features
            # visit_node = batch_data.visit_padded_node.float()
            # ehr_nodes_vec = batch_data.ehr_nodes.float()
            # 使用实际 batch 大小进行重排，避免最后一个 batch 大小变化导致错位
            curr_bs = int(batch.max().item() + 1)
            visits_per_patient = int(batch_data.visit_padded_node.shape[0] // curr_bs)
            
            # Reshape tensors for GraphCare format
            visit_node = batch_data.visit_padded_node.reshape(
                curr_bs, visits_per_patient, batch_data.visit_padded_node.shape[1]
            ).float()
            ehr_nodes_vec = batch_data.ehr_nodes.reshape(
                curr_bs, -1
            ).float()
            
            out = model(
                node_ids=node_ids,
                rel_ids=rel_ids,
                edge_index=edge_index,
                batch=batch,
                visit_node=visit_node,
                ehr_nodes=ehr_nodes_vec,
                in_drop=False,
            )
            logits = out[0] if isinstance(out, tuple) else out

            if mode == "binary":
                prob = torch.sigmoid(logits)
            elif mode in ("multilabel", "multiclass"):
                prob = torch.sigmoid(logits) if mode == "multilabel" else F.softmax(logits, dim=-1)
            else:
                prob = logits
            
            break  # Only process the single batch

    # Prepare output
    pid_val = sample_dataset[0].get('patient_id', None)
    result = {
        "patient_id": None if pid_val is None else str(pid_val),
        "sample_index": None if not args.sample_index else int(args.sample_index),
        "mode": mode,
        "logits": logits.detach().cpu().numpy().tolist(),
        "prob": prob.detach().cpu().numpy().tolist(),
    }

    # For drugrec, also return top-k indices and scores
    if task == "drugrec":
        k = min(10, prob.shape[-1])
        topv, topi = torch.topk(prob.view(-1), k)
        result.update({
            "topk_indices": topi.detach().cpu().numpy().tolist(),
            "topk_scores": topv.detach().cpu().numpy().tolist(),
        })
    if task == "procedure":
        k = min(10, prob.shape[-1])
        topv, topi = torch.topk(prob.view(-1), k)
        result.update({
            "topk_indices": topi.detach().cpu().numpy().tolist(),
            "topk_scores": topv.detach().cpu().numpy().tolist(),
        })

    print("[INFER] Single-sample inference done.")
    print(json.dumps({k: (v if k not in ["logits", "prob"] else f"shape={np.array(v).shape}") for k, v in result.items()}, ensure_ascii=False, indent=2))

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
}, allow_val_change=True)
print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

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
def evaluate(loader):
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
            
            if mode == "binary":
                y_prob = torch.sigmoid(logits)
            else:
                y_prob = F.softmax(logits, dim=-1)
            
            labels = batch_data.label.reshape(curr_bs, -1)
            
            y_true_all.append(labels.cpu().numpy())
            y_prob_all.append(y_prob.cpu().numpy())
    
    # 将 y_true/y_prob 展平为 1D，用于计算整体 AUC/PRAUC
    y_true_all = np.concatenate(y_true_all, axis=0).reshape(-1)
    y_prob_all = np.concatenate(y_prob_all, axis=0).reshape(-1)
    
    return y_true_all, y_prob_all

# Training loop with comprehensive WandB logging
print("Starting training...")
best_val_auc = 0
early_stop_indicator = 0
early_stop = 5

for epoch in range(1, epochs + 1):
    # Train
    train_loss, sparse_loss = train_one_epoch()
    
    # Validate
    y_true_val, y_prob_val = evaluate(val_loader)
    
    # Calculate comprehensive validation metrics (following graphcare.py but using probabilities for AUC/PR-AUC)
    if mode == "binary":
        y_pred_val = (y_prob_val >= 0.5).astype(int)
        
        val_pr_auc = average_precision_score(y_true_val, y_prob_val)
        val_roc_auc = roc_auc_score(y_true_val, y_prob_val)
        val_jaccard = jaccard_score(y_true_val, y_pred_val, average="macro", zero_division=1)
        val_acc = accuracy_score(y_true_val, y_pred_val)
        val_f1 = f1_score(y_true_val, y_pred_val, average="macro", zero_division=1)
        val_precision = precision_score(y_true_val, y_pred_val, average="macro", zero_division=1)
        val_recall = recall_score(y_true_val, y_pred_val, average="macro", zero_division=1)
    else:
        # multilabel (e.g., drugrec/procedure): 使用概率计算 PR-AUC/ROC-AUC，其他指标留空或后续扩展
        y_pred_val = np.argmax(y_prob_val, axis=-1)
        val_pr_auc = average_precision_score(y_true_val, y_prob_val)
        val_roc_auc = roc_auc_score(y_true_val, y_prob_val)
        val_jaccard = jaccard_score(y_true_val, y_pred_val, average="macro", zero_division=1)
        val_acc = accuracy_score(y_true_val, y_pred_val)
        val_f1 = f1_score(y_true_val, y_pred_val, average="macro", zero_division=1)
        val_precision = precision_score(y_true_val, y_pred_val, average="macro", zero_division=1)
        val_recall = recall_score(y_true_val, y_pred_val, average="macro", zero_division=1)
    
    # Model saving and early stopping
    if val_roc_auc >= best_val_auc:
        # Create weights directory if it doesn't exist
        os.makedirs('./data/weights', exist_ok=True)
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

# Final evaluation on test set
print("\nFinal evaluation on test set...")
y_true_test, y_prob_test = evaluate(test_loader)

if mode == "binary":
    y_pred_test = (y_prob_test >= 0.5).astype(int)
    test_pr_auc = average_precision_score(y_true_test, y_prob_test)
    test_roc_auc = roc_auc_score(y_true_test, y_prob_test)
    test_jaccard = jaccard_score(y_true_test, y_pred_test, average="macro", zero_division=1)
    test_acc = accuracy_score(y_true_test, y_pred_test)
    test_f1 = f1_score(y_true_test, y_pred_test, average="macro", zero_division=1)
    test_precision = precision_score(y_true_test, y_pred_test, average="macro", zero_division=1)
    test_recall = recall_score(y_true_test, y_pred_test, average="macro", zero_division=1)
else:
    y_pred_test = (y_prob_test >= 0.5).astype(int)
    test_pr_auc = average_precision_score(y_true_test, y_prob_test)
    test_roc_auc = roc_auc_score(y_true_test, y_prob_test)
    test_jaccard = jaccard_score(y_true_test, y_pred_test, average="macro", zero_division=1)
    test_acc = accuracy_score(y_true_test, y_pred_test)
    test_f1 = f1_score(y_true_test, y_pred_test, average="macro", zero_division=1)
    test_precision = precision_score(y_true_test, y_pred_test, average="macro", zero_division=1)
    test_recall = recall_score(y_true_test, y_pred_test, average="macro", zero_division=1)

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
