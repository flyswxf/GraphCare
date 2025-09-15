"""
GraphCare with Soft Sparsification for Mortality Prediction
"""

# 不使用该文件,使用runSparseModel.py
import sys
import os
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphcare import load_everything, get_mode_and_out_channels_and_loss_func, get_dataloader
from graphcare import label_ehr_nodes, get_rel_emb, label_k_hop_nodes
from graphcare_sparse_model import SparseGraphCare
from graphcare_ import split_by_patient
import torch
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, jaccard_score
import wandb
import logging

# Configuration
dataset = "mimic3" 
task = "mortality"
batch_size = 16
epochs = 5
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
print(f"Dataset: {dataset}, Task: {task}")

# Initialize logging and WandB (following graphcare.py style)
def get_logger(exp_name: str):
    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    os.makedirs('./training_logs', exist_ok=True)
    file_handler = logging.FileHandler(f'./training_logs/{exp_name}.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename.endswith(f'{exp_name}.log') for h in logger.handlers):
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.addHandler(stream_handler)

    return logger

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
}
run = wandb.init(project="GraphCareSparseTest", config=wandb_config)
exp_name = f"{dataset}_{task}_sparse_bs{batch_size}_ep{epochs}_lr{lr}"
logger = get_logger(exp_name)

# Load GraphCare data and graph
try:
    sample_dataset, graph, ent2id, rel2id, ent_emb, rel_emb, \
    map_cluster, map_cluster_inv, map_cluster_rel, map_cluster_rel_inv, \
    ccscm_id2clus, ccsproc_id2clus, atc3_id2clus = load_everything(dataset, task)
    
    print(f"Loaded {len(sample_dataset)} samples")
    print(f"Graph nodes: {graph.number_of_nodes()}, edges: {graph.number_of_edges()}")
    
except Exception as e:
    print(f"Error loading data: {e}")
    print("Please ensure GraphCare data files are available at the expected paths")
    sys.exit(1)

# Convert networkx graph to PyTorch Geometric format
G_tg = from_networkx(graph)
# 保持 G_tg 在 CPU 上，避免在 Dataset.__getitem__ 内部进行子图提取时出现 indices/device 不匹配错误；
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

# Convert embeddings to tensors
node_emb_tensor = torch.FloatTensor(ent_emb) if ent_emb is not None else None
rel_emb_tensor = torch.FloatTensor(rel_emb) if rel_emb is not None else None

model = SparseGraphCare(
    num_nodes=num_nodes,
    num_rels=num_rels,
    max_visit=max_visit,
    embedding_dim=embedding_dim,
    embedding_dim=embedding_dim,
    hidden_dim=128,
    out_channels=out_channels,
    layers=3,
    dropout=0.5,
    decay_rate=0.03,
    node_emb=node_emb_tensor,
    rel_emb=rel_emb_tensor,
    freeze=False,
    patient_mode="joint",
    use_alpha=False,
    use_beta=False,
    use_edge_attn=True,
    self_attn=0.,
    gnn="BAT",
    attn_init=None,
    drop_rate=0.,
    # Sparsification parameters
    use_sparsification=True,
    sparsification_ratio=0.1,    # Keep top 10% of edges
    l1_lambda=1e-4,             # L1 regularization strength
    connectivity_lambda=1e-3,    # Connectivity preservation strength
).to(device)

# Update wandb config with model params
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
    
    for batch_data in train_loader:
        batch_data = batch_data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        node_ids = batch_data.y
        rel_ids = batch_data.relation
        edge_index = batch_data.edge_index
        batch = batch_data.batch
        
        # Reshape tensors for GraphCare format
        visit_node = batch_data.visit_padded_node.reshape(
            batch_size, -1, batch_data.visit_padded_node.shape[1]
        ).float()
        ehr_nodes = batch_data.ehr_nodes.reshape(
            batch_size, -1
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
        labels = batch_data.label.reshape(batch_size, -1).float()
        pred_loss = loss_function(logits, labels)
        
        # Total loss
        total_loss_batch = pred_loss + sparse_loss
        total_loss_batch.backward()
        optimizer.step()
        
        total_loss += pred_loss.item()
        total_sparse_loss += sparse_loss.item() if torch.is_tensor(sparse_loss) else sparse_loss
    
    return total_loss / len(train_loader), total_sparse_loss / len(train_loader)

# Evaluation function
def evaluate(loader):
    model.eval()
    y_true_all = []
    y_prob_all = []
    
    with torch.no_grad():
        for batch_data in loader:
            batch_data = batch_data.to(device)
            
            node_ids = batch_data.y
            rel_ids = batch_data.relation
            edge_index = batch_data.edge_index
            batch = batch_data.batch
            
            visit_node = batch_data.visit_padded_node.reshape(
                batch_size, -1, batch_data.visit_padded_node.shape[1]
            ).float()
            ehr_nodes = batch_data.ehr_nodes.reshape(
                batch_size, -1
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
            
            labels = batch_data.label.reshape(batch_size, -1)
            
            y_true_all.append(labels.cpu().numpy())
            y_prob_all.append(y_prob.cpu().numpy())
    
    y_true_all = np.concatenate(y_true_all, axis=0)
    y_prob_all = np.concatenate(y_prob_all, axis=0)
    
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
    
    # Calculate comprehensive validation metrics (following graphcare.py)
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
        # Other modes (multiclass/multilabel) would be handled here
        val_pr_auc = average_precision_score(y_true_val, y_prob_val)
        val_roc_auc = roc_auc_score(y_true_val, y_prob_val)
        val_jaccard = 0
        val_acc = 0
        val_f1 = 0
        val_precision = 0
        val_recall = 0
    
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
    test_pr_auc = average_precision_score(y_true_test, y_prob_test)
    test_roc_auc = roc_auc_score(y_true_test, y_prob_test)
    test_jaccard = 0
    test_acc = 0
    test_f1 = 0
    test_precision = 0
    test_recall = 0

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

wandb.finish()