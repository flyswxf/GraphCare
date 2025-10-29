
"""
Baseline comparison: run multiple models (BAT/GAT/GIN/VisitRNN)
on the same dataset and log identical metrics to wandb.
Default: dataset=mimic3, task=drugrec, Heart enabled.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, jaccard_score, cohen_kappa_score
import wandb

# 允许从项目根目录导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/r/root/workspace/GraphCare')

from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader

from graphcare import load_everything, get_mode_and_out_channels_and_loss_func, get_dataloader
from graphcare import label_ehr_nodes, get_rel_emb, label_k_hop_nodes
from graphcare_ import split_by_patient
from graphcare_.model import GraphCare


class VisitRNN(nn.Module):
    """Simple RNN baseline using visit-wise aggregated node embeddings.
    Input: visit_padded_node as one-hot per visit; aggregated via node_emb.
    """
    def __init__(self, node_emb_weight: torch.Tensor, embedding_dim: int, hidden_dim: int, out_channels: int, dropout: float = 0.5):
        super().__init__()
        self.register_buffer('node_emb_weight', node_emb_weight.detach().float())
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.dropout = dropout
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_channels)

    def forward(self, *, visit_node: torch.Tensor, **kwargs):
        # visit_node: (B, V, N) one-hot/counts per visit; aggregate to embeddings
        # Normalize per-visit counts to avoid scale bias
        B, V, N = visit_node.shape
        # (B, V, N) @ (N, D) -> (B, V, D)
        visit_sum = visit_node.sum(dim=2, keepdim=True).clamp(min=1.0)
        visit_emb = (visit_node @ self.node_emb_weight) / visit_sum
        out, h = self.gru(visit_emb)
        # Use last hidden state
        h_last = h[-1]
        h_last = F.dropout(h_last, p=self.dropout, training=self.training)
        logits = self.fc(h_last)
        return logits


def build_graphcare_model(gnn: str, num_nodes: int, num_rels: int, max_visit: int, embedding_dim: int, out_channels: int,
                          node_emb: torch.Tensor, rel_emb: torch.Tensor, device: torch.device):
    model = GraphCare(
        num_nodes=num_nodes,
        num_rels=num_rels,
        max_visit=max_visit,
        embedding_dim=embedding_dim,
        hidden_dim=128,
        out_channels=out_channels,
        layers=3,
        dropout=0.5,
        decay_rate=0.03,
        node_emb=node_emb,
        rel_emb=rel_emb,
        freeze=False,
        patient_mode="joint",
        use_alpha=False,
        use_beta=True,
        use_edge_attn=True,
        gnn=gnn,
    ).to(device)
    return model


def evaluate(model, loader: DataLoader, device: torch.device, mode: str):
    model.eval()
    y_true_all = []
    y_prob_all = []
    with torch.no_grad():
        eval_pbar = tqdm(loader, desc='Evaluating')
        for batch_data in eval_pbar:
            batch_data = batch_data.to(device)
            batch = batch_data.batch
            curr_bs = int(batch.max().item() + 1)
            visits_per_patient = int(batch_data.visit_padded_node.shape[0] // curr_bs)
            visit_node = batch_data.visit_padded_node.reshape(
                curr_bs, visits_per_patient, batch_data.visit_padded_node.shape[1]
            ).float()

            if isinstance(model, VisitRNN):
                logits = model(visit_node=visit_node)
            else:
                node_ids = batch_data.y
                rel_ids = batch_data.relation
                edge_index = batch_data.edge_index
                ehr_nodes = batch_data.ehr_nodes.reshape(curr_bs, -1).float()
                logits = model(
                    node_ids=node_ids,
                    rel_ids=rel_ids,
                    edge_index=edge_index,
                    batch=batch,
                    visit_node=visit_node,
                    ehr_nodes=ehr_nodes,
                    in_drop=False,
                )

            if mode == "multiclass":
                y_prob = F.softmax(logits, dim=-1)
            else:
                y_prob = torch.sigmoid(logits)

            labels = batch_data.label.reshape(curr_bs, -1)
            y_true_all.append(labels.cpu().numpy())
            y_prob_all.append(y_prob.cpu().numpy())

    y_true_all = np.concatenate(y_true_all, axis=0)
    y_prob_all = np.concatenate(y_prob_all, axis=0)
    return y_true_all, y_prob_all


def train_one_epoch(model, optimizer, loss_function, train_loader: DataLoader, device: torch.device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, batch_data in pbar:
        batch_data = batch_data.to(device)
        optimizer.zero_grad()

        batch = batch_data.batch
        curr_bs = int(batch.max().item() + 1)
        visits_per_patient = int(batch_data.visit_padded_node.shape[0] // curr_bs)
        visit_node = batch_data.visit_padded_node.reshape(
            curr_bs, visits_per_patient, batch_data.visit_padded_node.shape[1]
        ).float()

        if isinstance(model, VisitRNN):
            logits = model(visit_node=visit_node)
        else:
            node_ids = batch_data.y
            rel_ids = batch_data.relation
            edge_index = batch_data.edge_index
            ehr_nodes = batch_data.ehr_nodes.reshape(curr_bs, -1).float()
            logits = model(
                node_ids=node_ids,
                rel_ids=rel_ids,
                edge_index=edge_index,
                batch=batch,
                visit_node=visit_node,
                ehr_nodes=ehr_nodes,
                in_drop=True,
            )

        labels = batch_data.label.reshape(curr_bs, -1).float()
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_description(f'Loss: {total_loss / (i + 1):.4f}')
    return total_loss / len(train_loader)


def main():
    parser = argparse.ArgumentParser(description="Baselines on GraphCare dataset")
    parser.add_argument('--dataset', type=str, default='mimic3', choices=['mimic3', 'mimic4'], help='Dataset to use')
    parser.add_argument('--task', type=str, default='drugrec', choices=['readmission', 'mortality', 'lenofstay', 'drugrec', 'procedure'], help='Task to run')
    parser.add_argument('--Heart', action='store_true', help='Enable Heart dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    args = parser.parse_args()

    # 默认使用 Heart 数据集 (与用户需求一致)
    Heart = True if not args.Heart else True
    dataset = args.dataset
    task = args.task
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Dataset: {dataset}, Task: {task}, Heart: {Heart}")

    # wandb 在线模式
    os.environ["WANDB_MODE"] = "online"

    # 加载与 SparseTest 一致的数据管线
    try:
        sample_dataset, graph, ent2id, rel2id, ent_emb, rel_emb, \
        map_cluster, map_cluster_inv, map_cluster_rel, map_cluster_rel_inv, \
        ccscm_id2clus, ccsproc_id2clus, atc3_id2clus = load_everything(
            dataset, task, inferMode=False, patient_id=None, index=None, Heart=Heart
        )
        print(f"Loaded {len(sample_dataset)} samples")
        print(f"Graph nodes: {graph.number_of_nodes()}, edges: {graph.number_of_edges()}")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    G_tg = from_networkx(graph)
    mode, out_channels, _loss_function = get_mode_and_out_channels_and_loss_func(task, sample_dataset, Heart)
    print(f"Task mode: {mode}, Output channels: {out_channels}")

    max_nodes = G_tg.num_nodes
    sample_dataset = label_ehr_nodes(task, sample_dataset, max_nodes, ccscm_id2clus, ccsproc_id2clus, atc3_id2clus)
    sample_dataset = label_k_hop_nodes(G_tg, sample_dataset, k=1)

    # 数据集划分与加载器
    train_dataset, val_dataset, test_dataset = split_by_patient(sample_dataset, [0.8, 0.1, 0.1], seed=528)
    train_loader, val_loader, test_loader = get_dataloader(G_tg, train_dataset, val_dataset, test_dataset, task, batch_size)

    # 准备嵌入
    node_emb_tensor = G_tg.x if hasattr(G_tg, 'x') and G_tg.x is not None else torch.FloatTensor(ent_emb)
    rel_emb_tensor = get_rel_emb(map_cluster_rel)
    embedding_dim = int(node_emb_tensor.shape[1])
    num_nodes = int(node_emb_tensor.shape[0])
    num_rels = int(rel_emb_tensor.shape[0])
    max_visit = sample_dataset[0]['visit_padded_node'].shape[0] if 'visit_padded_node' in sample_dataset[0] else 64

    # 统一的损失函数
    if mode == "multilabel":
        loss_function = nn.BCEWithLogitsLoss()
    elif mode == "multiclass":
        loss_function = nn.CrossEntropyLoss()
    else:
        loss_function = _loss_function

    # 基线模型列表
    baselines = [
        # {"name": "GraphCare-BAT", "type": "graphcare", "gnn": "BAT"},
        {"name": "GraphCare-GAT", "type": "graphcare", "gnn": "GAT"},
        {"name": "GraphCare-GIN", "type": "graphcare", "gnn": "GIN"},
        {"name": "VisitRNN", "type": "rnn"},
    ]

    for spec in baselines:
        run_name = f"{dataset}_{task}_{spec['name']}_bs{batch_size}_ep{epochs}_lr{lr}_{'Heart' if Heart else 'NoHeart'}"
        config = {
            "dataset": dataset,
            "task": task,
            "Heart": Heart,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "model": spec['name'],
        }
        run = wandb.init(project=f"{task}_Formal", config=config, name=run_name, notes="基线模型对比实验")

        # 构建模型
        if spec["type"] == "graphcare":
            model = build_graphcare_model(spec["gnn"], num_nodes, num_rels, max_visit, embedding_dim, out_channels,
                                          node_emb_tensor, rel_emb_tensor, device)
            wandb.config.update({
                "embedding_dim": embedding_dim,
                "hidden_dim": 128,
                "layers": 3,
                "dropout": 0.5,
                "decay_rate": 0.03,
                "gnn": spec["gnn"],
                "patient_mode": "joint",
            }, allow_val_change=True)
        else:
            model = VisitRNN(node_emb_weight=node_emb_tensor, embedding_dim=embedding_dim, hidden_dim=128, out_channels=out_channels, dropout=0.5).to(device)
            wandb.config.update({
                "embedding_dim": embedding_dim,
                "hidden_dim": 128,
                "layers": 1,
                "dropout": 0.5,
                "model_type": "VisitRNN",
            }, allow_val_change=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        print(f"Starting training for {spec['name']}...")
        best_val_auc = 0.0
        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(model, optimizer, loss_function, train_loader, device)
            y_true_val, y_prob_val = evaluate(model, val_loader, device, mode)

            if mode == "multiclass":
                y_pred_val = np.argmax(y_prob_val, axis=-1)
                y_true_cls = np.argmax(y_true_val, axis=-1)
                val_pr_auc = 0
                val_roc_auc = roc_auc_score(y_true_cls, y_prob_val, multi_class="ovr", average="weighted")
                val_jaccard = cohen_kappa_score(y_true_cls, y_pred_val)
                val_acc = accuracy_score(y_true_cls, y_pred_val)
                val_f1 = f1_score(y_true_cls, y_pred_val, average="weighted")
                val_precision = 0
                val_recall = 0
            elif mode == "multilabel":
                y_pred_val = (y_prob_val >= 0.5).astype(int)
                calc_y_true = y_true_val
                calc_y_prob = y_prob_val
                val_pr_auc = average_precision_score(calc_y_true, calc_y_prob, average="samples")
                val_roc_auc = roc_auc_score(calc_y_true, calc_y_prob, average="samples")
                val_jaccard = jaccard_score(calc_y_true, y_pred_val, average="samples", zero_division=1)
                val_acc = accuracy_score(calc_y_true, y_pred_val)
                val_f1 = f1_score(calc_y_true, y_pred_val, average="samples", zero_division=1)
                val_precision = precision_score(calc_y_true, y_pred_val, average="samples", zero_division=1)
                val_recall = recall_score(calc_y_true, y_pred_val, average="samples", zero_division=1)
            else:
                # 默认多标签
                y_pred_val = (y_prob_val >= 0.5).astype(int)
                val_pr_auc = average_precision_score(y_true_val, y_prob_val, average="samples")
                val_roc_auc = roc_auc_score(y_true_val, y_prob_val, average="samples")
                val_jaccard = jaccard_score(y_true_val, y_pred_val, average="samples", zero_division=1)
                val_acc = accuracy_score(y_true_val, y_pred_val)
                val_f1 = f1_score(y_true_val, y_pred_val, average="samples", zero_division=1)
                val_precision = precision_score(y_true_val, y_pred_val, average="samples", zero_division=1)
                val_recall = recall_score(y_true_val, y_pred_val, average="samples", zero_division=1)

            # 心源性休克（仅 Heart+drugrec）
            val_roc_auc_c = float('nan')
            val_pr_auc_c = float('nan')
            val_acc_c = float('nan')
            val_f1_c = float('nan')
            val_precision_c = float('nan')
            val_recall_c = float('nan')
            val_jaccard_c = float('nan')
            if Heart and task == 'drugrec' and mode == 'multilabel':
                c_idx = y_prob_val.shape[1] - 1
                y_true_c = y_true_val[:, c_idx]
                y_prob_c = y_prob_val[:, c_idx]
                y_pred_c = y_pred_val[:, c_idx]
                try:
                    val_pr_auc_c = average_precision_score(y_true_c, y_prob_c)
                    val_roc_auc_c = roc_auc_score(y_true_c, y_prob_c)
                except Exception:
                    pass
                val_acc_c = accuracy_score(y_true_c, y_pred_c)
                val_f1_c = f1_score(y_true_c, y_pred_c, zero_division=1)
                val_precision_c = precision_score(y_true_c, y_pred_c, zero_division=1)
                val_recall_c = recall_score(y_true_c, y_pred_c, zero_division=1)
                val_jaccard_c = jaccard_score(y_true_c, y_pred_c, zero_division=1)

            wandb_metrics = {
                "train/loss": train_loss,
                "val/pr_auc": val_pr_auc,
                "val/roc_auc": val_roc_auc,
                "val/acc": val_acc,
                "val/f1": val_f1,
                "val/precision": val_precision,
                "val/recall": val_recall,
                "val/jaccard": val_jaccard,
                "epoch": epoch,
            }
            if Heart and task == 'drugrec':
                wandb_metrics.update({
                    "val/cardiogenic_shock/roc_auc": val_roc_auc_c,
                    "val/cardiogenic_shock/pr_auc": val_pr_auc_c,
                    "val/cardiogenic_shock/acc": val_acc_c,
                    "val/cardiogenic_shock/f1": val_f1_c,
                    "val/cardiogenic_shock/precision": val_precision_c,
                    "val/cardiogenic_shock/recall": val_recall_c,
                    "val/cardiogenic_shock/jaccard": val_jaccard_c,
                })
            wandb.log(wandb_metrics)
            best_val_auc = max(best_val_auc, float(val_roc_auc))

        # 测试集评估
        print(f"\nFinal evaluation on test set for {spec['name']}...")
        y_true_test, y_prob_test = evaluate(model, test_loader, device, mode)
        if mode == "multiclass":
            y_pred_test = np.argmax(y_prob_test, axis=-1)
            y_true_cls_t = np.argmax(y_true_test, axis=-1)
            test_pr_auc = 0
            test_roc_auc = roc_auc_score(y_true_cls_t, y_prob_test, multi_class="ovr", average="weighted")
            test_jaccard = cohen_kappa_score(y_true_cls_t, y_pred_test)
            test_acc = accuracy_score(y_true_cls_t, y_pred_test)
            test_f1 = f1_score(y_true_cls_t, y_pred_test, average="weighted")
            test_precision = 0
            test_recall = 0
        else:
            y_pred_test = (y_prob_test >= 0.5).astype(int)
            calc_y_true_t = y_true_test
            calc_y_prob_t = y_prob_test
            test_pr_auc = average_precision_score(calc_y_true_t, calc_y_prob_t, average="samples")
            test_roc_auc = roc_auc_score(calc_y_true_t, calc_y_prob_t, average="samples")
            test_jaccard = jaccard_score(calc_y_true_t, y_pred_test, average="samples", zero_division=1)
            test_acc = accuracy_score(calc_y_true_t, y_pred_test)
            test_f1 = f1_score(calc_y_true_t, y_pred_test, average="samples", zero_division=1)
            test_precision = precision_score(calc_y_true_t, y_pred_test, average="samples", zero_division=1)
            test_recall = recall_score(calc_y_true_t, y_pred_test, average="samples", zero_division=1)

        # 心源性休克（仅 Heart+drugrec）
        test_roc_auc_c = float('nan')
        test_pr_auc_c = float('nan')
        test_acc_c = float('nan')
        test_f1_c = float('nan')
        test_precision_c = float('nan')
        test_recall_c = float('nan')
        test_jaccard_c = float('nan')
        if Heart and task == 'drugrec' and mode == 'multilabel':
            c_idx = y_prob_test.shape[1] - 1
            y_true_c_t = y_true_test[:, c_idx]
            y_prob_c_t = y_prob_test[:, c_idx]
            y_pred_c_t = y_pred_test[:, c_idx]
            try:
                test_pr_auc_c = average_precision_score(y_true_c_t, y_prob_c_t)
                test_roc_auc_c = roc_auc_score(y_true_c_t, y_prob_c_t)
            except Exception:
                pass
            test_acc_c = accuracy_score(y_true_c_t, y_pred_c_t)
            test_f1_c = f1_score(y_true_c_t, y_pred_c_t, zero_division=1)
            test_precision_c = precision_score(y_true_c_t, y_pred_c_t, zero_division=1)
            test_recall_c = recall_score(y_true_c_t, y_pred_c_t, zero_division=1)
            test_jaccard_c = jaccard_score(y_true_c_t, y_pred_c_t, zero_division=1)

        print(f"Test ROC-AUC: {test_roc_auc:.4f}")
        print(f"Test PR-AUC: {test_pr_auc:.4f}")
        if Heart and task == 'drugrec':
            print(f"Test Cardiogenic Shock ROC-AUC: {test_roc_auc_c:.4f}")
            print(f"Test Cardiogenic Shock PR-AUC: {test_pr_auc_c:.4f}")
            print(f"Test Cardiogenic Shock F1: {test_f1_c:.4f}")
            print(f"Test Cardiogenic Shock Accuracy: {test_acc_c:.4f}")
            print(f"Test Cardiogenic Shock Precision: {test_precision_c:.4f}")
            print(f"Test Cardiogenic Shock Recall: {test_recall_c:.4f}")
            print(f"Test Cardiogenic Shock Jaccard: {test_jaccard_c:.4f}")

        test_metrics = {
            "test/pr_auc": test_pr_auc,
            "test/roc_auc": test_roc_auc,
            "test/acc": test_acc,
            "test/f1": test_f1,
            "test/precision": test_precision,
            "test/recall": test_recall,
            "test/jaccard": test_jaccard,
        }
        if Heart and task == 'drugrec':
            test_metrics.update({
                "test/cardiogenic_shock/roc_auc": test_roc_auc_c,
                "test/cardiogenic_shock/pr_auc": test_pr_auc_c,
                "test/cardiogenic_shock/acc": test_acc_c,
                "test/cardiogenic_shock/f1": test_f1_c,
                "test/cardiogenic_shock/precision": test_precision_c,
                "test/cardiogenic_shock/recall": test_recall_c,
                "test/cardiogenic_shock/jaccard": test_jaccard_c,
            })
        wandb.log(test_metrics)
        wandb.run.summary["best/val_roc_auc"] = best_val_auc
        if Heart and task == 'drugrec':
            wandb.run.summary["best/cardiogenic_shock/test_roc_auc"] = test_roc_auc_c
            wandb.run.summary["best/cardiogenic_shock/test_pr_auc"] = test_pr_auc_c
            wandb.run.summary["best/cardiogenic_shock/test_f1"] = test_f1_c

        # 不保存权重、工件或阈值文件
        wandb.finish()


if __name__ == '__main__':
    main()
