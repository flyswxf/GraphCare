"""
Diagnostic script for SparseGraphCare metrics behavior.
It reproduces the data pipeline (like sparse-test.py), loads the saved weights
if available, runs a few validation batches, and prints detailed stats:
- shapes of tensors after reshape
- logits/probability stats
- label prevalence
- per-batch and cumulative metrics using probabilities (correct)
- per-batch and cumulative metrics using binary predictions (to mimic original graphcare.py)
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from torch_geometric.utils import from_networkx

from graphcare import load_everything, get_mode_and_out_channels_and_loss_func, get_dataloader
from graphcare import label_ehr_nodes, get_rel_emb, label_k_hop_nodes
from graphcare_sparse_model import SparseGraphCare
from graphcare_ import split_by_patient


def build_data_and_model(dataset="mimic3", task="mortality", batch_size=16, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    sample_dataset, graph, ent2id, rel2id, ent_emb, rel_emb, \
    map_cluster, map_cluster_inv, map_cluster_rel, map_cluster_rel_inv, \
    ccscm_id2clus, ccsproc_id2clus, atc3_id2clus = load_everything(dataset, task)

    G_tg = from_networkx(graph)

    # Task config
    mode, out_channels, _ = get_mode_and_out_channels_and_loss_func(task, sample_dataset)

    # Label EHR nodes and k-hop
    max_nodes = G_tg.num_nodes
    sample_dataset = label_ehr_nodes(task, sample_dataset, max_nodes, ccscm_id2clus, ccsproc_id2clus, atc3_id2clus)
    sample_dataset = label_k_hop_nodes(G_tg, sample_dataset, k=1)

    # Split
    train_dataset, val_dataset, test_dataset = split_by_patient(sample_dataset, [0.8, 0.1, 0.1], seed=528)

    # Dataloaders
    train_loader, val_loader, test_loader = get_dataloader(G_tg, train_dataset, val_dataset, test_dataset, task, batch_size)

    # Embeddings
    node_emb_tensor = G_tg.x if hasattr(G_tg, 'x') and G_tg.x is not None else torch.FloatTensor(ent_emb)
    rel_emb_tensor = get_rel_emb(map_cluster_rel)

    embedding_dim = int(node_emb_tensor.shape[1])
    num_nodes = int(max_nodes)
    num_rels = int(rel_emb_tensor.shape[0])
    max_visit = sample_dataset[0]['visit_padded_node'].shape[0] if 'visit_padded_node' in sample_dataset[0] else 64

    # Model
    model = SparseGraphCare(
        num_nodes=num_nodes,
        num_rels=num_rels,
        max_visit=max_visit,
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
        use_sparsification=True,
        sparsification_ratio=0.1,
        l1_lambda=1e-4,
        connectivity_lambda=1e-3,
    ).to(device)

    # Try load weights
    weights_path = f'./data/weights/saved_weights_{dataset}_{task}_sparse.pkl'
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Weights not found at {weights_path}, using randomly initialized model")

    return mode, model, val_loader, device


def batch_stats_and_metrics(mode, model, batch_data, device):
    model.eval()
    with torch.no_grad():
        batch_data = batch_data.to(device)
        node_ids = batch_data.y
        rel_ids = batch_data.relation
        edge_index = batch_data.edge_index
        batch = batch_data.batch

        curr_bs = int(batch.max().item() + 1)
        visits_per_patient = int(batch_data.visit_padded_node.shape[0] // curr_bs)

        visit_node = batch_data.visit_padded_node.reshape(
            curr_bs, visits_per_patient, batch_data.visit_padded_node.shape[1]
        ).float()
        ehr_nodes = batch_data.ehr_nodes.reshape(curr_bs, -1).float()
        labels = batch_data.label.reshape(curr_bs, -1).float()

        out = model(
            node_ids=node_ids,
            rel_ids=rel_ids,
            edge_index=edge_index,
            batch=batch,
            visit_node=visit_node,
            ehr_nodes=ehr_nodes,
            in_drop=False,
        )
        logits = out[0] if isinstance(out, tuple) else out

        # Stats
        logits_np = logits.detach().cpu().numpy().reshape(-1)
        y_prob = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        y_true = labels.detach().cpu().numpy().reshape(-1)

        # Validity checks
        print(f"  curr_bs={curr_bs}, visits_per_patient={visits_per_patient}")
        print(f"  logits shape={tuple(logits.shape)}, labels shape={tuple(labels.shape)}")
        print(f"  logits stats: mean={logits_np.mean():.6f}, std={logits_np.std():.6f}, min={logits_np.min():.6f}, max={logits_np.max():.6f}")
        print(f"  prob   stats: mean={y_prob.mean():.6f}, std={y_prob.std():.6f}, min={y_prob.min():.6f}, max={y_prob.max():.6f}")
        pos_rate = y_true.mean() if y_true.max() <= 1 else (y_true.sum()/len(y_true))
        print(f"  label prevalence (mean of y): {pos_rate:.6f}")

        # Metrics using probabilities (correct)
        try:
            ap_prob = average_precision_score(y_true, y_prob)
            auc_prob = roc_auc_score(y_true, y_prob)
        except Exception as e:
            ap_prob = float('nan')
            auc_prob = float('nan')
            print(f"  Warning: prob-metric error: {e}")

        # Metrics using binary predictions (mimic original)
        y_pred = (y_prob >= 0.5).astype(np.int32)
        try:
            ap_bin = average_precision_score(y_true, y_pred)
            auc_bin = roc_auc_score(y_true, y_pred)
        except Exception as e:
            ap_bin = float('nan')
            auc_bin = float('nan')
            print(f"  Warning: bin-metric error: {e}")
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=1)
        prec = precision_score(y_true, y_pred, zero_division=1)
        rec = recall_score(y_true, y_pred, zero_division=1)

        pos_pred_rate = y_pred.mean()
        print(f"  pos_pred_rate: {pos_pred_rate:.6f}")
        print(f"  Metrics (prob): AP={ap_prob:.6f}, ROC-AUC={auc_prob:.6f}")
        print(f"  Metrics (bin) : AP={ap_bin:.6f}, ROC-AUC={auc_bin:.6f}, Acc={acc:.6f}, F1={f1:.6f}, P={prec:.6f}, R={rec:.6f}")

        return y_true, y_prob, y_pred


def main():
    dataset = "mimic3"
    task = "mortality"
    batch_size = 16
    mode, model, val_loader, device = build_data_and_model(dataset, task, batch_size)

    print(f"Device: {device}, mode: {mode}")
    # Inspect some key parameter norms to see if model is updating
    mlp_w = model.MLP.weight.detach().abs().mean().item()
    mlp_b = model.MLP.bias.detach().abs().mean().item() if model.MLP.bias is not None else 0.0
    lin_node_w = model.lin_node.weight.detach().abs().mean().item()
    print(f"Param means | MLP.W={mlp_w:.6f}, MLP.b={mlp_b:.6f}, lin_node.W={lin_node_w:.6f}")

    y_true_all_prob, y_prob_all_prob = [], []
    y_true_all_bin, y_pred_all_bin = [], []

    for i, batch_data in enumerate(val_loader):
        print(f"\n== Batch {i+1} ==")
        y_true, y_prob, y_pred = batch_stats_and_metrics(mode, model, batch_data, device)

        y_true_all_prob.append(y_true)
        y_prob_all_prob.append(y_prob)
        y_true_all_bin.append(y_true)
        y_pred_all_bin.append(y_pred)

        if i >= 4:  # first 5 batches
            break

    # Cumulative metrics
    y_true_prob = np.concatenate(y_true_all_prob).reshape(-1)
    y_prob_prob = np.concatenate(y_prob_all_prob).reshape(-1)
    try:
        ap_prob = average_precision_score(y_true_prob, y_prob_prob)
        auc_prob = roc_auc_score(y_true_prob, y_prob_prob)
    except Exception as e:
        ap_prob, auc_prob = float('nan'), float('nan')
        print(f"Cum prob-metric error: {e}")

    y_true_bin = np.concatenate(y_true_all_bin).reshape(-1)
    y_pred_bin = np.concatenate(y_pred_all_bin).reshape(-1)
    try:
        ap_bin = average_precision_score(y_true_bin, y_pred_bin)
        auc_bin = roc_auc_score(y_true_bin, y_pred_bin)
    except Exception as e:
        ap_bin, auc_bin = float('nan'), float('nan')
        print(f"Cum bin-metric error: {e}")
    acc = accuracy_score(y_true_bin, y_pred_bin)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=1)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=1)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=1)

    print("\n== Cumulative over first 5 val batches ==")
    print(f"  Metrics (prob): AP={ap_prob:.6f}, ROC-AUC={auc_prob:.6f}")
    print(f"  Metrics (bin) : AP={ap_bin:.6f}, ROC-AUC={auc_bin:.6f}, Acc={acc:.6f}, F1={f1:.6f}, P={prec:.6f}, R={rec:.6f}")


if __name__ == "__main__":
    main()