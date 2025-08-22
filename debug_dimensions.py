"""
Debug script to analyze dimension mismatches in GraphCare
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graphcare import load_everything, label_ehr_nodes, label_k_hop_nodes
from graphcare_sparse_model import SparseGraphCare
from graphcare_ import split_by_patient
import torch
import numpy as np
from torch_geometric.utils import from_networkx

# New imports for sparse-test-like pipeline
from graphcare import get_mode_and_out_channels_and_loss_func, get_dataloader, get_rel_emb
import torch.nn.functional as F

print("=" * 60)
print("GraphCare Dimension Analysis")
print("=" * 60)

# Configuration: align to sparse-test default
dataset = "mimic3"
task = "mortality"

print(f"Dataset: {dataset}, Task: {task}")
print()

try:
    # Load GraphCare data
    print("Loading GraphCare data...")
    sample_dataset, graph, ent2id, rel2id, ent_emb, rel_emb, \
    map_cluster, map_cluster_inv, map_cluster_rel, map_cluster_rel_inv, \
    ccscm_id2clus, ccsproc_id2clus, atc3_id2clus = load_everything(dataset, task)
    
    print(f"✓ Loaded {len(sample_dataset)} samples")
    print(f"✓ Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Convert to PyTorch Geometric
    G_tg = from_networkx(graph)
    print(f"✓ PyG Graph: {G_tg.num_nodes} nodes, {G_tg.num_edges} edges")
    
    print()
    print("=" * 60)
    print("DIMENSION ANALYSIS")
    print("=" * 60)
    
    # 1. Entity and Relation embeddings
    print(f"1. Embedding Dimensions:")
    if ent_emb is not None:
        print(f"   - Entity embeddings shape: {np.array(ent_emb).shape}")
        print(f"   - Entity embedding dim: {np.array(ent_emb).shape[1]}")
    else:
        print(f"   - Entity embeddings: None")
    
    if rel_emb is not None:
        print(f"   - Relation embeddings shape: {np.array(rel_emb).shape}")
        print(f"   - Relation embedding dim: {np.array(rel_emb).shape[1]}")
    else:
        print(f"   - Relation embeddings: None")
    
    print(f"   - len(ent2id): {len(ent2id)}")
    print(f"   - len(rel2id): {len(rel2id)}")
    print()
    
    # 2. Graph node features
    print(f"2. Graph Node Features:")
    if hasattr(G_tg, 'x') and G_tg.x is not None:
        print(f"   - G_tg.x shape: {G_tg.x.shape}")
        print(f"   - Node feature dim: {G_tg.x.shape[1]}")
    else:
        print(f"   - G_tg.x: None (no node features)")
    print()
    
    # 3. Sample Dataset Structure
    print(f"3. Sample Dataset Structure:")
    sample = sample_dataset[0]
    print(f"   - Sample keys: {list(sample.keys())}")
    
    if 'visit_padded_node' in sample:
        vpn_shape = sample['visit_padded_node'].shape
        print(f"   - visit_padded_node shape: {vpn_shape}")
        print(f"     * max_visit: {vpn_shape[0]}")
        print(f"     * num_nodes_dim: {vpn_shape[1]}")
    
    # Label EHR nodes using G_tg.num_nodes to match sparse-test
    print()
    print(f"4. EHR Node Labeling with G_tg.num_nodes ({G_tg.num_nodes}):")
    sample_dataset = label_ehr_nodes(task, sample_dataset, G_tg.num_nodes, 
                                     ccscm_id2clus, ccsproc_id2clus, atc3_id2clus)
    print("   ✓ Labeled EHR nodes with consistent max_nodes")
    
    # k-hop labeling
    sample_dataset = label_k_hop_nodes(G_tg, sample_dataset, k=1)
    print("   ✓ Added k-hop labels")

    # 5. Attention Mechanism Dimension Check (sparse-test style)
    print()
    print("=" * 60)
    print("ATTENTION SHAPE TEST (sparse-test compatible)")
    print("=" * 60)

    # Build dataloaders
    mode, out_channels, _ = get_mode_and_out_channels_and_loss_func(task, sample_dataset)
    batch_size = 8
    train_dataset, val_dataset, test_dataset = split_by_patient(sample_dataset, [0.8, 0.1, 0.1], seed=528)
    train_loader, val_loader, test_loader = get_dataloader(G_tg, train_dataset, val_dataset, test_dataset, task, batch_size)
    print(f"   - mode: {mode}, out_channels: {out_channels}")
    print(f"   - train/val/test sizes: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")

    # Choose embeddings
    node_emb_tensor = G_tg.x if hasattr(G_tg, 'x') and G_tg.x is not None else torch.FloatTensor(ent_emb)
    rel_emb_tensor = get_rel_emb(map_cluster_rel)

    embedding_dim = int(node_emb_tensor.shape[1])
    num_nodes = int(G_tg.num_nodes)
    max_visit = sample['visit_padded_node'].shape[0]
    num_rels = int(rel_emb_tensor.shape[0])

    print(f"   - num_nodes: {num_nodes}, num_rels: {num_rels}, max_visit: {max_visit}, emb_dim: {embedding_dim}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SparseGraphCare(
        num_nodes=num_nodes,
        num_rels=num_rels,
        max_visit=max_visit,
        embedding_dim=embedding_dim,
        hidden_dim=64,
        out_channels=out_channels,
        layers=2,
        dropout=0.2,
        decay_rate=0.03,
        node_emb=node_emb_tensor,
        rel_emb=rel_emb_tensor,
        freeze=False,
        patient_mode="joint",
        use_alpha=True,
        use_beta=True,
        use_edge_attn=True,
        self_attn=0.,
        gnn="BAT",
        attn_init=None,
        drop_rate=0.,
        use_sparsification=False,
    ).to(device)

    # Pull one batch and run forward
    batch_data = next(iter(val_loader))
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

    print(f"   - visit_node shape (B,V,N): {tuple(visit_node.shape)}")
    print(f"   - ehr_nodes shape (B,N): {tuple(ehr_nodes.shape)}")

    # Sanity checks
    assert visit_node.shape[0] == curr_bs, "visit_node batch dim mismatch"
    assert visit_node.shape[1] == visits_per_patient, "visit_node visit dim mismatch"
    assert visit_node.shape[2] == num_nodes, f"visit_node last dim {visit_node.shape[2]} != num_nodes {num_nodes}"
    assert ehr_nodes.shape[0] == curr_bs and ehr_nodes.shape[1] == num_nodes, "ehr_nodes shape mismatch"

    with torch.no_grad():
        out = model(
            node_ids=node_ids,
            rel_ids=rel_ids,
            edge_index=edge_index,
            batch=batch,
            visit_node=visit_node,
            ehr_nodes=ehr_nodes,
            store_attn=True,
            in_drop=False,
        )
        if isinstance(out, dict):
            logits = out['logits']
        elif isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out

    print(f"   - logits shape: {tuple(logits.shape)} (expect (B, 1) for binary)")

    # Inspect stored attentions
    if hasattr(model, 'alpha_weights') and len(model.alpha_weights) > 0:
        print(f"   - alpha[layer0] shape: {tuple(model.alpha_weights[0].shape)} (expect (B,V,N))")
    if hasattr(model, 'beta_weights') and len(model.beta_weights) > 0:
        print(f"   - beta[layer0]  shape: {tuple(model.beta_weights[0].shape)} (expect (B,V,1))")
    if hasattr(model, 'attention_weights') and len(model.attention_weights) > 0:
        print(f"   - attn_edges[layer0] shape: {tuple(model.attention_weights[0].shape)} (expect (E,1))")

    print()
    print("✓ Attention shape test completed without runtime shape errors.")

    print()
    print("=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
except Exception as e:
    print(f"✗ Error during analysis: {e}")
    import traceback
    traceback.print_exc()