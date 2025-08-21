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

print("=" * 60)
print("GraphCare Dimension Analysis")
print("=" * 60)

# Configuration
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
    
    # 3. Dataset structure analysis
    print(f"3. Sample Dataset Structure:")
    sample = sample_dataset[0]
    print(f"   - Sample keys: {list(sample.keys())}")
    
    if 'visit_padded_node' in sample:
        vpn_shape = sample['visit_padded_node'].shape
        print(f"   - visit_padded_node shape: {vpn_shape}")
        print(f"     * max_visit: {vpn_shape[0]}")
        print(f"     * num_nodes_dim: {vpn_shape[1]}")
    
    # Label EHR nodes with different max_nodes values
    print()
    print(f"4. EHR Node Labeling Test:")
    
    # Test 1: Using len(ent2id) as in original sparse script
    print(f"   Test 1 - max_nodes = len(ent2id) = {len(ent2id)}:")
    try:
        test_dataset_1 = label_ehr_nodes(task, [sample_dataset[0]], len(ent2id), 
                                       ccscm_id2clus, ccsproc_id2clus, atc3_id2clus)
        ehr_shape_1 = test_dataset_1[0]['ehr_node_set'].shape
        print(f"     ✓ ehr_node_set shape: {ehr_shape_1}")
    except Exception as e:
        print(f"     ✗ Error: {e}")
    
    # Test 2: Using G_tg.num_nodes  
    print(f"   Test 2 - max_nodes = G_tg.num_nodes = {G_tg.num_nodes}:")
    try:
        test_dataset_2 = label_ehr_nodes(task, [sample_dataset[0]], G_tg.num_nodes, 
                                       ccscm_id2clus, ccsproc_id2clus, atc3_id2clus)
        ehr_shape_2 = test_dataset_2[0]['ehr_node_set'].shape
        print(f"     ✓ ehr_node_set shape: {ehr_shape_2}")
    except Exception as e:
        print(f"     ✗ Error: {e}")
    
    # Test 3: Using len(map_cluster) as in original graphcare.py
    print(f"   Test 3 - max_nodes = len(map_cluster) = {len(map_cluster)}:")
    try:
        test_dataset_3 = label_ehr_nodes(task, [sample_dataset[0]], len(map_cluster), 
                                       ccscm_id2clus, ccsproc_id2clus, atc3_id2clus)
        ehr_shape_3 = test_dataset_3[0]['ehr_node_set'].shape
        print(f"     ✓ ehr_node_set shape: {ehr_shape_3}")
    except Exception as e:
        print(f"     ✗ Error: {e}")
    
    print()
    print(f"5. Model Parameter Analysis:")
    
    # Create model with different embedding_dim values
    num_rels = len(rel2id)
    max_visit = sample['visit_padded_node'].shape[0] if 'visit_padded_node' in sample else 64
    
    print(f"   - num_rels: {num_rels}")
    print(f"   - max_visit: {max_visit}")
    
    # Test embedding dimensions
    if ent_emb is not None:
        actual_node_emb_dim = np.array(ent_emb).shape[1]
        print(f"   - Actual node embedding dim: {actual_node_emb_dim}")
        
        node_emb_tensor = torch.FloatTensor(ent_emb)
        print(f"   - node_emb_tensor shape: {node_emb_tensor.shape}")
    
    if rel_emb is not None:
        actual_rel_emb_dim = np.array(rel_emb).shape[1]
        print(f"   - Actual relation embedding dim: {actual_rel_emb_dim}")
        
        rel_emb_tensor = torch.FloatTensor(rel_emb)
        print(f"   - rel_emb_tensor shape: {rel_emb_tensor.shape}")
    
    print()
    print(f"6. Attention Mechanism Analysis:")
    
    # Check attention layer dimensions
    if ent_emb is not None:
        # Test different num_nodes values for alpha attention
        test_num_nodes = [len(ent2id), G_tg.num_nodes, len(map_cluster)]
        
        for i, num_nodes in enumerate(test_num_nodes, 1):
            print(f"   Test {i} - num_nodes = {num_nodes}:")
            
            # Calculate alpha attention parameters
            alpha_params = num_nodes * num_nodes
            print(f"     * Alpha attention parameters: {alpha_params:,}")
            
            if alpha_params > 1000000:  # 1M parameters
                print(f"     * ⚠️  WARNING: Too many parameters for alpha attention!")
            else:
                print(f"     * ✓ Alpha attention parameters manageable")
    
    print()
    print("=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    recommendations = []
    
    # Embedding dimension recommendations
    if ent_emb is not None:
        actual_emb_dim = np.array(ent_emb).shape[1]
        recommendations.append(f"Set embedding_dim = {actual_emb_dim} (matches pretrained embeddings)")
    
    # num_nodes recommendations
    vpn_last_dim = sample['visit_padded_node'].shape[1] if 'visit_padded_node' in sample else None
    if vpn_last_dim:
        if vpn_last_dim == G_tg.num_nodes:
            recommendations.append(f"Use num_nodes = G_tg.num_nodes = {G_tg.num_nodes} (matches visit_padded_node)")
        elif vpn_last_dim == len(map_cluster):
            recommendations.append(f"Use num_nodes = len(map_cluster) = {len(map_cluster)} (matches visit_padded_node)")
    
    # Attention recommendations
    if len(ent2id) > 10000:  # Large vocabulary
        recommendations.append("Consider disabling alpha attention (use_alpha=False) to avoid parameter explosion")
    
    # Model architecture recommendations
    recommendations.append("Remove unused RETAINLayer import to avoid pyhealth dependency")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    print()
    print("=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
except Exception as e:
    print(f"✗ Error during analysis: {e}")
    import traceback
    traceback.print_exc()