"""
Test script for SparseGraphCare model
"""
import torch
import torch.nn.functional as F
from graphcare_sparse_model import SparseGraphCare, EdgeScorer
import numpy as np

def test_edge_scorer():
    """Test EdgeScorer module"""
    print("Testing EdgeScorer...")
    
    # Create dummy data
    node_dim = 128
    edge_dim = 128
    num_nodes = 100
    num_edges = 200
    
    x = torch.randn(num_nodes, node_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, edge_dim)
    
    # Initialize EdgeScorer
    edge_scorer = EdgeScorer(node_dim, edge_dim)
    
    # Forward pass
    edge_scores = edge_scorer(x, edge_index, edge_attr)
    
    print(f"  Input shapes: x={x.shape}, edge_index={edge_index.shape}, edge_attr={edge_attr.shape}")
    print(f"  Output shape: {edge_scores.shape}")
    print(f"  Score range: [{edge_scores.min().item():.4f}, {edge_scores.max().item():.4f}]")
    
    # Check output validity
    assert edge_scores.shape == (num_edges, 1), f"Expected shape ({num_edges}, 1), got {edge_scores.shape}"
    assert torch.all(edge_scores >= 0) and torch.all(edge_scores <= 1), "Edge scores should be in [0, 1]"
    
    print("  ✓ EdgeScorer test passed!")
    return True

def test_sparse_graphcare():
    """Test SparseGraphCare model"""
    print("\nTesting SparseGraphCare...")
    
    # Model parameters
    num_nodes = 500
    num_rels = 50
    max_visit = 16
    embedding_dim = 128
    hidden_dim = 128
    out_channels = 1
    batch_size = 4
    
    # Create dummy data
    torch.manual_seed(42)
    
    # Node and relation IDs
    node_ids = torch.randint(0, num_nodes, (batch_size * 50,))
    rel_ids = torch.randint(0, num_rels, (200,))
    
    # Edge index
    edge_index = torch.randint(0, batch_size * 50, (2, 200))
    
    # Batch assignment
    batch = torch.repeat_interleave(torch.arange(batch_size), 50)
    
    # Visit node matrix
    visit_node = torch.randn(batch_size, max_visit, num_nodes)
    
    # EHR nodes (dummy)
    ehr_nodes = [torch.zeros(num_nodes) for _ in range(batch_size)]
    for i in range(batch_size):
        # Set some random nodes as EHR nodes
        ehr_nodes[i][torch.randint(0, num_nodes, (20,))] = 1.0
    
    # Initialize model with sparsification
    model = SparseGraphCare(
        num_nodes=num_nodes,
        num_rels=num_rels,
        max_visit=max_visit,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        out_channels=out_channels,
        layers=2,  # Reduce layers for testing
        dropout=0.3,
        use_sparsification=True,
        sparsification_ratio=0.2,
        l1_lambda=1e-4,
        connectivity_lambda=1e-3
    )
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test training mode
    model.train()
    with torch.no_grad():
        try:
            result = model(
                node_ids=node_ids,
                rel_ids=rel_ids,
                edge_index=edge_index,
                batch=batch,
                visit_node=visit_node,
                ehr_nodes=ehr_nodes
            )
            
            if isinstance(result, tuple):
                logits, sparse_loss = result
                print(f"  Training output: logits shape={logits.shape}, sparse_loss={sparse_loss.item():.6f}")
            else:
                logits = result
                print(f"  Training output: logits shape={logits.shape}")
                
            assert logits.shape == (batch_size, out_channels), f"Expected logits shape ({batch_size}, {out_channels}), got {logits.shape}"
            print("  ✓ Training mode test passed!")
            
        except Exception as e:
            print(f"  ✗ Training mode test failed: {e}")
            return False
    
    # Test evaluation mode
    model.eval()
    with torch.no_grad():
        try:
            logits = model(
                node_ids=node_ids,
                rel_ids=rel_ids,
                edge_index=edge_index,
                batch=batch,
                visit_node=visit_node,
                ehr_nodes=ehr_nodes
            )
            
            print(f"  Evaluation output: logits shape={logits.shape}")
            assert logits.shape == (batch_size, out_channels), f"Expected logits shape ({batch_size}, {out_channels}), got {logits.shape}"
            print("  ✓ Evaluation mode test passed!")
            
        except Exception as e:
            print(f"  ✗ Evaluation mode test failed: {e}")
            return False
    
    # Test with attention storage
    try:
        result_dict = model(
            node_ids=node_ids,
            rel_ids=rel_ids,
            edge_index=edge_index,
            batch=batch,
            visit_node=visit_node,
            ehr_nodes=ehr_nodes,
            store_attn=True
        )
        
        print(f"  Attention storage: {list(result_dict.keys())}")
        print("  ✓ Attention storage test passed!")
        
    except Exception as e:
        print(f"  ✗ Attention storage test failed: {e}")
        return False
    
    return True

def test_backward_compatibility():
    """Test backward compatibility with original GraphCare"""
    print("\nTesting backward compatibility...")
    
    # Import the compatibility class
    from graphcare_sparse_model import GraphCare
    
    # Create a GraphCare model (should disable sparsification)
    model = GraphCare(
        num_nodes=100,
        num_rels=20,
        max_visit=16,
        embedding_dim=64,
        hidden_dim=64,
        out_channels=1,
        layers=2
    )
    
    print(f"  GraphCare sparsification enabled: {model.use_sparsification}")
    assert not model.use_sparsification, "GraphCare should have sparsification disabled"
    print("  ✓ Backward compatibility test passed!")
    
    return True

def test_sparsification_loss():
    """Test sparsification loss computation"""
    print("\nTesting sparsification loss computation...")
    
    # Create dummy edge scores and edge index
    num_edges = 100
    edge_scores = torch.rand(num_edges, 1) * 0.8 + 0.1  # Random scores in [0.1, 0.9]
    edge_index = torch.randint(0, 50, (2, num_edges))
    
    # Create a dummy model to test loss computation
    model = SparseGraphCare(
        num_nodes=50,
        num_rels=10,
        max_visit=8,
        embedding_dim=32,
        hidden_dim=32,
        out_channels=1,
        layers=1,
        l1_lambda=0.1,
        connectivity_lambda=0.05
    )
    
    # Test loss computation
    total_loss, l1_loss, connectivity_loss = model.compute_sparsification_loss(edge_scores, edge_index)
    
    print(f"  Total loss: {total_loss.item():.6f}")
    print(f"  L1 loss: {l1_loss.item():.6f}")
    print(f"  Connectivity loss: {connectivity_loss.item():.6f}")
    
    # Check that losses are reasonable
    assert total_loss.item() >= 0, "Total loss should be non-negative"
    assert l1_loss.item() >= 0, "L1 loss should be non-negative"
    assert connectivity_loss.item() >= 0, "Connectivity loss should be non-negative"
    
    print("  ✓ Sparsification loss test passed!")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("SparseGraphCare Model Test Suite")
    print("=" * 60)
    
    all_passed = True
    
    # Run all tests
    tests = [
        test_edge_scorer,
        test_sparse_graphcare,
        test_backward_compatibility,
        test_sparsification_loss
    ]
    
    for test_func in tests:
        try:
            passed = test_func()
            all_passed = all_passed and passed
        except Exception as e:
            print(f"  ✗ {test_func.__name__} failed with exception: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! SparseGraphCare model is ready to use.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    print("=" * 60)