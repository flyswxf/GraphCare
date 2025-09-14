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

# Resolve sample index
# patient_id或sample_index任选其一
idx = None
if args.patient_id is not None:
    target_pid = str(args.patient_id)
    for i, p in enumerate(sample_dataset):
        if str(p.get('patient_id')) == target_pid:
            idx = i
            break
    if idx is None:
        print(f"[ERROR] patient_id={target_pid} not found in dataset")
        try:
            wandb.finish()
        except Exception:
            pass
        sys.exit(1)
elif args.sample_index is not None:
    if 0 <= int(args.sample_index) < len(sample_dataset):
        idx = int(args.sample_index)
    else:
        print(f"[ERROR] sample_index out of range: {args.sample_index} (0..{len(sample_dataset)-1})")
        try:
            wandb.finish()
        except Exception:
            pass
        sys.exit(1)
else:
    print("[ERROR] Inference mode requires --patient_id or --sample_index")
    try:
        wandb.finish()
    except Exception:
        pass
    sys.exit(1)

# Create a single-sample dataset and DataLoader for proper batch handling
from graphcare import Dataset
from torch_geometric.loader import DataLoader

# Create a dataset with just the target sample
single_sample_dataset = [sample_dataset[idx]]
inference_dataset = Dataset(G=G_tg, dataset=single_sample_dataset, task=task)
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
pid_val = sample_dataset[idx].get('patient_id', None)
result = {
    "patient_id": None if pid_val is None else str(pid_val),
    "sample_index": idx,
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