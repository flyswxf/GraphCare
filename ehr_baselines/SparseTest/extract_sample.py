"""
Extract a single sample from GraphCare sample_dataset and save it as:
- PKL file (pickle of the raw sample dict)
- Tree-structured TXT file (human-readable overview)

Usage examples:
  python -u ehr_baselines/SparseTest/extract_sample.py --dataset mimic3 --task readmission --index 0
  python -u ehr_baselines/SparseTest/extract_sample.py --dataset mimic3 --task drugrec --patient_id 12345 \
    --out_pkl data/samples/mimic3_drugrec_pid12345.pkl --out_txt data/samples/mimic3_drugrec_pid12345.txt
"""
import os
import sys
import argparse
import pickle
import io
import numpy as np
import torch
from typing import Any

# Add parent directory to path for imports (to access graphcare module)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/r/root/workspace/GraphCare')
from graphcare import load_everything


def summarize_leaf(x: Any) -> str:
    try:
        import pandas as pd  # optional, used only for type check
        has_pd = True
    except Exception:
        has_pd = False

    if isinstance(x, np.ndarray):
        return f"ndarray shape={tuple(x.shape)} dtype={x.dtype}"
    if torch.is_tensor(x):
        return f"tensor shape={tuple(x.shape)} dtype={x.dtype} device={x.device}"
    if has_pd:
        import pandas as pd  # type: ignore
        if isinstance(x, (pd.DataFrame, pd.Series)):
            return f"{type(x).__name__} shape={x.shape}"
    if isinstance(x, (set, frozenset)):
        s = list(x)
        preview = ", ".join(repr(v) for v in s[:5])
        more = f", +{len(s)-5} more" if len(s) > 5 else ""
        return f"{type(x).__name__}(size={len(s)}): {preview}{more}"
    if isinstance(x, (bytes, bytearray)):
        return f"{type(x).__name__}(len={len(x)})"
    if isinstance(x, str):
        return repr(x[:80] + ("..." if len(x) > 80 else ""))
    return repr(x)


def build_tree_text(obj: Any, max_depth: int = 4, max_items: int = 10) -> str:
    buf = io.StringIO()

    def rec(o: Any, prefix: str, depth: int):
        if depth > max_depth:
            buf.write(prefix + "... (max_depth reached)\n")
            return

        if isinstance(o, dict):
            buf.write(prefix + f"dict(len={len(o)})\n")
            for k, v in o.items():
                key_str = repr(k)
                if isinstance(v, (dict, list, tuple, set)) or torch.is_tensor(v) or isinstance(v, np.ndarray):
                    buf.write(prefix + f"└─ {key_str}:\n")
                    rec(v, prefix + "   ", depth + 1)
                else:
                    buf.write(prefix + f"└─ {key_str}: {summarize_leaf(v)}\n")
            return

        if isinstance(o, (list, tuple)):
            buf.write(prefix + f"{type(o).__name__}(len={len(o)})\n")
            n = min(len(o), max_items)
            for i in range(n):
                v = o[i]
                if isinstance(v, (dict, list, tuple, set)) or torch.is_tensor(v) or isinstance(v, np.ndarray):
                    buf.write(prefix + f"└─ [{i}]:\n")
                    rec(v, prefix + "   ", depth + 1)
                else:
                    buf.write(prefix + f"└─ [{i}]: {summarize_leaf(v)}\n")
            if len(o) > n:
                buf.write(prefix + f"└─ ... (+{len(o)-n} more)\n")
            return

        if isinstance(o, set):
            s = list(o)
            buf.write(prefix + f"set(len={len(s)})\n")
            n = min(len(s), max_items)
            for i in range(n):
                buf.write(prefix + f"└─ [#{i}]: {summarize_leaf(s[i])}\n")
            if len(s) > n:
                buf.write(prefix + f"└─ ... (+{len(s)-n} more)\n")
            return

        # numpy / tensor or others
        buf.write(prefix + summarize_leaf(o) + "\n")

    rec(obj, prefix="", depth=0)
    return buf.getvalue()


def main():
    parser = argparse.ArgumentParser(description="Extract single sample from sample_dataset")
    parser.add_argument('--dataset', type=str, default='mimic3', choices=['mimic3', 'mimic4'])
    parser.add_argument('--task', type=str, default='readmission', choices=['readmission', 'mortality', 'lenofstay', 'drugrec', 'procedure'])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--patient_id', type=str, help='Patient ID to locate sample')
    group.add_argument('--index', type=int, help='0-based index of sample in sample_dataset')
    parser.add_argument('--out_pkl', type=str, default=None, help='Output path for PKL')
    parser.add_argument('--out_txt', type=str, default=None, help='Output path for tree TXT')
    parser.add_argument('--max_depth', type=int, default=4, help='Max depth for tree print')
    parser.add_argument('--max_items', type=int, default=10, help='Max items per list/set in tree')

    args = parser.parse_args()

    # Load dataset
    sample_dataset, graph, ent2id, rel2id, ent_emb, rel_emb, \
        map_cluster, map_cluster_inv, map_cluster_rel, map_cluster_rel_inv, \
        ccscm_id2clus, ccsproc_id2clus, atc3_id2clus = load_everything(args.dataset, args.task)

    # Locate sample
    idx = None
    if args.patient_id is not None:
        target = str(args.patient_id)
        for i, p in enumerate(sample_dataset):
            if str(p.get('patient_id')) == target:
                idx = i
                break
        if idx is None:
            raise SystemExit(f"[ERROR] patient_id={target} not found in sample_dataset")
    else:
        if args.index < 0 or args.index >= len(sample_dataset):
            raise SystemExit(f"[ERROR] index out of range: {args.index} (0..{len(sample_dataset)-1})")
        idx = int(args.index)

    sample = sample_dataset[idx]
    pid = sample.get('patient_id')

    # Determine outputs
    default_stem = f"{args.dataset}_{args.task}_idx{idx}" if pid is None else f"{args.dataset}_{args.task}_pid{pid}"
    out_pkl = args.out_pkl or os.path.join('data', 'samples', default_stem + '.pkl')
    out_txt = args.out_txt or os.path.join('data', 'samples', default_stem + '.txt')

    os.makedirs(os.path.dirname(out_pkl), exist_ok=True)
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)

    # Save PKL
    with open(out_pkl, 'wb') as f:
        pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save tree TXT
    tree_text = build_tree_text(sample, max_depth=args.max_depth, max_items=args.max_items)
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write(tree_text)

    print(f"[DONE] Saved sample (index={idx}, patient_id={pid})")
    print(f"  PKL: {os.path.abspath(out_pkl)}")
    print(f"  TXT: {os.path.abspath(out_txt)}")


if __name__ == '__main__':
    main()