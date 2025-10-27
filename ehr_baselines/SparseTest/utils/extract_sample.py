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


def extract_sample_to_pkl(dataset='mimic3', task='readmission', patient_id=None, index=None, 
                          out_pkl=None, out_txt=None, max_depth=4, max_items=10, verbose=True, Heart=False):
    """
    Extract a single sample from GraphCare sample_dataset and save it as PKL file.
    
    Args:
        dataset: Dataset name ('mimic3' or 'mimic4')
        task: Task name ('readmission', 'mortality', 'lenofstay', 'drugrec', 'procedure')
        patient_id: Patient ID to locate sample (mutually exclusive with index)
        index: 0-based index of sample in sample_dataset (mutually exclusive with patient_id)
        out_pkl: Output path for PKL file (optional)
        out_txt: Output path for tree TXT file (optional)
        max_depth: Max depth for tree print
        max_items: Max items per list/set in tree
        verbose: Whether to print progress messages
        
    Returns:
        str: Absolute path to the saved PKL file
        
    Raises:
        ValueError: If neither patient_id nor index is provided, or if sample not found
    """
    if patient_id is None and index is None:
        raise ValueError("Either patient_id or index must be provided")
    if patient_id is not None and index is not None:
        raise ValueError("patient_id and index are mutually exclusive")
    
    # Determine default output paths
    if patient_id is not None:
        default_stem = f"{dataset}_{task}_pid{patient_id}"
    else:
        default_stem = f"{dataset}_{task}_idx{index}"
    
    default_pkl = os.path.join('ehr_baselines', 'SparseTest', 'samples', default_stem + '.pkl')
    default_txt = os.path.join('ehr_baselines', 'SparseTest', 'samples', default_stem + '.txt')
    
    out_pkl = out_pkl or default_pkl
    out_txt = out_txt or default_txt
    
    # Check if PKL file already exists
    if os.path.exists(out_pkl):
        if verbose:
            print(f"[INFO] PKL文件已存在，直接返回: {os.path.abspath(out_pkl)}")
        return os.path.abspath(out_pkl)
    
    if verbose:
        print(f"[INFO] PKL文件不存在，开始提取样本...")
        print(f"[INFO] 数据集: {dataset}, 任务: {task}")
        if patient_id is not None:
            print(f"[INFO] 患者ID: {patient_id}")
        else:
            print(f"[INFO] 样本索引: {index}")
    
    # Load dataset
    if verbose:
        print(f"[INFO] 正在加载数据集...")
    sample_dataset, graph, ent2id, rel2id, ent_emb, rel_emb, \
        map_cluster, map_cluster_inv, map_cluster_rel, map_cluster_rel_inv, \
        ccscm_id2clus, ccsproc_id2clus, atc3_id2clus = load_everything(dataset, task, Heart=Heart)
    
    # Locate sample
    idx = None
    if patient_id is not None:
        target = str(patient_id)
        for i, p in enumerate(sample_dataset):
            if str(p.get('patient_id')) == target:
                idx = i
                break
        if idx is None:
            raise ValueError(f"patient_id={target} not found in sample_dataset")
    else:
        if index < 0 or index >= len(sample_dataset):
            raise ValueError(f"index out of range: {index} (0..{len(sample_dataset)-1})")
        idx = int(index)
    
    sample = sample_dataset[idx]
    pid = sample.get('patient_id')
    
    if verbose:
        print(f"[INFO] 找到样本 (索引={idx}, 患者ID={pid})")
    
    # Create output directories
    os.makedirs(os.path.dirname(out_pkl), exist_ok=True)
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    
    # Save PKL
    if verbose:
        print(f"[INFO] 正在保存PKL文件...")
    with open(out_pkl, 'wb') as f:
        pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save tree TXT
    if verbose:
        print(f"[INFO] 正在保存TXT文件...")
    tree_text = build_tree_text(sample, max_depth=max_depth, max_items=max_items)
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write(tree_text)
    
    if verbose:
        print(f"[DONE] 样本提取完成 (索引={idx}, 患者ID={pid})")
        print(f"  PKL: {os.path.abspath(out_pkl)}")
        print(f"  TXT: {os.path.abspath(out_txt)}")
    
    return os.path.abspath(out_pkl)


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
    
    try:
        pkl_path = extract_sample_to_pkl(
            dataset=args.dataset,
            task=args.task,
            patient_id=args.patient_id,
            index=args.index,
            out_pkl=args.out_pkl,
            out_txt=args.out_txt,
            max_depth=args.max_depth,
            max_items=args.max_items,
            verbose=True
        )
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()