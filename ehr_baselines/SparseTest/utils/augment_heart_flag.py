#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone script: append cardiac flag to drugs_ind for each patient in sample_dataset
and overwrite the dataset file at its source path.

Usage examples:
  python -u ehr_baselines/SparseTest/utils/augment_heart_flag.py --dataset mimic3 --task drugrec --Heart
  python -u ehr_baselines/SparseTest/utils/augment_heart_flag.py --dataset mimic3 --task drugrec

Notes:
- Only applies to task 'drugrec'. Other tasks are skipped safely.
- Overwrites the same dataset file used for loading (Heart flag controls path variant).
"""

import os
import sys
import argparse
import csv
import pickle
from typing import Dict

import numpy as np
import torch

# Add project root to sys.path for importing data_prepare
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.join(CURRENT_DIR, '..', '..', '..')
sys.path.append(os.path.abspath(PROJECT_ROOT))

from data_prepare import load_dataset  # type: ignore


def get_dataset_file_path(dataset: str, task: str, Heart: bool) -> str:
    """Replicate data_prepare.py path logic for sample_dataset files."""
    if task in ("drugrec", "lenofstay"):
        base = "./data/ccscm_ccsproc"
        if Heart:
            filename = f"sample_dataset_{dataset}_{task}_Heart_th015.pkl"
        else:
            filename = f"sample_dataset_{dataset}_{task}_th015.pkl"
    elif task in ("mortality", "readmission"):
        base = "./data/ccscm_ccsproc_atc3"
        filename = f"sample_dataset_{dataset}_{task}_th015.pkl"
    elif task == "procedure":
        base = "./data/ccscm_atc3"
        filename = f"sample_dataset_{dataset}_{task}_th015.pkl"
    else:
        raise ValueError(f"Unsupported task: {task}")
    return os.path.join(base, filename)


def read_cardiac_flags(csv_path: str) -> Dict[int, int]:
    """Read patient_id -> cardiac flag mapping from CSV."""
    cardiac_map: Dict[int, int] = {}
    if not os.path.exists(csv_path):
        print(f"[HEART] Cardiac flags CSV not found at {csv_path}; skipping augmentation")
        return cardiac_map
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    pid = int(row.get('patient_id'))
                    flag = int(row.get('cardiac'))
                    cardiac_map[pid] = flag
                except Exception:
                    # Skip malformed rows
                    pass
    except Exception as e:
        print(f"[HEART] Failed reading cardiac flags: {e}")
    return cardiac_map


def augment_drugrec_with_cardiac(sample_dataset, cardiac_map: Dict[int, int]) -> int:
    """Append cardiac flag as extra channel to drugs_ind for each patient (drugrec task)."""
    if not cardiac_map:
        print("[HEART] Empty cardiac_map; nothing to append")
        return 0

    updated = 0
    for p in sample_dataset:
        try:
            pid = int(p.get('patient_id', -1))
        except Exception:
            pid = -1
        flag = float(cardiac_map.get(pid, 0))

        di = p.get('drugs_ind')
        if di is None:
            # If drugs_ind missing, skip this record safely
            continue

        if isinstance(di, torch.Tensor):
            p['drugs_ind'] = torch.cat([di.float(), torch.tensor([flag], dtype=torch.float32)], dim=0)
        else:
            arr = np.array(di, dtype=float)
            p['drugs_ind'] = torch.tensor(np.append(arr, flag), dtype=torch.float32)
        updated += 1
    return updated


def main():
    parser = argparse.ArgumentParser(description="Append cardiac flag to sample_dataset and overwrite source file")
    parser.add_argument('--dataset', type=str, default='mimic3', choices=['mimic3', 'mimic4'], help='Dataset to use')
    parser.add_argument('--task', type=str, default='drugrec', choices=['readmission', 'mortality', 'lenofstay', 'drugrec', 'procedure'], help='Task to run')
    parser.add_argument('--Heart', action='store_true', help='Use Heart dataset variant for path resolution and loading')
    args = parser.parse_args()

    dataset = args.dataset
    task = args.task
    Heart = args.Heart

    # Resolve dataset file path and load
    dataset_file = get_dataset_file_path(dataset, task, Heart)
    if not os.path.exists(dataset_file):
        print(f"[ERROR] Dataset file not found: {dataset_file}")
        sys.exit(1)

    print(f"[INFO] Loading processed dataset from {dataset_file}")
    sample_dataset = load_dataset(True, dataset=dataset, task=task, Heart=Heart)

    # Only apply to drugrec
    if task != 'drugrec':
        print(f"[INFO] Task '{task}' is not 'drugrec'; no augmentation applied.")
        print("[INFO] Exiting without changes.")
        sys.exit(0)

    # Read cardiac flags CSV (same relative path logic as runSparseModel.py)
    csv_path = os.path.join(CURRENT_DIR, '..', '..', 'dataPrepare', 'match_stats', 'cardiac_condition_flags.csv')
    csv_path = os.path.abspath(csv_path)
    print(f"[INFO] Reading cardiac flags from: {csv_path}")
    cardiac_map = read_cardiac_flags(csv_path)

    # Apply augmentation
    print("[INFO] Appending cardiac flag to drugs_ind ...")
    updated_count = augment_drugrec_with_cardiac(sample_dataset, cardiac_map)
    print(f"[HEART] Appended cardiac flag to drugs_ind for {updated_count} samples (total={len(sample_dataset)})")

    # Overwrite save
    try:
        os.makedirs(os.path.dirname(dataset_file), exist_ok=True)
        with open(dataset_file, 'wb') as f:
            pickle.dump(sample_dataset, f)
        print(f"[INFO] Overwritten dataset saved to: {dataset_file}")
    except Exception as e:
        print(f"[ERROR] Failed to save dataset to {dataset_file}: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()