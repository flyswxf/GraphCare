"""
Cluster mapper: selects most related cluster indices for add/remove based on keywords.

Reads KEYWORD_FILE, loads cluster embeddings according to task, computes similarity
to keyword embeddings (via embedding_retriever), and writes top-k indices to
CLUSTER_INDEX_FILE as JSON: {"add": [..], "remove": [..]}.
"""

import os
import json
from typing import List, Dict
import sys

sys.path.append(r'D:\\desktop\\code\\ICU\\GraphCare')

from constants import (
    KEYWORD_FILE,
    CLUSTER_INDEX_FILE,
    CLUSTERS_CCSPROC_CCSSCM,
    CLUSTERS_CCSPROC_CCSSCM_ATC3,
    CLUSTERS_CCSCM_ATC3,
    TOPK_CLUSTERS_ADD_DEFAULT,
    TOPK_CLUSTERS_REMOVE_DEFAULT,
    INFERENCE_RESULT_FILE,
)

try:
    from graphcare_.graph_generation.get_emb import embedding_retriever
except Exception:
    embedding_retriever = None


def _load_task() -> str:
    # Try to infer task from inference result ("task" field is not guaranteed)
    task = "drugrec"
    return task


def _cluster_file_for_task(task: str) -> str:
    if task in ("lenofstay", "drugrec"):
        return CLUSTERS_CCSPROC_CCSSCM
    if task in ("mortality", "readmission"):
        return CLUSTERS_CCSPROC_CCSSCM_ATC3
    if task == "procedure":
        return CLUSTERS_CCSCM_ATC3
    return CLUSTERS_CCSPROC_CCSSCM


def _cosine(u, v) -> float:
    import math
    if not u or not v:
        return 0.0
    dot = sum(a*b for a, b in zip(u, v))
    nu = math.sqrt(sum(a*a for a in u))
    nv = math.sqrt(sum(b*b for b in v))
    if nu == 0 or nv == 0:
        return 0.0
    return dot / (nu * nv)


def _embed(text: str):
    if embedding_retriever is None:
        return []
    try:
        return embedding_retriever(text)
    except Exception:
        return []


def _load_keywords() -> Dict[str, List[str]]:
    if not os.path.exists(KEYWORD_FILE):
        return {"add": [], "remove": []}
    with open(KEYWORD_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_clusters(path: str) -> Dict[str, Dict]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def map_keywords_to_clusters(task: str = None,
                             topk_add: int = TOPK_CLUSTERS_ADD_DEFAULT,
                             topk_remove: int = TOPK_CLUSTERS_REMOVE_DEFAULT) -> Dict[str, List[int]]:
    task = task or _load_task()
    cluster_file = _cluster_file_for_task(task)
    clusters = _load_clusters(cluster_file)
    keywords = _load_keywords()

    # Precompute keyword embeddings
    add_embs = [(_embed(k), k) for k in keywords.get("add", [])]
    rem_embs = [(_embed(k), k) for k in keywords.get("remove", [])]

    # Score clusters by summed cosine similarity across keywords
    add_scores = []
    rem_scores = []
    for idx_str, info in clusters.items():
        emb = info.get("embedding")
        if not emb:
            continue
        s_add = sum(_cosine(emb, e) for e, _k in add_embs)
        s_rem = sum(_cosine(emb, e) for e, _k in rem_embs)
        add_scores.append((int(idx_str), float(s_add)))
        rem_scores.append((int(idx_str), float(s_rem)))

    add_scores.sort(key=lambda x: x[1], reverse=True)
    rem_scores.sort(key=lambda x: x[1], reverse=True)

    add_indices = [i for i, _s in add_scores[:max(1, topk_add)]]
    rem_indices = [i for i, _s in rem_scores[:max(1, topk_remove)]]

    result = {"add": add_indices, "remove": rem_indices}

    os.makedirs(os.path.dirname(CLUSTER_INDEX_FILE), exist_ok=True)
    with open(CLUSTER_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


if __name__ == "__main__":
    map_keywords_to_clusters()