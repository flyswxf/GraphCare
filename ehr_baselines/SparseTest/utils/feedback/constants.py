"""
Centralized constants for feedback processing.

Update these paths if your project structure changes.
"""

import os

# Base directory for the project (absolute for Windows environment)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))

# Paths for feedback I/O (files live directly under utils/feedback)
FEEDBACK_DIR = os.path.join(BASE_DIR, "ehr_baselines", "SparseTest", "utils", "feedback")
RESPONSE_FILE = os.path.join(FEEDBACK_DIR, "response.txt")
RESULT_DIR = os.path.join(FEEDBACK_DIR, "result")
KEYWORD_FILE = os.path.join(RESULT_DIR, "keyword.txt")
CLUSTER_INDEX_FILE = os.path.join(RESULT_DIR, "clusterIndex.txt")

# Model output for context (optional)
INFERENCE_RESULT_FILE = os.path.join(BASE_DIR, "ehr_baselines", "SparseTest", "result", "inference_result_with_names.json")

# Embedding utility path (used by keyword/cluster mapping)
GET_EMB_MODULE_PATH = os.path.join(BASE_DIR, "graphcare_", "graph_generation", "get_emb.py")

# Clustering data roots (mirroring attention_init.ipynb logic)
CLUSTERING_ROOT = os.path.join(BASE_DIR, "clustering")
CLUSTERS_CCSPROC_CCSSCM = os.path.join(CLUSTERING_ROOT, "ccscm_ccsproc", "clusters_th015.json")
CLUSTERS_CCSPROC_CCSSCM_ATC3 = os.path.join(CLUSTERING_ROOT, "ccscm_ccsproc_atc3", "clusters_th015.json")
CLUSTERS_CCSCM_ATC3 = os.path.join(CLUSTERING_ROOT, "ccscm_atc3", "clusters_th015.json")

# Controls: number of keywords per group and top-k clusters
NUM_KEYWORDS_ADD_DEFAULT = 6
NUM_KEYWORDS_REMOVE_DEFAULT = 6
TOPK_CLUSTERS_ADD_DEFAULT = 10
TOPK_CLUSTERS_REMOVE_DEFAULT = 10

# Language indicators (used for rule-based parsing when LLM isn’t available)
ADD_HINTS = [
    "add", "include", "prefer", "need", "recommend", "increase",
    "添加", "加上", "加入", "需要", "推荐", "偏好", "增加", "希望"
]
REMOVE_HINTS = [
    "remove", "exclude", "avoid", "stop", "not", "do not",
    "移除", "去掉", "排除", "避免", "停止", "不", "不要", "取消"
]