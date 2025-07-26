from get_emb import embedding_retriever
import numpy as np
from tqdm import tqdm
import pickle
import os

SAVE_INTERVAL = 5000  # Save after processing every 1000 names
MAX_RETRIES = 30  # Retry up to 5 times if there's an error

# 获取脚本目录和项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "../..")
data_dir = os.path.join(project_root, "data")
concept_names_path = os.path.join(project_root, "KG_mapping/umls/concept_names.txt")

# Load previous embeddings if they exist
# 原始路径: /data/pj20/exp_data/umls_ent_emb_.pkl
try:
    with open(os.path.join(data_dir, 'umls_ent_emb_.pkl'), 'rb') as f:
        umls_ent_emb = pickle.load(f)
except FileNotFoundError:
    umls_ent_emb = []

# Loading and preprocessing the names
# 原始路径: /home/pj20/GraphCare/KG_mapping/umls/concept_names.txt
with open(concept_names_path, 'r', encoding='utf-8') as f:
    umls_ent = f.readlines()

umls_names = [line.split('\t')[1][:-1] for line in umls_ent]

# Skip names that are already processed
umls_names = umls_names[len(umls_ent_emb):]

# 顺序处理，不使用多线程，直接调用embedding_retriever
for idx, name in enumerate(tqdm(umls_names, total=len(umls_names))):
    current_total_idx = len(umls_ent_emb) + idx + 1
    # print(f"\n正在处理第 {current_total_idx} 项: {name}")
    
    # 跳过特定的敏感内容项目
    if "Does not enjoy having sex" in name:
        print(f"跳过敏感内容第 {current_total_idx} 项: {name}")
        # 添加一个零向量占位符
        umls_ent_emb.append([0.0] * 1024)  # 1024维零向量
        continue
    
    try:
        emb = embedding_retriever(term=name)
        umls_ent_emb.append(emb)
        # print(f"成功处理第 {current_total_idx} 项")
    except Exception as e:
        print(f"处理第 {current_total_idx} 项时出错: {e}")
        print(f"概念名称: {name}")
        # 如果是特定的403错误，跳过这一项并继续
        if "403" in str(e) or "API 访问被拒绝" in str(e):
            print(f"跳过第 {current_total_idx} 项，继续处理下一项")
            # 添加一个零向量占位符
            umls_ent_emb.append([0.0] * 1024)  # 1024维零向量
        else:
            raise e
    
    # Periodically save the data
    if (idx + 1) % SAVE_INTERVAL == 0:
        # 原始路径: /data/pj20/exp_data/umls_ent_emb_.pkl
        with open(os.path.join(data_dir, 'umls_ent_emb_.pkl'), 'wb') as f:
            pickle.dump(umls_ent_emb, f)
        print(f"已保存进度，当前处理了 {len(umls_ent_emb)} 项")

# Save the final data
# 原始路径: /data/pj20/exp_data/umls_ent_emb_.pkl
with open(os.path.join(data_dir, 'umls_ent_emb_.pkl'), 'wb') as f:
    pickle.dump(umls_ent_emb, f)
