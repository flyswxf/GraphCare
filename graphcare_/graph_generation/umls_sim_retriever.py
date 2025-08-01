import pickle
import numpy as np
from tqdm import tqdm
import multiprocessing
import os

def cosine_similarity(u, v):
    """
    计算两个向量的余弦相似度，安全处理零向量情况
    """
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    
    # 如果任一向量为零向量，返回0相似度
    if norm_u == 0 or norm_v == 0:
        return 0.0
    
    return np.dot(u, v) / (norm_u * norm_v)

def find_most_similar_embedding(target_emb, umls_ent_emb):
    max_similarity = -1
    max_index = None
    for idx, umls_emb in enumerate(umls_ent_emb):
        similarity = cosine_similarity(target_emb, umls_emb)
        if similarity > max_similarity:
            max_similarity = similarity
            max_index = idx
    if max_similarity > SIMILARITY_THRESHOLD:
        return max_index
    else:
        return None

def process_chunk(chunk, id2emb, umls_ent_emb, output_dict, process_id=None):
    # 简化处理，不在子进程中显示进度条，避免显示混乱
    for key in chunk:
        output_dict[key] = find_most_similar_embedding(id2emb[key], umls_ent_emb)

def sequential_mapping(id2emb, umls_ent_emb, desc="Processing"):
    """使用单进程处理，避免大文件多次加载导致的内存问题"""
    keys = list(id2emb.keys())
    output_dict = {}
    
    print(f"\n{desc}: 总共 {len(keys)} 个项目，使用单进程处理（避免内存问题）")
    
    # 显示进度
    for key in tqdm(keys, desc=f"{desc}"):
        output_dict[key] = find_most_similar_embedding(id2emb[key], umls_ent_emb)
    
    return output_dict

def parallel_mapping(id2emb, umls_ent_emb, num_processes, desc="Processing"):
    """如果数据量小，使用多进程；如果数据量大，自动切换到单进程"""
    # 检查umls_ent_emb的大小，如果太大则使用单进程
    umls_size_mb = len(umls_ent_emb) * len(umls_ent_emb[0]) * 8 / (1024 * 1024)  # 估算大小
    
    if umls_size_mb > 500:  # 如果超过500MB，使用单进程
        print(f"检测到大型嵌入文件 (~{umls_size_mb:.1f}MB)，使用单进程处理以避免内存问题")
        return sequential_mapping(id2emb, umls_ent_emb, desc)
    
    # 原有的多进程逻辑
    keys = list(id2emb.keys())
    chunks = np.array_split(keys, num_processes)
    manager = multiprocessing.Manager()
    output_dict = manager.dict()
    processes = []

    print(f"\n{desc}: 总共 {len(keys)} 个项目，使用 {num_processes} 个进程")
    
    for i, chunk in enumerate(chunks):
        p = multiprocessing.Process(target=process_chunk, args=(chunk, id2emb, umls_ent_emb, output_dict, i))
        processes.append(p)
        p.start()

    # 显示总体进度
    import time
    with tqdm(total=len(keys), desc=f"{desc}") as pbar:
        last_count = 0
        while any(p.is_alive() for p in processes):
            current_count = len(output_dict)
            if current_count > last_count:
                pbar.update(current_count - last_count)
                last_count = current_count
            time.sleep(0.5)  # 减少更新频率
        
        # 确保进度条显示完成
        final_count = len(output_dict)
        if final_count > last_count:
            pbar.update(final_count - last_count)

    for p in processes:
        p.join()

    return dict(output_dict)

if __name__ == "__main__":
    # 获取脚本目录和项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "../..")
    data_dir = os.path.join(project_root, "data")
    concept_names_path = os.path.join(project_root, "KG_mapping/umls/concept_names.txt")
    
    # 原始路径: /home/pj20/GraphCare/KG_mapping/umls/concept_names.txt
    with open(concept_names_path, 'r', encoding='utf-8') as f:
        umls_ent = f.readlines() 
    
    umls_ids = []
    umls_names = []

    for line in umls_ent:
        umls_id = line.split('\t')[0]
        umls_name = line.split('\t')[1][:-1]
        umls_names.append(umls_name)
        umls_ids.append(umls_id)
        
        
    # 原始路径: /data/pj20/exp_data/umls_ent_emb_.pkl
    with open(os.path.join(data_dir, 'umls_ent_emb_.pkl'), 'rb') as f:
        umls_ent_emb = pickle.load(f)
        
    # 原始路径: /data/pj20/exp_data/atc3_id2emb.pkl
    with open(os.path.join(data_dir, 'atc3_id2emb.pkl'), 'rb') as f:
        atc3_id2emb = pickle.load(f)
        
    # 原始路径: /data/pj20/exp_data/ccscm_id2emb.pkl
    with open(os.path.join(data_dir, 'ccscm_id2emb.pkl'), 'rb') as f:
        ccscm_id2emb = pickle.load(f)
        
    # 原始路径: /data/pj20/exp_data/ccsproc_id2emb.pkl
    with open(os.path.join(data_dir, 'ccsproc_id2emb.pkl'), 'rb') as f:
        ccsproc_id2emb = pickle.load(f)



    ccscm2umls, ccsproc2umls, atc32umls = {}, {}, {}

    # Define a similarity threshold
    SIMILARITY_THRESHOLD = 0.7
    # 减少进程数量以避免内存问题
    num_processes = min(8, multiprocessing.cpu_count())  # 最多使用8个进程
    print(f"使用 {num_processes} 个进程进行并行处理")

    # 任务1: 处理 CCSCM 到 UMLS 的映射
    ccscm_file = os.path.join(data_dir, 'ccscm2umls.pkl')
    if os.path.exists(ccscm_file):
        print("\n=== 任务 1/3: CCSCM->UMLS 映射文件已存在，跳过 ===")
        print(f"✓ 跳过 CCSCM->UMLS 映射: {ccscm_file}")
    else:
        print("\n=== 任务 1/3: 处理 CCSCM 到 UMLS 的映射 ===")
        ccscm2umls = parallel_mapping(ccscm_id2emb, umls_ent_emb, num_processes, "CCSCM->UMLS")
        
        # 立即保存 CCSCM 映射结果
        with open(ccscm_file, 'wb') as f:
            pickle.dump(ccscm2umls, f)
        print(f"✓ CCSCM->UMLS 映射已保存: {len(ccscm2umls)} 个映射 -> {ccscm_file}")
        del ccscm2umls  # 释放内存
    
    # 任务2: 处理 CCSPROC 到 UMLS 的映射
    ccsproc_file = os.path.join(data_dir, 'ccsproc2umls.pkl')
    if os.path.exists(ccsproc_file):
        print("\n=== 任务 2/3: CCSPROC->UMLS 映射文件已存在，跳过 ===")
        print(f"✓ 跳过 CCSPROC->UMLS 映射: {ccsproc_file}")
    else:
        print("\n=== 任务 2/3: 处理 CCSPROC 到 UMLS 的映射 ===")
        ccsproc2umls = parallel_mapping(ccsproc_id2emb, umls_ent_emb, num_processes, "CCSPROC->UMLS")
        
        # 立即保存 CCSPROC 映射结果
        with open(ccsproc_file, 'wb') as f:
            pickle.dump(ccsproc2umls, f)
        print(f"✓ CCSPROC->UMLS 映射已保存: {len(ccsproc2umls)} 个映射 -> {ccsproc_file}")
        del ccsproc2umls  # 释放内存
    
    # 任务3: 处理 ATC3 到 UMLS 的映射
    atc3_file = os.path.join(data_dir, 'atc32umls.pkl')
    if os.path.exists(atc3_file):
        print("\n=== 任务 3/3: ATC3->UMLS 映射文件已存在，跳过 ===")
        print(f"✓ 跳过 ATC3->UMLS 映射: {atc3_file}")
    else:
        print("\n=== 任务 3/3: 处理 ATC3 到 UMLS 的映射 ===")
        atc32umls = parallel_mapping(atc3_id2emb, umls_ent_emb, num_processes, "ATC3->UMLS")
        
        # 立即保存 ATC3 映射结果
        with open(atc3_file, 'wb') as f:
            pickle.dump(atc32umls, f)
        print(f"✓ ATC3->UMLS 映射已保存: {len(atc32umls)} 个映射 -> {atc3_file}")
        del atc32umls  # 释放内存
    
    print("\n🎉 所有映射任务处理完成！")
    print("📁 生成的文件:")
    print(f"   - {ccscm_file}")
    print(f"   - {ccsproc_file}")
    print(f"   - {atc3_file}")
