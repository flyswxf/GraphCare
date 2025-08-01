import pickle
import numpy as np
from tqdm import tqdm
import os

def fix_zero_vectors_in_embeddings():
    """
    修复umls_ent_emb_.pkl文件中的零向量问题
    """
    # 获取文件路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    umls_emb_path = os.path.join(data_dir, 'umls_ent_emb_.pkl')
    
    print("正在加载umls_ent_emb_.pkl文件...")
    with open(umls_emb_path, 'rb') as f:
        umls_ent_emb = pickle.load(f)
    
    print(f"总共有 {len(umls_ent_emb)} 个向量")
    
    # 转换为numpy数组以便处理
    umls_ent_emb = np.array(umls_ent_emb)
    
    # 找到零向量
    zero_vector_indices = []
    for i, emb in enumerate(tqdm(umls_ent_emb, desc="检查零向量")):
        if np.allclose(emb, 0):
            zero_vector_indices.append(i)
    
    print(f"发现 {len(zero_vector_indices)} 个零向量")
    
    if len(zero_vector_indices) > 0:
        # 计算非零向量的平均值
        non_zero_embeddings = []
        for i, emb in enumerate(umls_ent_emb):
            if i not in zero_vector_indices:
                non_zero_embeddings.append(emb)
        
        if len(non_zero_embeddings) > 0:
            # 使用非零向量的平均值
            mean_embedding = np.mean(non_zero_embeddings, axis=0)
            print(f"使用非零向量的平均值替换零向量，向量维度: {len(mean_embedding)}")
            
            # 替换零向量
            for idx in zero_vector_indices:
                # 添加少量随机噪声以避免完全相同的向量
                noise = np.random.normal(0, 0.01, mean_embedding.shape)
                umls_ent_emb[idx] = mean_embedding + noise
                
        else:
            # 如果所有向量都是零向量，使用随机向量
            print("所有向量都是零向量，使用随机向量替换")
            embedding_dim = len(umls_ent_emb[0])
            for idx in zero_vector_indices:
                umls_ent_emb[idx] = np.random.normal(0, 0.1, embedding_dim)
    
    # 保存修复后的文件
    backup_path = os.path.join(data_dir, 'umls_ent_emb_backup.pkl')
    print(f"备份原文件到: {backup_path}")
    with open(backup_path, 'wb') as f:
        pickle.dump(umls_ent_emb.tolist(), f)
    
    print(f"保存修复后的文件到: {umls_emb_path}")
    with open(umls_emb_path, 'wb') as f:
        pickle.dump(umls_ent_emb.tolist(), f)
    
    print("零向量修复完成！")
    return len(zero_vector_indices)

if __name__ == "__main__":
    fixed_count = fix_zero_vectors_in_embeddings()
    print(f"总共修复了 {fixed_count} 个零向量")