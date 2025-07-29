import numpy as np
import sys
import os

# 添加路径以便导入模块
sys.path.append('/root/workspace/GraphCare/graphcare_/graph_generation')

from umls_sim_retriever import cosine_similarity

def test_cosine_similarity_fix():
    """
    测试修复后的cosine_similarity函数是否能正确处理零向量
    """
    print("测试cosine_similarity函数的零向量处理...")
    
    # 测试正常向量
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    sim = cosine_similarity(v1, v2)
    print(f"正常向量相似度: {sim}")
    
    # 测试零向量
    zero_vector = np.array([0, 0, 0])
    normal_vector = np.array([1, 2, 3])
    
    # 零向量与正常向量
    sim1 = cosine_similarity(zero_vector, normal_vector)
    print(f"零向量与正常向量相似度: {sim1}")
    
    # 正常向量与零向量
    sim2 = cosine_similarity(normal_vector, zero_vector)
    print(f"正常向量与零向量相似度: {sim2}")
    
    # 两个零向量
    sim3 = cosine_similarity(zero_vector, zero_vector)
    print(f"两个零向量相似度: {sim3}")
    
    # 验证是否没有除零错误
    if sim1 == 0.0 and sim2 == 0.0 and sim3 == 0.0:
        print("✅ 零向量处理测试通过！")
        return True
    else:
        print("❌ 零向量处理测试失败！")
        return False

if __name__ == "__main__":
    test_cosine_similarity_fix()