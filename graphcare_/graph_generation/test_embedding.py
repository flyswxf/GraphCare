#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试embedding_retriever函数的简单脚本
"""

import sys
import os

# 添加当前目录到Python路径
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from get_emb import embedding_retriever

def test_embedding():
    """
    测试embedding_retriever函数
    """
    print("开始测试embedding_retriever函数...")
    
    # 测试用例
    test_cases = [
        "Hello world",
        "医学诊断",
        "machine learning",
        "人工智能"
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}: '{text}'")
        try:
            # 调用embedding_retriever函数
            embedding = embedding_retriever(text)
            
            # 检查返回结果
            if embedding is not None:
                print(f"✓ 成功获取embedding")
                print(f"  - 向量维度: {len(embedding)}")
                print(f"  - 前5个值: {embedding[:5]}")
                print(f"  - 向量类型: {type(embedding)}")
            else:
                print("✗ 返回结果为None")
                
        except Exception as e:
            print(f"✗ 发生错误: {str(e)}")
            print(f"  错误类型: {type(e).__name__}")
    
    print("\n测试完成！")

if __name__ == "__main__":
    test_embedding()