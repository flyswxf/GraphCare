#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试convert_indices_to_code.py的功能
"""

import json
import os
import sys

# 添加项目根目录到路径
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir, '../../..'))

from convert_indices_to_code import (
    load_atc_mapping, 
    convert_indices_to_codes, 
    convert_codes_to_names,
    process_inference_result
)
from data_prepare import load_dataset
from pyhealth.tokenizer import Tokenizer


def create_mock_inference_result():
    """创建模拟的推理结果用于测试"""
    mock_result = {
        "patient_id": "test_patient_001",
        "sample_index": 0,
        "mode": "test",
        "logits": [0.1, 0.8, 0.3, 0.9, 0.2, 0.7, 0.4, 0.6, 0.5, 0.15],
        "prob": [0.05, 0.25, 0.1, 0.3, 0.08, 0.22, 0.12, 0.18, 0.15, 0.06],
        "topk_indices": [3, 1, 5, 7, 8],  # 模拟top-5推荐的药物索引
        "topk_scores": [0.9, 0.8, 0.7, 0.6, 0.5]
    }
    return mock_result


def test_atc_mapping():
    """测试ATC映射加载功能"""
    print("=== 测试ATC映射加载 ===")
    
    project_root = os.path.join(current_dir, '../../..')
    atc_csv_path = os.path.join(project_root, 'resources/ATC.csv')
    
    if not os.path.exists(atc_csv_path):
        print(f"Error: ATC.csv not found at {atc_csv_path}")
        return False
    
    atc_mapping = load_atc_mapping(atc_csv_path)
    print(f"Loaded {len(atc_mapping)} ATC mappings")
    
    # 显示前10个映射
    print("\n前10个ATC映射示例:")
    for i, (code, name) in enumerate(list(atc_mapping.items())[:10]):
        print(f"  {code}: {name}")
    
    return len(atc_mapping) > 0


def test_tokenizer():
    """测试tokenizer功能"""
    print("\n=== 测试Tokenizer功能 ===")
    
    try:
        # 加载数据集
        sample_dataset = load_dataset(True, "mimic3", "drugrec")
        
        # 创建tokenizer
        label_tokenizer = Tokenizer(
            sample_dataset.get_all_tokens(key='drugs')
        )
        
        print(f"Tokenizer vocabulary size: {len(label_tokenizer.vocabulary.token2idx)}")
        
        # 显示前10个token
        print("\n前10个药物token示例:")
        tokens = list(label_tokenizer.vocabulary.token2idx.keys())[:10]
        for i, token in enumerate(tokens):
            idx = label_tokenizer.vocabulary.token2idx[token]
            print(f"  {token} -> {idx}")
        
        return label_tokenizer
        
    except Exception as e:
        print(f"Error testing tokenizer: {e}")
        return None


def test_conversion():
    """测试索引到代码的转换功能"""
    print("\n=== 测试索引转换功能 ===")
    
    # 获取tokenizer
    tokenizer = test_tokenizer()
    if tokenizer is None:
        return False
    
    # 测试索引转换
    test_indices = [0, 1, 2, 3, 4]
    codes = convert_indices_to_codes(test_indices, tokenizer)
    
    print(f"\n测试索引转换:")
    for idx, code in zip(test_indices, codes):
        print(f"  Index {idx} -> Code {code}")
    
    # 测试ATC映射
    project_root = os.path.join(current_dir, '../../..')
    atc_csv_path = os.path.join(project_root, 'resources/ATC.csv')
    atc_mapping = load_atc_mapping(atc_csv_path)
    
    names = convert_codes_to_names(codes, atc_mapping)
    
    print(f"\n测试代码到名称转换:")
    for code, name in zip(codes, names):
        print(f"  Code {code} -> Name {name}")
    
    return True


def test_full_pipeline():
    """测试完整的处理流程"""
    print("\n=== 测试完整处理流程 ===")
    
    # 创建模拟推理结果
    mock_result = create_mock_inference_result()
    
    # 保存到临时文件
    temp_input_path = os.path.join(current_dir, 'temp_inference_result.json')
    temp_output_path = os.path.join(current_dir, 'temp_inference_result_with_names.json')
    
    with open(temp_input_path, 'w', encoding='utf-8') as f:
        json.dump(mock_result, f, ensure_ascii=False, indent=2)
    
    print(f"Created mock inference result: {temp_input_path}")
    
    # 设置路径
    project_root = os.path.join(current_dir, '../../..')
    atc_csv_path = os.path.join(project_root, 'resources/ATC.csv')
    
    # 处理结果
    try:
        result = process_inference_result(
            inference_result_path=temp_input_path,
            output_path=temp_output_path,
            atc_csv_path=atc_csv_path,
            load_processed_dataset=True,
            dataset="mimic3",
            task="drugrec"
        )
        
        if result and os.path.exists(temp_output_path):
            print(f"\n处理成功！结果保存到: {temp_output_path}")
            
            # 显示结果
            with open(temp_output_path, 'r', encoding='utf-8') as f:
                processed_result = json.load(f)
            
            if 'recommendations' in processed_result:
                print(f"\n药物推荐结果 (共{len(processed_result['recommendations'])}个):")
                for rec in processed_result['recommendations']:
                    print(f"  {rec['rank']}. {rec['atc3_code']} - {rec['drug_name']} (score: {rec['score']:.4f})")
            
            # 清理临时文件
            os.remove(temp_input_path)
            os.remove(temp_output_path)
            
            return True
        else:
            print("处理失败")
            return False
            
    except Exception as e:
        print(f"Error in full pipeline test: {e}")
        import traceback
        traceback.print_exc()
        
        # 清理临时文件
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        
        return False


def main():
    """主测试函数"""
    print("开始测试convert_indices_to_code功能...\n")
    
    # 测试各个组件
    tests = [
        ("ATC映射加载", test_atc_mapping),
        ("索引转换", test_conversion),
        ("完整流程", test_full_pipeline)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"\n{test_name}: {'✓ 通过' if result else '✗ 失败'}")
        except Exception as e:
            print(f"\n{test_name}: ✗ 异常 - {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "="*50)
    print("测试总结:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {test_name}: {status}")
    
    print(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！convert_indices_to_code功能正常工作。")
    else:
        print("⚠️  部分测试失败，请检查相关功能。")


if __name__ == "__main__":
    main()