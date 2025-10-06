import json
import pandas as pd
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from data_prepare import load_dataset, load_embeddings
from pyhealth.tokenizer import Tokenizer


def load_atc_mapping(atc_csv_path):
    """
    加载ATC代码到药物名称的映射
    
    Args:
        atc_csv_path: ATC.csv文件路径
        
    Returns:
        dict: ATC代码到药物名称的映射字典
    """
    try:
        atc_df = pd.read_csv(atc_csv_path)
        # 创建ATC代码到名称的映射，只保留有效的代码和名称
        atc_mapping = {}
        for _, row in atc_df.iterrows():
            code = row['code']
            name = row['name']
            if pd.notna(code) and pd.notna(name):
                atc_mapping[code] = name
        return atc_mapping
    except Exception as e:
        print(f"Error loading ATC mapping: {e}")
        return {}


def convert_indices_to_codes(indices_list, tokenizer):
    """
    将索引列表转换为ATC3代码列表
    
    Args:
        indices_list: 索引列表
        tokenizer: PyHealth Tokenizer对象
        
    Returns:
        list: ATC3代码列表
    """
    try:
        # 使用tokenizer的vocabulary属性进行反向映射
        codes = []
        for idx in indices_list:
            if hasattr(tokenizer, 'vocabulary') and hasattr(tokenizer.vocabulary, 'idx2token'):
                # 通过索引获取对应的token（ATC3代码）
                if idx < len(tokenizer.vocabulary.idx2token):
                    code = tokenizer.vocabulary.idx2token[idx]
                    codes.append(code)
                else:
                    codes.append(f"<UNK_IDX_{idx}>")
            else:
                # 备用方法：如果没有直接的idx2token属性，尝试其他方式
                codes.append(f"<IDX_{idx}>")
        return codes
    except Exception as e:
        print(f"Error converting indices to codes: {e}")
        return [f"<ERROR_IDX_{idx}>" for idx in indices_list]


def convert_codes_to_names(codes_list, atc_mapping):
    """
    将ATC3代码列表转换为药物名称列表
    
    Args:
        codes_list: ATC3代码列表
        atc_mapping: ATC代码到名称的映射字典
        
    Returns:
        list: 药物名称列表
    """
    names = []
    for code in codes_list:
        if code in atc_mapping:
            names.append(atc_mapping[code])
        else:
            names.append(f"<UNKNOWN_CODE_{code}>")
    return names


def process_inference_result(inference_result_path, output_path, atc_csv_path, 
                           load_processed_dataset=True, dataset="mimic3", task="drugrec"):
    """
    处理推理结果，将索引转换为ATC3代码和药物名称
    
    Args:
        inference_result_path: inference_result.json文件路径
        output_path: 输出文件路径
        atc_csv_path: ATC.csv文件路径
        load_processed_dataset: 是否加载预处理数据集
        dataset: 数据集名称
        task: 任务名称
    """
    try:
        # 1. 加载推理结果
        print("Loading inference result...")
        with open(inference_result_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
        
        # 2. 加载数据集和tokenizer
        print("Loading dataset...")
        sample_dataset = load_dataset(load_processed_dataset, dataset, task)
        
        print("Creating tokenizer...")
        label_tokenizer = Tokenizer(
            sample_dataset.get_all_tokens(key='drugs')
        )
        
        # 3. 加载ATC映射
        print("Loading ATC mapping...")
        atc_mapping = load_atc_mapping(atc_csv_path)
        print(f"Loaded {len(atc_mapping)} ATC mappings")
        
        # 4. 处理topk_indices（如果存在）
        if 'topk_indices' in result:
            print("Converting topk_indices to codes and names...")
            topk_indices = result['topk_indices']
            topk_scores = result.get('topk_scores', [])
            
            # 转换索引到ATC3代码
            topk_codes = convert_indices_to_codes(topk_indices, label_tokenizer)
            
            # 转换ATC3代码到药物名称
            topk_names = convert_codes_to_names(topk_codes, atc_mapping)
            
            # 添加到结果中
            result['topk_codes'] = topk_codes
            result['topk_names'] = topk_names
            
            # 创建详细的推荐列表
            recommendations = []
            for i, (idx, code, name, score) in enumerate(zip(topk_indices, topk_codes, topk_names, topk_scores)):
                recommendations.append({
                    'rank': i + 1,
                    'index': idx,
                    'atc3_code': code,
                    'drug_name': name,
                    'score': score
                })
            result['recommendations'] = recommendations
            
            print(f"Converted {len(topk_indices)} drug recommendations")
            
            # 打印前5个推荐结果
            print("\nTop 5 drug recommendations:")
            for i, rec in enumerate(recommendations[:5]):
                print(f"  {rec['rank']}. {rec['atc3_code']} - {rec['drug_name']} (score: {rec['score']:.4f})")
        
        # 5. 保存结果（按现有文件自动递增 index）
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        index = 1
        while True:
            candidate = os.path.join(output_dir, f"{task}_final_inference_{index}.json")
            if not os.path.exists(candidate):
                final_output_path = candidate
                break
            index += 1

        print(f"Saving results to {final_output_path}...")
        with open(final_output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print("Conversion completed successfully!")
        return result
        
    except Exception as e:
        print(f"Error processing inference result: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 设置文件路径
    current_dir = os.path.dirname(__file__)
    project_root = os.path.join(current_dir, '../../..')
    
    inference_result_path = os.path.join(project_root, 'inference/inference_result.json')
    output_path = os.path.join(project_root, 'ehr_baselines', 'SparseTest', 'result', 'inference_result_with_names.json')
    atc_csv_path = os.path.join(project_root, 'resources/ATC.csv')
    
    # 检查文件是否存在
    if not os.path.exists(inference_result_path):
        print(f"Error: inference_result.json not found at {inference_result_path}")
        print("Please run the inference first to generate the result file.")
        sys.exit(1)
    
    if not os.path.exists(atc_csv_path):
        print(f"Error: ATC.csv not found at {atc_csv_path}")
        sys.exit(1)
    
    # 处理推理结果
    result = process_inference_result(
        inference_result_path=inference_result_path,
        output_path=output_path,
        atc_csv_path=atc_csv_path,
        load_processed_dataset=True,
        dataset="mimic3",
        task="drugrec"
    )
    
    if result:
        print(f"\nResults saved to: {output_path}")
        if 'recommendations' in result:
            print(f"Successfully converted {len(result['recommendations'])} drug recommendations")
    else:
        print("Failed to process inference result")
