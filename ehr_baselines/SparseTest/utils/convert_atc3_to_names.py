import os
import sys
import json
import argparse
import pandas as pd


def load_atc_mapping(atc_csv_path):
    """
    加载 ATC 代码到药物名称的映射。

    参数:
        atc_csv_path: ATC.csv 或 ATC_Chinese.csv 文件路径，需包含列 `code` 与 `name`

    返回:
        dict: { ATC代码(str) -> 药物名称(str) }
    """
    try:
        df = pd.read_csv(atc_csv_path)
        mapping = {}
        for _, row in df.iterrows():
            code = row.get('code')
            name = row.get('name')
            if pd.notna(code) and pd.notna(name):
                # 统一去除空白并大写代码
                code_str = str(code).strip().upper()
                name_str = str(name).strip()
                if code_str:
                    mapping[code_str] = name_str
        return mapping
    except Exception as e:
        print(f"Error loading ATC mapping from {atc_csv_path}: {e}")
        return {}


def convert_codes_to_names(codes_list, atc_mapping):
    """
    将 ATC3 代码列表转换为药物名称列表。

    参数:
        codes_list: list[str] ATC3 代码列表
        atc_mapping: dict 映射字典 { code -> name }

    返回:
        list[str]: 对应的药物名称列表（未知代码以 <UNKNOWN_CODE_...> 标记）
    """
    names = []
    for code in codes_list:
        if code is None:
            names.append("<UNKNOWN_CODE_None>")
            continue
        normalized = str(code).strip().upper()
        if normalized in atc_mapping:
            names.append(atc_mapping[normalized])
        else:
            names.append(f"<UNKNOWN_CODE_{normalized}>")
    return names


def read_codes_from_file(path):
    """
    从文件读取代码列表，支持：
    - JSON 文件: 形如 {"codes": ["A01", "B02", ...]}
    - 文本文件: 每行一个代码
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    codes = []
    if ext == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 支持 {"codes": [...]} 或直接为列表
        if isinstance(data, dict) and 'codes' in data:
            codes = data['codes']
        elif isinstance(data, list):
            codes = data
        else:
            raise ValueError("JSON input must be a list of codes or contain a 'codes' key")
    else:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                code = line.strip()
                if code:
                    codes.append(code)
    return codes


def main():
    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir, '../../..'))

    parser = argparse.ArgumentParser(description="将ATC3代码映射为药物名称")
    parser.add_argument(
        "--codes", "-c",
        help="逗号分隔的ATC3代码列表，如 A01,B02,C03",
        default=None
    )
    parser.add_argument(
        "--input", "-i",
        help="输入文件路径（JSON: {\"codes\": [...] } 或 文本：每行一个代码）",
        default=None
    )
    parser.add_argument(
        "--output", "-o",
        help="输出文件路径（JSON），默认写入到 ehr_baselines/SparseTest/result/atc3_names.json",
        default=os.path.join(project_root, 'ehr_baselines', 'SparseTest', 'result', 'atc3_names.json')
    )
    parser.add_argument(
        "--atc-csv", "-a",
        help="ATC映射文件路径（默认使用 resources/ATC_Chinese.csv）",
        default=os.path.join(project_root, 'resources', 'ATC_Chinese.csv')
    )
    parser.add_argument(
        "--dedupe",
        help="是否去重输入代码",
        action='store_true'
    )

    args = parser.parse_args()

    # 读取输入代码
    codes = []
    if args.codes:
        codes.extend([c.strip() for c in args.codes.split(',') if c.strip()])
    if args.input:
        codes_from_file = read_codes_from_file(os.path.abspath(args.input))
        codes.extend([str(c) for c in codes_from_file])

    if not codes:
        print("Error: 未提供任何ATC3代码。请使用 --codes 或 --input 指定。")
        sys.exit(1)

    if args.dedupe:
        # 保留原顺序的去重
        seen = set()
        deduped = []
        for c in codes:
            k = str(c).strip().upper()
            if k not in seen:
                seen.add(k)
                deduped.append(c)
        codes = deduped

    # 加载映射
    atc_csv_path = os.path.abspath(args.atc_csv)
    if not os.path.exists(atc_csv_path):
        print(f"Error: ATC映射文件未找到: {atc_csv_path}")
        sys.exit(1)

    print(f"Loading ATC mapping from {atc_csv_path}...")
    atc_mapping = load_atc_mapping(atc_csv_path)
    if not atc_mapping:
        print("Error: 映射加载失败或为空。")
        sys.exit(1)

    # 转换
    print("Converting ATC3 codes to names...")
    names = convert_codes_to_names(codes, atc_mapping)

    # 汇总未知代码
    unknown_codes = []
    for code, name in zip(codes, names):
        if isinstance(name, str) and name.startswith('<UNKNOWN_CODE_'):
            unknown_codes.append(str(code).strip().upper())

    # 输出目录
    output_path = os.path.abspath(args.output) if args.output else None
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    result = {
        'codes': [str(c).strip().upper() for c in codes],
        'names': names,
        'unknown_codes': unknown_codes,
        'mapping_size': len(atc_mapping)
    }

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Saved results to {output_path}")
    else:
        # 无输出路径则打印到控制台
        print(json.dumps(result, ensure_ascii=False, indent=2))

    # 展示前5条
    print("\nPreview (Top 5):")
    for i, (c, n) in enumerate(zip(result['codes'][:5], result['names'][:5]), start=1):
        print(f"  {i}. {c} -> {n}")


if __name__ == '__main__':
    main()