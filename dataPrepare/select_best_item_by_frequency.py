import os
import re
import argparse
import pandas as pd
from typing import List, Tuple, Dict, Any

# 可选：用于本地示例 simpleCHARTEVENT.csv 的演示性回退映射（真实环境请勿依赖）
SIM_ABBR_TO_ITEMIDS: Dict[str, List[int]] = {
    # "CO": [223834],  # 在示例CHARTEVENT中存在
    # 你也可以按需扩展："PAWP": [220224], "CI": [224665], "SVV": [220277]
}

DEFAULT_SYNONYMS: Dict[str, List[str]] = {
    # "CO": ["Cardiac Output"],
}


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

# 定义默认路径
def default_paths(query: str = "") -> Tuple[str, str, str]:
    here = os.path.abspath(os.path.dirname(__file__))
    root = os.path.abspath(os.path.join(here, os.pardir))
    data_dir = os.path.join(root, "data")
    # ditems = os.path.join(data_dir, "simpleD_items.csv")
    ditems = os.path.join(data_dir, "mimic3/D_ITEMS.csv")
    # chartevents = os.path.join(data_dir, "simpleCHARTEVENT.csv")
    chartevents = os.path.join(data_dir, "mimic3/CHARTEVENTS.csv")
    # 根据查询关键字生成不同的输出文件名
    if query:
        out = os.path.join(root, "dataPrepare", f"match_stats_{query}.csv")
    else:
        out = os.path.join(root, "dataPrepare", "match_stats.csv")
    return ditems, chartevents, out


def build_patterns(query: str, extra_terms: List[str]) -> List[re.Pattern]:
    pats: List[re.Pattern] = []
    # 精确令牌匹配（避免匹配到 glucose 这种包含 co 的词）
    token = re.escape(query)
    pats.append(re.compile(rf"(?i)(?<![A-Za-z0-9]){token}(?![A-Za-z0-9])"))
    for t in extra_terms:
        t = t.strip()
        if not t:
            continue
        # 短语匹配，允许空白变体（如Cardiac   Output）
        t_re = re.escape(t).replace(r"\ ", r"\s+")
        pats.append(re.compile(rf"(?i){t_re}"))
    return pats


def find_candidates(ditems: pd.DataFrame, query: str, extra_terms: List[str], search_in: str) -> pd.DataFrame:
    df = ditems.copy()
    df.columns = [c.strip().upper() for c in df.columns]
    # 仅保留 LINKSTO=chartevents
    df = df[df["LINKSTO"].fillna("").str.lower() == "chartevents"]

    pats = build_patterns(query, extra_terms)

    def match_row(row: pd.Series) -> bool:
        fields = []
        search_in_l = search_in.lower()
        if search_in_l in ("label", "both"):
            fields.append(str(row.get("LABEL", "")))
        if search_in_l in ("abbr", "both"):
            fields.append(str(row.get("ABBREVIATION", "")))
        text = " || ".join(fields)
        for p in pats:
            if p.search(text):
                return True
        return False

    cand = df[df.apply(match_row, axis=1)].copy()
    return cand[["ITEMID", "LABEL", "ABBREVIATION", "LINKSTO", "UNITNAME"]]


def count_chartevents_chunked(chartevents_path: str, itemids: List[int], chunk_size: int = 50000) -> pd.DataFrame:
    """分块读取CHARTEVENTS.csv并统计指定ITEMID的出现次数"""
    print(f"[INFO] 开始分块读取 {chartevents_path}，块大小: {chunk_size}")
    
    # 将itemids转换为set以提高查找效率
    itemids_set = set(itemids)
    
    # 初始化统计结果
    total_counts = {itemid: 0 for itemid in itemids}
    valuenum_counts = {itemid: 0 for itemid in itemids}
    unit_counts = {itemid: {} for itemid in itemids}
    
    chunk_num = 0
    processed_rows = 0
    try:
        # 分块读取CSV文件，只读取需要的列以节省内存
        usecols = ['ITEMID', 'VALUENUM', 'VALUEUOM']
        for chunk in pd.read_csv(chartevents_path, encoding="utf-8-sig", chunksize=chunk_size, 
                                low_memory=False, usecols=usecols):
            chunk_num += 1
            processed_rows += len(chunk)
            
            if chunk_num % 50 == 0:  # 减少打印频率
                print(f"[INFO] 已处理 {chunk_num} 个数据块，共 {processed_rows:,} 行")
            
            # 标准化列名
            chunk.columns = [c.strip().upper() for c in chunk.columns]
            
            # 使用向量化操作筛选目标ITEMID
            if 'ITEMID' not in chunk.columns:
                continue
                
            # 使用isin进行向量化筛选，比逐个循环快很多
            mask = chunk['ITEMID'].isin(itemids_set)
            filtered_chunk = chunk[mask]
            
            if filtered_chunk.empty:
                continue
            
            # 使用groupby进行向量化统计，避免循环
            itemid_counts = filtered_chunk.groupby('ITEMID').size()
            for itemid, count in itemid_counts.items():
                total_counts[itemid] += count
            
            # 统计非空VALUENUM的数量
            if 'VALUENUM' in filtered_chunk.columns:
                valuenum_stats = filtered_chunk.groupby('ITEMID')['VALUENUM'].apply(lambda x: x.notna().sum())
                for itemid, count in valuenum_stats.items():
                    valuenum_counts[itemid] += count
            
            # 统计单位（使用更高效的方法）
            if 'VALUEUOM' in filtered_chunk.columns:
                unit_stats = filtered_chunk.dropna(subset=['VALUEUOM']).groupby(['ITEMID', 'VALUEUOM']).size()
                for (itemid, unit), count in unit_stats.items():
                    if unit in unit_counts[itemid]:
                        unit_counts[itemid][unit] += count
                    else:
                        unit_counts[itemid][unit] = count
    
    except Exception as e:
        print(f"[ERROR] 读取文件时出错: {e}")
        raise
    
    print(f"[INFO] 完成处理，共处理 {chunk_num} 个数据块，总计 {processed_rows:,} 行")
    
    # 构建结果DataFrame
    results = []
    for itemid in itemids:
        # 找出最常见的单位
        if unit_counts[itemid]:
            most_common_unit = max(unit_counts[itemid], key=unit_counts[itemid].get)
        else:
            most_common_unit = None
            
        results.append({
            'ITEMID': itemid,
            'count_total': total_counts[itemid],
            'count_valuenum_notnull': valuenum_counts[itemid],
            'valueuom_mode': most_common_unit
        })
    
    return pd.DataFrame(results)


def count_chartevents(ce: pd.DataFrame, itemids: List[int]) -> pd.DataFrame:
    """原有的内存版本，保留用于小文件"""
    ce = ce.copy()
    ce.columns = [c.strip().upper() for c in ce.columns]
    sub = ce[ce["ITEMID"].isin(itemids)].copy()
    # 统计
    grp = sub.groupby("ITEMID")
    stats = grp.size().rename("count_total").to_frame()
    stats["count_valuenum_notnull"] = grp["VALUENUM"].apply(lambda s: s.notna().sum())
    # 取一个代表性的单位（出现最多的单位）
    unit_mode = grp["VALUEUOM"].agg(lambda s: s.dropna().mode().iloc[0] if s.dropna().size > 0 else None)
    stats["valueuom_mode"] = unit_mode
    stats.reset_index(inplace=True)
    return stats


def main():
    # 先解析参数以获取查询关键字
    parser = argparse.ArgumentParser(description="在D_ITEMS中检索与查询词相关的候选项（仅LINKSTO=chartevents），并在CHARTEVENTS中统计各ITEMID出现次数，输出计数并选择出现最多的项。")
    parser.add_argument("--query", required=True, help="查询关键字，如 CO")
    temp_args, _ = parser.parse_known_args()
    
    ditems_p, chartevents_p, out_p = default_paths(temp_args.query)

    # 重新创建完整的参数解析器
    parser = argparse.ArgumentParser(description="在D_ITEMS中检索与查询词相关的候选项（仅LINKSTO=chartevents），并在CHARTEVENTS中统计各ITEMID出现次数，输出计数并选择出现最多的项。")
    parser.add_argument("--ditems", default=ditems_p, help="D_ITEMS.csv 路径")
    parser.add_argument("--chartevents", default=chartevents_p, help="CHARTEVENTS.csv 路径")
    parser.add_argument("--query", required=True, help="查询关键字，如 CO")
    parser.add_argument("-t", "--term", action="append", default=None, help="附加检索短语，可多次提供，如 --term 'Cardiac Output'")
    parser.add_argument("--search_in", choices=["label", "abbr", "both"], default="both", help="检索字段：LABEL/ABBREVIATION/两者")
    parser.add_argument("--out", default=out_p, help="输出CSV路径")
    parser.add_argument("--use_sim_mapping", action="store_true", help="若在D_ITEMS中未找到候选，则尝试演示性回退（使用内置映射）")
    parser.add_argument("--chunk_size", type=int, default=50000, help="分块读取的块大小，默认50000行。内存较小时可以减少此值")
    args = parser.parse_args()

    # 读取D_ITEMS（通常较小，可以直接读取）
    ditems = read_csv(args.ditems)
    # CHARTEVENTS文件很大，稍后使用分块读取

    # 额外短语（若未提供，则加载默认同义词）
    extra_terms = args.term or DEFAULT_SYNONYMS.get(args.query, [])

    # 候选集合（仅LINKSTO=chartevents）
    cand = find_candidates(ditems, args.query, extra_terms, args.search_in)
    
    # 打印找到的匹配项
    if not cand.empty:
        print(f"[INFO] 在D_ITEMS中找到 {len(cand)} 个匹配项:")
        for _, row in cand.iterrows():
            print(f"  ITEMID={row['ITEMID']}, LABEL='{row['LABEL']}', ABBREVIATION='{row.get('ABBREVIATION', 'N/A')}'")
    else:
        print("[INFO] 在D_ITEMS中未找到任何匹配项")

    if cand.empty and args.use_sim_mapping:
        # 演示性回退：直接在CHARTEVENTS中统计映射到的ITEMID
        itemids = SIM_ABBR_TO_ITEMIDS.get(args.query, [])
        if not itemids:
            print(f"[WARN] 未找到候选，也没有可用的演示映射: query={args.query}")
            return
        # 使用分块读取处理大文件
        stats = count_chartevents_chunked(args.chartevents, itemids, args.chunk_size)
        # 构造输出（无D_ITEMS label信息）
        out = stats
        out["LABEL"] = args.query
        out["ABBREVIATION"] = args.query
        out = out[["ITEMID", "LABEL", "ABBREVIATION", "count_total", "count_valuenum_notnull", "valueuom_mode"]]
        out.sort_values(["count_total", "count_valuenum_notnull"], ascending=[False, False], inplace=True)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        out.to_csv(args.out, index=False, encoding="utf-8-sig")
        if out.empty:
            print("[INFO] 无匹配条目。")
            return
        best = out.iloc[0]
        print(f"[DONE:FALLBACK] 候选{len(out)}个，最佳：ITEMID={best['ITEMID']} LABEL={best['LABEL']} count={best['count_total']}")
        print(f"已写出: {args.out}")
        return

    if cand.empty:
        print("[INFO] 未在D_ITEMS中（且LINKSTO=chartevents）找到任何候选。")
        return

    # 在CHARTEVENTS中计数（使用分块读取）
    itemids = [int(x) for x in cand["ITEMID"].tolist()]
    stats = count_chartevents_chunked(args.chartevents, itemids, args.chunk_size)

    # 合并label/abbr信息
    merged = cand.merge(stats, on="ITEMID", how="left").fillna({"count_total": 0, "count_valuenum_notnull": 0})
    # 排序择优
    merged.sort_values(["count_total", "count_valuenum_notnull"], ascending=[False, False], inplace=True)

    # 写出结果
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    merged.to_csv(args.out, index=False, encoding="utf-8-sig")

    if merged.empty:
        print("[INFO] 无匹配条目。")
        return

    best = merged.iloc[0]
    print(f"[DONE] 候选{len(merged)}个，最佳：ITEMID={best['ITEMID']} LABEL={best['LABEL']} count={best['count_total']}")
    print(f"已写出: {args.out}")


if __name__ == "__main__":
    main()