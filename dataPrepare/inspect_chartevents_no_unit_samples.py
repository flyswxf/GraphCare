import argparse
import os
import pandas as pd
from typing import Dict, List, Set, Tuple


def list_match_stats_files(stats_dir: str) -> List[str]:
    files: List[str] = []
    for name in os.listdir(stats_dir):
        p = os.path.join(stats_dir, name)
        if os.path.isfile(p) and name.lower().endswith(".csv") and name.lower() != "itemids.txt":
            files.append(p)
    return sorted(files)


def read_stats_no_unit_itemids(path: str) -> List[int]:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    # 标准化列名为小写，便于兼容
    df.columns = [c.strip().lower() for c in df.columns]
    if "itemid" not in df.columns:
        return []
    # valueuom_mode 为空或缺失的项
    if "valueuom_mode" in df.columns:
        mask = df["valueuom_mode"].isna() | (df["valueuom_mode"].astype(str).str.strip() == "")
        sub = df[mask]
    else:
        # 如果没有该列，则认为没有可筛选的
        sub = df.iloc[0:0]
    # 返回唯一的 ITEMID（整数）
    itemids = []
    for v in sub["itemid"].dropna().unique().tolist():
        try:
            itemids.append(int(v))
        except Exception:
            # 忽略无法解析为整数的
            pass
    return itemids


def resolve_usecols(chartevents_path: str, preferred: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """根据文件实际列名解析要读取的列。

    返回: (usecols, name_map)
    - usecols: 实际存在且将被读取的列名（原始大小写）
    - name_map: 规范小写名 -> 实际列名 的映射
    """
    header = pd.read_csv(chartevents_path, nrows=0, low_memory=False)
    cols = header.columns.tolist()
    lower_to_actual: Dict[str, str] = {c.strip().lower(): c for c in cols}
    usecols: List[str] = []
    name_map: Dict[str, str] = {}
    for want in preferred:
        lw = want.lower()
        if lw in lower_to_actual:
            actual = lower_to_actual[lw]
            usecols.append(actual)
            name_map[lw] = actual
    # 必须包含 ITEMID
    if "itemid" not in name_map:
        raise RuntimeError("CHARTEVENTS 缺少 ITEMID 列，无法筛选样本")
    return usecols, name_map


def sample_rows_for_itemids(
    chartevents_path: str,
    itemids: List[int],
    per_item: int = 10,
    chunk_size: int = 100_000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """从 CHARTEVENTS 中为给定 ITEMID 采样每个最多 per_item 条记录。

    返回: (samples_df, summary_df)
    - samples_df: 包含所有采样行，具有列 ITEMID、VALUEUOM、VALUENUM、VALUE、CHARTTIME、SUBJECT_ID、HADM_ID、ICUSTAY_ID（若存在）
    - summary_df: 每个 ITEMID 的采样条数与样本中发现的去重单位列表
    """
    if not itemids:
        return pd.DataFrame(), pd.DataFrame()

    # 解析要读取的列
    preferred_cols = [
        "ITEMID",
        "VALUEUOM",
        "VALUENUM",
        "VALUE",
        "CHARTTIME",
        "SUBJECT_ID",
        "HADM_ID",
        "ICUSTAY_ID",
    ]
    usecols, name_map = resolve_usecols(chartevents_path, preferred_cols)

    # 待采样项目
    pending: Set[int] = set(int(x) for x in itemids)
    collected: Dict[int, List[pd.DataFrame]] = {iid: [] for iid in pending}

    # 分块读取并收集
    for chunk in pd.read_csv(
        chartevents_path,
        encoding="utf-8-sig",
        chunksize=chunk_size,
        low_memory=False,
        usecols=usecols,
    ):
        # 标准化列名为小写
        chunk.columns = [c.strip().lower() for c in chunk.columns]
        # 过滤当前需要的 ITEMID
        m = chunk[name_map["itemid"].lower()].isin(pending)
        sub = chunk[m].copy()
        if sub.empty:
            # 若无匹配则下一块
            if not pending:
                break
            continue

        # 按 ITEMID 分组，分别截取所需条数
        for iid, g in sub.groupby(name_map["itemid"].lower()):
            if iid not in pending:
                continue
            need = per_item - sum(len(df) for df in collected[iid])
            if need <= 0:
                continue
            collected[iid].append(g.head(need))
            # 若已满足，移出 pending
            if sum(len(df) for df in collected[iid]) >= per_item:
                pending.discard(iid)
        # 若所有项目已收集够，提前结束
        if not pending:
            break

    # 合并为一个 DataFrame，并补齐缺失列
    frames: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, object]] = []
    all_cols_l = [c.lower() for c in usecols]

    for iid in itemids:
        parts = collected.get(int(iid), [])
        if not parts:
            summary_rows.append({
                "ITEMID": int(iid),
                "samples_collected": 0,
                "distinct_uoms_in_samples": "",
                "nonnull_uom_count": 0,
            })
            continue
        df_i = pd.concat(parts, ignore_index=True)
        # 添加 ITEMID 一列（保证名称统一为大写 ITEMID）
        if "itemid" in df_i.columns:
            df_i.rename(columns={"itemid": "ITEMID"}, inplace=True)
        else:
            df_i["ITEMID"] = int(iid)

        frames.append(df_i)
        uom_col = "valueuom" if "valueuom" in df_i.columns else None
        if uom_col:
            uniq_uoms = sorted(set([str(x) for x in df_i[uom_col].dropna().unique().tolist() if str(x).strip() != ""]))
            nonnull_cnt = int(df_i[uom_col].notna().sum())
        else:
            uniq_uoms = []
            nonnull_cnt = 0
        summary_rows.append({
            "ITEMID": int(iid),
            "samples_collected": int(len(df_i)),
            "distinct_uoms_in_samples": ";".join(uniq_uoms),
            "nonnull_uom_count": nonnull_cnt,
        })

    samples_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    # 选取并按统一列顺序导出
    out_cols = []
    for c in ["ITEMID", "subject_id", "hadm_id", "icustay_id", "charttime", "value", "valuenum", "valueuom"]:
        # 包含大小写任意之一
        if c in samples_df.columns:
            out_cols.append(c)
        elif c.upper() in samples_df.columns:
            out_cols.append(c.upper())
    if samples_df.empty:
        samples_df = pd.DataFrame(columns=[c.upper() for c in out_cols])
    else:
        samples_df = samples_df[out_cols]

    summary_df = pd.DataFrame(summary_rows, columns=["ITEMID", "samples_collected", "distinct_uoms_in_samples", "nonnull_uom_count"])
    return samples_df, summary_df


def process_all(stats_dir: str, chartevents_path: str, out_dir: str, per_item: int = 10, chunk_size: int = 100_000):
    os.makedirs(out_dir, exist_ok=True)
    files = list_match_stats_files(stats_dir)
    if not files:
        print(f"[INFO] 未在 {stats_dir} 下找到任何 CSV 文件。")
        return

    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        itemids = read_stats_no_unit_itemids(f)
        if not itemids:
            print(f"[INFO] {base}: 无 valueuom_mode 为空的项目。")
            continue

        print(f"[INFO] {base}: 需要检查的 ITEMID 数量 = {len(itemids)}；开始采样…")
        samples_df, summary_df = sample_rows_for_itemids(
            chartevents_path,
            itemids,
            per_item=per_item,
            chunk_size=chunk_size,
        )

        # 写文件
        samples_out = os.path.join(out_dir, f"{base}_no_unit_samples.csv")
        summary_out = os.path.join(out_dir, f"{base}_no_unit_summary.csv")
        samples_df.to_csv(samples_out, index=False, encoding="utf-8-sig")
        summary_df.to_csv(summary_out, index=False, encoding="utf-8-sig")

        print(f"[DONE] {base}: 已写出样本 {len(samples_df)} 行 -> {samples_out}")
        print(f"[DONE] {base}: 汇总 {len(summary_df)} 项 -> {summary_out}")


def main():
    parser = argparse.ArgumentParser(description="根据 match_stats，抽取在 CHARTEVENTS 中无单位项目的样本行用于人工核查。")
    parser.add_argument("--stats_dir", type=str, default="./dataPrepare/match_stats", help="match_stats 目录路径")
    parser.add_argument("--chartevents", type=str, default="./data/mimic3/CHARTEVENTS.csv", help="CHARTEVENTS.csv 路径")
    parser.add_argument("--out_dir", type=str, default="./dataPrepare/no_unit_samples", help="输出目录")
    parser.add_argument("--per_item", type=int, default=10, help="每个 ITEMID 采样的最大条数")
    parser.add_argument("--chunk_size", type=int, default=100000, help="分块读取大小")
    args = parser.parse_args()

    process_all(args.stats_dir, args.chartevents, args.out_dir, per_item=args.per_item, chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()