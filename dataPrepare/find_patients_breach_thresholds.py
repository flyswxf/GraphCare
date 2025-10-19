# -*- coding: utf-8 -*-
"""
用途
- 遍历 CHARTEVENTS 表，依据 `dataPrepare/match_stats/chartevents_thresholds.csv` 中每个监测项的阈值（LOW/HIGH），
  一旦某病人的数值低于 LOW 或高于 HIGH，即记录该病人的 `SUBJECT_ID`（不重复）。

输入文件
- 阈值表: `dataPrepare/match_stats/chartevents_thresholds.csv`
  - 需要列: `ITEMID, LABEL, valueuom_mode, LOW, HIGH`
- CHARTEVENTS 数据表: CSV，需包含列: `SUBJECT_ID, ITEMID, VALUENUM, VALUE, VALUEUOM`
  - 示例文件: `data/simpleCHARTEVENT.csv`（参考 `process_chartevents_to_ranges.py`）

输出文件
- 病人ID列表: 文本文件（每行一个 SUBJECT_ID），默认写到 `dataPrepare/match_stats/patients_crossing_thresholds.txt`

用法示例
- 基本用法（按默认路径）:
    python dataPrepare/find_patients_breach_thresholds.py \
        --chartevents data/simpleCHARTEVENT.csv \
        --thresholds dataPrepare/match_stats/chartevents_thresholds.csv \
        --out dataPrepare/match_stats/patients_crossing_thresholds.txt

- 严格按单位过滤（只计算与阈值表 `valueuom_mode` 一致的记录）:
    python dataPrepare/find_patients_breach_thresholds.py \
        --chartevents data/simpleCHARTEVENT.csv \
        --unit_strict

参数说明
- --chartevents: CHARTEVENTS CSV 路径
- --thresholds: 阈值 CSV 路径（默认: dataPrepare/match_stats/chartevents_thresholds.csv）
- --out: 输出 txt 路径（默认: dataPrepare/match_stats/patients_crossing_thresholds.txt）
- --unit_strict: 是否严格按单位过滤（默认否，允许 VALUEUOM 为空或缺失）
- --chunksize: 逐块读取的行数（默认 500000，适合大文件）

注意
- 若某项 `LOW` 为空，则只判断高于 `HIGH`；若 `HIGH` 为空，则只判断低于 `LOW`。
- 若 `VALUENUM` 为空将尝试从 `VALUE` 中提取数字（例如 "85 mmHg" -> 85）。
- 仅根据数值比较，不做单位换算；如需换算请先清洗 CHARTEVENTS。
"""

import argparse
import os
import re
import sys
from typing import Dict, Optional, Tuple

import pandas as pd

_num_pat = re.compile(r"[-+]?\d+(?:\.\d+)?")


def to_float(s: object) -> Optional[float]:
    if s is None:
        return None
    try:
        if isinstance(s, str):
            s = s.strip()
            if s == "" or s.lower() in {"none", "nan", "null"}:
                return None
            # 尝试直接转
            try:
                return float(s)
            except ValueError:
                # 从字符串中抽取第一个数字
                m = _num_pat.search(s)
                if m:
                    return float(m.group(0))
                return None
        # 其他类型
        return float(s)
    except Exception:
        return None


def read_thresholds(path: str) -> Dict[int, Tuple[Optional[float], Optional[float], Optional[str]]]:
    df = pd.read_csv(path)
    # 统一列名大小写，去空格
    df.columns = [c.strip() for c in df.columns]
    req_cols = {"ITEMID", "LOW", "HIGH", "valueuom_mode"}
    if not req_cols.issubset(set(df.columns)):
        raise ValueError(f"阈值表缺少必要列: 期望 {req_cols}, 实际 {set(df.columns)}")

    mapping: Dict[int, Tuple[Optional[float], Optional[float], Optional[str]]] = {}
    for _, r in df.iterrows():
        try:
            itemid = int(r["ITEMID"])  # ITEMID 必须是整数
        except Exception:
            # 跳过非法 ITEMID
            continue
        low = to_float(r.get("LOW"))
        high = to_float(r.get("HIGH"))
        unit = str(r.get("valueuom_mode", "")).strip() or None
        mapping[itemid] = (low, high, unit)
    return mapping


def subject_ids_exceeding(chartevents_csv: str,
                           thresholds: Dict[int, Tuple[Optional[float], Optional[float], Optional[str]]],
                           unit_strict: bool = False,
                           chunksize: int = 500_000) -> set:
    ids = set()
    usecols = ["SUBJECT_ID", "ITEMID", "VALUENUM", "VALUE", "VALUEUOM"]
    itemid_set = set(thresholds.keys())

    # 逐块读取，适配大文件
    for chunk in pd.read_csv(chartevents_csv, usecols=usecols, chunksize=chunksize):
        # 标准化列名
        chunk.columns = [c.strip().upper() for c in chunk.columns]

        # 只保留阈值表中的 ITEMID
        sub = chunk[chunk["ITEMID"].isin(itemid_set)].copy()
        if sub.empty:
            continue

        # 单位过滤
        if unit_strict:
            # 仅保留 VALUEUOM 与阈值单位一致的，允许空值跳过（不计）
            rows = []
            for _, r in sub.iterrows():
                itemid = r.get("ITEMID")
                low, high, unit = thresholds.get(int(itemid), (None, None, None))
                vu = r.get("VALUEUOM")
                vu_norm = str(vu).strip() if pd.notna(vu) else ""
                if unit and vu_norm and vu_norm.lower() != unit.lower():
                    # 单位不匹配，跳过
                    continue
                rows.append(r)
            sub = pd.DataFrame(rows)
            if sub.empty:
                continue
        else:
            # 非严格：允许 VALUEUOM 为空或不匹配，不在此处过滤
            pass

        # 数值与阈值比较
        for _, r in sub.iterrows():
            vnum = r.get("VALUENUM")
            if pd.isna(vnum):
                vnum = to_float(r.get("VALUE"))
            else:
                vnum = to_float(vnum)
            if vnum is None:
                continue

            itemid = int(r.get("ITEMID"))
            low, high, _unit = thresholds.get(itemid, (None, None, None))

            breached = False
            if low is not None and vnum < low:
                breached = True
            if high is not None and vnum > high:
                breached = True

            if breached:
                sid = r.get("SUBJECT_ID")
                # 记录唯一 SUBJECT_ID
                if pd.notna(sid):
                    try:
                        ids.add(int(sid))
                    except Exception:
                        # 若无法转换为整数，按原字符串记录
                        ids.add(str(sid))
    return ids


def main():
    here = os.path.abspath(os.path.dirname(__file__))
    root = os.path.abspath(os.path.join(here, os.pardir))

    parser = argparse.ArgumentParser(description="Scan CHARTEVENTS against LOW/HIGH thresholds and list unique SUBJECT_IDs that breach.")
    parser.add_argument("--chartevents", required=True, help="Path to CHARTEVENTS CSV (columns: SUBJECT_ID, ITEMID, VALUENUM, VALUE, VALUEUOM)")
    parser.add_argument("--thresholds", default=os.path.join(here, "match_stats", "chartevents_thresholds.csv"), help="Path to thresholds CSV")
    parser.add_argument("--out", default=os.path.join(here, "match_stats", "patients_crossing_thresholds.txt"), help="Path to output txt file")
    parser.add_argument("--unit_strict", action="store_true", help="Only count rows whose VALUEUOM matches thresholds' valueuom_mode")
    parser.add_argument("--chunksize", type=int, default=500_000, help="Row count per chunk when streaming CHARTEVENTS")

    args = parser.parse_args()

    # 校验输入文件存在
    if not os.path.isfile(args.thresholds):
        print(f"[ERROR] 阈值文件不存在: {args.thresholds}")
        sys.exit(1)
    if not os.path.isfile(args.chartevents):
        print(f"[ERROR] CHARTEVENTS 文件不存在: {args.chartevents}")
        sys.exit(1)

    # 读取阈值映射
    thresholds = read_thresholds(args.thresholds)
    if not thresholds:
        print(f"[ERROR] 阈值映射为空，请检查 {args.thresholds}")
        sys.exit(1)

    # 遍历 CHARTEVENTS 收集病人ID
    ids = subject_ids_exceeding(args.chartevents, thresholds, unit_strict=args.unit_strict, chunksize=args.chunksize)
    print(f"[INFO] 共有 {len(ids)} 位病人出现越界指标（LOW/HIGH）")

    # 输出
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fw:
        for sid in sorted(ids):
            fw.write(f"{sid}\n")
    print(f"[DONE] 写出病人ID列表 -> {args.out}")


if __name__ == "__main__":
    main()