# -*- coding: utf-8 -*-
"""
用途
- 在 MIMIC-III 的 `D_ICD_DIAGNOSES` 表中搜索可能对应“心源性休克/心肌梗死”的 ICD-9 诊断条目，
  输出它们的代码与名称，并统计它们在 `DIAGNOSES_ICD` 中的出现次数（行数）。

输入文件
- 诊断字典: `data/mimic3/D_ICD_DIAGNOSES.csv`（列: ICD9_CODE, SHORT_TITLE, LONG_TITLE）
- 诊断记录: `data/mimic3/DIAGNOSES_ICD.csv`（列: SUBJECT_ID, HADM_ID, SEQ_NUM, ICD9_CODE）

输出文件
- 候选ICD条目统计: 默认 `dataPrepare/cardiac_icd_candidates.csv`
  - 列: icd9_code, short_title, long_title, count_rows, count_subjects, count_hadm

用法示例
- 使用默认关键词（中英文）:
    python dataPrepare/search_cardiac_icd_codes.py \
        --d_icd data/mimic3/D_ICD_DIAGNOSES.csv \
        --diagnoses data/mimic3/DIAGNOSES_ICD.csv \
        --out dataPrepare/cardiac_icd_candidates.csv

- 自定义关键词（逗号分隔，不区分大小写）:
    python dataPrepare/search_cardiac_icd_codes.py \
        --keywords "心源性休克,心肌梗死,cardiogenic shock,myocardial infarction,stemi,nstemi,ami" \
        --out dataPrepare/cardiac_icd_candidates.csv

参数说明
- --d_icd: D_ICD_DIAGNOSES CSV 路径（默认 data/mimic3/D_ICD_DIAGNOSES.csv）
- --diagnoses: DIAGNOSES_ICD CSV 路径（默认 data/mimic3/DIAGNOSES_ICD.csv）
- --keywords: 搜索关键词，逗号分隔；若不提供则使用内置中英文关键词
- --out: 输出 CSV 路径（默认 dataPrepare/cardiac_icd_candidates.csv）
- --chunksize: 读取 DIAGNOSES_ICD 的分块大小（默认 500000）

注意
- 关键词在 `SHORT_TITLE` 和 `LONG_TITLE` 中做不区分大小写的子串匹配。
- 统计包括总行数、去重的病人数（SUBJECT_ID）、去重的入院数（HADM_ID）。
"""

import argparse
import os
from typing import List, Set, Dict

import pandas as pd

DEFAULT_KEYWORDS = [
    # 中文
    # 
    # 英文
    "cardiogenic shock", "myocardial infarction", "acute myocardial infarction",
    "stemi", "nstemi", "ami", "heart attack",
    "cardiogenic", "myocardial"
]


def normalize_code(code: str) -> str:
    # 统一代码比较：去除空白，保留原格式；同时提供去点比较支持
    return (code or "").strip()


def normalize_code_nodot(code: str) -> str:
    return normalize_code(code).replace(".", "")


def pick_codes_by_keywords(d_icd_path: str, keywords: List[str]) -> pd.DataFrame:
    df = pd.read_csv(d_icd_path)
    df.columns = [c.strip().upper() for c in df.columns]
    # 规范列名
    for need in ("ICD9_CODE", "SHORT_TITLE", "LONG_TITLE"):
        if need not in df.columns:
            raise ValueError(f"D_ICD_DIAGNOSES 缺少列: {need}")
    # 关键词匹配（不区分大小写）
    kws = [k.strip().lower() for k in keywords if k.strip()]
    mask = pd.Series(False, index=df.index)
    for kw in kws:
        mask |= df["SHORT_TITLE"].astype(str).str.lower().str.contains(kw, na=False)
        mask |= df["LONG_TITLE"].astype(str).str.lower().str.contains(kw, na=False)
    return df.loc[mask, ["ICD9_CODE", "SHORT_TITLE", "LONG_TITLE"]].copy()


def count_occurrences(diagnoses_path: str, codes_df: pd.DataFrame, chunksize: int = 500_000) -> pd.DataFrame:
    # 映射集合：原码和去点码均纳入
    code_set_raw: Set[str] = set(normalize_code(c) for c in codes_df["ICD9_CODE"].astype(str))
    code_set_nodot: Set[str] = set(normalize_code_nodot(c) for c in code_set_raw)

    counts_rows: Dict[str, int] = {c: 0 for c in code_set_raw}
    subj_sets: Dict[str, Set[int]] = {c: set() for c in code_set_raw}
    hadm_sets: Dict[str, Set[int]] = {c: set() for c in code_set_raw}

    usecols = ["SUBJECT_ID", "HADM_ID", "ICD9_CODE"]
    for chunk in pd.read_csv(diagnoses_path, usecols=usecols, chunksize=chunksize):
        chunk.columns = [c.strip().upper() for c in chunk.columns]
        # 规范化代码（两种形式）
        chunk["ICD9_CODE_RAW"] = chunk["ICD9_CODE"].astype(str).str.strip()
        chunk["ICD9_CODE_NODOT"] = chunk["ICD9_CODE_RAW"].str.replace(".", "", regex=False)

        sub = chunk[chunk["ICD9_CODE_RAW"].isin(code_set_raw) | chunk["ICD9_CODE_NODOT"].isin(code_set_nodot)]
        if sub.empty:
            continue
        for _, r in sub.iterrows():
            code_raw = r["ICD9_CODE_RAW"]
            # 把无点形式映射回原集合的某个键（优先原码匹配）
            key = code_raw if code_raw in code_set_raw else r["ICD9_CODE_NODOT"]
            # 为简化统计，key取原码存在的项，否则取原码集合中第一个与无点同值的项
            if key not in counts_rows:
                # 回退：尝试在原集合中找到无点对应的项
                nm = normalize_code_nodot(code_raw)
                for c in code_set_raw:
                    if normalize_code_nodot(c) == nm:
                        key = c
                        break
            counts_rows[key] += 1
            sid = r.get("SUBJECT_ID")
            hadm = r.get("HADM_ID")
            try:
                subj_sets[key].add(int(sid))
            except Exception:
                pass
            try:
                hadm_sets[key].add(int(hadm))
            except Exception:
                pass

    out = codes_df.copy()
    out["count_rows"] = out["ICD9_CODE"].map(counts_rows).fillna(0).astype(int)
    out["count_subjects"] = out["ICD9_CODE"].map(lambda c: len(subj_sets.get(c, set()))).astype(int)
    out["count_hadm"] = out["ICD9_CODE"].map(lambda c: len(hadm_sets.get(c, set()))).astype(int)
    return out


def main():
    parser = argparse.ArgumentParser(description="Search D_ICD_DIAGNOSES for cardiogenic shock/myocardial infarction candidates and count occurrences in DIAGNOSES_ICD.")
    here = os.path.abspath(os.path.dirname(__file__))
    root = os.path.abspath(os.path.join(here, os.pardir))
    parser.add_argument("--d_icd", default=os.path.join(root, "data", "mimic3", "D_ICD_DIAGNOSES.csv"), help="Path to D_ICD_DIAGNOSES.csv")
    parser.add_argument("--diagnoses", default=os.path.join(root, "data", "mimic3", "DIAGNOSES_ICD.csv"), help="Path to DIAGNOSES_ICD.csv")
    parser.add_argument("--keywords", default=",".join(DEFAULT_KEYWORDS), help="Comma-separated keywords for matching (case-insensitive)")
    parser.add_argument("--out", default=os.path.join(root, "dataPrepare", "cardiac_icd_candidates.csv"), help="Output CSV path")
    parser.add_argument("--chunksize", type=int, default=500_000, help="Chunk size when streaming DIAGNOSES_ICD")

    args = parser.parse_args()

    if not os.path.isfile(args.d_icd):
        raise FileNotFoundError(f"D_ICD_DIAGNOSES not found: {args.d_icd}")
    if not os.path.isfile(args.diagnoses):
        raise FileNotFoundError(f"DIAGNOSES_ICD not found: {args.diagnoses}")

    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    codes_df = pick_codes_by_keywords(args.d_icd, keywords)
    if codes_df.empty:
        print("[WARN] No codes matched keywords. Outputting empty CSV.")
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        codes_df.assign(count_rows=0, count_subjects=0, count_hadm=0).to_csv(args.out, index=False, encoding="utf-8-sig")
        return

    out_df = count_occurrences(args.diagnoses, codes_df, chunksize=args.chunksize)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[DONE] Wrote {len(out_df)} candidates -> {args.out}")


if __name__ == "__main__":
    main()