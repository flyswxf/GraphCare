# -*- coding: utf-8 -*-
"""
用途
- 在越界病人名单（patients_crossing_thresholds.txt）范围内，
  读取 MIMIC-III 的 `DIAGNOSES_ICD` 表，检测病人是否存在“心源性休克/心肌梗死”的诊断代码，
  输出 `patient_id,bool` 两列（0=未发现指定诊断，1=发现）。

输入文件
- 越界病人名单: `dataPrepare/match_stats/patients_crossing_thresholds.txt`（每行一个 SUBJECT_ID）
- 诊断记录: `data/mimic3/DIAGNOSES_ICD.csv`（列: SUBJECT_ID, HADM_ID, SEQ_NUM, ICD9_CODE）
-（可选）诊断字典: `data/mimic3/D_ICD_DIAGNOSES.csv`（列: ICD9_CODE, SHORT_TITLE, LONG_TITLE）

输出文件
- 病人标记表: 默认 `dataPrepare/cardiac_condition_flags.csv`（列: patient_id, bool）

用法示例
- 按默认路径与自定义代码（逗号分隔）:
    python dataPrepare/detect_cardiac_conditions.py \
        --diagnoses data/mimic3/DIAGNOSES_ICD.csv \
        --patients_txt dataPrepare/match_stats/patients_crossing_thresholds.txt \
        --codes "785.51,410.90,410.71" \
        --out dataPrepare/cardiac_condition_flags.csv

- 从文件读取代码（每行一个 ICD9 代码）:
    python dataPrepare/detect_cardiac_conditions.py \
        --codes_file dataPrepare/my_selected_icd9.txt \
        --out dataPrepare/cardiac_condition_flags.csv

参数说明
- --diagnoses: DIAGNOSES_ICD CSV 路径（默认 data/mimic3/DIAGNOSES_ICD.csv）
- --patients_txt: 越界病人列表 TXT 路径（默认 dataPrepare/match_stats/patients_crossing_thresholds.txt）
- --codes: 逗号分隔 ICD9 代码集合（如 785.51 对应 cardiogenic shock，410.xx 对应 MI 变种）
- --codes_file: ICD9 代码文件路径（每行一个），与 --codes 合并去重
- --out: 输出 CSV 路径（默认 dataPrepare/cardiac_condition_flags.csv）
- --chunksize: 读取 DIAGNOSES_ICD 的分块大小（默认 500000）

注意
- 代码匹配同时支持带点与不带点形式（例如 "410.71" 与 "41071" 都能匹配）。
- 未在 DIAGNOSES_ICD 中出现的越界病人默认标记为 0。
- 若未提供任何代码，脚本将报错提示。
"""

import argparse
import os
from typing import Set, List

import pandas as pd


def normalize_code(code: str) -> str:
    return (code or "").strip()


def normalize_code_nodot(code: str) -> str:
    return normalize_code(code).replace(".", "")


def load_allowed_patients(patients_txt: str) -> Set[int]:
    ids: Set[int] = set()
    with open(patients_txt, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                ids.add(int(s))
            except Exception:
                # 若非纯数字，尝试忽略；也可以保留原字符串当作特殊ID
                pass
    return ids


def load_codes(codes_arg: str, codes_file: str) -> List[str]:
    codes: List[str] = []
    if codes_arg:
        codes.extend([c.strip() for c in codes_arg.split(",") if c.strip()])
    if codes_file and os.path.isfile(codes_file):
        with open(codes_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    codes.append(s)
    # 去重
    seen = set()
    uniq = []
    for c in codes:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def detect_flags(diagnoses_csv: str, allowed_patients: Set[int], selected_codes: List[str], chunksize: int = 500_000):
    # 初始化所有越界病人为0
    flags = {pid: 0 for pid in allowed_patients}

    # 代码集合（原码与去点）
    code_set_raw = set(normalize_code(c) for c in selected_codes)
    code_set_nodot = set(normalize_code_nodot(c) for c in selected_codes)

    usecols = ["SUBJECT_ID", "ICD9_CODE"]
    for chunk in pd.read_csv(diagnoses_csv, usecols=usecols, chunksize=chunksize):
        chunk.columns = [c.strip().upper() for c in chunk.columns]
        # 仅保留越界病人
        sub = chunk[chunk["SUBJECT_ID"].isin(allowed_patients)].copy()
        if sub.empty:
            continue
        sub["ICD9_CODE_RAW"] = sub["ICD9_CODE"].astype(str).str.strip()
        sub["ICD9_CODE_NODOT"] = sub["ICD9_CODE_RAW"].str.replace(".", "", regex=False)
        sub = sub[(sub["ICD9_CODE_RAW"].isin(code_set_raw)) | (sub["ICD9_CODE_NODOT"].isin(code_set_nodot))]
        if sub.empty:
            continue
        for pid in sub["SUBJECT_ID"].unique():
            flags[int(pid)] = 1
    return flags


def main():
    parser = argparse.ArgumentParser(description="Flag patients (in thresholds list) having selected cardiogenic shock/MI ICD9 codes in DIAGNOSES_ICD.")
    here = os.path.abspath(os.path.dirname(__file__))
    root = os.path.abspath(os.path.join(here, os.pardir))
    parser.add_argument("--diagnoses", default=os.path.join(root, "data", "mimic3", "DIAGNOSES_ICD.csv"), help="Path to DIAGNOSES_ICD.csv")
    parser.add_argument("--patients_txt", default=os.path.join(root, "dataPrepare", "match_stats", "patients_crossing_thresholds.txt"), help="Path to threshold-crossing patients TXT")
    parser.add_argument("--codes", default="", help="Comma-separated ICD9 codes to flag (e.g. 785.51,410.71,410.90)")
    parser.add_argument("--codes_file", default="", help="File path containing ICD9 codes (one per line)")
    parser.add_argument("--out", default=os.path.join(root, "dataPrepare", "cardiac_condition_flags.csv"), help="Output CSV path")
    parser.add_argument("--chunksize", type=int, default=500_000, help="Chunk size when streaming DIAGNOSES_ICD")

    args = parser.parse_args()

    if not os.path.isfile(args.diagnoses):
        raise FileNotFoundError(f"DIAGNOSES_ICD not found: {args.diagnoses}")
    if not os.path.isfile(args.patients_txt):
        raise FileNotFoundError(f"patients TXT not found: {args.patients_txt}")

    codes = load_codes(args.codes, args.codes_file)
    if not codes:
        raise ValueError("未提供任何 ICD9 代码。请使用 --codes 或 --codes_file 指定候选代码（可由 search_cardiac_icd_codes.py 输出后人工挑选）。")

    allowed = load_allowed_patients(args.patients_txt)
    if not allowed:
        print("[WARN] 越界病人列表为空，将输出空结果（无病人记录）。")
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        pd.DataFrame(columns=["patient_id", "bool"]).to_csv(args.out, index=False, encoding="utf-8-sig")
        return

    flags = detect_flags(args.diagnoses, allowed, codes, chunksize=args.chunksize)
    out_df = pd.DataFrame({"patient_id": list(flags.keys()), "bool": list(flags.values())})
    out_df.sort_values("patient_id", inplace=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[DONE] 标记 {len(out_df)} 位病人 -> {args.out}")


if __name__ == "__main__":
    main()