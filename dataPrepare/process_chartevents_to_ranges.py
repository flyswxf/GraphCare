import os
import re
import argparse
import pandas as pd
from typing import Optional, Tuple, Callable, Dict, Any, List


# Basic zh->en dictionary for common hemodynamic terms used in example
ZH_EN_MAP = {
    "肺动脉楔压": "Pulmonary Artery Wedge Pressure",
    "心输出量": "Cardiac Output",
    "心脏指数": "Cardiac Index",
    "全身血管阻力": "Systemic Vascular Resistance",
    "每搏量变异度": "Stroke Volume Variation",
}

# 用于本仓库提供的simpleCHARTEVENT.csv的演示性映射（仅用于模拟和示例）。
# 注意：这些ITEMID与真实生理意义未必一一对应，请在真实环境中关闭该回退或替换为可信映射。
SIM_ABBR_TO_ITEMID = {
    "CO": 223834,   # L/min 示例
    "PAWP": 220224, # mmHg 示例
    "CI": 224665,   # 数值在 1~2 区间的示例
    "SVV": 220277,  # % 示例（仅演示用途）
}


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def is_empty(x: Any) -> bool:
    if x is None:
        return True
    s = str(x).strip()
    return s == "" or s == "-"


def extract_name_abbr(param: str) -> Tuple[str, Optional[str]]:
    m = re.match(r"^(?P<name>[^()]+?)(?:\s*\((?P<abbr>[^)]+)\))?\s*$", str(param))
    if not m:
        return param.strip(), None
    name = m.group("name").strip()
    abbr = m.group("abbr")
    abbr = abbr.strip() if abbr else None
    return name, abbr


def english_candidates(zh_name: str, abbr: Optional[str]) -> List[str]:
    eng = ZH_EN_MAP.get(zh_name, None)
    cands = []
    if eng and abbr:
        cands.append(f"{eng} ({abbr})")
    if eng:
        cands.append(eng)
    if abbr:
        cands.append(abbr)
    return list(dict.fromkeys([c.lower() for c in cands]))  # unique and lowercased


def score_ditems_row(row: pd.Series, cand_labels_lower: List[str], abbr: Optional[str], unit: Optional[str]) -> int:
    score = 0
    label = str(row.get("LABEL", ""))
    abbr_row = str(row.get("ABBREVIATION", ""))
    unit_row = str(row.get("UNITNAME", ""))

    label_l = label.lower()
    abbr_l = abbr_row.lower()

    for c in cand_labels_lower:
        if label_l == c:
            score += 100
        elif c and c in label_l:
            score += 40

    if abbr:
        ab = abbr.lower()
        if ab and (abbr_l == ab):
            score += 60
        if ab and (ab in label_l):
            score += 30

    if unit and unit_row and unit_row.strip().lower() == str(unit).strip().lower():
        score += 10

    # Prefer chartevents-linked items
    linksto = str(row.get("LINKSTO", "")).lower()
    if linksto == "chartevents":
        score += 5

    return score


def find_best_ditem(ditems: pd.DataFrame, zh_param: str, abbr: Optional[str], unit: Optional[str]) -> Optional[pd.Series]:
    cands = english_candidates(zh_param, abbr)
    if not cands:
        # Fallback: use abbr only if present
        if abbr:
            cands = [abbr.lower()]
        else:
            cands = []

    best_row = None
    best_score = 0
    for _, row in ditems.iterrows():
        s = score_ditems_row(row, cands, abbr, unit)
        # 设定阈值，避免仅因LINKSTO/UNITNAME加分造成误匹配
        if s >= 30 and s > best_score:
            best_row = row
            best_score = s
    return best_row if best_score >= 30 else None


_num_pat = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _to_float(s: Any) -> Optional[float]:
    if pd.isna(s):
        return None
    try:
        return float(s)
    except Exception:
        m = _num_pat.search(str(s))
        if m:
            try:
                return float(m.group(0))
            except Exception:
                return None
        return None


def make_classifier(low_desc: str, normal_desc: str, high_desc: str) -> Tuple[Callable[[float], Optional[int]], str]:
    """
    Return: (classifier, description)
    classifier returns 0 (low), 1 (normal), 2 (high), or None if cannot classify.
    """
    l = (low_desc or "").strip()
    n = (normal_desc or "").strip()
    h = (high_desc or "").strip()

    def parse_bound(desc: str) -> Tuple[Optional[str], Optional[float]]:
        s = desc.replace("％", "%")
        s = s.replace("≥", ">=").replace("≤", "<=")
        s = s.replace("～", "-")
        s = s.replace("to", "-")
        s = s.strip()
        if not s or s == "-":
            return None, None
        if s.startswith("<="):
            return "<=", _to_float(s[2:])
        if s.startswith(">="):
            return ">=", _to_float(s[2:])
        if s.startswith("<"):
            return "<", _to_float(s[1:])
        if s.startswith(">"):
            return ">", _to_float(s[1:])
        # range a-b
        if "-" in s:
            parts = s.split("-")
            if len(parts) == 2:
                a = _to_float(parts[0])
                b = _to_float(parts[1])
                if a is not None and b is not None:
                    return "range", (a, b)
        # single number means equality threshold (treat as >= normal)
        v = _to_float(s)
        if v is not None:
            return "=", v
        return None, None

    n_type, n_val = parse_bound(n)
    l_type, l_val = parse_bound(l)
    h_type, h_val = parse_bound(h)

    desc_text = f"low:[{l}] normal:[{n}] high:[{h}]"

    # Prefer explicit normal range a-b
    if n_type == "range":
        a, b = n_val  # type: ignore
        def cls(v: float) -> Optional[int]:
            if v < a:
                return 0
            if v > b:
                return 2
            return 1
        return cls, desc_text

    # Normal defined as <= X
    if n_type in ("<=", "<"):
        x = n_val  # type: ignore
        inclusive = (n_type == "<=")
        def cls(v: float) -> Optional[int]:
            if (v <= x) if inclusive else (v < x):
                return 1
            return 2
        return cls, desc_text

    # Normal defined as >= X
    if n_type in (">=", ">"):
        x = n_val  # type: ignore
        inclusive = (n_type == ">=")
        def cls(v: float) -> Optional[int]:
            if (v >= x) if inclusive else (v > x):
                return 1
            return 0
        return cls, desc_text

    # Derive from abnormal bounds when normal is missing
    # Both abnormal bounds present
    if l_type in ("<=", "<", "=") and h_type in (">=", ">", "="):
        low_cut = l_val if l_type != "=" else l_val
        high_cut = h_val if h_type != "=" else h_val
        def cls(v: float) -> Optional[int]:
            if l_type == "<" and v < low_cut:
                return 0
            if l_type == "<=" and v <= low_cut:
                return 0
            if h_type == ">" and v > high_cut:
                return 2
            if h_type == ">=" and v >= high_cut:
                return 2
            # else normal between
            return 1
        return cls, desc_text

    # Only high abnormal present (e.g., ">=13")
    if h_type in (">=", ">", "=") and (not l_type or l == "-"):
        high_cut = h_val
        def cls(v: float) -> Optional[int]:
            if h_type == ">" and v > high_cut:
                return 2
            if h_type == ">=" and v >= high_cut:
                return 2
            if h_type == "=" and v == high_cut:
                return 2
            return 1  # otherwise normal
        return cls, desc_text

    # Only low abnormal present (e.g., "<6")
    if l_type in ("<=", "<", "=") and (not h_type or h == "-"):
        low_cut = l_val
        def cls(v: float) -> Optional[int]:
            if l_type == "<" and v < low_cut:
                return 0
            if l_type == "<=" and v <= low_cut:
                return 0
            if l_type == "=" and v == low_cut:
                return 0
            return 1
        return cls, desc_text

    # Fallback: cannot decide
    def cls_na(v: float) -> Optional[int]:
        return None
    return cls_na, desc_text


def process(my_items_path: str, ditems_path: str, chartevents_path: str, out_path: str, use_sim_mapping: bool = True) -> None:
    my_df = read_csv(my_items_path)
    ditems_df = read_csv(ditems_path)
    ce_df = read_csv(chartevents_path)

    # Normalize column names
    ditems_df.columns = [c.strip().upper() for c in ditems_df.columns]
    ce_df.columns = [c.strip().upper() for c in ce_df.columns]

    out_rows = []

    for idx, row in my_df.iterrows():
        zh_param = str(row.get("监测参数", "")).strip()
        if is_empty(zh_param):
            continue
        # Skip section header rows where most other fields are empty
        if is_empty(row.get("单位")) and is_empty(row.get("异常低值")) and is_empty(row.get("正常范围")) and is_empty(row.get("异常高值")):
            continue

        unit = str(row.get("单位", "")).strip()
        low_desc = str(row.get("异常低值", "")).strip()
        normal_desc = str(row.get("正常范围", "")).strip()
        high_desc = str(row.get("异常高值", "")).strip()

        base_name, abbr = extract_name_abbr(zh_param)
        ditem = find_best_ditem(ditems_df, base_name, abbr, unit)

        itemid = None
        label = None
        unitname = None

        if ditem is not None:
            itemid = ditem.get("ITEMID")
            label = ditem.get("LABEL")
            unitname = str(ditem.get("UNITNAME", "")).strip()
        elif use_sim_mapping and abbr and abbr in SIM_ABBR_TO_ITEMID:
            # 模拟映射回退
            itemid = SIM_ABBR_TO_ITEMID[abbr]
            label = abbr
            unitname = unit
        else:
            print(f"[WARN] 未在D_ITEMS中找到匹配项: {zh_param} (unit={unit}, abbr={abbr})")
            continue

        cls, rule_desc = make_classifier(low_desc, normal_desc, high_desc)

        # Filter CHARTEVENTS by ITEMID
        ce_sub = ce_df[ce_df["ITEMID"] == itemid].copy()
        if ce_sub.empty and use_sim_mapping and abbr and abbr in SIM_ABBR_TO_ITEMID:
            # 若D_ITEMS匹配的ITEMID在示例CHARTEVENTS中不存在，则尝试回退映射
            itemid_fallback = SIM_ABBR_TO_ITEMID[abbr]
            if itemid_fallback != itemid:
                ce_sub = ce_df[ce_df["ITEMID"] == itemid_fallback].copy()
                if not ce_sub.empty:
                    itemid = itemid_fallback
                    label = label or abbr

        if ce_sub.empty:
            print(f"[INFO] CHARTEVENTS中无记录: ITEMID={itemid} LABEL={label}")
            continue

        # Optional unit filter: ensure units match if provided
        if unit:
            ce_sub = ce_sub[(ce_sub["VALUEUOM"].fillna("").str.strip().str.lower() == unit.lower()) | (ce_sub["VALUEUOM"].isna())]
            if ce_sub.empty:
                print(f"[INFO] 单位不匹配或无可用记录: ITEMID={itemid} 期待单位={unit} 实际单位示例={unitname}")
                continue

        for _, r in ce_sub.iterrows():
            vnum = r.get("VALUENUM")
            if pd.isna(vnum):
                vnum = _to_float(r.get("VALUE"))
            vnum_f = _to_float(vnum)
            if vnum_f is None:
                continue
            cat = cls(vnum_f)
            if cat is None:
                continue
            out_rows.append({
                "monitor_param": zh_param,
                "abbr": abbr,
                "english_matched_label": label,
                "itemid": itemid,
                "subject_id": r.get("SUBJECT_ID"),
                "hadm_id": r.get("HADM_ID"),
                "icustay_id": r.get("ICUSTAY_ID"),
                "charttime": r.get("CHARTTIME"),
                "valueuom": r.get("VALUEUOM"),
                "valuenum_original": vnum_f,
                "value_category": cat,
                "range_rule": rule_desc,
            })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[DONE] 写出 {len(out_rows)} 条记录 -> {out_path}")


def default_paths() -> Tuple[str, str, str, str]:
    here = os.path.abspath(os.path.dirname(__file__))
    root = os.path.abspath(os.path.join(here, os.pardir))
    data_dir = os.path.join(root, "data")
    my_items = os.path.join(data_dir, "simpleMy_items.csv")
    ditems = os.path.join(data_dir, "simpleD_items.csv")
    chartevents = os.path.join(data_dir, "simpleCHARTEVENT.csv")
    out = os.path.join(root, "dataPrepare", "processed_chartevents_value_categories.csv")
    return my_items, ditems, chartevents, out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map CHARTEVENTS values into categorical ranges (0/1/2) based on simpleMy_items ranges.")
    my_items_p, ditems_p, ce_p, out_p = default_paths()
    parser.add_argument("--my_items", default=my_items_p, help="Path to simpleMy_items.csv")
    parser.add_argument("--ditems", default=ditems_p, help="Path to simpleD_items.csv (simulated D_ITEMS)")
    parser.add_argument("--chartevents", default=ce_p, help="Path to simpleCHARTEVENT.csv")
    parser.add_argument("--out", default=out_p, help="Output csv path")
    parser.add_argument("--no_sim_mapping", action="store_true", help="禁用示例数据的模拟映射回退")
    args = parser.parse_args()

    process(args.my_items, args.ditems, args.chartevents, args.out, use_sim_mapping=(not args.no_sim_mapping))