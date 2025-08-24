from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import uuid
import random
from typing import List, Dict, Any, Optional

app = FastAPI(title="GraphCare Demo Backend", version="1.0.0")

# CORS: allow all origins for demo purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store
SESSIONS: Dict[str, Dict[str, Any]] = {}


class MeasureItem(BaseModel):
    measure: str
    reason: Optional[str] = None


class SimilarCaseItem(BaseModel):
    measure: str
    frequency: float  # 0.0 - 1.0


class InferResponse(BaseModel):
    session_id: str
    probability: float
    recommended: List[MeasureItem]
    similar_cases: List[SimilarCaseItem]


class AugmentRequest(BaseModel):
    session_id: str
    text: str


def _safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def compute_base_probability(patient: Dict[str, Any]) -> float:
    # Heuristic demo scoring (not a medical device)
    score = 0.25
    vitals = patient.get("vitals", {}) or {}
    labs = patient.get("labs", {}) or {}
    history = patient.get("history", {}) or {}

    map_val = vitals.get("MAP")  # mean arterial pressure
    ci = vitals.get("CI")  # cardiac index
    pawp = vitals.get("PAWP")  # pulmonary artery wedge pressure
    hr = vitals.get("HR")
    lact = labs.get("lactate")
    ef = labs.get("EF")  # ejection fraction
    urine = labs.get("urine_output_6h") or labs.get("urine_output_24h")

    if map_val is not None and map_val < 65:
        score += 0.18
    if ci is not None and ci < 2.2:
        score += 0.17
    if pawp is not None and pawp > 18:
        score += 0.10
    if hr is not None and hr > 110:
        score += 0.05
    if lact is not None and lact >= 2:
        score += 0.12
    if ef is not None and ef < 35:
        score += 0.12
    if urine is not None:
        try:
            if urine < 0.5:  # ml/kg/h rough proxy
                score += 0.08
        except Exception:
            pass

    # history markers
    if (history.get("AMI_recent") or history.get("STEMI") or history.get("MI")):
        score += 0.08

    return max(0.01, min(0.98, score))


def analyze_text_adjustment(text: str) -> float:
    if not text:
        return 0.0
    t = text.lower()
    inc_keywords = [
        "低血压", "血压下降", "心率过快", "尿量减少", "少尿", "乳酸", "皮肤冰冷", "四肢冰冷", "皮肤湿冷",
        "st段抬高", "心肌梗死", "mi", "左室功能不全", "ef降低", "灌注不足", "意识模糊",
    ]
    dec_keywords = [
        "好转", "稳定", "无胸痛", "症状缓解", "灌注改善", "意识清醒", "血压稳定",
    ]
    delta = 0.0
    for kw in inc_keywords:
        if kw in t:
            delta += 0.04
    for kw in dec_keywords:
        if kw in t:
            delta -= 0.03
    return max(-0.25, min(0.25, delta))


def make_recommendations(prob: float, patient: Dict[str, Any]) -> List[MeasureItem]:
    recs: List[MeasureItem] = []
    vitals = patient.get("vitals", {}) or {}
    labs = patient.get("labs", {}) or {}

    if prob >= 0.7:
        recs.append(MeasureItem(measure="紧急升压支持（去甲肾上腺素优先）", reason="高风险休克：需快速恢复灌注压力"))
        recs.append(MeasureItem(measure="完善血流动力学监测（动脉置管/有创血压）", reason="实时评估MAP与用药反应"))
        recs.append(MeasureItem(measure="床旁超声/心电图，评估心功能与机械并发症", reason="明确病因指导治疗"))
        recs.append(MeasureItem(measure="评估机械循环支持（IABP/Impella/VA-ECMO）", reason="药物反应差时的升级策略"))
    elif prob >= 0.4:
        recs.append(MeasureItem(measure="升压药滴定维持MAP≥65 mmHg", reason="灌注保护"))
        recs.append(MeasureItem(measure="利尿剂/血管活性药物个体化调整", reason="容量与后负荷管理"))
        recs.append(MeasureItem(measure="动态监测乳酸与尿量", reason="判断组织灌注变化"))
        recs.append(MeasureItem(measure="心超评估泵功能", reason="决定是否需正性肌力药物"))
    else:
        recs.append(MeasureItem(measure="密切观察+基础监测", reason="当前风险较低"))
        recs.append(MeasureItem(measure="优化液体管理与镇痛/镇静", reason="避免诱发因素"))
        recs.append(MeasureItem(measure="必要时复查乳酸与心超", reason="动态评估风险"))

    if (vitals.get("PAWP") and vitals.get("PAWP") > 18) or (labs.get("BNP") and labs.get("BNP") > 400):
        recs.append(MeasureItem(measure="利尿与后负荷降低", reason="静脉淤血提示前负荷/后负荷过高"))

    return recs[:6]


def make_similar_cases(prob: float, seed: str) -> List[SimilarCaseItem]:
    rng = random.Random(seed)
    base = prob
    items = [
        ("去甲肾上腺素滴定", max(0.15, min(0.95, base + rng.uniform(-0.1, 0.1)))),
        ("正性肌力（多巴酚丁胺/米力农）", max(0.05, min(0.85, base - 0.1 + rng.uniform(-0.1, 0.1)))),
        ("IABP 评估/使用", max(0.05, min(0.7, base - 0.2 + rng.uniform(-0.1, 0.1)))),
        ("VA-ECMO 转诊/启动", max(0.02, min(0.5, base - 0.3 + rng.uniform(-0.1, 0.1)))),
    ]
    # Normalize to look like proportions used among相似病例
    total = sum(x[1] for x in items)
    if total <= 0:
        total = 1.0
    items = [(name, val / total) for name, val in items]
    return [SimilarCaseItem(measure=name, frequency=round(val, 3)) for name, val in items]


@app.post("/api/infer/upload", response_model=InferResponse)
async def infer_from_upload(file: UploadFile = File(...)):
    if file.content_type not in ("application/json", "text/json", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="请上传JSON文件（application/json）")
    raw = await file.read()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="无法解析JSON内容")

    session_id = str(uuid.uuid4())
    base_prob = compute_base_probability(payload)

    prob = base_prob
    recs = make_recommendations(prob, payload)
    sims = make_similar_cases(prob, seed=session_id)

    SESSIONS[session_id] = {"patient": payload, "notes": [], "prob": prob}

    return InferResponse(
        session_id=session_id,
        probability=round(prob, 3),
        recommended=recs,
        similar_cases=sims,
    )


@app.post("/api/infer/augment", response_model=InferResponse)
async def augment_with_text(body: AugmentRequest):
    session = SESSIONS.get(body.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session_id不存在或已过期")
    session["notes"].append(body.text)

    base_prob = compute_base_probability(session["patient"])  # base from patient
    delta = sum(analyze_text_adjustment(t) for t in session["notes"])  # cumulative text impact
    prob = max(0.01, min(0.98, base_prob + delta))
    session["prob"] = prob

    recs = make_recommendations(prob, session["patient"])
    sims = make_similar_cases(prob, seed=body.session_id)

    return InferResponse(
        session_id=body.session_id,
        probability=round(prob, 3),
        recommended=recs,
        similar_cases=sims,
    )


@app.get("/api/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)