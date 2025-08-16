from pyhealth.data import Patient, Visit
#  基础数据结构
# 1. Patient 对象
# ​代表一个患者的所有医疗记录
# ​属性​：
# patient_id：患者唯一标识
# visits：按时间排序的就诊记录列表
# 2. Visit 对象
# ​代表单次就诊​
# ​关键属性​：
# visit_id：就诊唯一标识
# encounter_time：入院时间
# discharge_time：出院时间
# discharge_status：出院状态（0=存活，1=死亡）
# get_code_list(table)：获取特定医疗代码（诊断/手术/药物）


# 所有函数共享以下核心逻辑：

# ​数据过滤​：
# 排除三无就诊（无诊断/手术/药物）
# 未来需排除未成年患者（TODO标记）
# ​历史累积​：
# python
# 复制
# # 首次就诊
# samples[0]["conditions"] = [首次诊断列表]

# # 第二次就诊
# samples[1]["conditions"] = [首次诊断] + [第二次诊断]

# # 第三次就诊
# samples[2]["conditions"] = [前两次诊断] + [第三次诊断]
# ​数据结构​：
# 每个样本包含：
# 就诊ID和患者ID
# 累积诊断/手术/药物历史
# 任务特定标签（药物/死亡/再入院/住院时长）
def drug_recommendation_fn(patient):
# 功能​：为患者生成药物推荐任务的训练样本
    samples = []
    for i in range(len(patient)):
        visit = patient[i]
# ​遍历患者所有就诊记录​：
# 提取每次就诊的诊断代码（DIAGNOSES_ICD表）
# 提取手术代码（PROCEDURES_ICD表）
# 提取处方药物代码（PRESCRIPTIONS表）
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        # ATC 3 level
        # drugs = [drug[:4] for drug in drugs]
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
# 构建初始样本​：
# 包含就诊ID、患者ID、诊断列表、手术列表、当前药物列表
# 单独存储完整药物列表（drugs_all）用于历史追踪
        samples.append(  
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "drugs_all": drugs,
            }
        )
    if len(samples) < 1:
        return []
    # add history
#     ​添加历史信息​：
# 首次就诊：历史记录仅包含自身数据
# 后续就诊：累加所有历史就诊的诊断/手术/药物记录
# 例如：第二次就诊的"conditions" = [第一次诊断] + [第二次诊断]
    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_all"] = [samples[0]["drugs_all"]]

    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_all"] = samples[i - 1]["drugs_all"] + [
            samples[i]["drugs_all"]
        ]

    return samples
# MIMIC-IV药物推荐函数
def drug_recommendation_mimic4_fn(patient: Patient):

    samples = []
    for i in range(len(patient)):
        visit: Visit = patient[i]
#         数据结构差异​：
# 使用小写表名（diagnoses_icd vs DIAGNOSES_ICD）
# 适配MIMIC-IV数据集结构
        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        # ATC 3 level
#         药物代码标准化​：
# 将药物代码截取前4位（ATC-3级分类）
# 例如："N02BE01" → "N02B"（镇痛药）
        drugs = [drug[:4] for drug in drugs]
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "drugs_all": drugs,
            }
        )
    # exclude: patients with less than 2 visit
#     患者过滤​：
# 排除就诊次数少于2的患者
# 确保有足够历史信息建模
    if len(samples) < 2:
        return []
    # add history
    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_all"] = [samples[0]["drugs_all"]]

    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_all"] = samples[i - 1]["drugs_all"] + [
            samples[i]["drugs_all"]
        ]

    return samples


def mortality_prediction_mimic3_fn(patient):
    # ​预测目标​：患者下次就诊是否死亡

    samples = []

    # we will drop the last visit
    for i in range(len(patient) - 1):
#         ​遍历非末次就诊​：
# 使用当前就诊预测下次就诊状态
# 末次就诊无未来状态，故排除
        visit = patient[i]
        next_visit = patient[i + 1]
# ​标签构建​：
# 下次就诊discharge_status=0（存活）→ 标签0
# discharge_status=1（死亡）→ 标签1
# 其他状态视为存活（标签0）
        if next_visit.discharge_status not in [0, 1]:
            mortality_label = 0
        else:
            mortality_label = int(next_visit.discharge_status)

        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        # exclude: visits without condition, procedure, and drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "label": mortality_label,
            }
        )
    if len(samples) < 1:
        return []
    # add history
    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs"] = [samples[0]["drugs"]]

    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs"] = samples[i - 1]["drugs"] + [
            samples[i]["drugs"]
        ]
    # no cohort selection
    return samples

# 再入院预测函数 (readmission_prediction_mimic3_fn)
# ​预测目标​：患者是否在15天内再入院
def readmission_prediction_mimic3_fn(patient: Patient, time_window=15):
    samples = []

    # we will drop the last visit
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]
# 时间差计算​：
# 当前就诊与下次就诊的时间间隔（天数）
# time_diff = (下次就诊时间 - 当前就诊时间).days
# ​标签定义​：
# time_diff < 15 → 再入院（标签1）
# time_diff ≥ 15 → 未再入院（标签0）
        # get time difference between current visit and next visit
        time_diff = (next_visit.encounter_time - visit.encounter_time).days
        readmission_label = 1 if time_diff < time_window else 0

        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "label": readmission_label,
            }
        )
    if len(samples) < 1:
        return []
    # add history
    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs"] = [samples[0]["drugs"]]

    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs"] = samples[i - 1]["drugs"] + [
            samples[i]["drugs"]
        ]
    # no cohort selection
    return samples
# 住院时长预测函数 (length_of_stay_prediction_mimic3_fn)
# ​预测目标​：住院天数分类
def length_of_stay_prediction_mimic3_fn(patient: Patient):
    samples = []

    for visit in patient:

        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
# 计算实际住院天数​：
# los_days = (出院时间 - 入院时间).days
        los_days = (visit.discharge_time - visit.encounter_time).days
# 这个函数写在了最下面，if los_days < 1: 类别0  # <24小时
# elif 1 <= los_days <= 7: 类别1-7  # 1-7天
# elif 8 <= los_days <= 14: 类别8  # 8-14天
# else: 类别9  # >14天
        los_category = categorize_los(los_days)

        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "label": los_category,
            }
        )
    if len(samples) < 1:
        return []
    # add history
    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs"] = [samples[0]["drugs"]]

    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs"] = samples[i - 1]["drugs"] + [
            samples[i]["drugs"]
        ]
    # no cohort selection
    return samples

# 多数据集支持​：
# MIMIC-III：大写表名
# MIMIC-IV：小写表名
# eICU：特殊表名（diagnosis/physicalExam）
def length_of_stay_prediction_mimic4_fn(patient: Patient):
    samples = []

    for visit in patient:

        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue

        los_days = (visit.discharge_time - visit.encounter_time).days
        los_category = categorize_los(los_days)

        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "label": los_category,
            }
        )
    if len(samples) < 1:
        return []
    # add history
    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs"] = [samples[0]["drugs"]]

    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs"] = samples[i - 1]["drugs"] + [
            samples[i]["drugs"]
        ]
    # no cohort selection
    return samples


def length_of_stay_prediction_eicu_fn(patient: Patient):
    samples = []

    for visit in patient:

        conditions = visit.get_code_list(table="diagnosis")
        procedures = visit.get_code_list(table="physicalExam")
        drugs = visit.get_code_list(table="medication")
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue

        los_days = (visit.discharge_time - visit.encounter_time).days
        los_category = categorize_los(los_days)

        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "label": los_category,
            }
        )
    if len(samples) < 1:
        return []
    # add history
    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs"] = [samples[0]["drugs"]]

    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs"] = samples[i - 1]["drugs"] + [
            samples[i]["drugs"]
        ]
    # no cohort selection
    return samples

def mortality_prediction_mimic4_fn(patient: Patient):
    samples = []

    # we will drop the last visit
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]

        if next_visit.discharge_status not in [0, 1]:
            mortality_label = 0
        else:
            mortality_label = int(next_visit.discharge_status)

        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "label": mortality_label,
            }
        )
    if len(samples) < 1:
        return []
    # add history
    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs"] = [samples[0]["drugs"]]

    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs"] = samples[i - 1]["drugs"] + [
            samples[i]["drugs"]
        ]
    # no cohort selection
    return samples

# 同上，多种数据集
def readmission_prediction_mimic4_fn(patient: Patient, time_window=15):
    samples = []

    # we will drop the last visit
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]

        # get time difference between current visit and next visit
        time_diff = (next_visit.encounter_time - visit.encounter_time).days
        readmission_label = 1 if time_diff < time_window else 0

        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "label": readmission_label,
            }
        )
    if len(samples) < 1:
        return []
    # add history
    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs"] = [samples[0]["drugs"]]

    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs"] = samples[i - 1]["drugs"] + [
            samples[i]["drugs"]
        ]
    # no cohort selection
    return samples

# 住院时长分类函数 (categorize_los)
# ​功能​：将住院天数转换为10个类别
# ​分类标准​：

# ​类别0​：住院<1天（短于24小时）
# ​类别1-7​：住院1-7天（每天一个类别）
# ​类别8​：住院8-14天（一周至两周）
# ​类别9​：住院>14天（超过两周）
def categorize_los(days: int):
    """Categorizes length of stay into 10 categories.

    One for ICU stays shorter than a day, seven day-long categories for each day of
    the first week, one for stays of over one week but less than two,
    and one for stays of over two weeks.

    Args:
        days: int, length of stay in days

    Returns:
        category: int, category of length of stay
    """
    # ICU stays shorter than a day
    if days < 1:
        return 0
    # each day of the first week
    elif 1 <= days <= 7:
        return days
    # stays of over one week but less than two
    elif 7 < days <= 14:
        return 8
    # stays of over two weeks
    else:
        return 9