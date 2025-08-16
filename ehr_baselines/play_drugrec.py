
# 这个脚本实现了从原始医疗数据到药物推荐模型的完整流程，核心是通过患者的诊断和手术历史预测所需的药物组合。


from pyhealth.datasets import MIMIC3Dataset
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.models import RNN
from task_fn import readmission_prediction_mimic3_fn, drug_recommendation_fn, length_of_stay_prediction_mimic3_fn, mortality_prediction_mimic3_fn
from pyhealth.trainer import Trainer
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "7"


# STEP 1: load data
# ​作用​：加载MIMIC-III医疗数据集
# ​参数说明​：
# root：数据集存储路径
# tables：需要加载的数据表（诊断、手术、处方）
# code_mapping：医疗编码转换规则
# ICD9CM → CCSCM：诊断编码标准化
# ICD9PROC → CCSPROC：手术编码标准化
# NDC → ATC：药品编码标准化
# dev：是否使用开发模式（小数据集）
# refresh_cache：是否刷新缓存
# base_dataset.stat()：打印数据集统计信息
base_dataset = MIMIC3Dataset(
    root="/shared/eng/pj20/mimiciii/1.4",
    tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
    code_mapping={"ICD9CM": "CCSCM", "ICD9PROC": "CCSPROC", "NDC": "ATC"},
    dev=False,
    refresh_cache=False,
)
base_dataset.stat()

# STEP 2: set task
# ​作用​：将基础数据集转换为药物推荐任务专用格式
# ​说明​：
# set_task(drug_recommendation_fn)：应用task_fn.py中的药物推荐处理函数
# stat()：打印任务数据集统计信息
# print(samples[0])：查看第一个样本数据格式
sample_dataset = base_dataset.set_task(drug_recommendation_fn)
sample_dataset.stat()
print(sample_dataset.samples[0])
# ​作用​：按患者划分训练集、验证集和测试集
# ​参数说明​：
# [0.8, 0.1, 0.1]：划分比例（80%训练，10%验证，10%测试）
# seed=528：随机种子确保可复现性
# ​按患者划分避免同一患者数据出现在不同集合
train_dataset, val_dataset, test_dataset = split_by_patient(
    sample_dataset, [0.8, 0.1, 0.1], seed=528
)
# ​作用​：创建PyTorch数据加载器
# ​参数说明​：
# batch_size=4：每批处理4个样本
# shuffle=True：训练集打乱顺序
# shuffle=False：验证/测试集保持顺序
train_dataloader = get_dataloader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = get_dataloader(val_dataset, batch_size=4, shuffle=False)
test_dataloader = get_dataloader(test_dataset, batch_size=4, shuffle=False)

# STEP 3: define model
# ​作用​：初始化循环神经网络模型
# ​参数说明​：
# dataset：参考数据集（用于获取词汇表）
# feature_keys：输入特征（诊断和手术）
# label_key：预测目标（药物）
# mode="multilabel"：多标签分类模式（一个患者需要多种药物）
# embedding_dim=128：词嵌入维度为128
model = RNN(
    dataset=sample_dataset,
    feature_keys=["conditions", "procedures"],
    label_key="drugs",
    mode="multilabel",
    embedding_dim=128,
)

# STEP 4: define trainer
# 作用​：创建模型训练器
# ​参数说明​：
# metrics：评估指标列表
# pr_auc_samples：样本级精确率-召回率曲线下面积
# roc_auc_samples：样本级ROC曲线下面积
# f1_samples：样本级F1分数
# jaccard_samples：样本级Jaccard相似系数
trainer = Trainer(model=model, metrics=["pr_auc_samples", "roc_auc_samples", "f1_samples", "jaccard_samples"])
# 作用​：执行模型训练
# ​参数说明​：
# epochs=50：训练50轮
# monitor="pr_auc_samples"：使用PR-AUC作为早停指标
# optimizer_params={"lr":1e-4}：优化器学习率为0.0001
trainer.train(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    epochs=50,
    monitor="pr_auc_samples",
    optimizer_params = {"lr": 1e-4},
)

# STEP 5: evaluate
print(trainer.evaluate(test_dataloader))
