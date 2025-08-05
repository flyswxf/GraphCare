from pyhealth.datasets import MIMIC3Dataset
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.models import RNN
from task_fn import readmission_prediction_mimic3_fn, drug_recommendation_fn, length_of_stay_prediction_mimic3_fn, mortality_prediction_mimic3_fn
from pyhealth.trainer import Trainer
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# STEP 1: load data
# 功能​：加载MIMIC-III医疗数据集
# ​参数详解​：
# root：数据集存储路径
# tables：加载诊断、手术和处方三张表
# code_mapping：医疗编码标准化
# dev=False：使用完整数据集
# refresh_cache=False：使用现有缓存
# stat()：打印数据集统计信息
base_dataset = MIMIC3Dataset(
    root="/shared/eng/pj20/mimiciii/1.4",
    tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
    code_mapping={"ICD9CM": "CCSCM", "ICD9PROC": "CCSPROC", "NDC": "ATC"},
    dev=False,
    refresh_cache=False,
)
base_dataset.stat()

# STEP 2: set task
# ​作用​：将基础数据转换为死亡率预测任务格式
# ​关键处理​：
# 使用mortality_prediction_mimic3_fn函数处理
# 标签：下次就诊是否死亡（0=存活，1=死亡）
# stat()：打印任务数据集统计
# print(samples[0])：查看第一个样本格式
sample_dataset = base_dataset.set_task(mortality_prediction_mimic3_fn)
sample_dataset.stat()
print(sample_dataset.samples[0])
# ​作用​：按患者划分训练集、验证集和测试集
# ​参数​：
# [0.8, 0.1, 0.1]：80%训练，10%验证，10%测试
# seed=528：固定随机种子确保可复现性
# ​重要​：按患者划分避免同一患者数据出现在不同集合
train_dataset, val_dataset, test_dataset = split_by_patient(
    sample_dataset, [0.8, 0.1, 0.1], seed=528
)
# 作用​：创建PyTorch数据加载器
# ​参数说明​：
# batch_size=4：每批处理4个样本
# shuffle=True：训练集打乱顺序
# shuffle=False：验证/测试集保持顺序

train_dataloader = get_dataloader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = get_dataloader(val_dataset, batch_size=4, shuffle=False)
test_dataloader = get_dataloader(test_dataset, batch_size=4, shuffle=False)

# STEP 3: define model
# ​作用​：初始化循环神经网络
# ​参数详解​：
# dataset：参考数据集（获取词汇表）
# feature_keys：输入特征（诊断、手术和药物）
# label_key="label"：预测目标（死亡率标签）
# mode="binary"：二分类模式（死亡/存活）
# embedding_dim=128：词嵌入维度
model = RNN(
    dataset=sample_dataset,
    feature_keys=["conditions", "procedures", "drugs"],
    label_key="label",
    mode="binary",
    embedding_dim=128,
)

# STEP 4: define trainer
# ​作用​：创建模型训练器
# ​评估指标​：
# pr_auc：精确率-召回率曲线下面积
# roc_auc：ROC曲线下面积
# ​选择理由​：二分类任务的标准评估指标
trainer = Trainer(model=model, metrics=['pr_auc', 'roc_auc'])
# 作用​：执行模型训练
# ​参数说明​：
# epochs=50：训练50轮
# monitor="roc_auc"：使用ROC-AUC选择最佳模型
# optimizer_params={"lr":1e-4}：学习率0.0001
trainer.train(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    epochs=50,
    monitor="roc_auc",
    optimizer_params = {"lr": 1e-4},
)

# STEP 5: evaluate
# ​作用​：在测试集上评估最终模型性能
# ​输出​：打印PR-AUC和ROC-AUC结果
print(trainer.evaluate(test_dataloader))
