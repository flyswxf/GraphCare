#!/bin/bash

# GraphCare 极端内存优化运行脚本
# 使用最小的参数设置来避免CUDA内存不足

echo "开始使用极端内存优化参数运行GraphCare..."
echo "参数设置: batch_size=4, hidden_dim=32, num_layers=1"
echo "这些参数可能会影响模型性能，但可以避免内存不足问题"

# 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

# 运行GraphCare，使用最小的参数
python graphcare.py \
    --dataset mimic3 \
    --task mortality \
    --kg GPT-KG \
    --batch_size 4 \
    --hidden_dim 32 \
    --epochs 20 \
    --lr 0.001 \
    --weight_decay 1e-5 \
    --dropout 0.3 \
    --num_layers 1 \
    --decay_rate 0.01 \
    --patient_mode joint \
    --alpha True \
    --beta True \
    --edge_attn True \
    --gnn BAT \
    --freeze False \
    --attn_init False \
    --in_drop_rate 0.0 \
    --kg_ratio 0.5 \
    --ehr_feat_ratio 0.5

echo "训练完成！"