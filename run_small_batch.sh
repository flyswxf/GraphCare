#!/bin/bash

# GraphCare 内存优化运行脚本
# 使用较小的batch_size和其他内存友好的参数

echo "开始使用内存优化参数运行GraphCare..."
echo "参数设置: batch_size=16, hidden_dim=64, num_layers=2"

# 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 运行GraphCare，使用较小的参数
python graphcare.py \
    --dataset mimic3 \
    --task mortality \
    --kg GPT-KG \
    --batch_size 16 \
    --hidden_dim 64 \
    --epochs 50 \
    --lr 0.001 \
    --weight_decay 1e-5 \
    --dropout 0.5 \
    --num_layers 2 \
    --decay_rate 0.01 \
    --patient_mode joint \
    --alpha True \
    --beta True \
    --edge_attn True \
    --gnn BAT \
    --freeze False \
    --attn_init False \
    --in_drop_rate 0.0 \
    --kg_ratio 1.0 \
    --ehr_feat_ratio 1.0

echo "训练完成！"