#!/bin/bash

# 批处理脚本：处理心源性休克和心肌梗死的检测指标
# 包含13个关键指标的数据收集和分析

echo "开始处理心源性休克和心肌梗死检测指标..."
echo "========================================"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/select_best_item_by_frequency.py"

# 检查Python脚本是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误: 找不到 select_best_item_by_frequency.py 脚本"
    exit 1
fi

# 创建输出目录
OUTPUT_DIR="$SCRIPT_DIR/match_stats"
mkdir -p "$OUTPUT_DIR"
echo "输出目录: $OUTPUT_DIR"

# 定义要处理的指标和对应的同义词
declare -A ITEMS=(
    ["MAP"]="Mean Arterial Pressure"
    ["CVP"]="Central Venous Pressure"
    ["PAWP"]="Pulmonary Artery Wedge Pressure,Pulmonary Capillary Wedge Pressure,PCWP"
    ["CO"]="Cardiac Output"
    ["CI"]="Cardiac Index"
    ["SVRI"]="Systemic Vascular Resistance Index"
    ["LVSWI"]="Left Ventricular Stroke Work Index"
    ["HR"]="Heart Rate"
    ["LAC"]="Lactate,Lactic Acid"
    ["URINE"]="Urine Output"
    ["K"]="Potassium,K+"
    ["PAO2FIO2"]="PaO2/FiO2,PF ratio,PaO2/FiO2 Ratio,P/F Ratio,PF,PF Ratio,Oxygen Index,OI,PaO2/FIO2,PaO2/FIO2 ratio"
    ["LVEF"]="Left Ventricular Ejection Fraction"
)

# 处理每个指标
for item in ${!ITEMS[@]}; do
# for item in "PAO2FIO2"; do
    echo ""
    echo "正在处理: $item"
    echo "--------------------------------------"
    
    # 构建命令参数，指定输出文件到match_stats目录
    OUTPUT_FILE="$OUTPUT_DIR/${item}.csv"
    cmd="python3 \"$PYTHON_SCRIPT\" --query \"$item\" --out \"$OUTPUT_FILE\""
    
    # 添加同义词参数
    IFS=',' read -ra SYNONYMS <<< "${ITEMS[$item]}"
    for synonym in "${SYNONYMS[@]}"; do
        if [ -n "$synonym" ]; then
            cmd="$cmd --term \"$synonym\""
        fi
    done
    
    # 添加其他参数
    cmd="$cmd --search_in both --chunk_size 50000"
    
    echo "执行命令: $cmd"
    
    # 执行命令
    eval $cmd
    
    if [ $? -eq 0 ]; then
        echo "✓ $item 处理完成"
    else
        echo "✗ $item 处理失败"
    fi
done

echo ""
echo "========================================"
echo "所有心源性休克和心肌梗死指标处理完成！"
echo "输出文件位置: $OUTPUT_DIR/"
echo "  - MAP.csv        (平均动脉压)"
echo "  - CVP.csv        (中心静脉压)"
echo "  - PAWP.csv       (肺动脉嵌顿压)"
echo "  - CO.csv         (心输出量)"
echo "  - CI.csv         (心脏指数)"
echo "  - SVRI.csv       (体循环阻力指数)"
echo "  - LVSWI.csv      (左心室做功指数)"
echo "  - HR.csv         (心率)"
echo "  - LAC.csv        (乳酸)"
echo "  - URINE.csv      (尿量)"
echo "  - K.csv          (血钾)"
echo "  - PAO2FIO2.csv   (氧合指数)"
echo "  - LVEF.csv       (左心室射血分数)"
echo "========================================"