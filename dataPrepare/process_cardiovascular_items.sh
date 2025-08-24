#!/bin/bash

# 批处理脚本：处理五个心血管监测指标
# 肺动脉楔压 (PAWP)、心输出量 (CO)、心脏指数 (CI)、全身血管阻力 (SVR)、每搏量变异度 (SVV)

echo "开始处理心血管监测指标..."
echo "======================================"

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
    ["PAWP"]="Pulmonary Artery Wedge Pressure,Pulmonary Capillary Wedge Pressure,PCWP"
    ["CO"]="Cardiac Output"
    ["CI"]="Cardiac Index"
    ["SVR"]="Systemic Vascular Resistance"
    ["SVV"]="Stroke Volume Variation"
)

# 处理每个指标
for item in "PAWP" "CO" "CI" "SVR" "SVV"; do
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
echo "======================================"
echo "所有指标处理完成！"
echo "输出文件位置: $OUTPUT_DIR/"
echo "  - PAWP.csv  (肺动脉楔压)"
echo "  - CO.csv    (心输出量)"
echo "  - CI.csv    (心脏指数)"
echo "  - SVR.csv   (全身血管阻力)"
echo "  - SVV.csv   (每搏量变异度)"
echo "======================================"