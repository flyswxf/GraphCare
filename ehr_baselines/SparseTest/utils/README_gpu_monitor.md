# GPU显存监控工具

这个脚本用于监控和检查GPU显存使用状态，特别适用于深度学习训练过程中的显存管理。

## 功能特性

- 实时显示GPU显存使用情况
- 支持多GPU环境
- 显示PyTorch显存分配信息
- 连续监控模式
- 保存监控数据到JSON文件
- 友好的可视化显示界面

## 使用方法

### 基本用法

```bash
# 查看当前GPU显存状态
python gpu_memory_monitor.py

# 显示PyTorch显存信息
python gpu_memory_monitor.py --pytorch
```

### 连续监控模式

```bash
# 连续监控，默认2秒刷新一次
python gpu_memory_monitor.py --monitor

# 自定义刷新间隔（5秒）
python gpu_memory_monitor.py --monitor --interval 5
```

### 保存监控数据

```bash
# 保存当前GPU状态到JSON文件
python gpu_memory_monitor.py --save gpu_status.json
```

## 命令行参数

- `--monitor, -m`: 启用连续监控模式
- `--interval, -i`: 设置监控刷新间隔（秒），默认2秒
- `--save, -s`: 保存GPU信息到指定的JSON文件
- `--pytorch, -p`: 显示PyTorch显存分配信息

## 输出信息说明

### GPU基本信息
- **GPU索引和名称**: 显示GPU编号和型号
- **总显存**: GPU的总显存容量
- **已用显存**: 当前已使用的显存量和百分比
- **可用显存**: 剩余可用的显存量
- **GPU利用率**: GPU计算单元的使用率
- **温度**: GPU当前温度（如果支持）

### PyTorch显存信息
- **PyTorch分配**: PyTorch实际分配的显存
- **PyTorch缓存**: PyTorch缓存的显存
- **PyTorch峰值**: 历史最大分配量

### 可视化显示
- 显存使用状态条：直观显示显存使用比例
- 实时刷新时间戳

## 使用场景

1. **训练前检查**: 确认显存是否足够
2. **训练过程监控**: 实时观察显存使用情况
3. **内存泄漏检测**: 发现异常的显存增长
4. **多GPU负载均衡**: 检查各GPU的使用情况
5. **性能调优**: 优化batch size和模型参数

## 依赖要求

- NVIDIA GPU驱动
- nvidia-smi工具
- Python 3.6+
- PyTorch（可选，用于显示PyTorch显存信息）

## 注意事项

1. 需要安装NVIDIA驱动和nvidia-smi工具
2. 在没有GPU的环境中会显示相应提示
3. 连续监控模式下按Ctrl+C停止
4. PyTorch显存信息需要安装PyTorch库

## 示例输出

```
================================================================================
GPU 显存状态监控
================================================================================

GPU 0: NVIDIA GeForce RTX 3080
  总显存:     10.0 GB
  已用显存:   2.5 GB (25.0%)
  可用显存:   7.5 GB
  GPU利用率:  45%
  温度:       65°C
  PyTorch分配: 2.1 GB
  PyTorch缓存: 2.3 GB
  PyTorch峰值: 3.2 GB
  显存使用:   [████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 25.0%
```

## 故障排除

1. **"nvidia-smi不可用"错误**: 检查NVIDIA驱动是否正确安装
2. **无法检测到GPU**: 确认GPU硬件连接和驱动状态
3. **PyTorch信息不显示**: 检查PyTorch是否正确安装