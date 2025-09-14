# 医疗决策模型用户反馈系统

这个系统允许用户通过自然语言反馈来调整医疗决策模型的预测结果，通过动态修改患者的节点表示来影响图神经网络的推理过程。

## 系统架构

```
用户自然语言反馈 → 提示词工程 → LLM理解 → 节点操作指令 → 图结构调整 → 重新预测
```

## 核心文件说明

### 1. `feedbackComponent.py`
- **功能**: 核心的反馈处理逻辑
- **主要函数**:
  - `parse_feedback_to_actions()`: 解析简单格式的反馈指令
  - `apply_user_actions_to_patient()`: 应用节点操作到患者数据
  - `recompute_with_feedback()`: 重新计算预测结果

### 2. `medical_feedback_prompt.py`
- **功能**: 提示词模板和医疗领域知识
- **主要内容**:
  - `MEDICAL_FEEDBACK_SYSTEM_PROMPT`: 系统级提示词
  - `MEDICAL_FEEDBACK_USER_PROMPT_TEMPLATE`: 用户提示词模板
  - `generate_feedback_prompt()`: 生成完整提示词的函数

### 3. `feedback_integration_example.py`
- **功能**: 集成示例和完整的处理流程
- **主要类**: `MedicalFeedbackProcessor` - 统一的反馈处理接口

## 使用方法

### 基础使用

```python
from feedback_integration_example import MedicalFeedbackProcessor

# 初始化处理器
processor = MedicalFeedbackProcessor()

# 处理用户反馈
result = processor.process_and_recompute(
    patient_id="12345",
    task_type="drugrec",  # 药物推荐任务
    current_prediction="推荐药物：阿司匹林、美托洛尔",
    user_feedback="患者有胃溃疡，不能用阿司匹林"
)

print(result)
```

### 集成LLM

```python
import openai

# 配置OpenAI客户端
client = openai.OpenAI(api_key="your-api-key")

# 使用LLM增强的处理器
processor = MedicalFeedbackProcessor(llm_client=client, model_name="gpt-3.5-turbo")

# 处理复杂的自然语言反馈
result = processor.process_and_recompute(
    patient_id="67890",
    task_type="mortality",
    current_prediction="死亡风险：0.35",
    user_feedback="患者还有严重的心力衰竭和慢性肾病，这些高危因素没有被充分考虑"
)
```

## 支持的反馈格式

### 1. 简单数字格式
- `+123`: 添加节点123
- `-456`: 移除节点456
- `+123,-456,+789`: 多个操作

### 2. 中文自然语言
- "添加糖尿病相关的节点"
- "删除阿司匹林，患者有胃溃疡"
- "考虑心力衰竭的风险因素"

### 3. 英文自然语言
- "Add heart failure as a risk factor"
- "Remove aspirin due to GI bleeding risk"
- "Consider renal impairment"

### 4. 混合格式
- "添加123，remove 456"
- "患者有心衰，+789，不要阿司匹林-234"

## 医疗任务类型

| 任务类型 | 描述 | 示例反馈 |
|---------|------|----------|
| `drugrec` | 药物推荐 | "患者对青霉素过敏，不能用抗生素" |
| `mortality` | 死亡率预测 | "患者有严重心衰，风险更高" |
| `readmission` | 再入院预测 | "患者依从性差，容易再入院" |
| `lenofstay` | 住院时长预测 | "需要长期康复，住院时间会更长" |

## 节点操作逻辑

### 添加节点 (+)
- 在 `ehr_node_set` 中将对应位置设为1
- 在 `visit_padded_node` 的最后一次就诊中添加该节点
- 更新 `node_set` 列表

### 移除节点 (-)
- 在 `ehr_node_set` 中将对应位置设为0
- 从所有就诊记录中移除该节点
- 从 `node_set` 列表中删除

### 安全机制
- 防止删空所有节点（会恢复到原始状态）
- 节点ID范围检查（0 ≤ node_id < max_nodes）
- 操作失败时的错误处理和回滚

## 提示词工程最佳实践

### 1. 系统提示词设计原则
- **医疗专业性**: 包含丰富的医疗领域知识
- **操作明确性**: 清晰定义输入输出格式
- **安全考虑**: 强调临床安全性和合理性
- **容错处理**: 处理模糊或矛盾的输入

### 2. 用户提示词优化
- **上下文信息**: 提供患者ID、任务类型、当前预测
- **结构化输入**: 使用模板确保信息完整
- **引导推理**: 分步骤引导模型分析

### 3. 输出格式控制
- **严格格式**: 要求特定的数字格式输出
- **多重解析**: 支持多种可能的输出格式
- **错误恢复**: 解析失败时的回退策略

## 扩展和定制

### 1. 添加新的医疗概念
在 `medical_feedback_prompt.py` 中扩展：
```python
COMMON_FEEDBACK_PATTERNS["new_concept"] = ["关键词1", "关键词2"]
```

### 2. 自定义节点映射
根据你的知识图谱结构更新 `NODE_TYPE_HINTS`：
```python
NODE_TYPE_HINTS = """
- 0-499: ICD-10疾病编码
- 500-999: ATC药物编码
- 1000-1499: 实验室检查
...
"""
```

### 3. 集成其他LLM
```python
class CustomLLMProcessor(MedicalFeedbackProcessor):
    def _call_llm_for_feedback(self, ...):
        # 实现你的LLM调用逻辑
        pass
```

## 性能优化建议

1. **缓存机制**: 对常见反馈模式进行缓存
2. **批处理**: 支持批量处理多个患者的反馈
3. **异步处理**: 对于大规模应用使用异步LLM调用
4. **模型选择**: 根据精度要求选择合适的LLM模型

## 错误处理

系统包含多层错误处理机制：
1. **直接解析**: 优先尝试规则解析
2. **LLM解析**: 复杂情况下使用LLM
3. **规则回退**: LLM失败时的规则回退
4. **安全恢复**: 操作失败时恢复原始状态

## 注意事项

1. **医疗安全**: 所有操作都应该经过医疗专业人员的验证
2. **数据隐私**: 确保患者数据的隐私保护
3. **模型限制**: 理解LLM在医疗领域的局限性
4. **持续监控**: 监控系统的预测质量和安全性

## 贡献指南

欢迎贡献代码和改进建议：
1. 提交Issue描述问题或建议
2. Fork项目并创建特性分支
3. 提交Pull Request
4. 确保代码通过测试和医疗安全检查