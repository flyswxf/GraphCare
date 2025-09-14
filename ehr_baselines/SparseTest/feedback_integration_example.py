# ===== 医疗反馈系统集成示例 =====

import re
import torch
import openai  # 或其他LLM API
from typing import List, Tuple, Dict, Any
from medical_feedback_prompt import (
    generate_feedback_prompt, 
    TASK_TYPE_DESCRIPTIONS,
    COMMON_FEEDBACK_PATTERNS
)
from feedbackComponent import (
    parse_feedback_to_actions,
    apply_user_actions_to_patient,
    recompute_with_feedback
)

class MedicalFeedbackProcessor:
    """
    医疗反馈处理器 - 集成自然语言理解和节点操作
    """
    
    def __init__(self, llm_client=None, model_name="gpt-3.5-turbo"):
        """
        初始化反馈处理器
        
        Args:
            llm_client: 语言模型客户端（如OpenAI客户端）
            model_name: 使用的模型名称
        """
        self.llm_client = llm_client
        self.model_name = model_name
        
    def process_natural_language_feedback(self, 
                                         patient_id: str,
                                         task_type: str,
                                         current_prediction: str,
                                         user_feedback: str) -> List[Tuple[str, int]]:
        """
        处理自然语言反馈，返回节点操作指令
        
        Args:
            patient_id: 患者ID
            task_type: 任务类型
            current_prediction: 当前预测结果
            user_feedback: 用户自然语言反馈
            
        Returns:
            节点操作列表 [(op, node_id), ...]
        """
        
        # 1. 首先尝试直接解析（处理简单格式）
        direct_actions = parse_feedback_to_actions(user_feedback)
        if direct_actions:
            print(f"直接解析成功: {direct_actions}")
            return direct_actions
            
        # 2. 使用LLM进行复杂自然语言理解
        if self.llm_client:
            try:
                llm_actions = self._call_llm_for_feedback(patient_id, task_type, current_prediction, user_feedback)
                if llm_actions:
                    print(f"LLM解析成功: {llm_actions}")
                    return llm_actions
            except Exception as e:
                print(f"LLM调用失败: {e}")
                
        # 3. 回退到规则基础的解析
        rule_actions = self._rule_based_parsing(user_feedback)
        print(f"规则解析结果: {rule_actions}")
        return rule_actions
        
    def _call_llm_for_feedback(self, patient_id: str, task_type: str, 
                              current_prediction: str, user_feedback: str) -> List[Tuple[str, int]]:
        """
        调用大语言模型处理反馈
        """
        # 生成提示词
        prompt = generate_feedback_prompt(patient_id, task_type, current_prediction, user_feedback)
        
        # 调用LLM（这里以OpenAI为例）
        response = self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": prompt.split('\n\n')[0]},  # system prompt
                {"role": "user", "content": prompt.split('\n\n')[1]}     # user prompt
            ],
            temperature=0.1,  # 低温度确保一致性
            max_tokens=200
        )
        
        llm_output = response.choices[0].message.content.strip()
        print(f"LLM原始输出: {llm_output}")
        
        # 解析LLM输出
        return self._parse_llm_output(llm_output)
        
    def _parse_llm_output(self, llm_output: str) -> List[Tuple[str, int]]:
        """
        解析LLM的输出为节点操作指令
        """
        actions = []
        
        # 提取数字和符号
        patterns = [
            r'([+\-])\s*(\d+)',  # +123, -456
            r'(\d+)\s*([+\-])',  # 123+, 456-
            r'add\s+(\d+)',      # add 123
            r'remove\s+(\d+)',   # remove 456
            r'(\d+)',            # 纯数字，需要根据上下文判断
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, llm_output, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    if match[0] in ['+', '-']:
                        actions.append((match[0], int(match[1])))
                    elif match[1] in ['+', '-']:
                        actions.append((match[1], int(match[0])))
                elif 'add' in llm_output.lower():
                    actions.append(('+', int(match)))
                elif 'remove' in llm_output.lower():
                    actions.append(('-', int(match)))
                    
        # 去重
        seen = set()
        dedup_actions = []
        for action in actions:
            if action not in seen:
                dedup_actions.append(action)
                seen.add(action)
                
        return dedup_actions
        
    def _rule_based_parsing(self, user_feedback: str) -> List[Tuple[str, int]]:
        """
        基于规则的反馈解析（回退方案）
        """
        actions = []
        text = user_feedback.lower()
        
        # 检查常见的医疗反馈模式
        if any(word in text for word in COMMON_FEEDBACK_PATTERNS["drug_allergy"]):
            # 药物过敏 - 可能需要移除某些药物节点
            # 这里需要根据具体的节点映射来实现
            pass
            
        if any(word in text for word in COMMON_FEEDBACK_PATTERNS["contraindication"]):
            # 禁忌症 - 移除相关节点
            pass
            
        if any(word in text for word in COMMON_FEEDBACK_PATTERNS["comorbidity"]):
            # 合并症 - 添加相关节点
            pass
            
        # 如果没有识别出具体模式，返回空列表
        return actions
        
    def process_and_recompute(self, 
                            patient_id: str,
                            task_type: str, 
                            current_prediction: str,
                            user_feedback: str,
                            topk: int = 5) -> Dict[str, Any]:
        """
        完整的反馈处理和重新计算流程
        
        Returns:
            包含原始结果、调整后结果和操作记录的字典
        """
        
        # 1. 处理自然语言反馈
        actions = self.process_natural_language_feedback(
            patient_id, task_type, current_prediction, user_feedback
        )
        
        if not actions:
            return {
                "status": "no_action",
                "message": "无法从反馈中识别出具体的操作指令",
                "original_prediction": current_prediction
            }
            
        # 2. 应用操作并重新计算
        try:
            # 构造反馈文本（转换为现有系统能理解的格式）
            feedback_text = ", ".join([f"{op}{nid}" for op, nid in actions])
            
            # 重新计算
            new_result = recompute_with_feedback(patient_id, feedback_text, topk)
            
            return {
                "status": "success",
                "actions_applied": actions,
                "feedback_text": feedback_text,
                "original_prediction": current_prediction,
                "new_result": new_result,
                "message": f"成功应用{len(actions)}个节点操作"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "actions_attempted": actions,
                "error_message": str(e),
                "original_prediction": current_prediction
            }

# 使用示例
def example_usage():
    """
    使用示例
    """
    
    # 初始化处理器（需要配置LLM客户端）
    # processor = MedicalFeedbackProcessor(llm_client=openai_client)
    processor = MedicalFeedbackProcessor()  # 不使用LLM的版本
    
    # 示例场景1：药物推荐任务
    result1 = processor.process_and_recompute(
        patient_id="12345",
        task_type="drugrec",
        current_prediction="推荐药物：阿司匹林、美托洛尔",
        user_feedback="患者有严重的胃溃疡病史，不能使用阿司匹林，建议考虑氯吡格雷"
    )
    print("场景1结果:", result1)
    
    # 示例场景2：死亡率预测任务
    result2 = processor.process_and_recompute(
        patient_id="67890",
        task_type="mortality",
        current_prediction="死亡风险：中等（0.35）",
        user_feedback="患者还有严重的心力衰竭和肾功能不全，风险应该更高"
    )
    print("场景2结果:", result2)
    
    # 示例场景3：简单格式反馈
    result3 = processor.process_and_recompute(
        patient_id="11111",
        task_type="drugrec",
        current_prediction="推荐药物列表",
        user_feedback="+123, -456, 添加789"
    )
    print("场景3结果:", result3)

if __name__ == "__main__":
    example_usage()