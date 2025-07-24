import os
from openai import OpenAI

# 读取ECNU API key
with open("../../resources/ecnu.key", 'r') as f:
    ecnu_key = f.readlines()[0].strip()

# 保留原有的OpenAI实现
# with open("../../resources/openai.key", 'r') as f:
#     openai_key = f.readlines()[0][:-1]

class ChatGPT:
    def __init__(self):
        # Setting the API key to use the OpenAI API
        import openai
        openai.api_key = openai_key
        self.messages = []

    def chat(self, message):
        import openai
        self.messages.append({"role": "user", "content": message})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.messages
        )
        # self.messages.append({"role": "assistant", "content": response["choices"][0]["message"].content})
        return response["choices"][0]["message"]

class ChatECNU:
    def __init__(self, model="ecnu-plus"):
        # 使用ECNU的API配置
        self.client = OpenAI(
            api_key=ecnu_key,
            base_url="https://chat.ecnu.edu.cn/open/api/v1"
        )
        self.model = model
        self.messages = []
    
    def chat(self, message):
        """发送消息并获取回复"""
        self.messages.append({"role": "user", "content": message})
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages
            )
            
            # 获取助手的回复
            assistant_message = completion.choices[0].message
            
            # 将助手的回复添加到消息历史中（可选）
            # self.messages.append({"role": "assistant", "content": assistant_message.content})
            
            return assistant_message
            
        except Exception as e:
            print(f"调用ECNU API时出错: {e}")
            return None
    
    def set_system_message(self, system_content):
        """设置系统消息"""
        # 如果已有系统消息，则替换；否则插入到开头
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0] = {"role": "system", "content": system_content}
        else:
            self.messages.insert(0, {"role": "system", "content": system_content})
    
    def clear_messages(self):
        """清空消息历史"""
        self.messages = []
    
    def get_available_models(self):
        """获取可用的模型列表"""
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            print(f"获取模型列表时出错: {e}")
            return []
