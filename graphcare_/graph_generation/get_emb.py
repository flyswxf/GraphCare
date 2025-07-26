import requests
import json
import os

# 原有的OpenAI实现（已注释）
# with open("../../resources/openai.key", 'r') as f:
#     openai_key = f.readlines()[0][:-1]

# 读取ECNU API key
script_dir = os.path.dirname(os.path.abspath(__file__))
key_path = os.path.join(script_dir, "../../resources/ecnu.key")
with open(key_path, 'r') as f:
    key = f.readlines()[0].strip()

def embedding_retriever(term):
    # 原有的OpenAI实现（已注释）
    # # Set up the API endpoint URL and request headers
    # url = "https://api.openai.com/v1/embeddings"
    # headers = {
    #     "Content-Type": "application/json",
    #     "Authorization": f"Bearer {openai_key}"
    # }
    # 
    # # Set up the request payload with the text string to embed and the model to use
    # payload = {
    #     "input": term,
    #     "model": "text-embedding-ada-002"
    #     # 该模型生成1536维的向量
    # }
    # 
    # # Send the request and retrieve the response
    # response = requests.post(url, headers=headers, data=json.dumps(payload))
    # 
    # # Extract the text embeddings from the response JSON
    # embedding = response.json()["data"][0]['embedding']
    # 
    # return embedding
    
    # 新的chatECNU实现
    # Set up the API endpoint URL and request headers for chatECNU
    url = "https://chat.ecnu.edu.cn/open/api/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
    }

    # Set up the request payload with the text string to embed and the ECNU model
    payload = {
        "input": term,
        "model": "ecnu-embedding-small"
        # 该模型生成1024维的向量
    }

    # 添加重试机制和频率控制
    import time
    max_retries = 5
    base_delay = 1.0  # 基础延迟时间
    
    for attempt in range(max_retries):
        try:
            # Send the request and retrieve the response
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
            
            # 检查响应状态码
            if response.status_code == 200:
                try:
                    # Extract the text embeddings from the response JSON
                    response_json = response.json()
                    embedding = response_json["data"][0]['embedding']
                    # 成功后添加小延迟避免频率限制
                    # time.sleep(base_delay)
                    return embedding
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"尝试 {attempt + 1}/{max_retries}: JSON 解析错误: {e}")
                    print(f"响应内容: {response.text}")
                    if attempt == max_retries - 1:
                        raise Exception(f"API 响应格式错误: {e}")
            elif response.status_code == 429:
                # 频率限制错误，需要更长的等待时间
                wait_time = 50  # 频率限制时等待更长时间
                print(f"尝试 {attempt + 1}/{max_retries}: API 频率限制 (429)")
                print(f"等待 {wait_time} 秒后重试...")
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                else:
                    raise Exception(f"API 频率限制，请稍后再试")
            elif response.status_code == 403:
                wait_time = 60  # 403错误时等待60秒
                print(f"尝试 {attempt + 1}/{max_retries}: API 访问被拒绝 (403)")
                print(f"响应内容: {response.text}")
                if attempt < max_retries - 1:
                    print(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"API 访问被拒绝，可能是网络代理或权限问题: {response.text}")
            else:
                print(f"尝试 {attempt + 1}/{max_retries}: API 请求失败，状态码: {response.status_code}")
                print(f"响应内容: {response.text}")
                if attempt == max_retries - 1:
                    raise Exception(f"API 请求失败: {response.status_code} - {response.text}")
                    
        except requests.exceptions.RequestException as e:
            print(f"尝试 {attempt + 1}/{max_retries}: 网络请求异常: {e}")
            if attempt == max_retries - 1:
                raise Exception(f"网络请求失败: {e}")
        
        # 等待后重试（非频率限制错误）
        if attempt < max_retries - 1 and response.status_code != 429:
            wait_time = (attempt + 1) * 3  # 递增等待时间
            print(f"等待 {wait_time} 秒后重试...")
            time.sleep(wait_time)