"""
Keyword extractor for user feedback using LLM + embeddings.

流程：
1) 读取 RESPONSE_FILE 的自然语言反馈；
2) 通过 ChatECNU 大模型将文本拆分成两组关键词："add" 与 "remove"；
3) 使用 embedding_retriever 为关键词与反馈文本生成向量，按相似度排序；
4) 截取前 k（由 num_add/num_remove 控制），写入 KEYWORD_FILE（JSON：{"add": [...], "remove": [...]}）。
"""

import os
import json
import re
from typing import List, Dict, Tuple
import sys

sys.path.append(r'D:\\desktop\\code\\ICU\\GraphCare')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 兼容包内导入与直接脚本运行的常量导入
try:
    from .constants import (
        RESPONSE_FILE,
        KEYWORD_FILE,
        INFERENCE_RESULT_FILE,
        NUM_KEYWORDS_ADD_DEFAULT,
        NUM_KEYWORDS_REMOVE_DEFAULT,
        ADD_HINTS,
        REMOVE_HINTS,
    )
except ImportError:
    from constants import (
        RESPONSE_FILE,
        KEYWORD_FILE,
        INFERENCE_RESULT_FILE,
        NUM_KEYWORDS_ADD_DEFAULT,
        NUM_KEYWORDS_REMOVE_DEFAULT,
        ADD_HINTS,
        REMOVE_HINTS,
    )

# Import embedding retriever
# try:
from graphcare_.graph_generation.get_emb import embedding_retriever
# except Exception:
#     embedding_retriever = None

# Import ChatECNU client (LLM)
# try:
from graphcare_.graph_generation.ChatGPT import ChatECNU

def _log(msg: str) -> None:
    print(f"[关键词提取] {msg}")
# except Exception:
#     ChatECNU = None


def _read_feedback_text() -> str:
    if not os.path.exists(RESPONSE_FILE):
        _log(f"未找到反馈文件: {RESPONSE_FILE}")
        return ""
    _log(f"读取反馈文件: {RESPONSE_FILE}")
    with open(RESPONSE_FILE, "r", encoding="utf-8") as f:
        content = f.read().strip()
    _log(f"反馈文本长度: {len(content)} 字符")
    preview = (content[:120] + "...") if len(content) > 120 else content
    _log(f"反馈预览: {preview}")
    return content




def _embed(text: str) -> List[float]:
    if embedding_retriever is None:
        _log("embedding_retriever 不可用，相关度排序将跳过或退化")
        return []
    try:
        return embedding_retriever(text)
    except Exception:
        _log("embedding_retriever 调用失败，忽略该向量")
        return []


def _cosine(u: List[float], v: List[float]) -> float:
    import math
    if not u or not v:
        return 0.0
    dot = sum(a*b for a, b in zip(u, v))
    nu = math.sqrt(sum(a*a for a in u))
    nv = math.sqrt(sum(b*b for b in v))
    if nu == 0 or nv == 0:
        return 0.0
    return dot / (nu * nv)


def _llm_extract_keywords(feedback_text: str, task: str,
                          num_add: int, num_remove: int) -> Dict[str, List[str]]:
    """
    调用 ChatECNU，将反馈文本拆分为两组关键词：{"add": [...], "remove": [...]}。
    若调用失败或解析失败，回退到基于提示词的简单规则解析。
    """
    # 优先使用 LLM
    if ChatECNU is not None:
        try:
            client = ChatECNU(model="ecnu-max")
            _log("使用 ChatECNU 进行关键词拆分")
            # 系统提示：定义任务与输出格式
            system_prompt = (
                "你是一名医疗知识图谱助手。根据用户反馈和任务类型，将文本拆分为两组关键词：\n"
                "- add：表示需要添加或强调的概念/药物/手术等；\n"
                "- remove：表示需要移除或弱化的概念/药物/手术等；\n"
                "请只输出JSON，包含两个数组字段：{\"add\": [...], \"remove\": [...]}。\n"
                f"每组最多返回合理的短语或词语，add不超过{num_add}个，remove不超过{num_remove}个。\n"
                "关键词应尽量短（1-4词），贴近医疗术语或ATC/CCS层级概念。"
            )
            client.set_system_message(system_prompt)

            # 用户提示：包含任务类型与反馈文本
            user_prompt = (
                f"任务类型：{task}\n"
                f"用户反馈：" + feedback_text + "\n"
                "请按要求只返回JSON，不要解释或添加其他文本。示例：\n"
                "{\n  \"add\": [\"antithrombotic agents\"],\n  \"remove\": [\"opioid analgesics\"]\n}"
            )
            msg = client.chat(user_prompt)
            if msg and getattr(msg, "content", None):
                content = msg.content
                # 尝试解析为JSON
                try:
                    parsed = json.loads(content)
                    add_list = [s.strip() for s in (parsed.get("add") or []) if s and isinstance(s, str)]
                    remove_list = [s.strip() for s in (parsed.get("remove") or []) if s and isinstance(s, str)]
                    _log(f"LLM 拆分完成：add={len(add_list)}，remove={len(remove_list)}")
                    return {"add": add_list, "remove": remove_list}
                except Exception:
                    # 内容不是纯JSON时，尝试用正则提取简单数组
                    try:
                        m_add = re.findall(r'"add"\s*:\s*\[(.*?)\]', content, flags=re.S)
                        m_remove = re.findall(r'"remove"\s*:\s*\[(.*?)\]', content, flags=re.S)
                        def _split_items(raw: str) -> List[str]:
                            items = re.findall(r'"(.*?)"', raw or "")
                            return [x.strip() for x in items if x.strip()]
                        add_list = _split_items(m_add[0]) if m_add else []
                        remove_list = _split_items(m_remove[0]) if m_remove else []
                        _log(f"LLM 文本解析完成：add={len(add_list)}，remove={len(remove_list)}")
                        return {"add": add_list, "remove": remove_list}
                    except Exception:
                        pass
        except Exception:
            _log("ChatECNU 调用失败，改用提示词规则回退")
            pass

    # 回退：基于提示词的简单规则解析
    add_list: List[str] = []
    remove_list: List[str] = []
    # 依据中文/英文提示词进行粗略分组
    parts = re.split(r"[，。、“”\";:,.!?\n\r]", feedback_text)
    for p in [x.strip() for x in parts if x.strip()]:
        low = p.lower()
        if any(h in p for h in REMOVE_HINTS) or any(h in low for h in [h for h in REMOVE_HINTS if h.isascii()]):
            remove_list.append(p)
        elif any(h in p for h in ADD_HINTS) or any(h in low for h in [h for h in ADD_HINTS if h.isascii()]):
            add_list.append(p)
    _log(f"规则回退拆分：add={len(add_list)}，remove={len(remove_list)}")
    return {"add": add_list, "remove": remove_list}


def extract_keywords(task: str,
                     num_add: int = NUM_KEYWORDS_ADD_DEFAULT,
                     num_remove: int = NUM_KEYWORDS_REMOVE_DEFAULT) -> Dict[str, List[str]]:
    """
    使用 LLM 将反馈拆分为关键词，再用 embedding 进行相关度排序与截断。
    返回 {"add": [...], "remove": [...]} 并写入 KEYWORD_FILE。
    """
    feedback = _read_feedback_text()
    _log(f"任务: {task}，目标关键词数 add={num_add}，remove={num_remove}")

    # 1) LLM 拆分
    llm_res = _llm_extract_keywords(feedback, task, num_add, num_remove)
    add_raw = [x for x in llm_res.get("add", []) if x]
    rem_raw = [x for x in llm_res.get("remove", []) if x]
    _log(f"原始候选：add={len(add_raw)}，remove={len(rem_raw)}")

    # 2) 用 embedding 与反馈相似度排序
    fb_emb = _embed(feedback)
    if fb_emb:
        _log("已获取反馈文本向量，开始相似度排序与截断")
    else:
        _log("反馈文本未获取到向量，相似度排序将退化为原始顺序去重截断")

    def _rank_and_cut(cands: List[str], topn: int) -> List[str]:
        if topn <= 0:
            return []
        scored: List[Tuple[str, float]] = []
        for c in cands:
            c_emb = _embed(c)
            s = _cosine(fb_emb, c_emb)
            scored.append((c, float(s)))
        scored.sort(key=lambda x: x[1], reverse=True)
        # 去重、截断
        uniq = []
        seen = set()
        for term, _s in scored:
            t = term.strip()
            if not t or t.lower() in seen:
                continue
            seen.add(t.lower())
            uniq.append(t)
            if len(uniq) >= topn:
                break
        return uniq

    add_keywords = _rank_and_cut(add_raw, num_add)
    remove_keywords = _rank_and_cut(rem_raw, num_remove)

    result = {"add": add_keywords, "remove": remove_keywords}
    _log(f"最终关键词：add={len(add_keywords)}，remove={len(remove_keywords)}")

    os.makedirs(os.path.dirname(KEYWORD_FILE), exist_ok=True)
    with open(KEYWORD_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    _log(f"关键词已写入: {KEYWORD_FILE}")

    return result


if __name__ == "__main__":
    # Default CLI run: try to read task from inference result if present
    task = "drugrec"
    _log("开始关键词提取流程")
    extract_keywords(task)
    _log("关键词提取流程完成")