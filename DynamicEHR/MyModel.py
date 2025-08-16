"""
DynamicEHR Model - Minimal Prototype
事件驱动记忆模型（EDM）+ 事件内概念交互混合器（EIM）+ 时间/状态编码
"""
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEncoder(nn.Module):
    """连续时间编码（最小原型：对数时间+线性层）"""
    def __init__(self, d_time: int):
        super().__init__()
        self.proj = nn.Linear(1, d_time)

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        # dt: [E] 事件间隔（已按序）
        dt = torch.log1p(torch.clamp(dt, min=0.0))
        return self.proj(dt.unsqueeze(-1))  # [E, d_time]


class StateChangeEncoder(nn.Module):
    """状态变化编码（最小原型：标量 -> 线性）"""
    def __init__(self, d_state: int):
        super().__init__()
        self.proj = nn.Linear(1, d_state)

    def forward(self, dstate: torch.Tensor) -> torch.Tensor:
        # dstate: [E] 事件状态变化幅度（如集合差异率）
        return self.proj(dstate.unsqueeze(-1))


class EventInteractionMixer(nn.Module):
    """事件内概念交互混合器（最小原型：均值池化）"""
    def __init__(self):
        super().__init__()

    def forward(self, concept_embeds: torch.Tensor) -> torch.Tensor:
        # concept_embeds: [N_concepts_in_event, d_embed]
        if concept_embeds.size(0) == 0:
            # 边界情况
            return torch.zeros((1, concept_embeds.size(1)), device=concept_embeds.device)
        return concept_embeds.mean(dim=0, keepdim=True)  # [1, d_embed]


class EventDrivenMemory(nn.Module):
    """事件驱动记忆（EDM）：指数时间衰减 + GRU更新"""
    def __init__(self, d_in: int, d_mem: int):
        super().__init__()
        self.d_mem = d_mem
        self.gru = nn.GRUCell(d_in, d_mem)
        self.gamma = nn.Parameter(torch.tensor(0.1))  # 时间衰减率

    def forward(self, event_repr: torch.Tensor, dt: torch.Tensor, patient_ptr: torch.Tensor) -> torch.Tensor:
        """
        event_repr: [E, d_in]
        dt: [E] 按事件顺序的时间间隔（不同患者拼接）
        patient_ptr: [B+1] 病人边界，指向事件索引
        返回：每个病人的最终记忆 [B, d_mem]
        """
        B = patient_ptr.numel() - 1
        m = torch.zeros((B, self.d_mem), device=event_repr.device)
        out = torch.zeros((B, self.d_mem), device=event_repr.device)
        
        for b in range(B):
            s, e = patient_ptr[b].item(), patient_ptr[b+1].item()
            h = torch.zeros((self.d_mem,), device=event_repr.device)
            for i in range(s, e):
                # 连续时间衰减
                h = h * torch.exp(-self.gamma * dt[i])
                h = self.gru(event_repr[i], h)
            out[b] = h
        return out


class MyModel(nn.Module):
    """DynamicEHR: 事件驱动的EHR时序模型（最小原型）"""
    def __init__(self, vocab_size: int, d_embed: int = 128, d_time: int = 16, d_state: int = 16, d_mem: int = 128,
                 num_types: int = 3, task: str = 'mortality'):
        super().__init__()
        self.task = task
        
        # 概念嵌入与类型嵌入（简单相加）
        self.concept_emb = nn.Embedding(vocab_size, d_embed)
        self.type_emb = nn.Embedding(num_types, d_embed)
        
        self.time_enc = TimeEncoder(d_time)
        self.state_enc = StateChangeEncoder(d_state)
        
        self.eim_proj = nn.Linear(d_embed, d_embed)  # 事件内投影
        self.eim = EventInteractionMixer()
        
        # 融合时间/状态的投影
        self.fuse = nn.Linear(d_embed + d_time + d_state, d_mem)
        
        # 事件驱动记忆
        self.edm = EventDrivenMemory(d_in=d_mem, d_mem=d_mem)
        
        # 任务头
        if task in ['mortality', 'readmission']:
            self.head = nn.Linear(d_mem, 1)
        elif task == 'drugrec':
            self.head = nn.Linear(d_mem, vocab_size)  # 简单用概念空间预测药物集合
        elif task == 'los':
            self.head = nn.Linear(d_mem, 1)
        else:
            self.head = nn.Linear(d_mem, 1)

    def forward(self, batch: Dict[str, torch.Tensor]):
        """
        输入batch包含：
        - batch_event_times: [E]
        - batch_concepts_flat: [C]
        - batch_type_ids_flat: [C]
        - batch_event_ptr: [E+1]
        - patient_ptr: [B+1]
        - labels: [B] 或 [*]
        """
        event_times = batch['batch_event_times']  # [E]
        concepts = batch['batch_concepts_flat']   # [C]
        type_ids = batch['batch_type_ids_flat']   # [C]
        event_ptr = batch['batch_event_ptr']      # [E+1]
        patient_ptr = batch['patient_ptr']        # [B+1]
        
        E = event_times.size(0)
        
        # 计算相邻事件的时间差（跨患者已按patient_ptr分段）
        dt = torch.zeros_like(event_times)
        dt[1:] = event_times[1:] - event_times[:-1]
        # 对每个患者的首事件置0
        dt[patient_ptr[1:-1]] = 0.0
        dt = torch.clamp(dt, min=0.0)
        
        # 构造事件表示
        event_reprs = []
        prev_event_concept_slice = None
        
        for e in range(E):
            s, t = event_ptr[e].item(), event_ptr[e+1].item()
            ce = self.concept_emb(concepts[s:t]) + self.type_emb(type_ids[s:t])  # [Ne, d]
            ce = self.eim_proj(ce)
            h_e = self.eim(ce)  # [1, d]
            
            # 状态变化幅度（最小版：概念集合大小变化的绝对差）
            if e == 0 or e in patient_ptr[1:].tolist():
                dstate_val = torch.tensor(0.0, device=concepts.device)
            else:
                prev_s, prev_t = event_ptr[e-1].item(), event_ptr[e].item()
                prev_set = set(concepts[prev_s:prev_t].tolist())
                curr_set = set(concepts[s:t].tolist())
                # Jaccard距离的1-相似度可作为变化幅度
                inter = len(prev_set & curr_set)
                union = max(1, len(prev_set | curr_set))
                jaccard_dist = 1.0 - inter / union
                dstate_val = torch.tensor(jaccard_dist, device=concepts.device)
            
            time_feat = self.time_enc(dt[e:e+1])        # [1, d_time]
            state_feat = self.state_enc(dstate_val.view(1))  # [1, d_state]
            
            h_e_aug = torch.cat([h_e, time_feat, state_feat], dim=-1)  # [1, d+d_time+d_state]
            h_e_aug = self.fuse(h_e_aug)  # [1, d_mem]
            event_reprs.append(h_e_aug)
        
        event_reprs = torch.cat(event_reprs, dim=0)  # [E, d_mem]
        
        # 事件驱动记忆 -> 病人级表示
        patient_repr = self.edm(event_reprs, dt, patient_ptr)  # [B, d_mem]
        
        # 任务头
        logits = self.head(patient_repr)
        
        return logits.squeeze(-1)