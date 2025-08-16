"""
DynamicEHR 训练脚本 - 最小原型
- 演示如何加载数据（占位符），构建模型并进行一次前向/训练循环
"""
import argparse
from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim

from MyModel import MyModel


def dummy_batch(B: int = 2, E: int = 5, C: int = 12, vocab_size: int = 100, device: str = 'cpu') -> Dict[str, torch.Tensor]:
    # 构造一个模拟batch，满足MyModel.forward的输入契约
    torch.manual_seed(42)
    batch_event_times = torch.sort(torch.rand(E))[0].to(device) * 10.0
    batch_concepts_flat = torch.randint(0, vocab_size, (C,), device=device)
    batch_type_ids_flat = torch.randint(0, 3, (C,), device=device)

    # 随机把C个概念切成E个事件
    cuts = torch.sort(torch.randint(1, C-1, (E-1,)))[0].tolist()
    ptr = [0] + cuts + [C]
    batch_event_ptr = torch.tensor(ptr, device=device)

    # 将E个事件切分成B个病人
    if E >= 2:
        p_cuts = sorted(set([0, E//2, E]))
    else:
        p_cuts = [0, E]
    patient_ptr = torch.tensor(p_cuts, device=device)

    labels = torch.randint(0, 2, (len(p_cuts)-1,), dtype=torch.float32, device=device)

    return {
        'batch_event_times': batch_event_times,
        'batch_concepts_flat': batch_concepts_flat,
        'batch_type_ids_flat': batch_type_ids_flat,
        'batch_event_ptr': batch_event_ptr,
        'patient_ptr': patient_ptr,
        'labels': labels,
    }


def train_one_step(model: MyModel, batch: Dict[str, torch.Tensor], optimizer, task: str = 'mortality'):
    model.train()
    logits = model(batch)
    if task in ['mortality', 'readmission', 'los']:
        labels = batch['labels']
        loss = nn.BCEWithLogitsLoss()(logits, labels)
    elif task == 'drugrec':
        # 简化：把labels视作多标签one-hot，这里用随机占位
        labels = torch.zeros_like(logits)
        loss = nn.BCEWithLogitsLoss()(logits, labels)
    else:
        labels = batch['labels']
        loss = nn.MSELoss()(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_size', type=int, default=100)
    parser.add_argument('--task', type=str, default='mortality')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    device = args.device
    model = MyModel(vocab_size=args.vocab_size, task=args.task).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    batch = dummy_batch(device=device, vocab_size=args.vocab_size)

    loss = train_one_step(model, batch, optimizer, task=args.task)
    print(f"One training step done. Loss={loss:.4f}")

    # 推理
    model.eval()
    with torch.no_grad():
        logits = model(batch)
        if args.task in ['mortality', 'readmission']:
            probs = torch.sigmoid(logits)
            print('Pred probs:', probs.detach().cpu().numpy())
        else:
            print('Logits:', logits.detach().cpu().numpy())


if __name__ == '__main__':
    main()