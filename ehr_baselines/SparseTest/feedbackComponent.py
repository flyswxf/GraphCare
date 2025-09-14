# ===== 用户反馈 -> 节点增删改 =====

def _last_active_visit_index(vpn: torch.Tensor) -> int:
    """返回 visit_padded_node 中最后一个非空就诊的行索引；若全为空则返回 0。
    vpn 形状为 (max_visit, max_nodes)。
    """
    if vpn.ndim != 2:
        return 0
    with torch.no_grad():
        sums = vpn.sum(dim=1)
        nonzero = torch.where(sums > 0)[0]
        if nonzero.numel() == 0:
            return 0
        return int(nonzero.max().item())

def parse_feedback_to_actions(feedback_text: str):
    """解析简单自然语言/指令为节点增删动作列表。
    支持格式：
    - "+123", "-45"（正负号+节点ID）
    - "添加123", "加上123", "加入123", "增加123"
    - "删除456", "去掉456", "移除456", "排除456"
    - "add 123", "remove 456"
    返回: [(op, node_id), ...]，op 为 "+" 或 "-"。
    """
    text = feedback_text.strip()
    actions = []

    # 1) 解析 +N / -N
    for sign, num in re.findall(r"([+\-])\s*(\d+)", text):
        actions.append((sign, int(num)))

    # 2) 解析中英文动词 + 数字
    add_words = ["添加", "加上", "加入", "增加", "include", "add"]
    del_words = ["删除", "去掉", "移除", "排除", "exclude", "remove"]
    for w in add_words:
        for num in re.findall(fr"{w}\s*(\d+)", text, flags=re.IGNORECASE):
            actions.append(("+", int(num)))
    for w in del_words:
        for num in re.findall(fr"{w}\s*(\d+)", text, flags=re.IGNORECASE):
            actions.append(("-", int(num)))

    # 去重但保留顺序
    seen = set()
    dedup = []
    for a in actions:
        if a not in seen:
            dedup.append(a)
            seen.add(a)
    return dedup

def apply_user_actions_to_patient(patient: dict, actions, max_nodes: int):
    """对单个 patient 字典应用增删节点动作，保持 node_set / ehr_node_set / visit_padded_node 一致性。
    注意：会就地修改 patient。
    """
    if not actions:
        return patient

    # 备份，防止删空
    old_node_set = list(patient.get('node_set', []))
    old_ehr = patient.get('ehr_node_set', None)
    old_vpn = patient.get('visit_padded_node', None)

    # 确保 tensor 类型
    if isinstance(patient['ehr_node_set'], np.ndarray):
        patient['ehr_node_set'] = torch.tensor(patient['ehr_node_set'])
    if isinstance(patient['visit_padded_node'], np.ndarray):
        patient['visit_padded_node'] = torch.tensor(patient['visit_padded_node'])

    node_set = set(int(x) for x in patient.get('node_set', []))
    ehr_vec = patient['ehr_node_set'].clone()
    vpn = patient['visit_padded_node'].clone()

    # 动作执行
    for op, nid in actions:
        if not (0 <= int(nid) < max_nodes):
            continue
        if op == "+":
            node_set.add(int(nid))
            if ehr_vec.shape[0] == max_nodes:
                ehr_vec[int(nid)] = 1
            # 将该节点标到“最近一次就诊”上
            last_idx = _last_active_visit_index(vpn)
            if vpn.shape[1] == max_nodes:
                vpn[last_idx, int(nid)] = 1
        elif op == "-":
            if int(nid) in node_set:
                node_set.remove(int(nid))
            if ehr_vec.shape[0] == max_nodes:
                ehr_vec[int(nid)] = 0
            # 从所有就诊中清除该节点
            if vpn.shape[1] == max_nodes:
                vpn[:, int(nid)] = 0

    # 防止删空
    if len(node_set) == 0:
        patient['node_set'] = old_node_set
        if old_ehr is not None:
            patient['ehr_node_set'] = old_ehr
        if old_vpn is not None:
            patient['visit_padded_node'] = old_vpn
        return patient

    # 写回
    patient['node_set'] = list(sorted(node_set))
    patient['ehr_node_set'] = ehr_vec
    patient['visit_padded_node'] = vpn
    return patient


def recompute_with_feedback(patient_id: str, feedback_text: str = None, topk: int = 5):
    """基于用户自然语言反馈对指定 patient 调整节点后，立刻走一遍前向，返回新预测。
    - patient_id: 与 sample_dataset[i]['patient_id'] 对应
    - feedback_text: 自然语言，如 "+123, -456" 或 "删除789, 添加321"
    返回：包含 logits、prob、以及若任务为 drugrec 则返回 topk 索引。
    """
    # 查找样本索引
    idx = None
    for i, p in enumerate(sample_dataset):
        if str(p.get('patient_id')) == str(patient_id):
            idx = i
            break
    if idx is None:
        raise ValueError(f"patient_id {patient_id} 未找到")

    # 应用反馈
    if feedback_text and feedback_text.strip():
        actions = parse_feedback_to_actions(feedback_text)
        apply_user_actions_to_patient(sample_dataset[idx], actions, max_nodes=max_nodes)

    # 取子图并推理
    data = get_subgraph(G_tg, sample_dataset, task, idx)
    data = data.to(device)

    node_ids = data.y
    rel_ids = data.relation
    edge_index = data.edge_index
    batch = data.batch

    # 单样本 reshape
    visit_node = data.visit_padded_node.reshape(1, -1, data.visit_padded_node.shape[1]).float()
    ehr_nodes_vec = data.ehr_nodes.reshape(1, -1).float()

    model.eval()
    with torch.no_grad():
        out = model(
            node_ids=node_ids,
            rel_ids=rel_ids,
            edge_index=edge_index,
            batch=batch,
            visit_node=visit_node,
            ehr_nodes=ehr_nodes_vec,
            in_drop=False,
        )
        logits = out[0] if isinstance(out, tuple) else out

        if mode == "binary":
            prob = torch.sigmoid(logits)
        elif mode in ("multilabel", "multiclass"):
            prob = torch.sigmoid(logits) if mode == "multilabel" else F.softmax(logits, dim=-1)
        else:
            prob = logits

    result = {
        "logits": logits.detach().cpu().numpy(),
        "prob": prob.detach().cpu().numpy(),
    }

    if task == "drugrec":
        # 返回 topk 建议（基于概率）
        k = min(topk, prob.shape[-1])
        topv, topi = torch.topk(prob.view(-1), k)
        result.update({
            "topk_indices": topi.detach().cpu().numpy().tolist(),
            "topk_scores": topv.detach().cpu().numpy().tolist(),
        })

    return result