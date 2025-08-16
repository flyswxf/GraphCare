"""
DynamicEHR Data Preparation Module
事件流数据预处理 - 将EHR数据转换为连续时间事件序列
"""
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import pandas as pd
from collections import defaultdict
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatientEvent:
    """单个患者事件的数据结构"""
    def __init__(self, timestamp: float, concepts: List[int], 
                 concept_types: List[str], vitals: Optional[Dict[str, float]] = None):
        self.timestamp = timestamp  # 连续时间戳(小时或天)
        self.concepts = concepts    # 医疗概念ID列表
        self.concept_types = concept_types  # 概念类型：'diag', 'proc', 'drug'
        self.vitals = vitals or {}  # 生理指标(可选)


class ConceptVocabulary:
    """统一概念词表，将诊断/手术/用药映射到同一空间"""
    def __init__(self):
        self.concept_to_id = {}
        self.id_to_concept = {}
        self.type_to_id = {'diag': 0, 'proc': 1, 'drug': 2}
        self.concept_counts = defaultdict(int)
        self.next_id = 1  # 保留0作为padding
        
    def add_concept(self, concept: str, concept_type: str) -> int:
        """添加概念并返回ID"""
        key = f"{concept_type}:{concept}"
        if key not in self.concept_to_id:
            self.concept_to_id[key] = self.next_id
            self.id_to_concept[self.next_id] = key
            self.next_id += 1
        self.concept_counts[key] += 1
        return self.concept_to_id[key]
    
    def get_id(self, concept: str, concept_type: str) -> int:
        """获取概念ID"""
        key = f"{concept_type}:{concept}"
        return self.concept_to_id.get(key, 0)  # 未知概念返回0
    
    def __len__(self):
        return self.next_id


class EventStreamDataset(Dataset):
    """事件流数据集"""
    def __init__(self, patient_events: List[List[PatientEvent]], 
                 labels: List[Dict], task: str = 'mortality'):
        self.patient_events = patient_events
        self.labels = labels
        self.task = task
        
    def __len__(self):
        return len(self.patient_events)
    
    def __getitem__(self, idx):
        events = self.patient_events[idx]
        label = self.labels[idx]
        
        # 提取事件时间序列
        event_times = [e.timestamp for e in events]
        
        # 扁平化所有概念
        concepts_flat = []
        type_ids_flat = []
        event_ptr = [0]
        
        for event in events:
            concepts_flat.extend(event.concepts)
            type_ids_flat.extend([self._get_type_id(t) for t in event.concept_types])
            event_ptr.append(len(concepts_flat))
        
        return {
            'event_times': torch.tensor(event_times, dtype=torch.float32),
            'concepts_flat': torch.tensor(concepts_flat, dtype=torch.long),
            'type_ids_flat': torch.tensor(type_ids_flat, dtype=torch.long),
            'event_ptr': torch.tensor(event_ptr, dtype=torch.long),
            'label': self._process_label(label)
        }
    
    def _get_type_id(self, concept_type: str) -> int:
        return {'diag': 0, 'proc': 1, 'drug': 2}.get(concept_type, 0)
    
    def _process_label(self, label: Dict):
        """根据任务处理标签"""
        if self.task == 'mortality':
            return torch.tensor(label.get('mortality', 0), dtype=torch.float32)
        elif self.task == 'readmission':
            return torch.tensor(label.get('readmission', 0), dtype=torch.float32)
        elif self.task == 'drugrec':
            # 多标签分类
            return torch.tensor(label.get('drugs', []), dtype=torch.float32)
        elif self.task == 'los':
            return torch.tensor(label.get('length_of_stay', 0.0), dtype=torch.float32)
        return torch.tensor(0, dtype=torch.float32)


def collate_event_batch(batch: List[Dict]) -> Dict:
    """批处理函数，将多个患者的事件流合并"""
    batch_size = len(batch)
    
    # 合并所有患者的事件时间
    batch_event_times = []
    batch_concepts_flat = []
    batch_type_ids_flat = []
    batch_event_ptr = [0]
    patient_ptr = [0]
    
    labels = []
    
    for i, sample in enumerate(batch):
        # 累积事件指针偏移
        event_offset = len(batch_concepts_flat)
        
        batch_event_times.extend(sample['event_times'].tolist())
        batch_concepts_flat.extend(sample['concepts_flat'].tolist())
        batch_type_ids_flat.extend(sample['type_ids_flat'].tolist())
        
        # 调整事件指针
        event_ptr = sample['event_ptr'].tolist()
        for j in range(1, len(event_ptr)):
            batch_event_ptr.append(event_ptr[j] + event_offset)
        
        # 病人边界
        patient_ptr.append(len(batch_event_times))
        labels.append(sample['label'])
    
    return {
        'batch_event_times': torch.tensor(batch_event_times, dtype=torch.float32),
        'batch_concepts_flat': torch.tensor(batch_concepts_flat, dtype=torch.long),
        'batch_type_ids_flat': torch.tensor(batch_type_ids_flat, dtype=torch.long),
        'batch_event_ptr': torch.tensor(batch_event_ptr, dtype=torch.long),
        'patient_ptr': torch.tensor(patient_ptr, dtype=torch.long),
        'labels': torch.stack(labels) if len(labels) > 0 and labels[0].numel() == 1 
                 else torch.tensor(labels, dtype=torch.float32)
    }


class EHRDataProcessor:
    """EHR数据处理器 - 从原始数据转换为事件流"""
    
    def __init__(self, time_unit='hour'):
        self.vocab = ConceptVocabulary()
        self.time_unit = time_unit  # 'hour' 或 'day'
        
    def process_mimic_data(self, sample_dataset: List[Dict], 
                          save_path: Optional[str] = None) -> Tuple[List[List[PatientEvent]], List[Dict]]:
        """
        处理MIMIC数据格式
        输入: GraphCare风格的sample_dataset
        输出: 事件流格式的数据
        """
        logger.info("开始处理MIMIC数据...")
        
        patient_events = []
        labels = []
        
        for patient_data in sample_dataset:
            # 提取患者事件序列
            events = self._extract_patient_events(patient_data)
            if len(events) == 0:
                continue
                
            patient_events.append(events)
            
            # 提取标签
            label = self._extract_labels(patient_data)
            labels.append(label)
        
        logger.info(f"处理完成: {len(patient_events)}个患者, 词表大小: {len(self.vocab)}")
        
        if save_path:
            self._save_processed_data(patient_events, labels, save_path)
            
        return patient_events, labels
    
    def _extract_patient_events(self, patient_data: Dict) -> List[PatientEvent]:
        """从单个患者数据提取事件序列"""
        events = []
        
        conditions = patient_data.get('conditions', [])
        procedures = patient_data.get('procedures', [])
        drugs = patient_data.get('drugs', [])
        
        # 假设visit级别的时间信息(需要根据实际数据调整)
        n_visits = len(conditions)
        
        for visit_idx in range(n_visits):
            # 构造虚拟时间戳 (实际应从数据中提取)
            timestamp = float(visit_idx * 24)  # 假设每次访问间隔24小时
            
            # 收集该visit的所有概念
            visit_concepts = []
            visit_types = []
            
            # 诊断
            if visit_idx < len(conditions):
                for cond in conditions[visit_idx]:
                    concept_id = self.vocab.add_concept(str(cond), 'diag')
                    visit_concepts.append(concept_id)
                    visit_types.append('diag')
            
            # 手术
            if visit_idx < len(procedures):
                for proc in procedures[visit_idx]:
                    concept_id = self.vocab.add_concept(str(proc), 'proc')
                    visit_concepts.append(concept_id)
                    visit_types.append('proc')
            
            # 用药
            if visit_idx < len(drugs):
                for drug in drugs[visit_idx]:
                    concept_id = self.vocab.add_concept(str(drug), 'drug')
                    visit_concepts.append(concept_id)
                    visit_types.append('drug')
            
            if len(visit_concepts) > 0:
                event = PatientEvent(
                    timestamp=timestamp,
                    concepts=visit_concepts,
                    concept_types=visit_types
                )
                events.append(event)
        
        # 按时间排序
        events.sort(key=lambda x: x.timestamp)
        return events
    
    def _extract_labels(self, patient_data: Dict) -> Dict:
        """提取任务标签"""
        return {
            'mortality': patient_data.get('label', 0),
            'readmission': patient_data.get('readmission_label', 0),
            'length_of_stay': patient_data.get('los_days', 0.0),
            'drugs': patient_data.get('future_drugs', [])
        }
    
    def _save_processed_data(self, patient_events: List[List[PatientEvent]], 
                           labels: List[Dict], save_path: str):
        """保存处理后的数据"""
        data = {
            'patient_events': patient_events,
            'labels': labels,
            'vocab': self.vocab
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"数据已保存到: {save_path}")


def load_processed_data(data_path: str) -> Tuple[List[List[PatientEvent]], List[Dict], ConceptVocabulary]:
    """加载预处理的数据"""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    return data['patient_events'], data['labels'], data['vocab']


def create_dataloader(patient_events: List[List[PatientEvent]], 
                     labels: List[Dict], 
                     task: str = 'mortality',
                     batch_size: int = 32,
                     shuffle: bool = True) -> DataLoader:
    """创建DataLoader"""
    dataset = EventStreamDataset(patient_events, labels, task)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_event_batch
    )


# 使用示例
if __name__ == "__main__":
    # 示例：处理数据
    processor = EHRDataProcessor()
    
    # 模拟数据 (实际使用时替换为真实数据)
    sample_data = [
        {
            'conditions': [['250.00', '401.9'], ['250.01']],
            'procedures': [['88.72'], ['99.04']],
            'drugs': [['A10AB01'], ['A10AB01', 'C03AA03']],
            'label': 1
        }
    ]
    
    # 处理数据
    patient_events, labels = processor.process_mimic_data(
        sample_data, 
        save_path='./DynamicEHR/processed_data.pkl'
    )
    
    # 创建DataLoader
    dataloader = create_dataloader(patient_events, labels, task='mortality', batch_size=2)
    
    # 测试批处理
    for batch in dataloader:
        print("批处理数据形状:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value)}")
        break