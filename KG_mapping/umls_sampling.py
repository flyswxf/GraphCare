import json
import numpy as np
import pickle as pkl
import random
import os
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def load_data(filename):
    with open(filename, 'rb') as f:
        return pkl.load(f)

def two_hop_mapping(mapping_dict, umls_cuis, triple_set, save_path):
    print('Now processing ' + save_path + '...')
    
    print("retrieving first hop triples...")
    first_hop_triples = defaultdict(set)
    for key in tqdm(mapping_dict.keys()):
        umls_id = mapping_dict[key]
        # 跳过None值，这些是没有找到匹配的UMLS映射
        if umls_id is None:
            continue
        umls_cui = umls_cuis[umls_id]
        # print(umls_cui)
        for triple in triple_set:
            if umls_cui in triple:
                first_hop_triples[key].add(triple)
    
    print('saving first hop triples...')
    # 确保目录存在
    os.makedirs(save_path, exist_ok=True)
    with open(save_path + '/first_hop_triples.pkl', 'wb') as f:
        pkl.dump(first_hop_triples, f)
    
    second_hop_triples = first_hop_triples.copy()
    second_hop_nodes = defaultdict(set)
    
    print("retrieving second hop nodes...")
    for key in tqdm(first_hop_triples.keys()):
        for triple in first_hop_triples[key]:
            second_hop_nodes[key].add(triple[0]) if triple[0] not in mapping_dict.keys() else None
            second_hop_nodes[key].add(triple[2]) if triple[2] not in mapping_dict.keys() else None
    
    print("retrieving second hop triples...")
    for key in tqdm(second_hop_nodes.keys()):
        nodes = second_hop_nodes[key]
        num_samples = min(5, len(nodes))
        random_five_nodes = random.sample(nodes, num_samples)

        for node in random_five_nodes:
            for triple in triple_set:
                if node in triple:
                    second_hop_triples[key].add(triple)
                    
    print('saving second hop triples...')
    # 确保目录存在
    os.makedirs(save_path, exist_ok=True)
    with open(save_path + '/second_hop_triples.pkl', 'wb') as f:
        pkl.dump(second_hop_triples, f)
        
    # Counting the entities, relations, and triples
    entities = {triple[0] for triple_group in second_hop_triples.values() for triple in triple_group}.union(
               {triple[2] for triple_group in second_hop_triples.values() for triple in triple_group})
    relations = {triple[1] for triple_group in second_hop_triples.values() for triple in triple_group}
    triple_count = sum(len(triple_group) for triple_group in second_hop_triples.values())

    return triple_count, len(entities), len(relations)


# 获取脚本目录和项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..")
data_dir = os.path.join(project_root, "data")

ccscm2umls = load_data(os.path.join(data_dir, 'ccscm2umls.pkl'))
ccsproc2umls = load_data(os.path.join(data_dir, 'ccsproc2umls.pkl'))
atc32umls = load_data(os.path.join(data_dir, 'atc32umls.pkl'))

umls_csv_path = os.path.join(project_root, 'KG_mapping/umls/umls.csv')
concepts_path = os.path.join(project_root, 'KG_mapping/umls/concepts.txt')

with open(umls_csv_path, 'r') as f:
    lines_1 = f.readlines()

with open(concepts_path, 'r') as f:
    umls_cuis = [line.strip() for line in f.readlines()]

triple_set = {(items[1].strip(), items[0].strip(), items[2].strip()) for items in (line.split('\t') for line in lines_1)}
print("Total triples: ", len(triple_set))
# print(triple_set.pop())

def threaded_two_hop_mapping(args):
    return two_hop_mapping(*args)

with ThreadPoolExecutor(max_workers=16) as executor:
    graphs_dir = os.path.join(project_root, 'graphs')
    datasets = [
        (ccscm2umls, os.path.join(graphs_dir, 'ccscm_umls')),
        (ccsproc2umls, os.path.join(graphs_dir, 'ccsproc_umls')),
        (atc32umls, os.path.join(graphs_dir, 'atc3_umls'))
    ]

    results = list(tqdm(executor.map(threaded_two_hop_mapping, [(d[0], umls_cuis, triple_set, d[1]) for d in datasets]), total=len(datasets)))

ccscm_counts, ccsproc_counts, atc3_counts = results

total_triples = ccscm_counts[0] + ccsproc_counts[0] + atc3_counts[0]
total_entities = ccscm_counts[1] + ccsproc_counts[1] + atc3_counts[1]
total_relations = ccscm_counts[2] + ccsproc_counts[2] + atc3_counts[2]


print("CCSCM: ", ccscm_counts)
print("CCSPROC: ", ccsproc_counts)
print("ATC3: ", atc3_counts)
print("Total: ", (total_triples, total_entities, total_relations))