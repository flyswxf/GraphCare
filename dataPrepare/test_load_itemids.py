import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataPrepare.chartevents_filter_prepare import load_itemids

# 测试CSV格式的itemids文件
csv_path = './dataPrepare/match_stats/itemids.csv'
itemids = load_itemids(csv_path)
print(f'从CSV文件加载的itemids数量: {len(itemids)}')
print(f'加载的itemids: {sorted(list(itemids))}')

# 测试TXT格式的itemids文件
txt_path = './dataPrepare/match_stats/itemids.txt'
itemids_txt = load_itemids(txt_path)
print(f'\n从TXT文件加载的itemids数量: {len(itemids_txt)}')
print(f'加载的itemids: {sorted(list(itemids_txt))}')