import json
import random


input_file = './trip_planning.json'
output_file = './trip_planning_s.jsonl'


with open(input_file, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)


all_keys = list(data.keys())

# 随机抽取50个样本（如果不足50个则全取）
sample_size = 50
sampled_keys = random.sample(all_keys, sample_size) if len(all_keys) >= sample_size else all_keys


with open(output_file, 'w', encoding='utf-8') as jsonl_file:
    for key in sampled_keys:
        value = data[key]
        if isinstance(value, dict):

            value['id'] = key
            jsonl_file.write(json.dumps(value, ensure_ascii=False) + '\n')
        else:

            record = {"id": key, "value": value}
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + '\n')

print(f"已将 {input_file} 中随机抽取的 {len(sampled_keys)} 个样本转换为 {output_file}")
