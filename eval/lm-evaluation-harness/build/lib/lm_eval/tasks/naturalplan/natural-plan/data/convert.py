import json

def convert_json(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    data_list = list(raw_data.values())

    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=2, ensure_ascii=False)
    
    print(f"转换完成，结果保存在 {output_file}")

if __name__ == '__main__':
    input_file = './trip_planning.json'
    output_file = './trip_planning_new.json'
    convert_json(input_file, output_file)