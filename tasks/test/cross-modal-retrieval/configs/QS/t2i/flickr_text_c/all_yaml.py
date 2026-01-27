import yaml
import os

# 原始YAML文件路径
original_yaml_path = 'retrieval_flickr.yaml'

replacements = {
    'annotation_active': ['1'],
    'annotation_back_trans': ['1'],
    'annotation_casual': ['1'],
    'annotation_formal': ['1'],
    'annotation_ip': ['6', '7'],
    'annotation_KeyboardAug': ['1', '2', '3', '4', '5', '6', '7'],
    'annotation_OcrAug': ['1', '2', '3', '4', '5', '6', '7'],
    'annotation_passive': ['1'],
    'annotation_RandomCharAug_delete': ['1', '2', '3', '4', '5', '6', '7'],
    'annotation_RandomCharAug_insert': ['1', '2', '3', '4', '5', '6', '7'],
    'annotation_RandomCharAug_substitute': ['1', '2', '3', '4', '5', '6', '7'],
    'annotation_RandomCharAug_swap': ['1', '2', '3', '4', '5', '6', '7'],
    'annotation_rd': ['6', '7'],
    'annotation_ri': ['6', '7'],
    'annotation_rs': ['6', '7'],
    'annotation_sr': ['6', '7']
}

with open(original_yaml_path, 'r') as file:
    yaml_str = file.read()

# 遍历替换规则，生成新的YAML文件
for annotation_type, sequences in replacements.items():
    if isinstance(sequences, list):  # 如果sequences是一个列表
        for sequence in sequences:
            # 构建新的ann_root路径
            new_ann_root = f'{annotation_type}/{sequence}/'

            # 创建新的YAML内容
            new_yaml_str = yaml_str.replace('flickr_annotations/', new_ann_root)

            # 构建新的YAML文件名
            new_yaml_filename = f"{annotation_type}_{sequence}.yaml"

            # 写入新的YAML文件
            with open(new_yaml_filename, 'w') as new_file:
                new_file.write(new_yaml_str)

            print(f"Generated {new_yaml_filename}")
    else:  # 如果sequences不是一个列表（只有一个值）
        # 这里可以添加处理单个序列值的逻辑
        pass