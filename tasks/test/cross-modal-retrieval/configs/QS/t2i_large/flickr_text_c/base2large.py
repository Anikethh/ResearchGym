import os
import yaml

# 获取当前目录下所有的.yaml 文件
yaml_files = [f for f in os.listdir('.') if f.endswith('.yaml')]

for file in yaml_files:
    # 读取 YAML 文件
    with open(file, 'r') as f:
        lines = f.readlines()

    # 找到包含 pretrained 的行并修改其值
    for i, line in enumerate(lines):
        if 'pretrained:' in line and line.strip() == "pretrained: './weights/model_base_retrieval_flickr.pth'":
            lines[i] = "pretrained: './weights/model_large_retrieval_flickr.pth'\n"

        if 'vit:' in line and line.strip() == "vit: 'base'":
            lines[i] = "vit: 'large'\n"

        if 'vit_ckpt_layer:' in line and line.strip() == "vit_ckpt_layer: 4":
            lines[i] = "vit_ckpt_layer: 10\n"

    # 将修改后的内容写回文件
    with open(file, 'w') as f:
        f.writelines(lines)