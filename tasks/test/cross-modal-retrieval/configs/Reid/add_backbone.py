import os

def add_backbone_to_yaml(file_path, backbone_line="backbone: 'clip_reid'"):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 检查最后一行是否已经包含 'backbone:'
    if lines and lines[-1].strip() == backbone_line:
        return  # 如果已经有相同的最后一行，直接返回

    # 在文件最后添加 backbone: 'clip_reid'
    with open(file_path, 'a') as f:
        f.write(f"\n{backbone_line}\n")

def process_directory(root_dir):
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.yaml'):
                file_path = os.path.join(subdir, file)
                add_backbone_to_yaml(file_path)

if __name__ == "__main__":
    current_directory = os.getcwd()  # 获取当前工作目录
    process_directory(current_directory)
