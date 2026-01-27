import os
import yaml

"""
Helper script for batch-updating YAML config paths.
Edit folder_path and replacements before use.
"""

folder_path = "./configs"

replacements = {
    # "<absolute/old/path>": "<relative/new/path>",
}

def batch_replace_yaml(folder: str, mapping: dict):
    for filename in os.listdir(folder):
        if filename.endswith(".yaml"):
            file_path = os.path.join(folder, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            for old, new in mapping.items():
                content = content.replace(old, new)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Updated {filename}")

if __name__ == "__main__":
    batch_replace_yaml(folder_path, replacements)
