# Description:
#   This script generates a JSON manifest file for the Celeb-DF-v2 dataset.
#   It reads the official testing video list, scans for corresponding preprocessed
#   frame files, and creates a JSON file in the format expected by the
#   DeepfakeBench data loader (abstract_dataset.py).

import os
import json
from pathlib import Path
from tqdm import tqdm
import glob

# ==============================================================================
# --- Paths to Configure ---
# ==============================================================================

# 1. Set the full path to the original 'List_of_testing_videos.txt' file.
#    This file is part of the original Celeb-DF-v2 dataset download.
TXT_FILE_PATH = Path("/path/to/List_of_testing_videos.txt")

# 2. Set the full path to the TOP-LEVEL directory of your PREPROCESSED dataset.
#    This directory should contain the 'Celeb-real', 'Celeb-synthesis', etc. folders.
PREPROCESSED_DATA_ROOT = Path("path/to/training/dataset/Celeb-DF-v2")

# ==============================================================================

def main():
    """
    Main function to generate the dataset JSON manifest.
    """
    # --- Output Path Configuration ---
    # This script should be run from the 'training/' directory.
    # The output JSON will be saved to '../preprocessing/dataset_json/',
    # which is the location expected by the grade.sh script.
    output_dir = Path("../preprocessing/dataset_json/")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json_path = output_dir / "Celeb-DF-v2.json"

    # --- JSON Structure Initialization ---
    # This is the nested dictionary structure required by abstract_dataset.py.
    final_json_structure = {
        "Celeb-DF-v2": {
            "Celeb-real": {"train": {}, "test": {}},
            "Celeb-synthesis": {"train": {}, "test": {}},
            "YouTube-real": {"train": {}, "test": {}}
        }
    }

    # --- Label Mapping ---
    # Maps dataset folder names to the specific label keys found in test_config.yaml.
    label_mapping = {
        "Celeb-real": "CelebDFv2_real",
        "Celeb-synthesis": "CelebDFv2_fake",
        "YouTube-real": "CelebDFv2_real"
    }

    # --- File Processing ---
    print(f"Reading test video list from: '{TXT_FILE_PATH}'")
    if not TXT_FILE_PATH.exists():
        raise FileNotFoundError(f"Error: The file '{TXT_FILE_PATH}' was not found. Please check the path.")

    with open(TXT_FILE_PATH, 'r') as f:
        test_video_lines = f.readlines()

    print(f"Found {len(test_video_lines)} videos in the test list.")
    print(f"Scanning for preprocessed frames in: '{PREPROCESSED_DATA_ROOT}'")

    for line in tqdm(test_video_lines, desc="Generating JSON Manifest"):
        parts = line.strip().split()
        if len(parts) != 2:
            continue

        _, video_path_str = parts
        video_path = Path(video_path_str)
        
        subfolder_name = video_path.parent.name
        video_id = video_path.stem

        # Construct the path to the folder containing the preprocessed frames
        frames_folder_path = PREPROCESSED_DATA_ROOT / subfolder_name / "frames" / video_id
        
        if not frames_folder_path.is_dir():
            continue

        # Find all .png files in the directory
        frame_files = sorted(glob.glob(os.path.join(frames_folder_path, '*.png')))
        
        if not frame_files:
            continue

        # Create relative paths as expected by the data loader
        relative_frame_paths = [
            str(Path("Celeb-DF-v2") / subfolder_name / "frames" / video_id / Path(f).name)
            for f in frame_files
        ]

        # Get the correct label string from our mapping
        correct_label_str = label_mapping.get(subfolder_name)
        if not correct_label_str:
            print(f"Warning: No label mapping found for '{subfolder_name}'. Skipping.")
            continue
        
        video_entry = {
            "label": correct_label_str,
            "frames": relative_frame_paths
        }
        
        final_json_structure["Celeb-DF-v2"][subfolder_name]["test"][video_id] = video_entry

    # --- Save the final JSON file ---
    with open(output_json_path, 'w') as f:
        json.dump(final_json_structure, f)

    print(f"\nProcess complete. Manifest file saved to: '{output_json_path}'")
    # Check if the generated file has content
    if os.path.getsize(output_json_path) < 200: # An empty structure is ~130 bytes
         print("\nWARNING: The generated JSON file appears to be empty or nearly empty.")
         print("Please double-check that the 'PREPROCESSED_DATA_ROOT' path is correct.")

if __name__ == "__main__":
    main()