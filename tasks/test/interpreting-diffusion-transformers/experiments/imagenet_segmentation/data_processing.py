"""
    Code for processing the segmentation dataset into a better format. 
"""
import os
import h5py
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
from torch.utils import data
from nltk.corpus import wordnet as wn

def get_english_name_for_index(h5py_file, index):
    id_bytes = h5py_file[h5py_file['/value/id'][index, 0]]
    synset_code = b"".join(id_bytes).decode('utf-16').strip()
    pos = synset_code[0] # First character indicates POS ('n' for noun, 'v' for verb, etc.)
    offset = int(synset_code[1:].split('_')[0])  # Remaining part is the numeric offset
    synset = wn.synset_from_pos_and_offset(pos, offset) # Get the synset object
    english_name = synset.lemmas()[0].name().replace('_', ' ')

    return english_name

def process_dataset(
    directory: str="data/imagenet_segmentation",
):
    # Make the files
    if not os.path.exists(f"{directory}"):
        os.makedirs(f"{directory}")
    if not os.path.exists(f"{directory}/images"):
        os.makedirs(f"{directory}/images")
    if not os.path.exists(f"{directory}/segmentation_masks"):
        os.makedirs(f"{directory}/segmentation_masks")
    # Load the imagenet class map json
    with open(f"data/imagenet_class_map.json", "r") as f:
        imagenet_class_map = json.load(f)
        imagenet_class_to_simplified_name = imagenet_class_map["categories"]
    # Make a pandas dataframe
    df = pd.DataFrame(
        columns=["image_path", "segmentation_mask_path", "simplified_name"]
    )
    # Load the .mat file 
    h5py_file = h5py.File("data/gtsegs_ijcv.mat", "r")
    # Iterate through the data
    for index in range(h5py_file['/value/id'].shape[0]):
        # print(
        #     np.array(h5py_file[h5py_file['/value/target'][index, 0]])
        # )
        # Load the image
        img = np.array(
            h5py_file[
                h5py_file['/value/img'][index, 0]
            ]
        ).transpose((2, 1, 0))
        # Load the target segmentation
        target = np.array(
            h5py_file[h5py_file[h5py_file['/value/gt'][index, 0]][0, 0]]
        ).transpose((1, 0))
        # Get the english name
        english_name = get_english_name_for_index(h5py_file, index)
        print(f"English name: {english_name}")
        # Get the simplified name
        simplified_name = imagenet_class_to_simplified_name[english_name]
        # Save the image
        img_path = f"{directory}/images/{index}.png"
        Image.fromarray(img).save(img_path)
        # Save the target segmentation
        target_path = f"{directory}/segmentation_masks/{index}.png"
        Image.fromarray(target).save(target_path)
        # Add the row to the pandas dataframe
        df = pd.concat([
            df,
            pd.DataFrame(
                {
                    "image_path": [img_path],
                    "segmentation_mask_path": [target_path],
                    "imagenet_name": [english_name],
                    # "imagenet_class_index": [index],
                    "simplified_name": [simplified_name]
                },
                index=[index]
            )
        ])
        # Save the pandas data frame 
        df.to_csv(f"{directory}/data.csv")

class ImagenetSegmentation(data.Dataset):
    CLASSES = 2

    def __init__(
        self,
        directory: str="data/imagenet_segmentation",
        transform=None,
        target_transform=None
    ):
        self.directory = directory
        
        if not os.path.exists(f"{self.directory}/data.csv"):
            process_dataset(directory=self.directory)
        # Load the csv as a dataframe
        self.df = pd.read_csv(f"{self.directory}/data.csv")
        self.data_length = len(self.df)

    def __getitem__(self, index):
        # Load the image file
        img = Image.open(f"{self.directory}/images/{index}.png").convert("RGB")
        # Load the target segmentation file
        target = Image.open(f"{self.directory}/segmentation_masks/{index}.png")
        # Load the simplified name
        simplified_name = self.df.iloc[index]["simplified_name"]

        return img, target, simplified_name

    def __len__(self):
        return self.data_length

if __name__ == "__main__":
    process_dataset()