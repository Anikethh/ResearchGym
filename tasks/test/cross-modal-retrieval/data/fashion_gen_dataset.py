import os
import json
import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
import pickle
from PIL import Image
import numpy as np
from data.utils import pre_caption

class fashion_gen_retrieval_eval_image(Dataset):
    """加载 fashion-gen 数据集（下游任务）。

    Args:
        root (string): 数据集根目录
        args (object): 参数对象
    """
    def __init__(self, transform, root, ann_root, split, max_words=77):
        self.image_root = os.path.join(root, 'extracted_valid_images')

        self.transform = transform
        tir_root = os.path.join(root, 'full_valid_info_PAI')
        self.pkls_tir = sorted([os.path.join(tir_root, f) for f in os.listdir(tir_root)])

        self.size = len(self.pkls_tir)

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        self.id_to_data = {}

        # self.size=100
        for i in range(self.size):

            value = self.pkl_loader(self.pkls_tir[i])
            captions_list = value['captions']
            img_name = value['img_name']
            
            object_id = img_name.split('_')[0]

            if object_id not in self.id_to_data:
                self.id_to_data[object_id] = {'images': [], 'captions': []}

            self.id_to_data[object_id]['images'].append(img_name)

            if isinstance(captions_list, str):
                captions_list = [captions_list]

            # if not id_to_data[object_id]['captions']:
            self.id_to_data[object_id]['captions'].extend(captions_list)

        img_id = 0
        txt_id = 0

        for object_id, data in self.id_to_data.items():
            captions = []
            for caption in data['captions']:
                self.text.append(pre_caption(caption, max_words))
                captions.append(txt_id)
                self.txt2img[txt_id] = []
                txt_id += 1

            for img_name in data['images']:
                self.image.append(img_name)
                self.img2txt[img_id] = captions
                for cap_id in captions:
                    self.txt2img[cap_id].append(img_id)
                img_id += 1
        

    def __getitem__(self, index):

        image_path = os.path.join(self.image_root, self.image[index])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index    

    def __len__(self):
        return len(self.image)
    
    def pkl_loader(self, pkl_path):
        """从 *.pkl 加载文本"""
        with open(pkl_path, 'rb') as f:
            info_dict = pickle.load(f)
            return info_dict

class fashion_gen_retrieval_eval_text(Dataset):
    """加载 fashion-gen 数据集（下游任务）。

    Args:
        root (string): 数据集根目录
        args (object): 参数对象
    """
    def __init__(self, transform, root, ann_root, split, max_words=77):
        self.image_root = os.path.join(root, 'extracted_valid_images')

        self.transform = transform
        tir_root = os.path.join(root, 'full_valid_info_PAI')
        self.pkls_tir = sorted([os.path.join(tir_root, f) for f in os.listdir(tir_root)])

        self.size = len(self.pkls_tir)

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        self.id_to_data = {}

        # self.size=100
        for i in range(self.size):

            value = self.pkl_loader(self.pkls_tir[i])
            captions_list = value['captions']
            img_name = value['img_name']
            
            object_id = img_name.split('_')[0]

            if object_id not in self.id_to_data:
                self.id_to_data[object_id] = {'images': [], 'captions': []}

            self.id_to_data[object_id]['images'].append(img_name)

            if isinstance(captions_list, str):
                captions_list = [captions_list]

            # if not id_to_data[object_id]['captions']:
            self.id_to_data[object_id]['captions'].extend(captions_list)
            # print(captions_list)

        img_id = 0
        txt_id = 0

        for object_id, data in self.id_to_data.items():
            captions = []
            for caption in data['captions']:
                self.text.append(pre_caption(caption, max_words))
                captions.append(txt_id)
                self.txt2img[txt_id] = []
                txt_id += 1

            for img_name in data['images']:
                self.image.append(img_name)
                self.img2txt[img_id] = captions
                for cap_id in captions:
                    self.txt2img[cap_id].append(img_id)
                img_id += 1

                
    def __getitem__(self, index):

        caption = self.text[index]
        return caption, index  

    def __len__(self):
        return len(self.text)
    
    def pkl_loader(self, pkl_path):
        """从 *.pkl 加载文本"""
        with open(pkl_path, 'rb') as f:
            info_dict = pickle.load(f)
            return info_dict
