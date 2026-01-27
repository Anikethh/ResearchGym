import os
import json
import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image

from data.utils import pre_caption
    
class nocaps_entire_domain_eval_image(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30):  
        '''
        image_root (string): Root directory of images (e.g. nocaps/)
        ann_root (string): directory to store the annotation file
        '''
        filename = 'annotations.json'
        
        self.annotation = json.load(open(os.path.join(ann_root, filename), 'r'))
        self.transform = transform
        self.image_root = image_root
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        img_id = 0
        
        for ann in self.annotation:
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for caption in ann['caption']:
                self.text.append(pre_caption(caption, max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = [img_id]
                txt_id += 1
            img_id += 1

    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):    
        image_path = os.path.join(self.image_root, self.image[index])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, index    

class nocaps_entire_domain_eval_text(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30):  
        '''
        image_root (string): Root directory of images (e.g. nocaps/)
        ann_root (string): directory to store the annotation file
        '''
        filename = 'annotations.json'
        
        self.annotation = json.load(open(os.path.join(ann_root, filename), 'r'))
        self.transform = transform
        self.image_root = image_root
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        img_id = 0
        
        for ann in self.annotation:
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for caption in ann['caption']:
                self.text.append(pre_caption(caption, max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = [img_id]
                txt_id += 1
            img_id += 1

    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index):    
        caption = self.text[index]
        
        return caption, index