import os
from os.path import join
import torch
import json
from tqdm import tqdm

from torch.utils.data import Dataset
from PIL import Image
from data.cls_to_names import object365_classes
from pycocotools.coco import COCO

class OBJECT365(Dataset):
    def __init__(self, set_id, dataset_dir, transform):

        self.object365_classes = object365_classes
        self.dataset_dir = dataset_dir

        self.dataset_root = join(self.dataset_dir, 'Images/val/val')
        self.coco_instance_json_file = join(self.dataset_dir, 'Annotations/val/val.json')
        coco = COCO(self.coco_instance_json_file)
        self.valset_ids = coco.getImgIds()

        instance_info = {}
        with open(self.coco_instance_json_file, 'r') as f:
            instance_info = json.load(f)

        clsid2clsidx = {}
        clsidx2clsid = {}
        clsid2clsname = {}
        for idx, cat_info in enumerate(instance_info["categories"]):
            clsid2clsidx[cat_info['id']] = idx
            clsidx2clsid[idx] = cat_info['id']
            clsid2clsname[cat_info['id']] = cat_info['name']
        
        self.test = []
        for idx, imgid in tqdm(enumerate(self.valset_ids)):
            label = []
            img_path = self.dataset_root + f"/obj365_val_{str(imgid).zfill(12)}.jpg"
            annIds = coco.getAnnIds(imgIds = imgid)
            anns = coco.loadAnns(annIds)
            for ann in anns:
                label.append(clsid2clsidx[ann['category_id']])
            label = list(set(label))
            self.test.append([img_path, label])
                
        self.transform = transform
        
    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        img_path, label = self.test[idx]

        image = Image.open(open(img_path, "rb")).convert("RGB")
        image = self.transform(image)
        target = torch.LongTensor(label)

        return image, target