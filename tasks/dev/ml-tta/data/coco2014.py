import os
from os.path import join
import torch
import json
from tqdm import tqdm

from torch.utils.data import Dataset
from PIL import Image

from pycocotools.coco import COCO

class COCO2014(Dataset):
    def __init__(self, set_id, dataset_dir, transform):

        self.dataset_dir = dataset_dir
        if set_id == 'coco2014':
            coco2014_val = os.path.join(self.dataset_dir, "annotations/instances_val2014.json")
        elif set_id == 'coco2017':
            coco2014_val = os.path.join(self.dataset_dir, "annotations/instances_val2017.json")
        self.coco_val = COCO(coco2014_val)
        self.ids_val = self.coco_val.getImgIds()

        categories = self.coco_val.loadCats(self.coco_val.getCatIds())
        categories.sort(key=lambda x: x['id'])

        classes = {}
        coco_labels = {}
        coco_labels_inverse = {}
        for c in categories:
            coco_labels[len(classes)] = c['id']
            coco_labels_inverse[c['id']] = len(classes)
            classes[c['name']] = len(classes)

        def load_annotations(coco_, img_idlist, image_index, filter_tiny=True):
            tmp_id = image_index if (img_idlist is None) else img_idlist[image_index]
            annotations_ids = coco_.getAnnIds(imgIds=tmp_id, iscrowd=False)
            annotations = []

            if len(annotations_ids) == 0:
                return annotations

            coco_annotations = coco_.loadAnns(annotations_ids)
            for idx, a in enumerate(coco_annotations):
                if filter_tiny and (a['bbox'][2] < 1 or a['bbox'][3] < 1):
                    continue
                annotations += [coco_labels_inverse[a['category_id']]]

            return annotations
        
        self.test = []
        for idx, imgid in tqdm(enumerate(self.ids_val)):
            ip = "val2014" if set_id == 'coco2014' else "val2017"
            img_path = self.dataset_dir + f"/{ip}/{self.coco_val.loadImgs(imgid)[0]['file_name']}"
            label = load_annotations(self.coco_val, None, imgid, filter_tiny=False)
            label = list(set(label))
            if label:
                if len(label) == 5:
                    self.test.append([img_path, label])
        # self.test = self.test[::50]
        self.transform = transform
        print(f"length of test: {len(self.test)}")

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        img_path, label = self.test[idx]

        image = Image.open(open(img_path, "rb")).convert("RGB")
        image = self.transform(image)
        target = torch.LongTensor(label)

        return image, img_path, target