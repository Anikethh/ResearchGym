import os
from os.path import join
import torch
import json
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset
from PIL import Image
from data.cls_to_names import nuswide_classes

class NUSWIDE(Dataset):
    def __init__(self, set_id, dataset_dir, transform):

        self.dataset_dir = dataset_dir
        self.nuswide_classes = nuswide_classes
    
        self.image_dir = os.path.join(self.dataset_dir, "Flickr")
        self.cls_name_list = self.read_name_list(join(self.dataset_dir, 'Concepts81.txt'), False)
        self.im_name_list_test = self.read_name_list(join(self.dataset_dir, 'ImageList/TestImagelist.txt'), False)
        print('NUS-WIDE test {} images. '.format(len(self.im_name_list_test)))

        path_labels = os.path.join(self.dataset_dir, 'TrainTestLabels')
        num_classes = len(self.nuswide_classes)

        test_labels = defaultdict(list)
        for i in tqdm(range(num_classes)):
            file_ = os.path.join(path_labels, 'Labels_'+self.nuswide_classes[i]+'_Test.txt')
            cls_labels = []
            with open(file_, 'r') as f:
                for j, line in enumerate(f):
                    tmp = line.strip()
                    if tmp == '1':
                        test_labels[j].append(i)
        
        self.test = []
        for i, name in tqdm(enumerate(self.im_name_list_test)):
            img_path=self.image_dir + '/' + '/'.join(name.split('\\'))
            label=test_labels[i]
            label = list(set(label))
            if label:
                self.test.append([img_path, label])

        self.transform = transform

    def read_name_list(self, path, if_split=True):
        ret = []
        with open(path, 'r') as f:
            for line in f:
                if if_split:
                    tmp = line.strip().split(' ')
                    ret.append(tmp[0])
                else:
                    tmp = line.strip()
                    ret.append(tmp)
        return ret
    
    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        img_path, label = self.test[idx]

        image = Image.open(open(img_path, "rb")).convert("RGB")
        image = self.transform(image)
        target = torch.LongTensor(label)

        return image, target