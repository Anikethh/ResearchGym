import os
from os.path import join
from collections import defaultdict
import torch
from torch.utils.data import Dataset
from data.cls_to_names import voc2007_classes
from PIL import Image

class VOC2007(Dataset):
    def __init__(self, set_id, dataset_dir, transform):
        
        if set_id == 'voc2007':
            phase = 'test'
            self.dataset_dir = os.path.join(dataset_dir, "VOC2007")
        elif set_id == 'voc2012':
            phase = 'val'
            self.dataset_dir = dataset_dir

        self.image_dir = os.path.join(self.dataset_dir, "JPEGImages")
        self.im_name_list = self.read_name_list(join(self.dataset_dir, 'ImageSets/Main/trainval.txt'))
        self.im_name_list_val = self.read_name_list(join(self.dataset_dir, f'ImageSets/Main/{phase}.txt'))
        print('VOC2007 train total {} images, test total {} images. '.format(len(self.im_name_list), len(self.im_name_list_val)))
        self.voc2007_classes = voc2007_classes

        test_data_imname2label = self.read_object_labels(self.dataset_dir, phase=phase)

        self.test = []
        for i, name in enumerate(self.im_name_list_val):
            img_path = self.image_dir+'/{}.jpg'.format(name)
            label = test_data_imname2label[name]
            label = list(set(label))
            if label:
                self.test.append([img_path, label])

        self.transform = transform
    
    def read_object_labels(self, path, phase):
        path_labels = os.path.join(path, 'ImageSets', 'Main')
        labeled_data = defaultdict(list)
        num_classes = len(self.voc2007_classes)

        for i in range(num_classes):
            file = os.path.join(path_labels, self.voc2007_classes[i] + '_' + phase + '.txt')
            data_ = self.read_image_label(file)

            for (name, label) in data_.items():
                if label == 1:
                    labeled_data[name].append(i)
        return labeled_data

    def read_image_label(self, file):
        data_ = dict()
        with open(file, 'r') as f:
            for line in f:
                tmp = line.strip().split(' ')
                name = tmp[0]
                label = int(tmp[-1])
                data_[name] = label
        return data_

    def read_name_list(self, path):
        ret = []
        with open(path, 'r') as f:
            for line in f:
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

        return image, img_path, target
