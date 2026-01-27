import torch
import numpy as np

class ClassBalancedSampler:
    def __init__(self, labels, num_batches, batch_size):
        self.num_batches = num_batches
        self.y_array = labels
        self.classes = np.unique(self.y_array)
        self.batch_size = batch_size
        self.n_classes = len(self.classes)

    def __len__(self):
        return self.num_batches

    
    def __iter__(self):
        per_class_num = max(self.batch_size // self.n_classes, 1)
        for n in range(self.num_batches):
            indexes = np.arange(len(self.y_array))
            cls_indexes = []
            for l in self.classes:
                sel_indexes = indexes[self.y_array == l]
                idxes = np.random.choice(sel_indexes, per_class_num, replace=True)
                cls_indexes.append(idxes)
                
            batch = np.concatenate(cls_indexes)
            
            yield torch.tensor(batch)


class RandomSampler:
    def __init__(self, labels, num_batches, batch_size):
        self.num_batches = num_batches
        self.y_array = labels
        self.classes = np.unique(self.y_array)
        self.batch_size = batch_size
        self.n_classes = len(self.classes)

    def __len__(self):
        return self.num_batches

    
    def __iter__(self):
        for n in range(self.num_batches):
            indexes = np.arange(len(self.y_array))
            batch = np.random.choice(indexes, self.batch_size, replace=True)
            yield torch.tensor(batch)


class JointRandomSampler:
    def __init__(self, labels1, labels2, num_batches, batch_size):
        self.num_batches = num_batches
        self.y_array1 = labels1
        self.y_array2 = labels2
        self.classes = np.unique(np.concatenate([np.unique(self.y_array1),np.unique(self.y_array2)]))
        self.batch_size = batch_size
        self.n_classes = len(self.classes)

    def __len__(self):
        return self.num_batches

    
    def __iter__(self):
        indexes1 = np.arange(len(self.y_array1))
        indexes2 = np.arange(len(self.y_array2))
        cls_indexes = [indexes2[self.y_array2==y] for y in self.classes]
        for n in range(self.num_batches):
            batch1 = np.random.choice(indexes1, self.batch_size, replace=True)
            batch2 = np.zeros(self.batch_size).astype(int)
            classes = self.y_array1[batch1]
            for i, y in enumerate(self.classes):
                num = (classes==y).sum().item()
                batch2[classes==y] = np.random.choice(cls_indexes[i], num, replace=True)
            batch1 = torch.tensor(batch1)
            batch2 = torch.tensor(batch2)
            yield torch.stack([batch1,batch2],dim=1)



class GroupBalancedSampler:
    def __init__(self, group_labels, num_batches, batch_size):
        self.num_batches = num_batches
        self.group_array = group_labels
        self.groups = np.unique(self.group_array)
        self.batch_size = batch_size
        self.n_groups = len(self.groups)

    def __len__(self):
        return self.num_batches

    
    def __iter__(self):
        per_group_num = max(self.batch_size // self.n_groups, 1)
        for n in range(self.num_batches):
            indexes = np.arange(len(self.group_array))
            group_indexes = []
            for g in self.groups:
                sel_indexes = indexes[self.group_array == g]
                idxes = np.random.choice(sel_indexes, per_group_num, replace=True)
                group_indexes.append(idxes)
                
            batch = np.concatenate(group_indexes)
            
            yield torch.tensor(batch)





class ShortcutSampler:
    def __init__(self, mis_index_dict, cor_index_dict, mis_pred_label_dict, cor_pred_label_dict, num_batches, batch_size):
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.mis_index_dict = mis_index_dict
        self.cor_index_dict = cor_index_dict
        self.mis_pred_label_dict = mis_pred_label_dict
        self.cor_pred_label_dict = cor_pred_label_dict

    def __len__(self):
        return self.num_batches

    
    def __iter__(self):
        for n in range(self.num_batches):
            keys = list(self.mis_index_dict.keys())
            batch_key = np.random.choice(keys, self.batch_size, replace=True)
            batch1 = []
            batch2 = []
            for i in range(self.batch_size//2):
                # if i % 2 == 0:
                key = batch_key[i]
                mis_indexes = self.mis_index_dict[key]
                idx = np.random.choice(np.arange(len(mis_indexes)))
                batch2.append(mis_indexes[idx])
                mis_key = self.mis_pred_label_dict[key][idx].item()
                # assert mis_key != key, "mis_label should not be the same as key"
                cor_indexes = self.cor_index_dict[mis_key]
                idx = np.random.choice(np.arange(len(cor_indexes)))
                batch1.append(cor_indexes[idx])
                
                # else:
                #     key = batch_key[i]
                #     cor_indexes = self.cor_index_dict[key]
                #     idx = np.random.choice(np.arange(len(cor_indexes)))
                #     batch1.append(cor_indexes[idx])
                #     mis_indexes = self.mis_index_dict[key]

                #     idx = np.random.choice(np.arange(len(mis_indexes)))
                #     batch2.append(mis_indexes[idx])


            yield torch.cat([torch.tensor(batch1),torch.tensor(batch2)])