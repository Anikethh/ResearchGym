# create sparse annos --zyrant

import argparse
import time
from os import path as osp

import mmcv
import numpy as np

from mmdet3d.core.bbox import limit_period
import copy


def create_scannet_sparse_infos(root_dir, out_dir, pkl_files, random_annos = False, oneclassoneanno = False):
    print(f'{pkl_files} is adding sparse annos')
    if root_dir == out_dir:
        print(f'Warning, you are overwriting '
              f'the original data under {root_dir}.')
        time.sleep(3)
    for pkl_file in pkl_files:
        in_path = osp.join(root_dir, pkl_file)
        print(f'Reading from input file: {in_path}.')
        a = mmcv.load(in_path)
        if 'train' in pkl_file:
            print('Start adding:')
            for item in mmcv.track_iter_progress(a):
                print(item)
                info = {}
                full_annos = item['annos']
                num_full_annos = full_annos['gt_num']

                #################sence sparse anno select#####################
                if random_annos:
                    num_labeled = [1]
                    for num in num_labeled:
                            annotations = {}
                            num_sparse_gt = min(num, num_full_annos)
                            annotations['gt_num'] = num_sparse_gt 
                            if num_sparse_gt != 0:

                                choices = np.random.choice(num_full_annos, num_sparse_gt, replace=False)
                            
                                annotations['name'] = full_annos['name'][choices]
                                # default names are given to aligned bbox for compatibility
                                # we also save unaligned bbox info with marked names
                                annotations['location'] = full_annos['location'][choices]
                                annotations['dimensions'] = full_annos['dimensions'][choices]
                                annotations['gt_boxes_upright_depth'] = full_annos['gt_boxes_upright_depth'][choices]
                                annotations['unaligned_location'] = full_annos['unaligned_location'][choices]
                                annotations['unaligned_dimensions'] = full_annos['unaligned_dimensions'][choices]
                                annotations[
                                    'unaligned_gt_boxes_upright_depth'] =  full_annos['unaligned_gt_boxes_upright_depth'][choices]
                                annotations['index'] = np.arange(
                                    num_sparse_gt, dtype=np.int32)
                                annotations['class'] = full_annos['class'][choices]

                            annotations['axis_align_matrix'] = full_annos['axis_align_matrix']  # 4x4

                            info['annos_'+ str(num)] = annotations 

                #######################################################

                ##############one class one anno#######################
                if oneclassoneanno:
                    annotations = {}

                    if num_full_annos != 0:
                        unique_class = np.unique(full_annos['class'])
                        choices = [np.random.choice(np.where(full_annos['class'] == val)[0]) for val in unique_class]
                        num_sparse_gt = min(len(choices), num_full_annos)
                        annotations['gt_num'] = num_sparse_gt 
                        annotations['name'] = full_annos['name'][choices]
                        # default names are given to aligned bbox for compatibility
                        # we also save unaligned bbox info with marked names
                        annotations['location'] = full_annos['location'][choices]
                        annotations['dimensions'] = full_annos['dimensions'][choices]
                        annotations['gt_boxes_upright_depth'] = full_annos['gt_boxes_upright_depth'][choices]
                        annotations['unaligned_location'] = full_annos['unaligned_location'][choices]
                        annotations['unaligned_dimensions'] = full_annos['unaligned_dimensions'][choices]
                        annotations[
                            'unaligned_gt_boxes_upright_depth'] =  full_annos['unaligned_gt_boxes_upright_depth'][choices]
                        annotations['index'] = np.arange(
                            num_sparse_gt, dtype=np.int32)
                        annotations['class'] = full_annos['class'][choices]
                    else:
                        annotations['gt_num'] = num_full_annos 

                    annotations['axis_align_matrix'] = full_annos['axis_align_matrix']  # 4x4

                    info['annos_ocoa'] = annotations 

                #######################################################

                item.update(info)
            
        else:
            print('pass')
        out_path = osp.join(out_dir, pkl_file)
        print(f'Writing to output file: {out_path}.')
        mmcv.dump(a, out_path, 'pkl')



def create_sunrgbd_sparse_infos(root_dir, out_dir, pkl_files, random_annos = False, oneclassoneanno = False):
    print(f'{pkl_files} is adding sparse annos')
    if root_dir == out_dir:
        print(f'Warning, you are overwriting '
              f'the original data under {root_dir}.')
        time.sleep(3)
    for pkl_file in pkl_files:
        in_path = osp.join(root_dir, pkl_file)
        print(f'Reading from input file: {in_path}.')
        a = mmcv.load(in_path)

        if 'train' in pkl_file:
            print('Start adding:')
            for i, item in enumerate(mmcv.track_iter_progress(a)):

                info = {}
                full_annos = item['annos']
                num_full_annos = full_annos['gt_num']

                if random_annos:
                    num_labeled = [1]
                    for num in num_labeled:
                        annotations = {}

                        num_sparse_gt = min(num, num_full_annos)
                        annotations['gt_num'] = num_sparse_gt

                        if annotations['gt_num'] != 0:

                            choices = np.random.choice(num_full_annos, num_sparse_gt, replace=False)

                            annotations['name'] = full_annos['name'][choices]
                            annotations['bbox'] = full_annos['bbox'][choices]
                            annotations['location'] = full_annos['location'][choices]
                            annotations['dimensions'] = full_annos['dimensions'][choices]  # lwh (depth) format
                            annotations['rotation_y'] = full_annos['rotation_y'][choices]
                            annotations['index'] = np.arange(num_sparse_gt, dtype=np.int32)
                            annotations['class'] = full_annos['class'][choices]
                            annotations['gt_boxes_upright_depth'] = full_annos['gt_boxes_upright_depth'][choices]  # (K,8)
                            
                        info['annos_'+ str(num)] = annotations
                

                if oneclassoneanno:

                    annotations = {}
                    if num_full_annos != 0:
                        unique_class = np.unique(full_annos['class'])
                        choices = [np.random.choice(np.where(full_annos['class'] == val)[0]) for val in unique_class]
                        num_sparse_gt = min(len(choices), num_full_annos)
                        annotations['gt_num'] = num_sparse_gt
                        annotations['name'] = full_annos['name'][choices]
                        annotations['bbox'] = full_annos['bbox'][choices]
                        annotations['location'] = full_annos['location'][choices]
                        annotations['dimensions'] = full_annos['dimensions'][choices]  # lwh (depth) format
                        annotations['rotation_y'] = full_annos['rotation_y'][choices]
                        annotations['index'] = np.arange(num_sparse_gt, dtype=np.int32)
                        annotations['class'] = full_annos['class'][choices]
                        annotations['gt_boxes_upright_depth'] = full_annos['gt_boxes_upright_depth'][choices]  # (K,8)                
                    else:
                        annotations['gt_num'] = num_full_annos 

                    info['annos_ocoa'] = annotations

                item.update(info)

        else:
            print('pass')

        out_path = osp.join(out_dir, pkl_file)
        print(f'Writing to output file: {out_path}.')
        mmcv.dump(a, out_path, 'pkl')



parser = argparse.ArgumentParser(description='Create sparse annos.')
parser.add_argument('--dataset', default='sunrgbd', help='name of the dataset')
parser.add_argument(
    '--root-dir',
    type=str,
    default='/opt/data/private/all_data/sunrgbd/all_sparse_anno',
    help='specify the root dir of dataset')
parser.add_argument(
    '--out-dir',
    type=str,
    default='/opt/data/private/all_data/sunrgbd/all_sparse_anno',
    help='specify the out dir of dataset')
parser.add_argument('--random_annos', default=False, help='Random annos')
parser.add_argument('--oneclassoneanno', default=False, help='One class one anno')

args = parser.parse_args()

# 
if __name__ == '__main__':
    if args.out_dir is None:
        args.out_dir = args.root_dir
    elif args.dataset == 'scannet':
        pkl_files = ['scannet_infos_train.pkl']
        create_scannet_sparse_infos(
            root_dir=args.root_dir, out_dir=args.out_dir, pkl_files=pkl_files, 
            random_annos = args.random_annos, oneclassoneanno = args.oneclassoneanno)
    elif args.dataset == 'sunrgbd':
        pkl_files = ['sunrgbd_infos_train.pkl']
        create_sunrgbd_sparse_infos(
            root_dir=args.root_dir, out_dir=args.out_dir, pkl_files=pkl_files, 
            random_annos = args.random_annos, oneclassoneanno = args.oneclassoneanno)
