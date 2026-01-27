import argparse

import time

from copy import deepcopy

from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from config_loader import load_config_from_path

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models

from clip.custom_clip import get_coop
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import Summary, AverageMeter, ProgressMeter, set_random_seed, mAP
from data.cls_to_names import *

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def select_confident_samples(logits, top, flag=False):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=flag)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx

def ranking_loss(y_pred, y_true, scale_ = 2.0, margin_ = 1):
    y_pred *= scale_
    y_true_ = y_true.float()
    tmp = margin_ - y_pred[:, None, :] + y_pred[:, :, None]
    partial_losses = torch.maximum(torch.zeros_like(tmp), tmp)
    loss = partial_losses * y_true_[:, None, :] * (1 - y_true_[:, :, None])
    loss = torch.sum(loss, dim=-1)
    loss = torch.sum(loss, dim=-1)
    return torch.mean(loss)

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

def test_time_tuning(model, inputs, optimizer, scaler, args):
    
    for j in range(args.tta_steps):

        with torch.cuda.amp.autocast():
            output, output_neg = model(inputs)

            output, selected_idx = select_confident_samples(output, args.selection_p, flag=False)
            loss_entropy = avg_entropy(output)
            
            output_neg, selected_idx_neg = select_confident_samples(output_neg, args.selection_p, flag=False)
            loss_ranking = avg_entropy(output_neg)
            
            # loss_ranking = ranking_loss(output_neg, labels, scale_ = 1.0, margin_ = 1)

            loss = loss_entropy + loss_ranking
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return

def main(args):
    
    set_random_seed(args.seed)
    print("Use GPU: {} for training".format(args.gpu))

    classnames = eval("{}_classes".format(args.test_sets.lower()))
    model = get_coop(args.arch, args.test_sets, args.gpu, args.n_ctx, args.ctx_init, maple=args.maple, promptkd=args.promptkd, is_bind=args.is_bind)
    if args.maple:
        model.load_pretrained(ckpt_path=args.maple)
    if args.coop:
        print("Use pre-trained soft prompt (CoOp) as initialization")
        pretrained_ctx = torch.load(args.coop)['state_dict']['ctx']
        assert pretrained_ctx.size()[0] == args.n_ctx
        with torch.no_grad():
            model.prompt_learner.ctx.copy_(pretrained_ctx)
            model.prompt_learner.ctx_init_state = pretrained_ctx
            
            model.prompt_learner.ctx_neg.copy_(pretrained_ctx)
            model.prompt_learner.ctx_init_state_neg = pretrained_ctx
    model_state = None

    for name, param in model.named_parameters():
        if "prompt_learner.ctx" not in name:
            param.requires_grad_(False)
    
    print("=> Model created: visual backbone {}".format(args.arch))
    
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    # trainable_param = model.prompt_learner.parameters()
    if args.promptkd:
        param_groups = [
            {'params': model.model_teacher.prompt_learner.ctx, 'lr': args.lr1},
            {'params': model.model_teacher.prompt_learner.ctx_neg, 'lr': args.lr2}
        ]
    else:
        param_groups = [
            {'params': model.prompt_learner.ctx, 'lr': args.lr1},
            {'params': model.prompt_learner.ctx_neg, 'lr': args.lr2}
        ]
    optimizer = torch.optim.AdamW(param_groups)
    optim_state = deepcopy(optimizer.state_dict())
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    print('=> Using native Torch AMP. Training in mixed precision.')

    cudnn.benchmark = True
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    datasets = args.test_sets.split("/")
    results = {}
    for set_id in datasets:
        base_transform = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.resolution)])
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size-1, 
                                        augmix=len(set_id)>1)
        batchsize = 1

        print("evaluating: {}".format(set_id))
        classnames = eval("{}_classes".format(set_id.lower()))
        model.reset_classnames(classnames, args.arch)

        val_dataset = build_dataset(set_id, data_transform, args.data, mode=args.dataset_mode)
        print("number of test samples: {}".format(len(val_dataset)))
        val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batchsize, shuffle=True,
                    num_workers=args.workers, pin_memory=True)
            
        results[set_id] = test_time_adapt_eval(set_id, val_loader, model, model_state, optimizer, optim_state, scaler, args)

def test_time_adapt_eval(set_id, val_loader, model, model_state, optimizer, optim_state, scaler, args):
    batch_time = AverageMeter('Time:', ':5.3f', Summary.NONE)
    mAP_meter_image = AverageMeter('mAP_image:', ':5.3f', Summary.VAL)
    mAP_meter_text = AverageMeter('mAP_text:', ':5.3f', Summary.VAL)
    mAP_meter_merge_post_5 = AverageMeter('mAP_merge_post_5:', ':5.3f', Summary.VAL)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, mAP_meter_image, mAP_meter_text, mAP_meter_merge_post_5],
        prefix='Test: ')

    model.eval()
    with torch.no_grad():
        model.reset()
    end = time.time()

    pred_list_image = []
    pred_list_text = []
    pred_list_merge_post_5 = []
    target_list = []

    for i, (images, img_name, target) in tqdm(enumerate(val_loader)):
        for k in range(len(images)):
            images[k] = images[k].cuda(args.gpu, non_blocking=True)
        image = images[0]

        if set_id not in ['coco2014', 'coco2017', 'voc2007', 'voc2012', 'nuswide', 'object365']:
            target = target.cuda(args.gpu, non_blocking=True)
        if args.tpt:
            images = torch.cat(images, dim=0)

        if args.tta_steps > 0:
            with torch.no_grad():
                model.reset()

        # # example
        # with torch.no_grad():
        #     with torch.cuda.amp.autocast():
        #         output_src, output_neg_src = model(image)

        if args.tpt:
            optimizer.load_state_dict(optim_state)
            test_time_tuning(model, images, optimizer, scaler, args)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output, output_neg = model(image)
                # print(f"output_max: {output.max()}, output_min: {output.min()}")

                pred_list_image.append(output.cpu())
                pred_list_text.append(output_neg.cpu())

                tar = torch.tensor([0] * output.shape[1])
                for t in target[0].tolist():
                    tar[t] = 1
                target_list.append(tar)

        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0 or i == len(val_loader) - 1:

            if set_id in ['coco2014', 'coco2017', 'voc2007', 'voc2012', 'nuswide', 'object365', 'lvis']:
                
                pred_list_npy_image = torch.cat(pred_list_image,dim=0)
                pred_list_npy_text = torch.cat(pred_list_text,dim=0)
                
                tmp1_ = (pred_list_npy_image - pred_list_npy_image.min()) / (pred_list_npy_image.max() - pred_list_npy_image.min())
                tmp2_ = (pred_list_npy_text - pred_list_npy_text.min()) / (pred_list_npy_text.max() - pred_list_npy_text.min())
                
                pred_list_npy_merge_post_5 = (tmp1_ * 0.5 + tmp2_ * 0.5).numpy()

                pred_list_npy_image = pred_list_npy_image.numpy()
                pred_list_npy_text = pred_list_npy_text.numpy()
                
                target_list_npy = torch.stack(target_list).numpy()

                map_image = mAP(target_list_npy, pred_list_npy_image)
                mAP_meter_image.update(map_image, image.size(0))

                map_text = mAP(target_list_npy, pred_list_npy_text)
                mAP_meter_text.update(map_text, image.size(0))

                map_merge_post_5 = mAP(target_list_npy, pred_list_npy_merge_post_5)
                mAP_meter_merge_post_5.update(map_merge_post_5, image.size(0))

            progress.display(i)

    progress.display_summary()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    parser.add_argument('--data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='A/R/V/K/I', help='test dataset (multiple datasets split by slash)')
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
    parser.add_argument('--lr1', '--learning-rate1', default=0.01, type=float,metavar='LR', help='initial learning rate', dest='lr1')
    parser.add_argument('--lr2', '--learning-rate2', default=0.001, type=float,metavar='LR', help='initial learning rate', dest='lr2')
    parser.add_argument('-p', '--print-freq', default=200, type=int,metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=0, type=int,help='GPU id to use.')
    parser.add_argument('--tpt', type=int, default=0, help='run test-time prompt tuning')
    parser.add_argument('--selection_p', default=0.1, type=float, help='confidence selection percentile')
    parser.add_argument('--tta_steps', default=1, type=int, help='test-time-adapt steps')
    parser.add_argument('--n_ctx', default=16, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default=None, type=str, help='init tunable prompts')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--coop', type=str, default="")
    parser.add_argument('--is_bind', type=int, default=0)
    parser.add_argument('--is_pre', type=int, default=0)
    parser.add_argument('--maple', type=str, default="")
    args = parser.parse_args()
    
    main(args)