import utils
import numpy as np
import time
import datetime
import torch
import torch.nn.functional as F
import torch.distributed as dist
# from models.blip_retrieval_tta import get_all_text_embeds, get_all_image_embeds
import models.blip_tta as BLIP
import models.clip_tta as CLIP
import models.clip_reid_tta as CLIP_Reid
import utils
from losses import *
from ddp import *

@torch.no_grad()
def evaluation(model, test_loader_text, test_loader_image, device, args, config):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    print('Computing features for evaluation...')
    start_time = time.time()  

    texts = test_loader_text.dataset.text   
    # text_embeds=get_all_text_embeds(test_loader_text, model, device, args)
    # image_embeds=get_all_image_embeds(test_loader_image, model, device, args)
    if config['backbone']=='blip':
        text_embeds=BLIP.get_text_embeds_blip(test_loader_text, model, device, args)
        image_embeds=BLIP.get_image_embeds_blip(test_loader_image, model, device, args)
    elif config['backbone']=='clip':
        text_embeds=CLIP.get_text_embeds_clip(test_loader_text, model, device, args)
        image_embeds=CLIP.get_image_embeds_clip(test_loader_image, model, device, args)
    elif config['backbone']=='clip_reid':
        text_embeds=CLIP_Reid.get_text_embeds_clip_reid(test_loader_text, model, device, args)
        image_embeds=CLIP_Reid.get_image_embeds_clip_reid(test_loader_image, model, device, args)

    #blip_score
    sims_matrix = F.normalize(image_embeds) @ F.normalize(text_embeds).t()
    score_matrix_i2t = torch.full((len(test_loader_image.dataset.image),len(texts)),-100.0).to(device)
    
    num_tasks = utils.get_world_size()
    rank = utils.get_rank() 
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)


    for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 100, header)): 
        score_matrix_i2t[start+i] = sims
        
    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts),len(test_loader_image.dataset.image)),-100.0).to(device)
    
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)    


    for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 100, header)): 
        score_matrix_t2i[start+i] = sims

    if args.distributed:
        dist.barrier()   
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM) 
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)   

    # Optional: print distribution stats can be added by users as needed
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()

@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, args, data_loader, idx_trans=None):
    if args.retrieval=='i2t':
        txt2img=data_loader.dataset.txt2img
        img2txt = [data_loader.dataset.img2txt[idx] for idx in idx_trans]
    elif args.retrieval=='t2i':
        img2txt=data_loader.dataset.img2txt
        txt2img = [data_loader.dataset.txt2img[idx] for idx in idx_trans]
    else:
        img2txt=data_loader.dataset.img2txt
        txt2img=data_loader.dataset.txt2img

    # Images->Text 
    ranks = np.zeros(scores_i2t.shape[0])
    average_precisions_i2t = []
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        relevant_indices = img2txt[index]
        relevant_positions = []
        for i in relevant_indices:
            tmp = np.where(inds == i)[0][0]
            relevant_positions.append(tmp)
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

        # Calculate average precision
        relevant_positions.sort()
        precisions = [(i + 1) / (pos + 1) for i, pos in enumerate(relevant_positions)]
        average_precision = np.mean(precisions) if precisions else 0
        average_precisions_i2t.append(average_precision)

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
  
    # Text->Images 
    ranks = np.zeros(scores_t2i.shape[0])
    average_precisions_t2i = []


    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        relevant_indices = txt2img[index]
        relevant_positions = []
        for i in relevant_indices:
            tmp = np.where(inds == i)[0][0]
            relevant_positions.append(tmp)
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

        # Calculate average precision
        relevant_positions.sort()
        precisions = [(i + 1) / (pos + 1) for i, pos in enumerate(relevant_positions)]
        average_precision = np.mean(precisions) if precisions else 0
        average_precisions_t2i.append(average_precision)

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)        

    tr_mean = round((tr1 + tr5 + tr10) / 3, 2)
    ir_mean = round((ir1 + ir5 + ir10) / 3, 2)
    r_mean = round((tr_mean + ir_mean) / 2, 2)

    # Calculate R_sum
    r_sum = tr1 + tr5 + tr10 + ir1 + ir5 + ir10

    # Calculate mAP and format to two decimal places
    mAP_i2t = round(np.mean(average_precisions_i2t) * 100, 2)
    mAP_t2i = round(np.mean(average_precisions_t2i) * 100, 2)

    eval_result =  {
        'txt_r1': tr1,
        'txt_r5': tr5,
        'txt_r10': tr10,
        'txt_r_mean': tr_mean,
        'txt_map': mAP_i2t,
        'img_r1': ir1,
        'img_r5': ir5,
        'img_r10': ir10,
        'img_r_mean': ir_mean,
        'img_map': mAP_t2i,
        'r_mean': r_mean,
        'r_sum': r_sum
    }
    return eval_result