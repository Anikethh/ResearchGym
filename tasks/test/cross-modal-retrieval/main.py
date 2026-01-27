import argparse, os, time, datetime, json, random
import numpy as np, torch, torch.backends.cudnn as cudnn
import ruamel.yaml as yaml
from pathlib import Path
from tqdm import tqdm

import utils
from data import create_dataset, create_sampler, create_loader
from ddp import *
from evaluation import evaluation, itm_eval
from models.tta_baselines.set_model import set_optimizer, set_model, set_temperature
import models.blip_tta as BLIP, models.clip_tta as CLIP, models.clip_reid_tta as CLIP_Reid

def test_time_tune(adapt_model, data_loader_text, data_loader_image, optimizer, device, config, args):
    model_without_ddp = adapt_model.model.module
    print("Computing features for evaluation...")
    start_time = time.time()

    len_image = len(data_loader_image.dataset.image)
    len_text = len(data_loader_image.dataset.text)
    
    score_matrix_i2t = torch.full((len_image, len_text), -100.0).to(device)
    score_matrix_t2i = torch.full((len_text, len_image), -100.0).to(device)

    idx_list = []
    adapt_model.eval()
    
    with torch.no_grad():
        if args.only_visual:
            if config["backbone"] == "blip":
                all_text_embeds = BLIP.get_text_embeds_blip(data_loader_text, model_without_ddp, device, args)
            elif config["backbone"] == "clip":
                all_text_embeds = CLIP.get_text_embeds_clip(data_loader_text, model_without_ddp, device, args)
            elif config["backbone"] == "clip_reid":
                all_text_embeds = CLIP_Reid.get_text_embeds_clip_reid(data_loader_text, model_without_ddp, device, args)
            
            model_without_ddp.set_text_features(all_text_embeds.detach())
        else:
            if config["backbone"] == "blip":
                all_image_embeds = BLIP.get_image_embeds_blip(data_loader_image, model_without_ddp, device, args)
            elif config["backbone"] == "clip":
                all_image_embeds = CLIP.get_image_embeds_clip(data_loader_image, model_without_ddp, device, args)
            elif config["backbone"] == "clip_reid":
                all_image_embeds = CLIP_Reid.get_image_embeds_clip_reid(data_loader_image, model_without_ddp, device, args)
            
            model_without_ddp.set_image_features(all_image_embeds.detach())
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("loss_total", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    
    bs = config["batch_size_total"]
    
    print("Test time adaptation...")
    
    with torch.no_grad():
        if args.only_visual:
            for i, (image, idx) in enumerate(tqdm(data_loader_image, desc="Processing", leave=False)):
                
                image = image.to(device)
                idx_list.append(concat_all_gather(torch.tensor(idx).to(device)))
                
                sims_matrix = adapt_model(image, device, args, metric_logger)
                
                # Evaluation
                start, end = i * bs, min(score_matrix_i2t.size(0), (i + 1) * bs)
                if len_image % 2 == 1 and end == len_image:
                    sims_matrix = sims_matrix[:-1]
                
                score_matrix_i2t[start:end] = sims_matrix
        else:
            for i, (text, idx) in enumerate(tqdm(data_loader_text, desc="Processing", leave=False)):
                
                idx_list.append(concat_all_gather(torch.tensor(idx).to(device)))
                
                sims_matrix = adapt_model(text, device, args, metric_logger)
                
                # Evaluation
                start, end = i * bs, min(score_matrix_t2i.size(0), (i + 1) * bs)
                score_matrix_t2i[start:end] = sims_matrix
    
    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Test time adaptation time {total_time_str}")
    
    idx_tensor = torch.cat(idx_list)
    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy(), idx_tensor.cpu().numpy()

def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed); cudnn.benchmark = True

    print("Creating retrieval dataset")
    args.backbone = config['backbone']
    ds = create_dataset(f'retrieval_{config["dataset"]}', config, resize=224) if config['backbone']=='clip' \
         else create_dataset(f'retrieval_{config["dataset"]}', config)
    ds_text, ds_image = ds if isinstance(ds, tuple) else (None, None)
    
    if args.distributed:
        nt, gr = utils.get_world_size(), utils.get_rank()
        if args.retrieval == 't2i':
            samplers = create_sampler([ds_text], [True], nt, gr) + create_sampler([ds_image], [False], nt, gr)
        elif args.retrieval == 'i2t':
            samplers = create_sampler([ds_text], [False], nt, gr) + create_sampler([ds_image], [True], nt, gr)
        else:
            samplers = [None, None]
        config['batch_size_text'] = config['batch_size_image'] = config['batch_size_total'] // nt
    else:
        samplers = [None, None]
    
    loader_text, loader_image = create_loader([ds_text, ds_image], samplers,
                                    batch_size=[config['batch_size_text'], config['batch_size_image']],
                                    num_workers=[1, 1], is_trains=[False, False],
                                    collate_fns=[None, None], config=config)
    
    print("Creating model")
    # Set learning rate and weight decay
    if args.retrieval == 'i2t': 
        config['init_lr'], config['weight_decay'] = 3e-4, 1e-4
    else:
        config['init_lr'] = 3e-4 if config['backbone'] in ['clip', 'clip_reid'] else 3e-5
        config['weight_decay'] = 1e-4

    if config['backbone'] == 'blip':
        model = BLIP.blip_retrieval(pretrained=config['pretrained'], image_size=config['image_size'],
                                    vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'],
                                    vit_ckpt_layer=config['vit_ckpt_layer'])
    elif config['backbone'] == 'clip':
        if config['dataset'] == "fashion_gen_detail":
            model = CLIP.clip_retrieval(pretrained=config["pretrained"], device=device, fashion=True)
        else:
            model = CLIP.clip_retrieval(device=device)
    elif config['backbone'] == 'clip_reid':
        model = CLIP_Reid.reid_clip_retrieval(pretrained=config["pretrained"], args=args)
        
    model = model.to(device)    
    if args.retrieval is not None:
        args.only_visual = True if args.retrieval == 'i2t' else False
        freeze_fn = {'blip': BLIP.freeze_parameters, 'clip': CLIP.freeze_parameters, 'clip_reid': CLIP_Reid.freeze_parameters}
        model = freeze_fn[config['backbone']](model, only_visual=args.only_visual)
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        collect_fn = {'blip': BLIP.collect_params, 'clip': CLIP.collect_params, 'clip_reid': CLIP_Reid.collect_params}
        params = collect_fn[config['backbone']](model.module, only_visual=args.only_visual)[0]
        optimizer = set_optimizer(params, config, args=args)
    else:
        optimizer = None

    if args.retrieval is not None:
        model = set_model(model, optimizer, args)

    print("Start training")
    t0 = time.time()
    if args.retrieval is not None:
        score_i2t, score_t2i, idx_trans = test_time_tune(model, loader_text, loader_image, optimizer, device, config, args)
        test_result = itm_eval(score_i2t, score_t2i, args, loader_image, idx_trans)
    else:
        score_i2t, score_t2i = evaluation(model, loader_text, loader_image, device, args, config=config)
        test_result = itm_eval(score_i2t, score_t2i, args, loader_image)
        
    if args.retrieval == "i2t":
        print("Test Result:", {k: test_result[k] for k in ['txt_r1', 'txt_r5', 'txt_r10', 'txt_r_mean', 'txt_map']})
    elif args.retrieval == "t2i":
        print("Test Result:", {k: test_result[k] for k in ['img_r1', 'img_r5', 'img_r10', 'img_r_mean', 'img_map']})
    else:
        print("Test Result:", test_result)
    
    with open(os.path.join(args.output_dir, "evaluate.txt"), "a") as f:
        f.write(json.dumps(test_result) + "\n")
    print("Training time:", str(datetime.timedelta(seconds=int(time.time()-t0))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/QGS/i2t/retrieval_coco.yaml')
    parser.add_argument('--output_dir', default='output/Retrieval_flickr')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--distributed', type=bool, default=True)
    parser.add_argument('--retrieval', type=str, default=None)
    parser.add_argument('--tta_steps', type=int, default=3)
    parser.add_argument('--temperature', type=float, default=0.02)
    parser.add_argument('--method', default='tent')
    args = parser.parse_args()
    
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    args = set_temperature(args, config)
    
    main(args, config)
