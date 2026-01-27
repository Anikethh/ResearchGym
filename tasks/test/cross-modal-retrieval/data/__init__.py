import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from data.coco_dataset import coco_retrieval_eval_text, coco_retrieval_eval_image
from data.flickr30k_dataset import flickr30k_retrieval_eval_text, flickr30k_retrieval_eval_image
from data.fashion_gen_dataset import fashion_gen_retrieval_eval_text, fashion_gen_retrieval_eval_image
from data.cuhk_pedes_dataset import cuhk_pedes_retrieval_eval_text, cuhk_pedes_retrieval_eval_image
from data.icfg_pedes_dataset import icfg_pedes_retrieval_eval_text, icfg_pedes_retrieval_eval_image
from data.nocaps_in_domain_dataset import nocaps_in_domain_eval_image, nocaps_in_domain_eval_text
from data.nocaps_near_domain_dataset import nocaps_near_domain_eval_image, nocaps_near_domain_eval_text
from data.nocaps_out_domain_dataset import nocaps_out_domain_eval_image, nocaps_out_domain_eval_text
from data.nocaps_entire_domain_dataset import nocaps_entire_domain_eval_image, nocaps_entire_domain_eval_text
from data.fashion_gen_detail_dataset import fashion_gen_detail_retrieval_eval_text, fashion_gen_detail_retrieval_eval_image

def create_dataset(dataset, config, min_scale=0.5, resize=384):
    
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    transform_test = transforms.Compose([
        transforms.Resize((resize, resize),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])  
    transform_test_reid = transforms.Compose([
            transforms.Resize((384, 128)),
            transforms.ToTensor(),
            normalize,
        ])
    

    if dataset=='retrieval_coco':           
        test_text_dataset = coco_retrieval_eval_text(transform_test, config['image_root'], config['ann_root'], 'test')     
        test_image_dataset = coco_retrieval_eval_image(transform_test, config['image_root'], config['ann_root'], 'test')         
        return test_text_dataset, test_image_dataset  
    
    elif dataset=='retrieval_flickr':          
        test_text_dataset = flickr30k_retrieval_eval_text(transform_test, config['image_root'], config['ann_root'], 'test')     
        test_image_dataset = flickr30k_retrieval_eval_image(transform_test, config['image_root'], config['ann_root'], 'test')          
        return test_text_dataset, test_image_dataset 
    
    elif dataset=='retrieval_fashion_gen':
        test_text_dataset = fashion_gen_retrieval_eval_text(transform_test, config['image_root'], config['ann_root'], 'test')     
        test_image_dataset = fashion_gen_retrieval_eval_image(transform_test, config['image_root'], config['ann_root'], 'test')   
        return test_text_dataset, test_image_dataset
    
    elif dataset=='retrieval_cuhk_pedes':
        test_text_dataset = cuhk_pedes_retrieval_eval_text(transform_test_reid, config['image_root'], config['ann_root'], 'test')     
        test_image_dataset = cuhk_pedes_retrieval_eval_image(transform_test_reid, config['image_root'], config['ann_root'], 'test')          
        return test_text_dataset, test_image_dataset 
    
    elif dataset=='retrieval_icfg_pedes':
        test_text_dataset = icfg_pedes_retrieval_eval_text(transform_test_reid, config['image_root'], config['ann_root'], 'test')     
        test_image_dataset = icfg_pedes_retrieval_eval_image(transform_test_reid, config['image_root'], config['ann_root'], 'test')          
        return test_text_dataset, test_image_dataset 
    
    elif dataset=='retrieval_nocaps_in_domain':
        test_text_dataset = nocaps_in_domain_eval_text(transform_test, config['image_root'], config['ann_root'])     
        test_image_dataset = nocaps_in_domain_eval_image(transform_test, config['image_root'], config['ann_root'])          
        return test_text_dataset, test_image_dataset 
    
    elif dataset=='retrieval_nocaps_out_domain':
        test_text_dataset = nocaps_out_domain_eval_text(transform_test, config['image_root'], config['ann_root'])     
        test_image_dataset = nocaps_out_domain_eval_image(transform_test, config['image_root'], config['ann_root'])          
        return test_text_dataset, test_image_dataset 
    
    elif dataset=='retrieval_nocaps_near_domain':
        test_text_dataset = nocaps_near_domain_eval_text(transform_test, config['image_root'], config['ann_root'])     
        test_image_dataset = nocaps_near_domain_eval_image(transform_test, config['image_root'], config['ann_root'])          
        return test_text_dataset, test_image_dataset 
    
    elif dataset=='retrieval_nocaps_entire_domain':
        test_text_dataset = nocaps_entire_domain_eval_text(transform_test, config['image_root'], config['ann_root'])     
        test_image_dataset = nocaps_entire_domain_eval_image(transform_test, config['image_root'], config['ann_root'])          
        return test_text_dataset, test_image_dataset 
    
    elif dataset=='retrieval_fashion_gen_detail':
        test_text_dataset = fashion_gen_detail_retrieval_eval_text(transform_test, config['image_root'], config['ann_root'], 'test')    
        test_image_dataset = fashion_gen_detail_retrieval_eval_image(transform_test, config['image_root'], config['ann_root'], 'test')  
        return test_text_dataset, test_image_dataset 

    
    
def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns, config):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
        shuffle = False
        drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    
