import torch
from models.tta_baselines.tent import Tent
from models.tta_baselines.sar import SAR, SAM
# from models.tta_baselines.eata import EATA
from models.tta_baselines.read import READ
from models.tta_baselines.shot import SHOT
# from models.tta_baselines.deyo import DeYO

def set_optimizer(params, config, args):
    if args.method=='sar':
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(params=params, base_optimizer=base_optimizer, lr=config['init_lr'], weight_decay=config['weight_decay'])
    else:
        optimizer = torch.optim.AdamW(params=params, lr=config['init_lr'], weight_decay=config['weight_decay']) 
    return optimizer

def set_model(model, optimizer, args, margin_e0=0):
    if args.method=='tent':
        model=Tent(model, optimizer, steps=args.tta_steps)
    elif args.method=='sar':
        model=SAR(model, optimizer, steps=args.tta_steps, margin_e0=0.40)
    elif args.method=='read':
        model=READ(model, optimizer, steps=args.tta_steps)
    elif args.method=='shot':
        model=SHOT(model, optimizer, steps=args.tta_steps, threshold=0.9, clf_coeff=0.1)
    # elif args.method=='eata':
    #     model=EATA(model, optimizer, steps=args.tta_steps, fishers=None, e_margin=0.40)
    # elif args.method=='deyo':
    #     model=DeYO(model, args, optimizer, steps=args.tta_steps, deyo_margin=0.50, margin_e0=0.40)
    else:
        assert False, "Not correct model name."

    return model

def set_temperature(args, config):
    dataset_temperature_map = {
        'flickr': 0.01,
        'coco': 0.01,
        'fashion_gen': 0.001,
        'fashion_gen_detail': 0.001,
        'cuhk_pedes': 0.0001,
        'icfg_pedes': 0.0001,
        'nocaps_entire_domain': 0.01,
        'nocaps_in_domain': 0.01,
        'nocaps_near_domain': 0.01,
        'nocaps_out_domain': 0.01
    }
    
    dataset = config["dataset"] 
    temperature = dataset_temperature_map.get(dataset, 0.01) 
    args.temperature = temperature 
    return args


    

        
    
