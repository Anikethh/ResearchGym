import torch
import argparse
from functools import partial
import matplotlib.pyplot as plt

from txai.utils.data.preprocess import zip_x_time_y
from txai.utils.data import process_Synth, process_PAM, process_Epilepsy
from txai.synth_data.generate_spikes import SpikeTrainDataset
from txai.utils.data import EpiDataset#, PAMDataset

from txai.models.encoders.transformer_simple import TransformerMVTS
# from txai.models.gumbelmask_model import GumbelMask
# from txai.models.kumamask_model import TSKumaMask_TransformerPred as TSKumaMask


# Import all explainers:
from txai.baselines import TSR
from captum.attr import IntegratedGradients, Saliency, DeepLift, GradientShap
from txai.utils.baseline_comp.run_dynamask import run_dynamask

def get_explainer(key, args, device = None):

    key = key.lower()

    needs_training = False

    if key == 'fit':
        # Need to ensure we have generator
        needs_training = True
        pass

    elif key == 'dyna':
        explainer = partial(run_dynamask, device = device)

    elif key == 'tsr':
        def explainer(model, x, time, y):
            GradExplainer = Saliency(model)
            x = x.unsqueeze(0)
            time = time.transpose(0,1)
            out = TSR(GradExplainer, x, y, additional_forward_args = (time, None, True, False))
            return torch.from_numpy(out).to(device)

    elif key == 'ig_our22':
        def explainer(model, x, time, y): 
            IG = IntegratedGradients(model)
            # Transform inputs to captum-like (batch first):
            x = x.transpose(0, 1)
            time = time.transpose(0,1)
            
            attr_first = IG.attribute(x, target=y, baselines= (x * 0), additional_forward_args = (time, None, True))

            q50 = torch.quantile(attr_first.reshape(-1), 0.5)
            q50 = 0.0

            baselines = x.clone()
            baselines[attr_first > q50] = 0.0
            
            attr_second = IG.attribute(x, target=y, baselines= baselines, additional_forward_args = (time, None, True))
            attr_second[attr_first <= q50] = attr_first[attr_first <= q50]
            return attr_second

    elif key == 'ig':
        if args.cf == 1:  
            def explainer(model, x, time, y): 
                IG = IntegratedGradients(model)
                # Transform inputs to captum-like (batch first):
                x = x.transpose(0, 1)
                time = time.transpose(0,1)
            
                baselines = torch.zeros_like(x)
                baselines[:, 1:, :] = x[:, :-1, :]
                attr = IG.attribute(x, target = y, baselines=baselines, additional_forward_args = (time, None, True))
            
                return attr
        else:
            def explainer(model, x, time, y): 
                IG = IntegratedGradients(model)
                # Transform inputs to captum-like (batch first):
                x = x.transpose(0, 1)
                time = time.transpose(0,1)
                
                attr = IG.attribute(x, target=y, additional_forward_args = (time, None, True))

                return attr
        
    
    elif key == 'ig_online':
        def explainer(model, x, time, y): 
            IG = IntegratedGradients(model)
            # Transform inputs to captum-like (batch first):
            x = x.transpose(0, 1)
            time = time.transpose(0,1)
            
            attr = torch.zeros_like(x)
            for t in range(x.shape[1]):
                baselines = x.clone()
                baselines[:, t, :] = 0
                attr[:, t, :] = IG.attribute(
                    x,
                    target=y,
                    baselines=baselines,
                    additional_forward_args=(time, None, True),
                )[:, t, :]
            
            return attr
        
    elif key == 'ig_online_feature':
        def explainer(model, x, time, y): 
            IG = IntegratedGradients(model)
            # Transform inputs to captum-like (batch first):
            x = x.transpose(0, 1)
            time = time.transpose(0,1)
            
            attr_f = torch.zeros_like(x)
            for f in range(x.shape[2]):
                baselines = x.clone()
                baselines[:, :, f] = 0
                attr_f[:, :, f] = IG.attribute(
                    x,
                    target=y,
                    baselines=baselines,
                    additional_forward_args=(time, None, True),
                )[:, :, f]
                
            attr_t = torch.zeros_like(x)
            for t in range(x.shape[1]):
                baselines = x.clone()
                baselines[:, t, :] = 0
                attr_t[:, t, :] = IG.attribute(
                    x,
                    target=y,
                    baselines=baselines,
                    additional_forward_args=(time, None, True),
                )[:, t, :]
            
            return attr_t + attr_f
    
    elif key == 'ig_feature':
        def explainer(model, x, time, y): 
            IG = IntegratedGradients(model)
            
            # Transform inputs to captum-like (batch first):
            x = x.transpose(0, 1)
            time = time.transpose(0,1)
            
            attr = torch.zeros_like(x)
            for f in range(x.shape[2]):
                baselines = x.clone()
                baselines[:, :, f] = 0
                attr[:, :, f] = IG.attribute(
                    x,
                    target=y,
                    baselines=baselines,
                    additional_forward_args=(time, None, True),
                )[:, :, f]
            
            return attr
    
    elif key == 'ig_point':
        def explainer(model, x, time, y): 
            IG = IntegratedGradients(model)
            # Transform inputs to captum-like (batch first):
            x = x.transpose(0, 1)
            time = time.transpose(0,1)
            
            attr = torch.zeros_like(x)
            for t in range(x.shape[1]):
                for f in range(x.shape[2]):
                    baselines = x.clone()
                    baselines[:, t, f] = 0
                    attr[:, t, f] = IG.attribute(
                        x,
                        target=y,
                        baselines=baselines,
                        additional_forward_args=(time, None, True),
                    )[:, t, f]
            
            return attr
    
    elif key == 'deeplift':
        if args.cf == 1:  
            def explainer(model, x, time, y): 
                IG = DeepLift(model)
                # Transform inputs to captum-like (batch first):
                x = x.transpose(0, 1)
                time = time.transpose(0,1)
            
                baselines = torch.zeros_like(x)
                baselines[:, 1:, :] = x[:, :-1, :]
                attr = IG.attribute(x, target = y, baselines=baselines, additional_forward_args = (time, None, True))
            
                return attr
        else:
            def explainer(model, x, time, y): 
                IG = DeepLift(model)
                # Transform inputs to captum-like (batch first):
                x = x.transpose(0, 1)
                time = time.transpose(0,1)
                
                attr = IG.attribute(x, target=y, additional_forward_args = (time, None, True))

                return attr
    
    elif key == 'gradientshap':
        if args.cf == 1:  
            def explainer(model, x, time, y): 
                IG = GradientShap(model)
                # Transform inputs to captum-like (batch first):
                x = x.transpose(0, 1)
                time = time.transpose(0,1)
            
                baselines = torch.zeros_like(x)
                baselines[:, 1:, :] = x[:, :-1, :]
                attr = IG.attribute(x, target = y, baselines=baselines, additional_forward_args = (time, None, True))
            
                return attr
        else:
            def explainer(model, x, time, y): 
                IG = GradientShap(model)
                # Transform inputs to captum-like (batch first):
                x = x.transpose(0, 1)
                time = time.transpose(0,1)
                
                attr = IG.attribute(x, target=y, additional_forward_args = (time, None, True))

                return attr
            # if args.cf == 1:          
            #     baselines = torch.zeros_like(x)
            #     baselines[:, 1:, :] = x[:, :-1, :]
            #     attr = IG.attribute(x, target = y, baselines=baselines, additional_forward_args = (time, None, True))
            # else:
            #     attr = IG.attribute(x, target=y, additional_forward_args = (time, None, True))
            # target_list = [0, 1, 2, 3]
            # target_list.remove(y.cpu().numpy())
            
            # attr = IG.attribute(x, target = y, baselines=baselines, additional_forward_args = (time, None, True))
            # for i in target_list:
            #     attr -= IG.attribute(x, target = i, baselines=baselines, additional_forward_args = (time, None, True))
            
            # return attr

    elif key == 'random':
        def explainer(model, x, time, y):
            return torch.randn_like(x).squeeze(1).float()

    elif key == 'attn':
        pass

    elif key == 'attngrad':
        pass

    elif key == 'attnnorm':
        pass

    elif key == 'model':
        def explainer(model, x, time, y):
            #model.eval()
            pred, mask = model(x.unsqueeze(1), time, captum_input = False)
            return mask

    #elif key == 'model_adv':
        #def model_adv(extractor, predictor, x, time, y):

    else:
        raise NotImplementedError('Cannot find explainer "{}"'.format(key))

    return explainer, needs_training

def get_dataset(data, split, device = None):
    '''
    Gets dataset based on only string entry for data and split given by number
    '''

    data = data.lower()

    if data == 'pam':
        train, val, test = process_PAM(split_no = split, device = device, 
            base_path = '/home/owq978/TimeSeriesXAI/datasets/PAMAP2data/', gethalf = True)
    
    elif (data == 'epi') or (data == 'epilepsy'):
        train, val, test = process_Epilepsy(split_no = split, device = device, 
            base_path = '/home/owq978/TimeSeriesXAI/datasets/Epilepsy/')

    elif (data == 'spike'):
        D = process_Synth(split_no = split, device = device, 
            base_path = '/home/owq978/TimeSeriesXAI/datasets/Spike/simple/')
        
        train = D['train_loader']
        val = D['val']
        test = D['test']

    return train, val, test