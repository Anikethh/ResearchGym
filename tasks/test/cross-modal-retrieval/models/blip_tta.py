from models.med import BertConfig, BertModel
from transformers import BertTokenizer
import torch
from torch import nn
import torch.nn.functional as F
from models.blip import create_vit, init_tokenizer, load_checkpoint
from ddp import *

def get_text_embeds_blip(data_loader, model, device, args):
    num_length=len(data_loader.dataset.text)
    text_embeds = torch.zeros(num_length, 256).to(device)

    for text, index in data_loader:
        text_input = model.tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device)
        text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
        text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)
        text_embeds[index] = text_embed

    # All-reduce to aggregate embeddings across GPUs
    if args.distributed:
        dist.all_reduce(text_embeds, op=dist.ReduceOp.SUM)
    if torch.any(torch.all(text_embeds == 0, dim=1)):
        raise ValueError("There is at least one row in text_embeds that is all zeros.")

    return text_embeds

def get_image_embeds_blip(data_loader, model, device, args):
    num_length=len(data_loader.dataset.image)
    image_embeds = torch.zeros(num_length, 256).to(device)

    for image, index in data_loader:
        image = image.to(device)
        image_feat = model.visual_encoder(image)
        image_embed = model.vision_proj(image_feat[:, 0, :])
        image_embed = F.normalize(image_embed, dim=-1)
        image_embeds[index] = image_embed

    # All-reduce to aggregate embeddings across GPUs
    if args.distributed:
        dist.all_reduce(image_embeds, op=dist.ReduceOp.SUM)
    if torch.any(torch.all(image_embeds == 0, dim=1)):
        raise ValueError("There is at least one row in text_embeds that is all zeros.")

    return image_embeds

class BLIP_Retrieval(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 384,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                      
                 embed_dim = 256,     
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)          
        text_width = self.text_encoder.config.hidden_size
        
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.image_features=None
        self.text_features=None

    def set_image_features(self, image_features=None):
        self.image_features = image_features

    def set_text_features(self,text_features=None):
        self.text_features = text_features

    def encode_image(self, image):
        image_embeds = self.visual_encoder(image)      
        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]), dim=1)  
        assert self.text_features is not None, "Error: text is None"
        return image_feat

    def encode_text(self, text, device):
        text_input = self.tokenizer(text, padding='max_length', truncation=True, max_length=35, 
                              return_tensors="pt").to(device) 
        text_output = self.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:,0,:]), dim=1)
        assert self.image_features is not None, "Error: image is None"
        return text_feat
    
    def forward_output(self, modality_query, device, args):
        if args.retrieval=='i2t':
            modality_gallery_feat_all=self.text_features
            modality_query_feat=self.encode_image(modality_query)
            modality_query_feat=all_gather_with_grad(modality_query_feat)
        else:
            modality_gallery_feat_all=self.image_features
            modality_query_feat=self.encode_text(modality_query, device)
            modality_query_feat=all_gather_with_grad(modality_query_feat)

        sim_matrix = modality_query_feat @ modality_gallery_feat_all.t()
        
        sim_inter = sim_matrix/args.temperature
        return sim_inter
    
    def forward_output_without_ddp(self, modality_query, device, args):
        if args.retrieval=='i2t':
            modality_gallery_feat_all=self.text_features
            modality_query_feat=self.encode_image(modality_query)
        else:
            modality_gallery_feat_all=self.image_features
            modality_query_feat=self.encode_text(modality_query, device)

        sim_matrix = modality_query_feat @ modality_gallery_feat_all.t()
        
        sim_inter = sim_matrix/args.temperature
        return sim_inter

    def forward_tta(self, modality_query, device, queue_list, max_queue_size, update_signal, step, args):
        if args.retrieval=='i2t':
            modality_gallery_feat_all=self.text_features
            modality_query_feat=self.encode_image(modality_query)
            modality_query_feat=all_gather_with_grad(modality_query_feat)
        else:
            modality_gallery_feat_all=self.image_features
            modality_query_feat=self.encode_text(modality_query, device)
            modality_query_feat=all_gather_with_grad(modality_query_feat)

        sim_matrix = modality_query_feat @ modality_gallery_feat_all.t()
        nearest_neighbors_indices = (sim_matrix).argmax(dim=1)
        modality_gallery_feat = modality_gallery_feat_all[nearest_neighbors_indices]

        if (step==0 and update_signal):
            queue_list=update_queue(modality_query_feat, modality_gallery_feat, queue_list, args.con_ratio, max_queue_size, args)

        margin, entropy_queue=get_current_value(queue_list)
        outputs = (modality_query_feat @ modality_gallery_feat.t())
        sim_inter = outputs /args.temperature

        loss_REM=entropy_loss_against_noisy(sim_inter, entropy_queue)
        loss_UNI=center_uniform_loss(modality_query_feat, t=args.t)

        target_modality_gap=compute_modality_gap(modality_query_feat, modality_gallery_feat)
        loss_EMG=(target_modality_gap-margin)**2

        return loss_REM, loss_UNI, loss_EMG, queue_list, sim_matrix
    
    def forward_tta_untrain(self, modality_query, device, queue_list, max_queue_size, update_signal, step, args, scale):
        if args.retrieval=='i2t':
            modality_gallery_feat_all=self.text_features
            modality_query_feat=self.encode_image(modality_query)
            modality_query_feat=all_gather_with_grad(modality_query_feat)
        else:
            modality_gallery_feat_all=self.image_features
            modality_query_feat=self.encode_text(modality_query, device)
            modality_query_feat=all_gather_with_grad(modality_query_feat)

        sim_matrix = modality_query_feat @ modality_gallery_feat_all.t()
        nearest_neighbors_indices = (sim_matrix).argmax(dim=1)
        modality_gallery_feat = modality_gallery_feat_all[nearest_neighbors_indices]

        if (step==0 and update_signal):
            queue_list=update_queue(modality_query_feat, modality_gallery_feat, queue_list, args.con_ratio, max_queue_size, args)

        margin, entropy_queue=get_current_value(queue_list)

        target_modality_gap=compute_modality_gap(modality_query_feat, modality_gallery_feat)

        # #uniformity
        center_query=torch.mean(modality_query_feat, dim=0)
        modality_query_feat=scale*(modality_query_feat-center_query)+center_query

        #modality gap
        shift=torch.mean(modality_gallery_feat, dim=0)-center_query
        rate=margin/target_modality_gap
        modality_query_feat=modality_query_feat+(1-rate)*shift
        modality_query_feat=F.normalize(modality_query_feat)
        sim_matrix=modality_query_feat @ modality_gallery_feat_all.t()

        return queue_list, sim_matrix

def blip_retrieval(pretrained='',**kwargs):
    model = BLIP_Retrieval(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        print("missing keys:")
        print(msg.missing_keys)
    
    return model 

def freeze_parameters(model,only_visual):
    model.train()
    model.requires_grad_(False)
    if only_visual:
        print("only_visual")
        for name, param in model.visual_encoder.named_parameters():
            if ('norm' in name) or ('Norm' in name):
                param.requires_grad_(True)
    else:
        print("only_text")
        for name, param in model.text_encoder.named_parameters():
            if ('norm' in name) or ('Norm' in name): 
                param.requires_grad_(True)
    return model

def collect_params(model, only_visual):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    if only_visual:
        for nm, m in model.visual_encoder.named_modules():
            if isinstance(m, (nn.LayerNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")
    else:
        for nm, m in model.text_encoder.named_modules():
            if isinstance(m, (nn.LayerNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")

    return params, names

