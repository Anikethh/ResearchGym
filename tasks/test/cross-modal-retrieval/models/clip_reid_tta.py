import torch
import torch.nn.functional as F
from ddp import *
from .clip_reid_model import build_CLIP_from_openai_pretrained, convert_weights
import torch
import torch.nn as nn

class IRRA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained('ViT-B/16', (384, 128), 16)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 

        self.image_features=None
        self.text_features=None

    def set_image_features(self, image_features=None):
        self.image_features = image_features

    def set_text_features(self,text_features=None):
        self.text_features = text_features
    
    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        x = x[:, 0, :].float()
        return F.normalize(x, dim=1)

    def encode_text(self, text, device=torch.device('cuda')):
        text=text.to(device)
        x = self.base_model.encode_text(text)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()
        return F.normalize(x, dim=1)

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
        sim_inter = (modality_query_feat @ modality_gallery_feat.t()) /args.temperature

        loss_REM=entropy_loss_against_noisy(sim_inter, entropy_queue)
        loss_UNI=center_uniform_loss(modality_query_feat, t=args.t)

        target_modality_gap=compute_modality_gap(modality_query_feat, modality_gallery_feat)
        loss_EMG=(target_modality_gap-margin)**2

        return loss_REM, loss_UNI, loss_EMG, queue_list, sim_matrix

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

def reid_clip_retrieval(pretrained, args):
    model = IRRA(args)
    convert_weights(model)
    if pretrained:
        model, msg = load_checkpoint(model,pretrained)
        print("missing keys:")
        print(msg['missing_keys']) 
    return model

def load_checkpoint(model, pretrained):
    checkpoint = torch.load(pretrained)
    model.load_state_dict(checkpoint['model'], strict=False)
    msg = {}
    msg['missing_keys'] = [k for k in checkpoint['model'].keys() if k not in model.state_dict().keys()]
    return model, msg

def freeze_parameters(model,only_visual):

    model.train()
    model.requires_grad_(False)
    if only_visual:
        print("only_visual")

        for name, param in model.base_model.visual.named_parameters():
            if ('ln' in name):
                param.requires_grad_(True)
    else:
        print("only_text")
        for name, param in model.base_model.transformer.named_parameters():
            if ('ln' in name): 
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
        for nm, m in model.base_model.visual.named_modules():
            if isinstance(m, (nn.LayerNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")
    else:
        for nm, m in model.base_model.transformer.named_modules():
            if isinstance(m, (nn.LayerNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")

    return params, names

def get_text_embeds_clip_reid(data_loader, model, device, args):
    num_length=len(data_loader.dataset.text)
    text_embeds = torch.zeros(num_length, 512).to(device)

    for text, index in data_loader:
        # text=text.to(device)
        text_output = model.encode_text(text, device)
        text_embed=F.normalize(text_output,dim=-1)
        text_embeds[index] = text_embed

    # All-reduce to aggregate embeddings across GPUs
    if args.distributed:
        dist.all_reduce(text_embeds, op=dist.ReduceOp.SUM)
    if torch.any(torch.all(text_embeds == 0, dim=1)):
        raise ValueError("There is at least one row in text_embeds that is all zeros.")

    return text_embeds

def get_image_embeds_clip_reid(data_loader, model, device, args):
    num_length=len(data_loader.dataset.image)
    image_embeds = torch.zeros(num_length, 512).to(device)

    for image, index in data_loader:
        image = image.to(device) 
        image_embed = model.encode_image(image)           
        image_embed = F.normalize(image_embed,dim=-1)   
        image_embeds[index] = image_embed

    # All-reduce to aggregate embeddings across GPUs
    if args.distributed:
        dist.all_reduce(image_embeds, op=dist.ReduceOp.SUM)
    if torch.any(torch.all(image_embeds == 0, dim=1)):
        raise ValueError("There is at least one row in text_embeds that is all zeros.")

    return image_embeds
