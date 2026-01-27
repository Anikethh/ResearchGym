import pickle
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn

from maple_clip import clip
from maple_clip.simple_tokenizer import SimpleTokenizer as _Tokenizer


from maple_clip.model import convert_weights

_tokenizer = _Tokenizer()

class Feature_Trans_Module_two_layer(nn.Module):
    def __init__(self, input_dim=100, out_dim=256):
        super(Feature_Trans_Module_two_layer, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, out_dim, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, 1)
        )
    def forward(self, input_feat):
        
        final_feat = self.conv1(input_feat.unsqueeze(-1).unsqueeze(-1))
        
        return final_feat.squeeze(-1).squeeze(-1)
        
def load_clip_to_cpu_teacher():
    
    model_path = '/mnt/workspace/workgroup/jinmu/test_time_adaption/TPT_TEXT_MLC/~/.cache/clip/ViT-L-14.pt'
    print("CLIP Teacher name is ViT-L-14")
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    # We default use PromptSRC to pretrain our teacher model
    design_details = {"trainer": 'IVLP',
                        "vision_depth": 9,
                        "language_depth": 9,
                        "vision_ctx": 4,
                        "language_ctx": 4}
    
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model


def load_clip_to_cpu():
    model_path = '/mnt/workspace/workgroup/jinmu/test_time_adaption/TPT_TEXT_MLC/~/.cache/clip/ViT-B-16.pt'
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {"trainer": 'IVLP',
                      "vision_depth": 9,
                      "language_depth": 9,
                      "vision_ctx": 4,
                      "language_ctx": 4}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        
        # print(f'------prompts size is {prompts.size()}------')
        # print(f'------tokenized prompts size is {tokenized_prompts.size()}------')

        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class VLPromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, is_teacher):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 4
        ctx_init = "a_photo_of_a"
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        
        self.trainer_name = "PromptKD"
        self.train_modal = "cross"
        
        ctx_init_neg = ctx_init
        if ctx_init and n_ctx <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
            
            ctx_init_neg = ctx_init_neg.replace("_", " ")
            prompt_neg = clip.tokenize(ctx_init_neg)
            with torch.no_grad():
                embedding_neg = clip_model.token_embedding(prompt_neg).type(dtype)
            ctx_vectors_neg = embedding_neg[0, 1: 1 + n_ctx, :]
            prompt_prefix_neg = ctx_init_neg
            
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print(f"Independent V-L design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print(f"Number of context words (tokens) for Vision prompting: 4")
        
        self.prompt_prefix = prompt_prefix
        self.ctx_init_state = ctx_vectors.detach().clone()
        self.ctx = nn.Parameter(ctx_vectors)
        
        self.prompt_prefix_neg = prompt_prefix_neg
        self.ctx_init_state_neg = ctx_vectors_neg.detach().clone()
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)

        classnames = [name.replace("_", " ") for name in classnames]
        
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        
        prompts_neg = [prompt_prefix_neg + " " + name + "." for name in classnames]
        tokenized_prompts_neg = torch.cat([clip.tokenize(p) for p in prompts_neg])  # (n_cls, n_tkn)
        
        print(f'classnames size is {len(classnames)}')

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            embedding_neg = clip_model.token_embedding(tokenized_prompts_neg).type(dtype)
        
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.tokenized_prompts_neg = tokenized_prompts_neg  # torch.Tensor
        # self.name_lens = name_lens

        if self.train_modal == "cross":
            self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
            
            self.register_buffer("token_prefix2", embedding[:, :1, :])  # SOS
            self.register_buffer("token_suffix2", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
            
            self.register_buffer("token_prefix_neg", embedding_neg[:, :1, :])  # SOS
            self.register_buffer("token_suffix_neg", embedding_neg[:, 1 + n_ctx:, :])  # CLS, EOS
            
            self.register_buffer("token_prefix2_neg", embedding_neg[:, :1, :])  # SOS
            self.register_buffer("token_suffix2_neg", embedding_neg[:, 1 + n_ctx:, :])  # CLS, EOS
            
    def reset(self):

        self.ctx.copy_(self.ctx_init_state) # to be optimized
        self.ctx_neg.copy_(self.ctx_init_state_neg) # to be optimized

    def construct_prompts(self, ctx, prefix, suffix, ctx_neg, prefix_neg, suffix_neg):

        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        prompts_neg = torch.cat([prefix_neg, ctx_neg, suffix_neg], dim=1)

        return prompts, prompts_neg

    def forward(self):
        ctx = self.ctx
        ctx_neg = self.ctx_neg
        
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            ctx_neg = ctx_neg.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        
        prefix_neg = self.token_prefix_neg
        suffix_neg = self.token_suffix_neg

        if self.trainer_name == "PromptKD" and self.train_modal == "base2novel":
            
            prefix = torch.cat([prefix, self.token_prefix2], dim=0)
            suffix = torch.cat([suffix, self.token_suffix2], dim=0)
            
            prefix_neg = torch.cat([prefix_neg, self.token_prefix2_neg], dim=0)
            suffix_neg = torch.cat([suffix_neg, self.token_suffix2_neg], dim=0)

        prompts, prompts_neg = self.construct_prompts(ctx, prefix, suffix, ctx_neg, prefix_neg, suffix_neg)

        return prompts, prompts_neg

class CustomCLIP(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.total_epochs = 20
        self.n_cls = len(classnames)
        
        self.VPT_image_trans = Feature_Trans_Module_two_layer(512, 768)
        
        self.VPT_image_trans = self.VPT_image_trans.cuda()
        convert_weights(self.VPT_image_trans)

    def forward(self, image):
        logit_scale = self.logit_scale.exp()
        
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = self.VPT_image_trans(image_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features, logit_scale


class CustomCLIP_teacher(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.prompt_learner = VLPromptLearner(classnames, clip_model, True)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.tokenized_prompts_neg = self.prompt_learner.tokenized_prompts_neg
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model).cuda()
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
        # self.clip_model = clip_model
        
    def forward(self, image=None):
        
        prompts, prompts_neg = self.prompt_learner()
        
        
        # cls2coco = {}
        # coco_classname = coco_classname_synonyms = []
        # for c in coco_classname:
        #     for cc in c:
        #         cls2coco[cc] = c[0]
        #         cc = cc.replace(' ', '')
        #         cls2coco[cc] = c[0]
        # # with open('../../prompt_modal_tuning/nips24/cache/train_new_label/test_01_coco_925.pkl', 'rb') as f:
        #     # coco_data = pickle.load(f)
        # # with open('../../prompt_modal_tuning/nips24/cache/train_new_label/test_01_nuswide_925.pkl', 'rb') as f:
        #     # nuswide_data = pickle.load(f)
        # with open('../../prompt_modal_tuning/nips24/cache/train_new_label/test_01_obj_925.pkl', 'rb') as f:
        #     obj_data = pickle.load(f)
        # captions = obj_data[0]
        # cls_list = []
        # caption_list = []
        # embed_list = []
        # step = 32
        # for idx in tqdm(range(0, len(captions), step)):
        #     data = captions[idx:idx+step]
        #     cls_cap = [t.split('&&') for t in data]
        #     text_inputs = torch.cat([clip.tokenize(c[0]) for c in cls_cap]).to(prompts.device)
        #     with torch.no_grad():
        #         text_inputs_embedding = self.clip_model.token_embedding(text_inputs).type(self.dtype)
        #         text_features = self.text_encoder(text_inputs_embedding, text_inputs)
        #     text_features /= text_features.norm(dim=-1, keepdim=True)
        #     text_features = text_features.cpu().numpy()
        #     for i in range(len(cls_cap)):
        #         new_cls = []
        #         for x in cls_cap[i][1].split(','):
        #             if x in cls2coco:
        #                 new_cls.append(cls2coco[x])
        #         if new_cls:
        #             cls_list.append(new_cls)
        #             caption_list.append(cls_cap[i][0])
        #             embed_list.append(text_features[i])
        # print(f"number of text data: {len(cls_list)}")
        # with open('text_data/vit-b-16/promptkd_object365_cls_captions_embed.pkl', 'wb') as f:
        #     pickle.dump([cls_list, caption_list, embed_list], f)
        
        
        tokenized_prompts = self.tokenized_prompts
        tokenized_prompts_neg = self.tokenized_prompts_neg
        
        text_features = self.text_encoder(prompts.cuda(), tokenized_prompts.cuda())
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        text_features_neg = self.text_encoder(prompts_neg.cuda(), tokenized_prompts_neg.cuda())
        text_features_neg = text_features_neg / text_features_neg.norm(dim=-1, keepdim=True)
        
        return text_features, text_features_neg


class PromptKD(nn.Module):
    def __init__(self, is_bind, device, test_set, classnames, clip_arch):

        super().__init__()
        self.is_bind = is_bind
        self.device = device
        self.n_cls = len(classnames)
        
        clip_model = load_clip_to_cpu()
        clip_model_teacher = load_clip_to_cpu_teacher()
        
        clip_model = clip_model.float()
        clip_model_teacher = clip_model_teacher.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(classnames, clip_model)
        self.model_teacher = CustomCLIP_teacher(classnames, clip_model_teacher)
        self.token_embedding = clip_model_teacher.token_embedding
        
        model_path = './pretrain_weights/promptkd/teacher-model-best-base2new.pth.tar'
            
        self.train_modal = "cross"
        
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint["state_dict"]
        
        if "prompt_learner.token_prefix" in state_dict:
            del state_dict["prompt_learner.token_prefix"]
        if "prompt_learner.token_prefix2" in state_dict:
            del state_dict["prompt_learner.token_prefix2"]

        if "prompt_learner.token_suffix" in state_dict:
            del state_dict["prompt_learner.token_suffix"]
        if "prompt_learner.token_suffix2" in state_dict:
            del state_dict["prompt_learner.token_suffix2"]
        
        self.model_teacher.load_state_dict(state_dict, strict=False)
        
        with torch.no_grad():
            
            self.model_teacher.prompt_learner.ctx.copy_(state_dict['prompt_learner.ctx'])
            self.model_teacher.prompt_learner.ctx_init_state = state_dict['prompt_learner.ctx']
            
            self.model_teacher.prompt_learner.ctx_neg.copy_(state_dict['prompt_learner.ctx'])
            self.model_teacher.prompt_learner.ctx_init_state_neg = state_dict['prompt_learner.ctx']
        
        for name in ['VLPromptLearner']:
            model_path = './pretrain_weights/promptkd/student-model-best.pth.tar'

            checkpoint = torch.load(model_path, map_location="cpu")
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]
            if "prompt_learner.token_prefix2" in state_dict:
                del state_dict["prompt_learner.token_prefix2"]
                
            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]
            if "prompt_learner.token_suffix2" in state_dict:
                del state_dict["prompt_learner.token_suffix2"]
                
            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self.model.load_state_dict(state_dict, strict=False)
            
        arch2pkl = {
            "ViT-B/16" : 'vit-b-16',
        }
        
        test_set2pkl = {
            "coco2014" : 'coco',
            "coco2017" : 'coco',
            "voc2007" : 'voc',
            "voc2012" : 'voc',
            "nuswide" : 'nuswide',
            "object365" : 'object365',
        }

        with open(f'text_data/{arch2pkl[clip_arch]}/promptkd_{test_set2pkl[test_set]}_cls_captions_embed.pkl', 'rb') as f:
            self.cls_list, self.cap_list, self.embed_list = pickle.load(f)

        print(len(self.cls_list))
        self.embed_list = np.stack(self.embed_list)
        self.embed_list = torch.from_numpy(self.embed_list).to(device).t()
            
        # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.model_teacher.prompt_learner.reset()
    
    def reset_classnames(self, classnames, *args):
        n_cls = len(classnames)
        n_ctx = self.model_teacher.prompt_learner.n_ctx
        
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        
        prompts = [self.model_teacher.prompt_learner.prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)  # (n_cls, n_tkn)
        
        prompts_neg = [self.model_teacher.prompt_learner.prompt_prefix_neg + " " + name + "." for name in classnames]
        tokenized_prompts_neg = torch.cat([clip.tokenize(p) for p in prompts_neg]).to(self.device)  # (n_cls, n_tkn)
        
        with torch.no_grad():
            embedding = self.token_embedding(tokenized_prompts).type(self.model.dtype)
            embedding_neg = self.token_embedding(tokenized_prompts_neg).type(self.model.dtype)
        
        self.model_teacher.prompt_learner.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.model_teacher.prompt_learner.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        self.model_teacher.prompt_learner.register_buffer("token_prefix2", embedding[:, :1, :])  # SOS
        self.model_teacher.prompt_learner.register_buffer("token_suffix2", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        
        self.model_teacher.prompt_learner.register_buffer("token_prefix_neg", embedding_neg[:, :1, :])  # SOS
        self.model_teacher.prompt_learner.register_buffer("token_suffix_neg", embedding_neg[:, 1 + n_ctx:, :])  # CLS, EOS
        self.model_teacher.prompt_learner.register_buffer("token_prefix2_neg", embedding_neg[:, :1, :])  # SOS
        self.model_teacher.prompt_learner.register_buffer("token_suffix2_neg", embedding_neg[:, 1 + n_ctx:, :])  # CLS, EOS

        self.model_teacher.prompt_learner.n_cls = n_cls
        self.model_teacher.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.model_teacher.tokenized_prompts_neg = tokenized_prompts_neg  # torch.Tensor
        self.name_lens = name_lens

        if hasattr(self, "text_features"):
            delattr(self, "text_features")
            
    def forward(self, image):
        
        text_features, text_features_neg = self.model_teacher(image)
        
        with torch.no_grad():
            image_features, logit_scale = self.model(image)
        
        logits = logit_scale * image_features @ text_features.t()
        
        if image.shape[0] == 1:
            logits_neg = logit_scale * image_features @ text_features_neg.t()
            return logits, logits_neg
        
        img_cap_sims = image_features @ self.embed_list
        top_sims_idx = torch.argsort(img_cap_sims, dim=1, descending=True)[:,:16].flatten(0,1)
        top_sims_idx_unique = torch.unique(top_sims_idx)
        top_sims_caps = self.embed_list[:, top_sims_idx_unique].t()
        logits_neg = logit_scale * top_sims_caps @ text_features_neg.t()

        labels_names = [self.cls_list[i.item()] for i in top_sims_idx]
        labels_names_unique = [self.cls_list[i.item()] for i in top_sims_idx_unique]
        # labels = torch.zeros((len(logits_neg), len(self.model_teacher.prompt_learner.classnames)), dtype=torch.float32).to(logits_neg.device)
        # for i, names in enumerate(labels_names):
        #     for na in names:
        #         labels[i, self.model_teacher.prompt_learner.classnames.index(na)] = 1
                
        # max_k = 0
        
        if self.is_bind:
            for i, k in enumerate(labels_names_unique):
                top_k_values, top_k_indices = torch.topk(logits_neg[i], len(k))
                max_value = top_k_values.max()
                logits_neg[i, top_k_indices] = max_value
                # max_k = max(max_k, len(k))
            for i, k in enumerate(labels_names[::16]):
                top_k_values, top_k_indices = torch.topk(logits[i], len(k))
                max_value = top_k_values.max()
                logits[i, top_k_indices] = max_value
                # max_k = max(max_k, len(k))
            
        # top5_indices = logits.topk(max_k, dim=1)[1]
        # mask = torch.zeros_like(logits)
        # for i in range(logits.shape[0]):
        #     mask[i, top5_indices[i]] = 1
        # logits = torch.where(mask == 1, logits.max(dim=1, keepdim=True)[0].expand_as(logits), logits)

        return logits, logits_neg