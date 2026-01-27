
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from maple_clip.maple import CustomCLIP as MaPLeCLIP
from clip import load, tokenize
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from data.imagnet_prompts import imagenet_classes
from data.fewshot_datasets import fewshot_datasets
from data.cls_to_names import *
from maple_clip.promptkd import PromptKD

_tokenizer = _Tokenizer()

DOWNLOAD_ROOT='~/.cache/clip'

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, batch_size=None, n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False):
        super().__init__()
        n_cls = len(classnames)
        self.learned_cls = learned_cls
        dtype = clip_model.dtype
        self.dtype = dtype
        self.device = clip_model.visual.conv1.weight.device
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim
        self.batch_size = batch_size

        # self.ctx, prompt_prefix = self.reset_prompt(ctx_dim, ctx_init, clip_model)
        ctx_init_neg = ctx_init

        if ctx_init:
            print("Initializing the contect with given words: [{}]".format(ctx_init))
            ctx_init = ctx_init.replace("_", " ")
            if '[CLS]' in ctx_init:
                ctx_list = ctx_init.split(" ")
                split_idx = ctx_list.index("[CLS]")
                ctx_init = ctx_init.replace("[CLS] ", "")
                ctx_position = "middle"
            else:
                split_idx = None
            self.split_idx = split_idx
            n_ctx = len(ctx_init.split(" "))

            prompt = tokenize(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

            ctx_init_neg = ctx_init_neg.replace("_", " ")
            prompt_neg = tokenize(ctx_init_neg).to(self.device)
            with torch.no_grad():
                embedding_neg = clip_model.token_embedding(prompt_neg).type(dtype)
            ctx_vectors_neg = embedding_neg[0, 1 : 1 + n_ctx, :]
            prompt_prefix_neg = ctx_init_neg
        else:
            print("Random initialization: initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
            
            ctx_vectors_neg = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors_neg, std=0.02)
            prompt_prefix_neg = " ".join(["X"] * n_ctx)
        
        self.prompt_prefix = prompt_prefix
        self.prompt_prefix_neg = prompt_prefix_neg

        print(f'Initial context: "{prompt_prefix}"')
        print(f'Initial context_neg: "{prompt_prefix_neg}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx_init_state = ctx_vectors.detach().clone()
        self.ctx = nn.Parameter(ctx_vectors) # to be optimized

        self.ctx_init_state_neg = ctx_vectors_neg.detach().clone()
        self.ctx_neg = nn.Parameter(ctx_vectors_neg) # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        prompts_neg = [prompt_prefix_neg + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)
        tokenized_prompts_neg = torch.cat([tokenize(p) for p in prompts_neg]).to(self.device)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            embedding_neg = clip_model.token_embedding(tokenized_prompts_neg).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_prefix_neg", embedding_neg[:, :1, :])  # SOS

        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS
        self.register_buffer("token_suffix_neg", embedding_neg[:, 1 + n_ctx :, :])  # CLS, EOS

        self.ctx_init = ctx_init
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        self.ctx_init_neg = ctx_init_neg
        self.tokenized_prompts_neg = tokenized_prompts_neg  # torch.Tensor

        self.name_lens = name_lens
        self.class_token_position = ctx_position
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.classnames = classnames

    def reset(self):

        ctx_vectors = self.ctx_init_state
        self.ctx.copy_(ctx_vectors) # to be optimized

        ctx_vectors_neg = self.ctx_init_state_neg
        self.ctx_neg.copy_(ctx_vectors_neg) # to be optimized

    def reset_classnames(self, classnames, arch):
        self.n_cls = len(classnames)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        prompts_neg = [self.prompt_prefix_neg + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)
        tokenized_prompts_neg = torch.cat([tokenize(p) for p in prompts_neg]).to(self.device)

        clip, _, _ = load(arch, device=self.device, download_root=DOWNLOAD_ROOT)

        with torch.no_grad():
            embedding = clip.token_embedding(tokenized_prompts).type(self.dtype)
            embedding_neg = clip.token_embedding(tokenized_prompts_neg).type(self.dtype)

        self.token_prefix = embedding[:, :1, :]
        self.token_suffix = embedding[:, 1 + self.n_ctx :, :]

        self.token_prefix_neg = embedding_neg[:, :1, :]
        self.token_suffix_neg = embedding_neg[:, 1 + self.n_ctx :, :]

        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts
        self.tokenized_prompts_neg = tokenized_prompts_neg
        self.classnames = classnames

    def forward(self, init=None):

        ctx_neg = self.ctx_neg.unsqueeze(0).expand(self.n_cls, -1, -1)
        prompts_neg = torch.cat([self.token_prefix_neg, ctx_neg, self.token_suffix_neg], dim=-2)

        ctx = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prompts = torch.cat([self.token_prefix, ctx, self.token_suffix], dim=-2)

        return prompts, prompts_neg


class ClipTestTimeTuning(nn.Module):
    def __init__(self, is_bind, device, test_set, classnames, batch_size, criterion='cosine', arch="ViT-L/14",
                        n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False):
        super(ClipTestTimeTuning, self).__init__()
        clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.is_bind = is_bind
        self.image_encoder = clip.visual
        self.text_encoder = TextEncoder(clip)
        self.logit_scale = clip.logit_scale.data
        # prompt tuning
        self.prompt_learner = PromptLearner(clip, classnames, batch_size, n_ctx, ctx_init, ctx_position, learned_cls)
        self.criterion = criterion
        
    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.prompt_learner.reset()

    def reset_classnames(self, classnames, arch):
        self.prompt_learner.reset_classnames(classnames, arch)

    def get_text_features(self):

        text_features = []
        text_features_neg = []

        prompts, prompts_neg = self.prompt_learner()

        tokenized_prompts = self.prompt_learner.tokenized_prompts
        t_features = self.text_encoder(prompts, tokenized_prompts)
        text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        text_features = torch.stack(text_features, dim=0)

        tokenized_prompts_neg = self.prompt_learner.tokenized_prompts_neg
        t_features_neg = self.text_encoder(prompts_neg, tokenized_prompts_neg)
        text_features_neg.append(t_features_neg / t_features_neg.norm(dim=-1, keepdim=True))
        text_features_neg = torch.stack(text_features_neg, dim=0)

        return torch.mean(text_features, dim=0), torch.mean(text_features_neg, dim=0)

    def inference(self, image):
        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))

        text_features, text_features_neg = self.get_text_features()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        logits_neg = logit_scale * image_features @ text_features_neg.t()
        return logits, logits_neg

    def forward(self, input):
        if isinstance(input, Tuple):
            view_0, view_1, view_2 = input
            return self.contrast_prompt_tuning(view_0, view_1, view_2)
        elif len(input.size()) == 2:
            return self.directional_prompt_tuning(input)
        else:
            return self.inference(input)


def get_coop(clip_arch, test_set, device, n_ctx, ctx_init, learned_cls=False, maple="", promptkd="", is_bind=False):
    if test_set in fewshot_datasets:
        classnames = eval("{}_classes".format(test_set.lower()))
    elif test_set == 'bongard':
        if learned_cls:
            classnames = ['X', 'X']
        else:
            classnames = ['True', 'False']
    else:
        classnames = imagenet_classes

    if maple:
        model = MaPLeCLIP(
            is_bind=is_bind,
            device=device,
            test_set=test_set,
            classnames=classnames,
            clip_arch=clip_arch,
            n_ctx=2,
            ctx_init="a_photo_of_a",
            prompt_depth=3,
            freeze_text=True,
            freeze_vision=True,
            cache_text_features=True
        )
    elif promptkd:
        model = PromptKD(is_bind, device, test_set, classnames, clip_arch=clip_arch)
    else:
        model = ClipTestTimeTuning(is_bind, device, test_set, classnames, None, arch=clip_arch, n_ctx=n_ctx, ctx_init=ctx_init, learned_cls=learned_cls)

    return model

