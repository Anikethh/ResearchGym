import copy
import torch
import torch.nn as nn
from maple_clip import clip
import pickle
import numpy as np
from tqdm import tqdm 
from maple_clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(arch, n_ctx):
    url = clip._MODELS[arch]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'MaPLe',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": n_ctx}
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model.float()


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class MultiModalPromptLearner(nn.Module):
    def __init__(self, n_ctx, ctx_init, prompt_depth, clip_model, classnames):
        super().__init__()
        n_ctx = n_ctx
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # Default is 1, which is compound shallow prompting
        assert prompt_depth >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = prompt_depth  # max=12, but will create 11 such shared prompts

        ctx_init_neg = ctx_init
        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            print("context init", ctx_init)
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
            
            print("context init neg", ctx_init_neg)
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
        
        self.prompt_prefix = prompt_prefix
        self.prompt_prefix_neg = prompt_prefix_neg
        
        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim, 768)
        
        self.ctx_init_state = ctx_vectors.detach().clone()
        self.ctx = nn.Parameter(ctx_vectors)
        
        self.ctx_init_state_neg = ctx_vectors_neg.detach().clone()
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)

        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)
        
        self.classnames = classnames
        
    def reset(self):

        self.ctx.copy_(self.ctx_init_state) # to be optimized
        self.ctx_neg.copy_(self.ctx_init_state_neg) # to be optimized

    def construct_prompts(self, ctx, prefix, suffix, prefix_neg, ctx_neg, suffix_neg, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

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
        
        prompts, prompts_neg = self.construct_prompts(ctx, prefix, suffix, prefix_neg, ctx_neg, suffix_neg)

        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        return prompts, prompts_neg, self.proj(self.ctx), self.proj(self.ctx_neg), self.compound_prompts_text, visual_deep_prompts   # pass here original, as for visual 768 is required


class CustomCLIP(nn.Module):
    def __init__(self, is_bind, device, test_set, classnames, clip_arch, n_ctx, ctx_init, prompt_depth, freeze_text, freeze_vision, cache_text_features):
        super().__init__()
        self.is_bind = is_bind
        self.n_ctx = n_ctx
        self.ctx_init = ctx_init
        self.prompt_depth = prompt_depth
        self.arch = clip_arch
        self.freeze_text = freeze_text
        self.freeze_vision = freeze_vision
        self.cache_text_features = cache_text_features
        
        clip_model = load_clip_to_cpu(self.arch, self.n_ctx)
        self.prompt_learner = MultiModalPromptLearner(n_ctx, ctx_init, prompt_depth, clip_model, classnames)
        
        self.token_embedding = clip_model.token_embedding
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
            
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

        with open(f'text_data/{arch2pkl[clip_arch]}/maple_{test_set2pkl[test_set]}_cls_captions_embed.pkl', 'rb') as f:
            self.cls_list, self.cap_list, self.embed_list = pickle.load(f)

        print(len(self.cls_list))
        self.embed_list = np.stack(self.embed_list)
        self.embed_list = torch.from_numpy(self.embed_list).to(device).t()
        # self.clip_model = clip_model

    @property
    def device(self):
        return self.text_encoder.text_projection.device

    @property
    def temp(self):
        return 1/self.logit_scale.exp()
    
    def load_pretrained(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        
        for key in state_dict:
            state_dict[key] = state_dict[key].float()

        # Ignore fixed token vectors
        if "prompt_learner.token_prefix" in state_dict:
            del state_dict["prompt_learner.token_prefix"]

        if "prompt_learner.token_suffix" in state_dict:
            del state_dict["prompt_learner.token_suffix"]

        print("Loading weights from {}".format(ckpt_path))
        # set strict=False
        msg = self.load_state_dict(state_dict, strict=False)
        print("Loaded MaPLe weights.")
        print(f"Missing Keys: {msg.missing_keys}")
        print(f"Unexpected keys: {msg.unexpected_keys}")
        
        with torch.no_grad():
            self.prompt_learner.ctx_init_state = state_dict['prompt_learner.ctx']
            self.prompt_learner.ctx_init_state_neg = state_dict['prompt_learner.ctx']
            
        
    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.prompt_learner.reset()
    
    def reset_classnames(self, classnames, *args):
        n_cls = len(classnames)
        
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [self.prompt_learner.prompt_prefix + " " + name + "." for name in classnames]
        prompts_neg = [self.prompt_learner.prompt_prefix_neg + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)  # (n_cls, n_tkn)
        tokenized_prompts_neg = torch.cat([clip.tokenize(p) for p in prompts_neg]).to(self.device)  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = self.token_embedding(tokenized_prompts).type(self.dtype)
            embedding_neg = self.token_embedding(tokenized_prompts_neg).type(self.dtype)

        self.prompt_learner.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.prompt_learner.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx:, :])  # CLS, EOS
        
        self.prompt_learner.register_buffer("token_prefix_neg", embedding_neg[:, :1, :])  # SOS
        self.prompt_learner.register_buffer("token_suffix_neg", embedding_neg[:, 1 + self.n_ctx:, :])  # CLS, EOS

        self.prompt_learner.n_cls = n_cls
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.tokenized_prompts_neg = tokenized_prompts_neg  # torch.Tensor
        self.name_lens = name_lens

        if hasattr(self, "text_features"):
            delattr(self, "text_features")

    def get_text_features(self):
        
        tokenized_prompts = self.tokenized_prompts
        tokenized_prompts_neg = self.tokenized_prompts_neg
        
        prompts, prompts_neg, shared_ctx, shared_ctx_neg, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
        
        # cls2coco = {}
        # nuswide_classname_synonyms = [
        #     ['scale'],
        #     ['tape'],
        #     ['chicken'],
        #     ['hurdle'],
        #     ['game board'],
        #     ['baozi'],
        #     ['target'],
        #     ['plants pot'],
        #     ['toothbrush'],
        #     ['projector'],
        #     ['cheese'],
        #     ['candy'],
        #     ['durian'],
        #     ['dumbbell'],
        #     ['gas stove'],
        #     ['lion'],
        #     ['french fries'],
        #     ['bench'],
        #     ['power outlet'],
        #     ['faucet'],
        #     ['storage box'],
        #     ['crab'],
        #     ['helicopter'],
        #     ['chainsaw'],
        #     ['antelope'],
        #     ['hamimelon'],
        #     ['jellyfish'],
        #     ['kettle'],
        #     ['marker'],
        #     ['clutch'],
        #     ['lettuce'],
        #     ['toilet'],
        #     ['oven'],
        #     ['baseball'],
        #     ['drum'],
        #     ['hanger'],
        #     ['toaster'],
        #     ['bracelet'],
        #     ['cherry'],
        #     ['tissue '],
        #     ['watermelon'],
        #     ['basketball'],
        #     ['cleaning products'],
        #     ['tent'],
        #     ['fire hydrant'],
        #     ['truck'],
        #     ['rice cooker'],
        #     ['microscope'],
        #     ['tablet'],
        #     ['stuffed animal'],
        #     ['golf ball'],
        #     ['CD'],
        #     ['eggplant'],
        #     ['bowl'],
        #     ['desk'],
        #     ['eagle'],
        #     ['slippers'],
        #     ['horn'],
        #     ['carpet'],
        #     ['notepaper'],
        #     ['peach'],
        #     ['saw'],
        #     ['surfboard'],
        #     ['facial cleanser'],
        #     ['corn'],
        #     ['folder'],
        #     ['violin'],
        #     ['watch'],
        #     ['glasses'],
        #     ['shampoo'],
        #     ['pizza'],
        #     ['asparagus'],
        #     ['mushroom'],
        #     ['steak'],
        #     ['suitcase'],
        #     ['table tennis  paddle'],
        #     ['mango'],
        #     ['boots'],
        #     ['necklace'],
        #     ['noodles'],
        #     ['volleyball'],
        #     ['baseball bat'],
        #     ['nuts'],
        #     ['stroller'],
        #     ['pumpkin'],
        #     ['strawberry'],
        #     ['pear'],
        #     ['luggage'],
        #     ['sandals'],
        #     ['liquid soap'],
        #     ['handbag'],
        #     ['flashlight'],
        #     ['trombone'],
        #     ['remote'],
        #     ['shovel'],
        #     ['ladder'],
        #     ['cake'],
        #     ['pomegranate'],
        #     ['clock'],
        #     ['vent'],
        #     ['cymbal'],
        #     ['iron'],
        #     ['okra'],
        #     ['pasta'],
        #     ['lantern'],
        #     ['broom'],
        #     ['fire extinguisher'],
        #     ['snowboard'],
        #     ['rice'],
        #     ['swing'],
        #     ['cow'],
        #     ['van'],
        #     ['tuba'],
        #     ['book'],
        #     ['swan'],
        #     ['lamp'],
        #     ['race car'],
        #     ['egg'],
        #     ['avocado'],
        #     ['guitar'],
        #     ['radio'],
        #     ['sneakers'],
        #     ['eraser'],
        #     ['measuring cup'],
        #     ['sushi'],
        #     ['deer'],
        #     ['parrot'],
        #     ['scissors'],
        #     ['balloon'],
        #     ['tortoise'],
        #     ['meat balls'],
        #     ['cat'],
        #     ['electric drill'],
        #     ['comb'],
        #     ['sausage'],
        #     ['bar soap'],
        #     ['hamburger'],
        #     ['pepper'],
        #     ['router'],
        #     ['spring rolls'],
        #     ['american football'],
        #     ['egg tart'],
        #     ['tape measure'],
        #     ['banana'],
        #     ['gun'],
        #     ['billiards'],
        #     ['picture'],
        #     ['paper towel'],
        #     ['bus'],
        #     ['goldfish'],
        #     ['computer box'],
        #     ['potted plant'],
        #     ['ship'],
        #     ['ambulance'],
        #     ['dog'],
        #     ['medal'],
        #     ['butterfly'],
        #     ['hair dryer'],
        #     ['globe'],
        #     ['french horn'],
        #     ['board eraser'],
        #     ['tea pot'],
        #     ['telephone'],
        #     ['mop'],
        #     ['broccoli'],
        #     ['dolphin'],
        #     ['chair'],
        #     ['hat'],
        #     ['tripod'],
        #     ['traffic light'],
        #     ['hot dog'],
        #     ['pot'],
        #     ['car'],
        #     ['dining table'],
        #     ['crosswalk sign'],
        #     ['tomato'],
        #     ['barrel'],
        #     ['washing machine'],
        #     ['polar bear'],
        #     ['tie'],
        #     ['monkey'],
        #     ['green beans'],
        #     ['cucumber'],
        #     ['cookies'],
        #     ['suv'],
        #     ['brush'],
        #     ['carrot'],
        #     ['tennis racket'],
        #     ['helmet'],
        #     ['sink'],
        #     ['stool'],
        #     ['flower'],
        #     ['radiator'],
        #     ['fishing rod'],
        #     ['Life saver'],
        #     ['lighter'],
        #     ['bread'],
        #     ['radish'],
        #     ['human'],
        #     ['traffic cone'],
        #     ['knife'],
        #     ['grapes'],
        #     ['cellphone'],
        #     ['trophy'],
        #     ['urinal'],
        #     ['cup'],
        #     ['paint brush'],
        #     ['mouse'],
        #     ['soccer'],
        #     ['cutting'],
        #     ['wheelchair'],
        #     ['Accordion'],
        #     ['goose'],
        #     ['red cabbage'],
        #     ['plate'],
        #     ['saxophone'],
        #     ['laptop'],
        #     ['facial mask'],
        #     ['onion'],
        #     ['motorbike'],
        #     ['canned'],
        #     ['lobster'],
        #     ['toiletries'],
        #     ['earphone'],
        #     ['flag'],
        #     ['Bread'],
        #     ['trumpet'],
        #     ['parking meter'],
        #     ['garlic'],
        #     ['skateboard'],
        #     ['pie'],
        #     ['barbell'],
        #     ['yak'],
        #     ['stapler'],
        #     ['tangerine'],
        #     ['zebra'],
        #     ['traffic sign'],
        #     ['bottle'],
        #     ['hotair balloon'],
        #     ['sailboat'],
        #     ['llama'],
        #     ['blackboard'],
        #     ['coffee machine'],
        #     ['flute'],
        #     ['pencil case'],
        #     ['ice cream'],
        #     ['combine with bowl'],
        #     ['kite'],
        #     ['microphone'],
        #     ['fork'],
        #     ['hoverboard'],
        #     ['blender'],
        #     ['skating and skiing shoes'],
        #     ['nightstand'],
        #     ['toothpaste'],
        #     ['poker card'],
        #     ['fan'],
        #     ['orange'],
        #     ['chopsticks'],
        #     ['pig'],
        #     ['bathtub'],
        #     ['glove'],
        #     ['golf club'],
        #     ['refrigerator'],
        #     ['rickshaw'],
        #     ['candle'],
        #     ['mirror'],
        #     ['microwave'],
        #     ['converter'],
        #     ['airplane'],
        #     ['lemon'],
        #     ['head phone'],
        #     ['tricycle'],
        #     ['bear'],
        #     ['backpack'],
        #     ['apple'],
        #     ['trolley'],
        #     ['tong'],
        #     ['papaya'],
        #     ['cello'],
        #     ['camel'],
        #     ['binoculars'],
        #     ['cabbage'],
        #     ['umbrella'],
        #     ['cigar'],
        #     ['pomelo'],
        #     ['cabinet'],
        #     ['keyboard'],
        #     ['horse'],
        #     ['duck'],
        #     ['combine with glove'],
        #     ['pine apple'],
        #     ['potato'],
        #     ['air conditioner'],
        #     ['pliers'],
        #     ['fire truck'],
        #     ['hockey stick'],
        #     ['elephant'],
        #     ['sports car'],
        #     ['toy'],
        #     ['mangosteen'],
        #     ['rabbit'],
        #     ['bicycle'],
        #     ['giraffe'],
        #     ['screwdriver'],
        #     ['spoon'],
        #     ['sheep'],
        #     ['key'],
        #     ['wine glass'],
        #     ['treadmill'],
        #     ['extension cord'],
        #     ['shrimp'],
        #     ['ring'],
        #     ['boat'],
        #     ['green vegetables'],
        #     ['coffee table'],
        #     ['pitaya'],
        #     ['shark'],
        #     ['basket'],
        #     ['wild bird'],
        #     ['carriage'],
        #     ['slide'],
        #     ['fish'],
        #     ['frisbee'],
        #     ['hammer'],
        #     ['printer'],
        #     ['plum'],
        #     ['towel'],
        #     ['camera'],
        #     ['speaker'],
        #     ['pickup truck'],
        #     ['high heels'],
        #     ['bow tie'],
        #     ['pigeon'],
        #     ['coconut'],
        #     ['machinery vehicle'],
        #     ['sofa'],
        #     ['bed'],
        #     ['tennis ball'],
        #     ['dates'],
        #     ['street lights'],
        #     ['paddle'],
        #     ['calculator'],
        #     ['starfish'],
        #     ['chips'],
        #     ['train'],
        #     ['kiwi fruit'],
        #     ['belt'],
        #     ['monitor'],
        #     ['skis'],
        #     ['leather shoes'],
        #     ['sandwich'],
        #     ['Electronic stove and gas stove'],
        #     ['penguin'],
        #     ['surveillance camera'],
        #     ['cue'],
        #     ['scallop'],
        #     ['green onion'],
        #     ['seal'],
        #     ['crane'],
        #     ['donkey'],
        #     ['pen'],
        #     ['donut'],
        #     ['pillow'],
        #     ['trash bin'],
        # ]
        # for c in nuswide_classname_synonyms:
        #     for cc in c:
        #         cls2coco[cc] = c[0]
        #         cc = cc.replace(' ', '')
        #         cls2coco[cc] = c[0]
        # with open('../../prompt_modal_tuning/nips24/cache/train_new_label/test_01_coco_925.pkl', 'rb') as f:
        #     coco_data = pickle.load(f)
        # with open('../../prompt_modal_tuning/nips24/cache/train_new_label/test_01_nuswide_925.pkl', 'rb') as f:
        #     nuswide_data = pickle.load(f)
        # with open('../../prompt_modal_tuning/nips24/cache/train_new_label/test_01_obj_925.pkl', 'rb') as f:
        #     obj_data = pickle.load(f)
        # captions = coco_data[0] + obj_data[0]
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
        #         text_features = self.text_encoder(text_inputs_embedding, text_inputs, deep_compound_prompts_text)
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
        # with open('text_data/vit-b-16/maple_object365_cls_captions_embed.pkl', 'wb') as f:
        #     pickle.dump([cls_list, caption_list, embed_list], f)
        
        
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        text_features_neg = self.text_encoder(prompts_neg, tokenized_prompts_neg, deep_compound_prompts_text)
        text_features_neg = text_features_neg / text_features_neg.norm(dim=-1, keepdim=True)
        
        return text_features, text_features_neg
    

    def get_image_features(self, image):
        
        prompts, prompts_neg, shared_ctx, shared_ctx_neg, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
        
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        image_features_neg = self.image_encoder(image.type(self.dtype), shared_ctx_neg, deep_compound_prompts_vision)
        image_features_neg = image_features_neg / image_features_neg.norm(dim=-1, keepdim=True)
        
        return image_features, image_features_neg


    def forward(self, image, label=None):
        logit_scale = self.logit_scale.exp()

        text_features, text_features_neg = self.get_text_features()
        image_features, image_features_neg = self.get_image_features(image)
        
        logits = logit_scale * image_features @ text_features.t()
        
        if image.shape[0] == 1:
            logits_neg = logit_scale * image_features_neg @ text_features_neg.t()
            return logits, logits_neg

        img_cap_sims = image_features @ self.embed_list
        top_sims_idx = torch.argsort(img_cap_sims, dim=1, descending=True)[:,:16].flatten(0,1)
        top_sims_idx_unique = torch.unique(top_sims_idx)
        top_sims_caps = self.embed_list[:, top_sims_idx_unique].t()
        logits_neg = logit_scale * top_sims_caps @ text_features_neg.t()

        labels_names = [self.cls_list[i.item()] for i in top_sims_idx]
        labels_names_unique = [self.cls_list[i.item()] for i in top_sims_idx_unique]
                
        if self.is_bind:
            for i, k in enumerate(labels_names_unique):
                top_k_values, top_k_indices = torch.topk(logits_neg[i], len(k))
                max_value = (top_k_values.max() - top_k_values).detach()
                logits_neg[i, top_k_indices] += max_value
            for i, k in enumerate(labels_names[::16]):
                top_k_values, top_k_indices = torch.topk(logits[i], len(k))
                max_value = (top_k_values.max() - top_k_values).detach()
                logits[i, top_k_indices] += max_value

        return logits, logits_neg

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
