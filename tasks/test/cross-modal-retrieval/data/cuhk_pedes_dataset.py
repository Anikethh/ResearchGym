import os
import json
import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from PIL import Image
from data.utils import pre_caption
import os.path as op
import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re


def read_json(fpath):
    with open(fpath, 'r') as f:
        return json.load(f)

class cuhk_pedes_retrieval_eval_image(Dataset):
    """加载 fashion-gen 数据集（下游任务）。

    Args:
        root (string): 数据集根目录
        args (object): 参数对象
    """
    def __init__(self, transform, root, ann_root, split='test', max_words=77):
        self.dataset_dir = root
        self.image_root = op.join(self.dataset_dir, 'imgs/')

        self.anno_path = op.join(self.dataset_dir, 'reid_raw.json')
        self._check_before_run()

        self.annos = self._split_anno(self.anno_path, split)

        self.tokenizer = SimpleTokenizer(bpe_path="./data/bpe_simple_vocab_16e6.txt.gz")

        self.transform = transform

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        self.id_to_data = {}

        for anno in self.annos:
            captions_list = anno['captions']
            img_name = anno['file_path']
            object_id = anno['id']

            if object_id not in self.id_to_data:
                self.id_to_data[object_id] = {'images': [], 'captions': []}
            self.id_to_data[object_id]['images'].append(img_name)
            self.id_to_data[object_id]['captions'].extend(captions_list)

        img_id = 0
        txt_id = 0

        for object_id, data in self.id_to_data.items():
            captions = []
            for caption in data['captions']:
                self.text.append(tokenize(caption, tokenizer=self.tokenizer, text_length=max_words, truncate=True))
                captions.append(txt_id)
                self.txt2img[txt_id] = []
                txt_id += 1

            for img_name in data['images']:
                self.image.append(img_name)
                self.img2txt[img_id] = captions
                for cap_id in captions:
                    self.txt2img[cap_id].append(img_id)
                img_id += 1

    
    def _split_anno(self, anno_path: str, split: str):
        test_annos=[]
        annos = read_json(anno_path)
        
        for anno in annos:
            if anno['split'] == split:
                test_annos.append(anno)
        return test_annos

    def __getitem__(self, index):

        image_path = os.path.join(self.image_root, self.image[index])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index    

    def __len__(self):
        return len(self.image)
    
    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not os.path.exists(self.image_root):
            raise RuntimeError("'{}' is not available".format(self.image_root))
        if not os.path.exists(self.anno_path):
            raise RuntimeError("'{}' is not available".format(self.anno_path))
    

class cuhk_pedes_retrieval_eval_text(Dataset):
    """加载 fashion-gen 数据集（下游任务）。

    Args:
        root (string): 数据集根目录
        args (object): 参数对象
    """
    def __init__(self, transform, root, ann_root, split='test', max_words=77):
        self.dataset_dir = root
        self.image_root = op.join(self.dataset_dir, 'imgs/')

        self.anno_path = op.join(self.dataset_dir, 'reid_raw.json')
        self._check_before_run()

        self.annos = self._split_anno(self.anno_path, split)

        self.tokenizer = SimpleTokenizer(bpe_path="./data/bpe_simple_vocab_16e6.txt.gz")

        self.transform = transform

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        img_id = 0
        txt_id = 0

        self.id_to_data = {}

        for anno in self.annos:
            captions_list = anno['captions']
            img_name = anno['file_path']
            object_id = anno['id']

            if object_id not in self.id_to_data:
                self.id_to_data[object_id] = {'images': [], 'captions': []}

            self.id_to_data[object_id]['images'].append(img_name)
            self.id_to_data[object_id]['captions'].extend(captions_list)

        img_id = 0
        txt_id = 0

        for object_id, data in self.id_to_data.items():
            captions = []
            for caption in data['captions']:
                self.text.append(tokenize(caption, tokenizer=self.tokenizer, text_length=max_words, truncate=True))
                captions.append(txt_id)
                self.txt2img[txt_id] = []
                txt_id += 1

            for img_name in data['images']:
                self.image.append(img_name)
                self.img2txt[img_id] = captions
                for cap_id in captions:
                    self.txt2img[cap_id].append(img_id)
                img_id += 1
        
    
    def _split_anno(self, anno_path: str, split: str):
        test_annos=[]
        annos = read_json(anno_path)
        
        for anno in annos:
            if anno['split'] == split:
                test_annos.append(anno)
        return test_annos

    def __getitem__(self, index):
        caption = self.text[index]
        
        return caption, index  

    def __len__(self):
        return len(self.text)
    
    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not os.path.exists(self.image_root):
            raise RuntimeError("'{}' is not available".format(self.image_root))
        if not os.path.exists(self.anno_path):
            raise RuntimeError("'{}' is not available".format(self.anno_path))
        

def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result

@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        
        vocab.pop(-1) # remove last one in vocab(jekyll) to keep vocab_size unchanged
        vocab.extend(['<|mask|>', '<|startoftext|>', '<|endoftext|>']) # vocab_size 49408
        # vocab.extend(['<|startoftext|>', '<|endoftext|>']) # vocab_size 49408
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|mask|>': '<|mask|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|mask\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text