import os
import random
from pathlib import Path
from argparse import ArgumentParser
import torch
from transformers import (
    AutoConfig,
    BertForMaskedLM,
    BertTokenizer,
    AutoTokenizer,
    DataCollatorForWholeWordMask,
    Trainer,
    TrainingArguments,
    set_seed,
    TrainerCallback,
)
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import re

# Set GPU if available
if torch.cuda.is_available():
    device = torch.device('cuda')
    n_gpu = torch.cuda.device_count()
    print(f'Number of available GPUs: {n_gpu}')
else:
    device = torch.device('cpu')
    n_gpu = 0
    print('CUDA is not available. Using CPU.')

print('Using device:', device)

def ensure_dir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    return dir_path

parser = ArgumentParser()
parser.add_argument('--data_dir', required=True, type=str)
parser.add_argument('--model_save_dir', required=True, type=str)
parser.add_argument('--cache_dir', default=None, type=str)
parser.add_argument('--vocab_file', default=None, type=str)
args = parser.parse_args()

model_revision = 'main'
model_name = 'allenai/scibert_scivocab_uncased'
cache_dir = ensure_dir(args.cache_dir) if args.cache_dir else None
output_dir = ensure_dir(args.model_save_dir)
vocab_file = args.vocab_file

#set_seed(SEED)

config_kwargs = {
    'cache_dir': cache_dir,
    'revision': model_revision,
    'use_auth_token': None,
}
config = AutoConfig.from_pretrained(model_name, **config_kwargs)

# Create a custom tokenizer
tokenizer = BertTokenizer(vocab_file=vocab_file)

# Check vocabulary size
vocab_size = len(tokenizer)
print(f"HybridTokenizer(MatSciBERT) Vocabulary Size: {vocab_size}")

class MSC_Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.data.items()}
        return item

    def __len__(self):
        return len(self.data['input_ids'])

def load_and_split_data(data_dir, split_ratio=0.15):
    all_data = {'input_ids': [], 'attention_mask': []}
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
    for file_name in tqdm(file_list, desc="Loading data"):
        file_path = os.path.join(data_dir, file_name)
        data = torch.load(file_path)
        for key in all_data:
            all_data[key].extend(data[key])
    
    total_size = len(all_data['input_ids'])
    indices = list(range(total_size))
    random.shuffle(indices)
    
    split_idx = int(total_size * split_ratio)
    val_indices = indices[:split_idx]
    train_indices = indices[split_idx:]
    
    train_data = {key: [all_data[key][i] for i in train_indices] for key in all_data}
    val_data = {key: [all_data[key][i] for i in val_indices] for key in all_data}
    
    return train_data, val_data

train_data, val_data = load_and_split_data(args.data_dir)

train_dataset = MSC_Dataset(train_data)
eval_dataset = MSC_Dataset(val_data)
print(len(train_dataset), len(eval_dataset))

# Load base BERT model and tokenizer
bert_base_model_name = 'allenai/scibert_scivocab_uncased'
bert_base_tokenizer = AutoTokenizer.from_pretrained(bert_base_model_name)
bert_base_model = BertForMaskedLM.from_pretrained(bert_base_model_name)

model = BertForMaskedLM.from_pretrained(
    model_name,
    from_tf=False,
    config=config,
    cache_dir=cache_dir,
    revision=model_revision,
    use_auth_token=None,
)
model.resize_token_embeddings(len(tokenizer))

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    gradient_accumulation_steps=128 // (128 * n_gpu),
    evaluation_strategy='steps',
    save_strategy='steps',
    save_steps=50000,
    eval_steps=50000,
    logging_dir=os.path.join(output_dir, 'logs'),
    logging_steps=128,
    load_best_model_at_end=True,
    warmup_ratio=0.048,
    learning_rate=1e-4,
    weight_decay=1e-2,
    adam_beta1=0.9,
    adam_beta2=0.98,
    adam_epsilon=1e-6,
    max_grad_norm=0.0,
    num_train_epochs=30,
    seed=42,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# Start training
trainer.train()


